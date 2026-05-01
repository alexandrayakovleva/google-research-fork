# coding=utf-8
"""Validate ReasonIF data-collection runs against a ReasonIF JSONL dataset."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def _extract_final_answer(answer_text: str, source: str) -> str:
  """Normalize an answer using reasonIF/src/eval_utils.py logic."""
  answer_text = answer_text.strip()
  if source in ("gsm8k", "amc", "aime"):
    matches = re.findall(r"([+-]?\d*\.?\d+)", answer_text)
    if matches:
      answer = matches[-1].strip()
      try:
        return str(int(answer))
      except ValueError:
        return str(answer)
    return answer_text

  if source in ("arc", "gpqa"):
    single_letters = re.findall(r"[ABCD]", answer_text)
    if single_letters:
      return single_letters[0]
    return answer_text.strip()

  raise ValueError(f"Unsupported source: {source}")


def _read_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
      if not line.strip():
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
  return rows


def _load_test_rows(test_jsonl: Path) -> dict[str, dict]:
  rows = _read_jsonl(test_jsonl)
  by_prompt = {}
  duplicates = []
  for row in rows:
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
      raise ValueError(f"Dataset row has no string prompt: {row}")
    if prompt in by_prompt:
      duplicates.append(prompt)
    by_prompt[prompt] = row
  if duplicates:
    raise ValueError(f"Found {len(duplicates)} duplicate prompt(s) in {test_jsonl}")
  return by_prompt


def _single_instruction_id(row: dict) -> str:
  instruction_ids = row.get("instruction_id_list")
  if not isinstance(instruction_ids, list) or len(instruction_ids) != 1:
    raise ValueError(f"Expected exactly one instruction_id in row: {row}")
  return str(instruction_ids[0])


def _prediction_from_response(response: str, source: str) -> str | None:
  match = ANSWER_RE.search(response)
  if match is None:
    return None
  return _extract_final_answer(match.group(1), source)


def _normalize_gold_answer(answer: object, source: str) -> str:
  return _extract_final_answer(str(answer), source)


def _score_task(response: str, test_row: dict) -> tuple[float, str | None, str]:
  source = test_row.get("source")
  if not isinstance(source, str) or not source:
    raise ValueError(f"Matched test row is missing source: {test_row}")
  if "answer" not in test_row:
    raise ValueError(f"Matched test row is missing answer: {test_row}")

  prediction = _prediction_from_response(response, source)
  gold = _normalize_gold_answer(test_row["answer"], source)
  if prediction is None:
    return 0.0, None, gold
  return float(prediction == gold), prediction, gold


def _default_output_path(runs_dir: Path) -> Path:
  if runs_dir.name == "runs":
    return runs_dir.parent / f"{runs_dir.parent.name}.txt"
  return runs_dir.with_suffix(".txt")


def _format_average(value: float | None) -> str:
  return "nan" if value is None else f"{value:.6f}"


def _format_std(values) -> str:
  values = list(values)
  if not values:
    return "nan"
  return f"{pstdev(values):.6f}"


def _format_se(values) -> str:
  values = list(values)
  if not values:
    return "nan"
  return f"{pstdev(values) / math.sqrt(len(values)):.6f}"


def validate(test_jsonl: Path, runs_dir: Path) -> tuple[str, Path]:
  by_prompt = _load_test_rows(test_jsonl)
  response_paths = sorted(runs_dir.glob("*/response.jsonl"))
  if not response_paths:
    raise ValueError(f"No */response.jsonl files found under {runs_dir}")

  rows = []
  unmatched = []
  missing_score_before = []
  for response_path in response_paths:
    payloads = _read_jsonl(response_path)
    if len(payloads) != 1:
      raise ValueError(f"Expected exactly one JSON object in {response_path}, got {len(payloads)}")
    payload = payloads[0]
    prompt = payload.get("prompt")
    if prompt not in by_prompt:
      unmatched.append(str(response_path))
      continue

    test_row = by_prompt[prompt]
    instruction_id = _single_instruction_id(test_row)
    if _single_instruction_id(payload) != instruction_id:
      raise ValueError(
          f"Instruction mismatch for {response_path}: "
          f"run={payload.get('instruction_id_list')} test={test_row.get('instruction_id_list')}"
      )

    score_before = payload.get("score_before")
    if score_before is None:
      missing_score_before.append(str(response_path))
      score_before_float = 0.0
    else:
      score_before_float = float(score_before)

    task_score, prediction, gold = _score_task(str(payload.get("response", "")), test_row)
    rows.append({
        "response_path": response_path,
        "instruction_id": instruction_id,
        "score_before": score_before_float,
        "task_score": task_score,
        "prediction": prediction,
        "gold": gold,
        "source": test_row.get("source"),
    })

  if not rows:
    raise ValueError("No response rows matched the test dataset.")

  by_instruction: dict[str, list[dict]] = defaultdict(list)
  for row in rows:
    by_instruction[row["instruction_id"]].append(row)

  lines = []
  lines.append(f"test_jsonl: {test_jsonl}")
  lines.append(f"runs_dir: {runs_dir}")
  lines.append(f"response_files: {len(response_paths)}")
  lines.append(f"matched: {len(rows)}")
  lines.append(f"unmatched: {len(unmatched)}")
  lines.append(f"missing_score_before: {len(missing_score_before)}")
  lines.append("")
  lines.append("overall")
  score_before_values = [row["score_before"] for row in rows]
  task_score_values = [row["task_score"] for row in rows]
  lines.append(f"score_before_avg: {_format_average(mean(score_before_values))}")
  lines.append(f"score_before_std: {_format_std(score_before_values)}")
  lines.append(f"score_before_se: {_format_se(score_before_values)}")
  lines.append(f"task_score_avg: {_format_average(mean(task_score_values))}")
  lines.append(f"task_score_std: {_format_std(task_score_values)}")
  lines.append(f"task_score_se: {_format_se(task_score_values)}")
  lines.append("")
  lines.append("by_instruction_id")
  for instruction_id in sorted(by_instruction):
    group = by_instruction[instruction_id]
    group_score_before_values = [row["score_before"] for row in group]
    group_task_score_values = [row["task_score"] for row in group]
    lines.append(
        f"{instruction_id}\t"
        f"n={len(group)}\t"
        f"score_before_avg={_format_average(mean(group_score_before_values))}\t"
        f"score_before_std={_format_std(group_score_before_values)}\t"
        f"score_before_se={_format_se(group_score_before_values)}\t"
        f"task_score_avg={_format_average(mean(group_task_score_values))}\t"
        f"task_score_std={_format_std(group_task_score_values)}\t"
        f"task_score_se={_format_se(group_task_score_values)}"
    )

  if unmatched:
    lines.append("")
    lines.append("unmatched_response_jsonl")
    lines.extend(unmatched)

  if missing_score_before:
    lines.append("")
    lines.append("missing_score_before_response_jsonl")
    lines.extend(missing_score_before)

  output_path = _default_output_path(runs_dir)
  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return "\n".join(lines), output_path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test-jsonl", type=Path, required=True)
  parser.add_argument("--runs-dir", type=Path, required=True)
  args = parser.parse_args()

  report, output_path = validate(args.test_jsonl, args.runs_dir)
  print(report)
  print(f"\nSaved scores to: {output_path}")


if __name__ == "__main__":
  main()
