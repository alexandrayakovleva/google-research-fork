# coding=utf-8
"""Replace number_words rows in a ReasonIF dataset with fixed rows."""

import argparse
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT = DATA_DIR / "reasonif_train_dataset_guidelines_scores.jsonl"
DEFAULT_FIXED_NUM_WORDS = DATA_DIR / "reasonif_train_num_words_longer_limits.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "reasonif_train_dataset_guidelines_scores_longer_limits.jsonl"
NUMBER_WORDS_SUFFIX = "number_words"


def _has_number_words_instruction(row: dict) -> bool:
  instruction_ids = row.get("instruction_id_list", [])
  if not isinstance(instruction_ids, list):
    return False
  return any(str(instruction_id).split(":")[-1] == NUMBER_WORDS_SUFFIX
             for instruction_id in instruction_ids)


def _read_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as input_file:
    for line_no, line in enumerate(input_file, 1):
      if not line.strip():
        continue
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
  return rows


def _fixed_rows_by_key(path: Path) -> dict[object, dict]:
  rows_by_key = {}
  duplicate_keys = []
  for row in _read_jsonl(path):
    if not _has_number_words_instruction(row):
      raise ValueError(f"Fixed row is not a number_words task: {row}")
    key = row.get("key")
    if key in rows_by_key:
      duplicate_keys.append(key)
    rows_by_key[key] = row
  if duplicate_keys:
    raise ValueError(
        f"Found duplicate key(s) in {path}: {duplicate_keys[:10]}")
  return rows_by_key


def replace_num_words_rows(
    input_path: Path,
    fixed_num_words_path: Path,
    output_path: Path,
) -> Counter:
  """Writes a full dataset copy with number_words rows replaced by key."""
  fixed_rows = _fixed_rows_by_key(fixed_num_words_path)
  counts = Counter()
  seen_fixed_keys = set()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with input_path.open(encoding="utf-8") as input_file:
    with output_path.open("w", encoding="utf-8") as output_file:
      for line_no, line in enumerate(input_file, 1):
        if not line.strip():
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError as exc:
          raise ValueError(
              f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc

        if _has_number_words_instruction(row):
          key = row.get("key")
          if key not in fixed_rows:
            raise ValueError(
                f"Missing fixed number_words row for key {key!r} "
                f"at {input_path}:{line_no}")
          row = fixed_rows[key]
          seen_fixed_keys.add(key)
          counts["replaced"] += 1
        else:
          counts["kept"] += 1

        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

  unused_fixed_keys = set(fixed_rows) - seen_fixed_keys
  if unused_fixed_keys:
    sample = sorted(unused_fixed_keys, key=str)[:10]
    raise ValueError(
        f"{len(unused_fixed_keys)} fixed row(s) were not used. "
        f"Sample keys: {sample}")

  return counts


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Create a full ReasonIF dataset copy, replacing only number_words "
          "rows with fixed rows from a separate JSONL file."
      )
  )
  parser.add_argument(
      "--input",
      type=Path,
      default=DEFAULT_INPUT,
      help=f"Full input dataset JSONL. Defaults to {DEFAULT_INPUT}.",
  )
  parser.add_argument(
      "--fixed-num-words",
      type=Path,
      default=DEFAULT_FIXED_NUM_WORDS,
      help=(
          "JSONL containing fixed number_words rows. "
          f"Defaults to {DEFAULT_FIXED_NUM_WORDS}."
      ),
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=DEFAULT_OUTPUT,
      help=f"Output full dataset JSONL. Defaults to {DEFAULT_OUTPUT}.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  counts = replace_num_words_rows(
      args.input,
      args.fixed_num_words,
      args.output,
  )
  total = counts["kept"] + counts["replaced"]
  print(f"Wrote {total} rows to {args.output}")
  print(f"Kept {counts['kept']} non-number_words rows")
  print(f"Replaced {counts['replaced']} number_words rows")


if __name__ == "__main__":
  main()
