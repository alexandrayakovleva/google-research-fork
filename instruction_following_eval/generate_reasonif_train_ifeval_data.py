import argparse
import json
import os
import random
import re
import sys
from typing import Any, Callable, Optional

try:
  from instruction_following_eval import instructions_registry
except ModuleNotFoundError:
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  from instruction_following_eval import instructions_registry

from datasets import load_dataset


TARGET_INSTRUCTION_IDS = [
    # "reasonif:language:reasoning_language",
    # "reasonif:length_constraint_checkers:number_words",
    # "reasonif:change_case:english_capital",
    # "reasonif:startend:end_checker",
    # "reasonif:detectable_format:json_format",
    # "reasonif:punctuation:no_comma",
    "agent_reasoning:answer_draft_marker",
    "agent_reasoning:sections",
    "agent_reasoning:xml_draft_review",
    "agent_reasoning:two_approaches",
]

DEFAULT_REASONIF_WORD_LIMIT = {
    "aime": 860,
    "amc": 181,
    "arc": 38,
    "gpqa": 392,
    "gsm8k": 52,
}

# PROMPT_TEMPLATE = (
#     "Think step-by-step, and place only your final answer inside the tags "
#     "`<answer>` and `</answer>`. Format your reasoning according to the "
#     "following rule: **{constraint_text}**\n\n"
#     "Here is the question:\n\n"
#     "{question_statement}"
# )
PROMPT_TEMPLATE = (
    "{question_statement}\n\n"
    "<thinking guidelines>\n{constraint_text}\n</thinking guidelines>\n\n"
    "Place the final answer inside the tags `<answer>` and `</answer>`."
)


def _normalize_text(text: str) -> str:
  text = text.replace("\r\n", "\n")
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def _get_first_present(example: dict[str, Any], keys: list[str]) -> Optional[Any]:
  for key in keys:
    if key in example and example[key] is not None:
      return example[key]
  return None


def _format_choice_block(labels: list[str], choices: list[str]) -> str:
  return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def _build_mcq_question(stem: str, choices: list[str]) -> str:
  labels = ["A", "B", "C", "D"][: len(choices)]
  return (
      "Read the following multiple-choice question and select the most "
      "appropriate option.\n"
      f"{stem}\n"
      f"{_format_choice_block(labels, choices)}"
  )


def _extract_gsm8k_question(example: dict[str, Any], rng: random.Random) -> str:
  del rng
  question = _get_first_present(example, ["question", "problem"])
  if not isinstance(question, str) or not question.strip():
    raise ValueError("Missing GSM8K question text.")
  return question.strip()


def _extract_gsm8k_answer(example: dict[str, Any]) -> str:
  answer = _get_first_present(example, ["answer", "final_answer", "target"])
  if not isinstance(answer, str) or not answer.strip():
    raise ValueError("Missing GSM8K answer.")
  return answer.strip()


def _extract_aimo_question(example: dict[str, Any], rng: random.Random) -> str:
  del rng
  question = _get_first_present(example, ["problem", "question", "Question"])
  if not isinstance(question, str) or not question.strip():
    raise ValueError("Missing AIMO question text.")
  return question.strip()


def _extract_aimo_answer(example: dict[str, Any]) -> str:
  answer = _get_first_present(example, ["answer", "Answer", "target"])
  if answer is None:
    raise ValueError("Missing AIMO answer.")
  answer_text = str(answer).strip()
  if not answer_text:
    raise ValueError("Missing AIMO answer.")
  return answer_text


def _extract_arc_question(example: dict[str, Any], rng: random.Random) -> str:
  del rng
  stem = _get_first_present(
      example,
      ["question", "question_stem", "questionStem"],
  )
  if not isinstance(stem, str) or not stem.strip():
    raise ValueError("Missing ARC question stem.")

  choices_obj = example.get("choices")
  choices: list[str] = []
  if isinstance(choices_obj, dict):
    raw_choices = choices_obj.get("text")
    if isinstance(raw_choices, list):
      choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
  elif isinstance(choices_obj, list):
    for choice in choices_obj:
      if isinstance(choice, dict):
        choice_text = _get_first_present(choice, ["text", "label_text", "content"])
        if isinstance(choice_text, str) and choice_text.strip():
          choices.append(choice_text.strip())

  if not choices:
    raise ValueError("Missing ARC choices.")
  return _build_mcq_question(stem.strip(), choices)


def _extract_arc_answer(example: dict[str, Any]) -> str:
  answer_key = _get_first_present(example, ["answerKey", "answer_key", "label"])
  if not isinstance(answer_key, str) or not answer_key.strip():
    raise ValueError("Missing ARC answer key.")
  normalized_key = answer_key.strip()

  choices_obj = example.get("choices")
  if isinstance(choices_obj, dict):
    labels = choices_obj.get("label")
    texts = choices_obj.get("text")
    if isinstance(labels, list) and isinstance(texts, list):
      for label, text in zip(labels, texts):
        if str(label).strip() == normalized_key and str(text).strip():
          return str(text).strip()
  elif isinstance(choices_obj, list):
    for choice in choices_obj:
      if not isinstance(choice, dict):
        continue
      label = _get_first_present(choice, ["label", "key"])
      text = _get_first_present(choice, ["text", "label_text", "content"])
      if (
          isinstance(label, str)
          and label.strip() == normalized_key
          and isinstance(text, str)
          and text.strip()
      ):
        return text.strip()

  return normalized_key


def _extract_gpqa_question(example: dict[str, Any], rng: random.Random) -> str:
  stem = _get_first_present(example, ["Question", "question", "prompt"])
  if not isinstance(stem, str) or not stem.strip():
    raise ValueError("Missing GPQA question stem.")

  correct = _get_first_present(example, ["Correct Answer", "correct_answer"])
  incorrects = [
      _get_first_present(example, ["Incorrect Answer 1", "incorrect_answer_1"]),
      _get_first_present(example, ["Incorrect Answer 2", "incorrect_answer_2"]),
      _get_first_present(example, ["Incorrect Answer 3", "incorrect_answer_3"]),
  ]
  if (
      isinstance(correct, str)
      and correct.strip()
      and all(isinstance(choice, str) and choice.strip() for choice in incorrects)
  ):
    choices = [correct.strip(), *(choice.strip() for choice in incorrects)]
    rng.shuffle(choices)
    return _build_mcq_question(stem.strip(), choices)

  question_text = stem.strip()
  if "A." in question_text and "B." in question_text:
    return question_text
  raise ValueError("Unsupported GPQA schema.")


def _extract_gpqa_answer(example: dict[str, Any]) -> str:
  answer = _get_first_present(
      example,
      ["Correct Answer", "correct_answer", "answer", "Answer"],
  )
  if answer is None:
    raise ValueError("Missing GPQA answer.")
  answer_text = str(answer).strip()
  if not answer_text:
    raise ValueError("Missing GPQA answer.")
  return answer_text


def _mcq_stem(question: str) -> str:
  text = question.strip()
  prefix = (
      "Read the following multiple-choice question and select the most "
      "appropriate option."
  )
  if text.startswith(prefix):
    text = text[len(prefix):].lstrip()
  lines = text.splitlines()
  stem_lines = []
  option_pattern = re.compile(r"^[A-D]\.\s")
  for line in lines:
    if option_pattern.match(line.strip()):
      break
    stem_lines.append(line)
  return _normalize_text("\n".join(stem_lines))


def _default_key_fn(question: str) -> str:
  return _normalize_text(question)


def _mcq_key_fn(question: str) -> str:
  return _mcq_stem(question)


def _allow_all(_: dict[str, Any]) -> bool:
  return True


def _year_at_most(max_year: int) -> Callable[[dict[str, Any]], bool]:
  def _predicate(example: dict[str, Any]) -> bool:
    year = _get_first_present(example, ["Year", "year"])
    try:
      return int(year) <= max_year
    except (TypeError, ValueError):
      return False

  return _predicate


SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "pools": [
            {
                "dataset_id": "openai/gsm8k",
                "candidates": [("main", "train"), (None, "train")],
                "extract_question": _extract_gsm8k_question,
                "extract_answer": _extract_gsm8k_answer,
                "question_key": _default_key_fn,
                "include_example": _allow_all,
                "name": "gsm8k_main",
            },
        ],
    },
    "amc": {
        "pools": [
            {
                "dataset_id": "rawsh/2024_AMC12",
                "candidates": [(None, "train"), ("default", "train")],
                "extract_question": _extract_aimo_question,
                "extract_answer": _extract_aimo_answer,
                "question_key": _default_key_fn,
                "include_example": _allow_all,
                "name": "amc_2024",
            },
        ],
    },
    "aime": {
        "pools": [
            {
                "dataset_id": "lchen001/AIME1983_2024",
                "candidates": [(None, "train"), ("default", "train")],
                "extract_question": _extract_aimo_question,
                "extract_answer": _extract_aimo_answer,
                "question_key": _default_key_fn,
                "include_example": _year_at_most(2021),
                "name": "aime_1983_2021",
            },
            {
                "dataset_id": "rawsh/aime_2025",
                "candidates": [(None, "train"), ("default", "train")],
                "extract_question": _extract_aimo_question,
                "extract_answer": _extract_aimo_answer,
                "question_key": _default_key_fn,
                "include_example": _allow_all,
                "name": "aime_2025",
            },
            {
                "dataset_id": "math-ai/aime26",
                "candidates": [(None, "test"), ("default", "test")],
                "extract_question": _extract_aimo_question,
                "extract_answer": _extract_aimo_answer,
                "question_key": _default_key_fn,
                "include_example": _allow_all,
                "name": "aime_2026",
            },
        ],
    },
    "gpqa": {
        "pools": [
            {
                "dataset_id": "Idavidrein/gpqa",
                "candidates": [
                    ("gpqa_extended", "train"),
                    ("gpqa_main", "train"),
                    ("gpqa_diamond", "train"),
                    ("main", "train"),
                    (None, "train"),
                ],
                "extract_question": _extract_gpqa_question,
                "extract_answer": _extract_gpqa_answer,
                "question_key": _mcq_key_fn,
                "include_example": _allow_all,
                "name": "gpqa_primary",
            },
            {
                "dataset_id": "fingertap/GPQA-Diamond",
                "candidates": [(None, "test"), ("default", "test")],
                "extract_question": _extract_aimo_question,
                "extract_answer": _extract_gpqa_answer,
                "question_key": _mcq_key_fn,
                "include_example": _allow_all,
                "name": "gpqa_diamond_fallback",
            },
        ],
    },
    "arc": {
        "pools": [
            {
                "dataset_id": "allenai/ai2_arc",
                "candidates": [("ARC-Challenge", "train"), ("ARC-Challenge", "validation")],
                "extract_question": _extract_arc_question,
                "extract_answer": _extract_arc_answer,
                "question_key": _mcq_key_fn,
                "include_example": _allow_all,
                "name": "arc_challenge",
            },
        ],
    },
}


def _load_first_available_dataset(
    dataset_id: str,
    candidates: list[tuple[Optional[str], str]],
):
  errors = []
  for config_name, split_name in candidates:
    try:
      dataset = load_dataset(dataset_id, config_name, split=split_name)
      return dataset, config_name, split_name
    except Exception as exc:  # pylint: disable=broad-exception-caught
      errors.append(f"config={config_name!r}, split={split_name!r}: {exc}")
  raise RuntimeError(
      f"Unable to load dataset '{dataset_id}'. Tried: " + " | ".join(errors)
  )


def _is_access_or_availability_error(error: Exception) -> bool:
  text = str(error).lower()
  markers = (
      "gated dataset",
      "ask for access",
      "authorization",
      "forbidden",
      "401",
      "403",
      "not found",
      "couldn't reach",
      "cannot connect",
      "connection",
      "timed out",
  )
  return any(marker in text for marker in markers)


def _build_instruction(
    source: str,
    inst_id: str,
) -> tuple[str, dict[str, Any]]:
  instruction_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
  instruction = instruction_cls(inst_id)
  build_kwargs: dict[str, Any] = {}
  if inst_id == "reasonif:length_constraint_checkers:number_words":
    build_kwargs["num_words"] = DEFAULT_REASONIF_WORD_LIMIT[source]
  description = instruction.build_description(**build_kwargs)
  kwargs = instruction.get_instruction_args() or {}
  return description, kwargs


def _load_test_question_keys(
    test_dataset_path: str,
) -> dict[str, set[str]]:
  with open(test_dataset_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

  keys_by_source: dict[str, set[str]] = {source: set() for source in SOURCE_SPECS}
  for item in test_data:
    source = item.get("source")
    question = item.get("question")
    if source not in SOURCE_SPECS or not isinstance(question, str):
      continue
    key_fn: Callable[[str], str] = SOURCE_SPECS[source]["pools"][0]["question_key"]
    keys_by_source[source].add(key_fn(question))
  return keys_by_source


def _sample_records_for_pool(
    dataset: Any,
    remaining_limit: int,
    rng: random.Random,
    excluded_keys: set[str],
    extract_question: Callable[[dict[str, Any], random.Random], str],
    extract_answer: Callable[[dict[str, Any]], str],
    key_fn: Callable[[str], str],
    include_example: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  examples = list(dataset)
  rng.shuffle(examples)

  records: list[dict[str, Any]] = []
  stats = {
      "requested": remaining_limit,
      "accepted": 0,
      "skipped_duplicate_or_test": 0,
      "skipped_invalid": 0,
      "skipped_filtered_out": 0,
  }

  for example in examples:
    if not include_example(example):
      stats["skipped_filtered_out"] += 1
      continue

    try:
      question = extract_question(example, rng)
      answer = extract_answer(example)
    except ValueError:
      stats["skipped_invalid"] += 1
      continue

    question_key = key_fn(question)
    if question_key in excluded_keys:
      stats["skipped_duplicate_or_test"] += 1
      continue

    stats["accepted"] += 1
    excluded_keys.add(question_key)

    records.append({"question": question, "answer": answer})
    if len(records) >= remaining_limit:
      break

  return records, stats


def generate_records(
    per_source_limit: int,
    seed: int,
    test_dataset_path: str,
    sources: list[str],
    skip_unavailable_sources: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
  rng = random.Random(seed)
  test_question_keys = _load_test_question_keys(test_dataset_path)

  all_records: list[dict[str, Any]] = []
  report: dict[str, dict[str, Any]] = {}

  next_key = 0
  for source in sources:
    spec = SOURCE_SPECS[source]
    source_written = 0
    source_stats = {
        "requested": per_source_limit,
        "written": 0,
        "status": "ok",
        "pools": [],
    }

    for pool in spec["pools"]:
      remaining_limit = per_source_limit - source_written
      if remaining_limit <= 0:
        break
      try:
        dataset, config_name, split_name = _load_first_available_dataset(
            pool["dataset_id"], pool["candidates"]
        )
      except RuntimeError as exc:
        if skip_unavailable_sources and _is_access_or_availability_error(exc):
          source_stats["pools"].append(
              {
                  "name": pool["name"],
                  "dataset_id": pool["dataset_id"],
                  "written": 0,
                  "status": "skipped_unavailable",
                  "reason": str(exc),
              }
          )
          continue
        raise

      pool_records, pool_stats = _sample_records_for_pool(
          dataset=dataset,
          remaining_limit=remaining_limit,
          rng=rng,
          excluded_keys=test_question_keys[source],
          extract_question=pool["extract_question"],
          extract_answer=pool["extract_answer"],
          key_fn=pool["question_key"],
          include_example=pool["include_example"],
      )

      for record in pool_records:
        inst_id = rng.choice(TARGET_INSTRUCTION_IDS)
        description, kwargs = _build_instruction(source, inst_id)
        prompt = PROMPT_TEMPLATE.format(
            constraint_text=description,
            question_statement=record["question"],
        )
        all_records.append(
            {
                "key": next_key,
                "prompt": prompt,
                "answer": record["answer"],
                "instruction_id_list": [inst_id],
                "kwargs": [kwargs],
                "source": source,
            }
        )
        next_key += 1
        source_written += 1

      source_stats["pools"].append(
          {
              "name": pool["name"],
              "dataset_id": pool["dataset_id"],
              "config": config_name,
              "split": split_name,
              "written": len(pool_records),
              "status": "ok",
              **pool_stats,
          }
      )

    source_stats["written"] = source_written
    if source_written == 0 and all(
        pool.get("status") == "skipped_unavailable"
        for pool in source_stats["pools"]
    ):
      source_stats["status"] = "skipped_unavailable"
    report[source] = source_stats

  return all_records, report


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Create IF-Eval style JSONL training data with one sampled ReasonIF "
          "instruction per question."
      )
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Output JSONL file path.",
  )
  parser.add_argument(
      "--per_source_limit",
      type=int,
      default=1200,
      help="Maximum number of retained examples per source dataset.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for dataset shuffling and instruction sampling.",
  )
  parser.add_argument(
      "--test_dataset_path",
      type=str,
      default="/Users/sasha/repos/reasonIF/data/reasonIF_dataset.json",
      help="Path to the original ReasonIF test JSON used for exclusion.",
  )
  parser.add_argument(
      "--sources",
      nargs="+",
      default=["gsm8k", "amc", "aime", "gpqa", "arc"],
      choices=sorted(SOURCE_SPECS.keys()),
      help="Sources to include.",
  )
  parser.add_argument(
      "--skip_unavailable_sources",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Skip gated or unavailable datasets instead of failing the whole run. "
          "Use --no-skip-unavailable-sources to fail fast."
      ),
  )
  args = parser.parse_args()

  records, report = generate_records(
      per_source_limit=args.per_source_limit,
      seed=args.seed,
      test_dataset_path=args.test_dataset_path,
      sources=args.sources,
      skip_unavailable_sources=args.skip_unavailable_sources,
  )

  with open(args.output, "w", encoding="utf-8") as f:
    for row in records:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  print(f"Wrote {len(records)} samples to {args.output}")
  print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
