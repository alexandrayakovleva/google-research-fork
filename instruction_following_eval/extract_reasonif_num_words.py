# coding=utf-8
"""Extract ReasonIF rows that use the number_words instruction."""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "reasonif_train_dataset_guidelines_scores.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "reasonif_train_num_words.jsonl"
)
NUMBER_WORDS_SUFFIX = "number_words"


def _has_number_words_instruction(row: dict) -> bool:
  instruction_ids = row.get("instruction_id_list", [])
  if not isinstance(instruction_ids, list):
    return False
  return any(str(instruction_id).split(":")[-1] == NUMBER_WORDS_SUFFIX
             for instruction_id in instruction_ids)


def extract_num_words(input_path: Path, output_path: Path) -> Counter:
  """Writes number_words rows from input_path to output_path unchanged."""
  counts = Counter()
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

        if not _has_number_words_instruction(row):
          continue

        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[str(row.get("source", ""))] += 1

  return counts


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Extract ReasonIF number_words rows into a separate JSONL file."
  )
  parser.add_argument(
      "--input",
      type=Path,
      default=DEFAULT_INPUT,
      help=f"Input JSONL path. Defaults to {DEFAULT_INPUT}.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=DEFAULT_OUTPUT,
      help=f"Output JSONL path. Defaults to {DEFAULT_OUTPUT}.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  counts = extract_num_words(args.input, args.output)
  total = sum(counts.values())
  print(f"Wrote {total} rows to {args.output}")
  for source in sorted(counts):
    print(f"{source}: {counts[source]}")


if __name__ == "__main__":
  main()
