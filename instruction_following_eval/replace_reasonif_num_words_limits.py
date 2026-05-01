# coding=utf-8
"""Create a ReasonIF number_words copy with expanded word limits."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "data" / "reasonif_train_num_words.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "reasonif_train_num_words_longer_limits.jsonl"
)

WORD_LIMITS = {
    "aime": (860, 860),
    "amc": (181, 381),
    "arc": (38, 238),
    "gpqa": (392, 692),
    "gsm8k": (52, 252),
}
PROMPT_LIMIT_RE = re.compile(r"When reasoning, use less than \d+ words\.")


def _number_words_kwargs(row: dict) -> list[dict]:
  return [
      kwarg for kwarg in row.get("kwargs", [])
      if isinstance(kwarg, dict) and "num_words" in kwarg
  ]


def replace_limits(
    input_path: Path,
    output_path: Path,
    rng: random.Random,
) -> Counter:
  """Writes output_path with updated num_words limits and prompt text."""
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

        source = row.get("source")
        if source not in WORD_LIMITS:
          raise ValueError(
              f"Unsupported source at {input_path}:{line_no}: {source!r}")

        kwargs = _number_words_kwargs(row)
        if len(kwargs) != 1:
          raise ValueError(
              f"Expected exactly one num_words kwarg at {input_path}:{line_no}, "
              f"got {len(kwargs)}")

        lower, upper = WORD_LIMITS[source]
        num_words = rng.randint(lower, upper)
        kwargs[0]["num_words"] = num_words

        prompt = row.get("prompt")
        if not isinstance(prompt, str):
          raise ValueError(f"Missing prompt at {input_path}:{line_no}")
        updated_prompt, replacements = PROMPT_LIMIT_RE.subn(
            f"When reasoning, use less than {num_words} words.",
            prompt,
            count=1,
        )
        if replacements != 1:
          raise ValueError(
              f"Expected one prompt word-limit guideline at "
              f"{input_path}:{line_no}, got {replacements}")
        row["prompt"] = updated_prompt

        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[source] += 1

  return counts


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Copy ReasonIF number_words rows and replace num_words limits with "
          "expanded randomized ranges."
      )
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
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="Optional random seed for reproducible limits.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  counts = replace_limits(args.input, args.output, random.Random(args.seed))
  total = sum(counts.values())
  print(f"Wrote {total} rows to {args.output}")
  for source in sorted(counts):
    lower, upper = WORD_LIMITS[source]
    if lower == upper:
      print(f"{source}: {counts[source]} rows, limit {lower}")
    else:
      print(f"{source}: {counts[source]} rows, range [{lower}, {upper}]")


if __name__ == "__main__":
  main()
