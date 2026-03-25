import argparse
import json
import os
import random
import re
import sys
from typing import Any

try:
  from instruction_following_eval import instructions_registry
except ModuleNotFoundError:
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  from instruction_following_eval import instructions_registry

from datasets import load_dataset


TARGET_INSTRUCTION_ID = "detectable_format:text_splitter"
MIN_NUM_PARAGRAPHS = 10
MAX_NUM_PARAGRAPHS = 25

_START_ARTICLE = "_START_ARTICLE_"
_START_SECTION = "_START_SECTION_"
_START_PARAGRAPH = "_START_PARAGRAPH_"
_NEWLINE = "_NEWLINE_"


def _normalize_whitespace(text: str) -> str:
  text = text.replace(_NEWLINE, " ")
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def _extract_paragraphs(wiki40b_text: str) -> list[str]:
  """Extract normalized article paragraphs from a wiki40b article."""
  if not isinstance(wiki40b_text, str) or not wiki40b_text.strip():
    return []

  chunks = wiki40b_text.split(_START_PARAGRAPH)
  paragraphs: list[str] = []
  for chunk in chunks[1:]:
    # Stop paragraph content at the next structural marker if present.
    for marker in (_START_SECTION, _START_ARTICLE):
      if marker in chunk:
        chunk = chunk.split(marker, 1)[0]
    paragraph = _normalize_whitespace(chunk)
    if paragraph:
      paragraphs.append(paragraph)
  return paragraphs


def _merge_paragraphs(paragraphs: list[str]) -> str:
  return " ".join(paragraphs).strip()


def _build_instruction(
    merged_text: str, num_paragraphs: int
) -> tuple[str, dict[str, Any]]:
  instruction_cls = instructions_registry.INSTRUCTION_DICT[TARGET_INSTRUCTION_ID]
  instruction = instruction_cls(TARGET_INSTRUCTION_ID)
  description = instruction.build_description(
      text=merged_text,
      num_paragraphs=num_paragraphs,
  )
  kwargs = instruction.get_instruction_args() or {}
  return description, kwargs


def generate_records(
    dataset: Any,
    num_samples: int,
    seed: int,
    min_chars: int,
    max_tries_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
  rng = random.Random(seed)
  shuffled_dataset = dataset.shuffle(seed=seed)

  records: list[dict[str, Any]] = []
  source_idx = 0
  stats = {
      "skipped_missing_text": 0,
      "skipped_too_short": 0,
  }

  max_tries = num_samples * max_tries_multiplier

  for example in shuffled_dataset:
    if len(records) >= num_samples or source_idx >= max_tries:
      break

    raw_text = example.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
      stats["skipped_missing_text"] += 1
      source_idx += 1
      continue

    paragraphs = _extract_paragraphs(raw_text)
    merged_text = _merge_paragraphs(paragraphs)
    if len(merged_text) < min_chars:
      stats["skipped_too_short"] += 1
      source_idx += 1
      continue

    num_paragraphs = rng.randint(MIN_NUM_PARAGRAPHS, MAX_NUM_PARAGRAPHS)
    prompt, kwargs = _build_instruction(merged_text, num_paragraphs)

    records.append(
        {
            "key": source_idx,
            "prompt": prompt,
            "instruction_id_list": [TARGET_INSTRUCTION_ID],
            "kwargs": [kwargs],
            "source_wikidata_id": example.get("wikidata_id"),
            "source_num_paragraphs": len(paragraphs),
        }
    )
    source_idx += 1

  if len(records) < num_samples:
    raise RuntimeError(
        "Only generated "
        f"{len(records)} samples out of requested {num_samples}. "
        f"Increase max_tries_multiplier or relax filters."
    )

  return records, stats


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Create IF-Eval style text-splitter prompts from Wikipedia."
  )
  parser.add_argument(
      "--hf_dataset",
      type=str,
      default="google/wiki40b",
      help="HF dataset id.",
  )
  parser.add_argument(
      "--hf_config",
      type=str,
      default="en",
      help="HF dataset config name.",
  )
  parser.add_argument(
      "--split",
      type=str,
      default="train",
      help="HF split name.",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=5000,
      help="Number of prompts to generate.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed.",
  )
  parser.add_argument(
      "--min_chars",
      type=int,
      default=6000,
      help="Minimum merged text length in characters.",
  )
  parser.add_argument(
      "--max_tries_multiplier",
      type=int,
      default=50,
      help="Maximum source examples scanned is num_samples * this value.",
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Output JSONL file path.",
  )
  args = parser.parse_args()

  dataset = load_dataset(
      args.hf_dataset,
      args.hf_config,
      split=args.split,
  )

  records, stats = generate_records(
      dataset=dataset,
      num_samples=args.num_samples,
      seed=args.seed,
      min_chars=args.min_chars,
      max_tries_multiplier=args.max_tries_multiplier,
  )

  with open(args.output, "w", encoding="utf-8") as f:
    for row in records:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  print(
      f"Wrote {len(records)} samples to {args.output}. "
      f"Skipped missing text: {stats['skipped_missing_text']}, "
      f"too short: {stats['skipped_too_short']}."
  )


if __name__ == "__main__":
  main()
