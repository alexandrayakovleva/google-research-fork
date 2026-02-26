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


TARGET_INSTRUCTION_IDS = [
    "change_case:capital_word_frequency",
    "combination:repeat_prompt",
    "keywords:forbidden_words",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "startend:end_checker"
]

# Conservative secret-like patterns to redact from source prompts.
_SECRET_PATTERNS = [
    r"hf_[A-Za-z0-9]{20,}",  # Hugging Face tokens
    r"sk-[A-Za-z0-9]{20,}",  # OpenAI-style keys
    r"ysk-[A-Za-z0-9]{20,}",  # observed key variant in scraped data
    r"AIza[0-9A-Za-z\-_]{20,}",  # Google API keys
]
_SECRET_REGEX = [re.compile(p) for p in _SECRET_PATTERNS]


def _contains_secret(text: str) -> bool:
  """Return True if text appears to contain a secret token."""
  return any(regex.search(text) for regex in _SECRET_REGEX)


def _extract_prompt(example: dict[str, Any]) -> str:
  """Extract user prompt, prioritizing Infinity-Instruct style schema."""
  # Infinity-Instruct commonly stores dialogs in:
  # conversations: [{"from": "human", "value": "..."} ...]
  conversations = example.get("conversations")
  if isinstance(conversations, list):
    for turn in conversations:
      if not isinstance(turn, dict):
        continue
      speaker = str(turn.get("from", turn.get("role", ""))).lower()
      if speaker in {"human", "user"}:
        value = turn.get("value", turn.get("content"))
        if isinstance(value, str) and value.strip():
          return value.strip()

  raise ValueError("No usable prompt field found in dataset example.")


def _is_conflict(a: str, b: str, conflicts: dict[str, set[str]]) -> bool:
  return b in conflicts.get(a, set()) or a in conflicts.get(b, set())


def _sample_instruction_ids(rng: random.Random) -> list[str]:
  """Sample 1-2 instruction IDs from the target set, conflict-safe."""
  # n = rng.choice((1, 2))
  n = 1
  if n == 1:
    return [rng.choice(TARGET_INSTRUCTION_IDS)]

  conflicts = instructions_registry.conflict_make(
      {k: set(v) for k, v in instructions_registry.INSTRUCTION_CONFLICTS.items()}
  )
  candidates = TARGET_INSTRUCTION_IDS[:]
  rng.shuffle(candidates)
  chosen: list[str] = []
  for inst_id in candidates:
    if all(not _is_conflict(inst_id, existing, conflicts) for existing in chosen):
      chosen.append(inst_id)
    if len(chosen) == 2:
      break

  if not chosen:
    return [rng.choice(TARGET_INSTRUCTION_IDS)]
  return chosen


def _build_instruction(
    inst_id: str, base_prompt: str
) -> tuple[str, dict[str, Any]]:
  instruction_cls = instructions_registry.INSTRUCTION_DICT[inst_id]
  instruction = instruction_cls(inst_id)
  build_kwargs: dict[str, Any] = {}
  if inst_id == "combination:repeat_prompt":
    build_kwargs["prompt_to_repeat"] = base_prompt
  text = instruction.build_description(**build_kwargs)
  kwargs = instruction.get_instruction_args() or {}
  return text, kwargs


def generate_records(
    dataset: Any,
    num_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
  rng = random.Random(seed)
  records: list[dict[str, Any]] = []
  source_idx = 0
  skipped_for_secrets = 0

  for example in dataset:
    if len(records) >= num_samples:
      break
    try:
      base_prompt = _extract_prompt(example)
    except ValueError:
      source_idx += 1
      continue

    if _contains_secret(base_prompt):
      skipped_for_secrets += 1
      source_idx += 1
      continue

    instruction_ids = _sample_instruction_ids(rng)
    descriptions = []
    kwargs_list = []
    for inst_id in instruction_ids:
      description, kwargs = _build_instruction(inst_id, base_prompt)
      descriptions.append(description)
      kwargs_list.append(kwargs)

    prompt = base_prompt + "\n\n" + " ".join(descriptions)
    records.append(
        {
            "key": source_idx,
            "prompt": prompt,
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs_list,
        }
    )
    source_idx += 1

  if len(records) < num_samples:
    raise RuntimeError(
        f"Only generated {len(records)} samples; requested {num_samples}."
    )
  return records, skipped_for_secrets


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Create IF-Eval style prompts from InfinityInstruct Gen."
  )
  parser.add_argument(
      "--hf_dataset",
      type=str,
      default="BAAI/Infinity-Instruct",
      help="HF dataset id.",
  )
  parser.add_argument(
      "--hf_config",
      type=str,
      default="Gen",
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
      default=25000,
      help="Number of prompts to generate.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for instruction sampling.",
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

  records, skipped_for_secrets = generate_records(
      dataset=dataset,
      num_samples=args.num_samples,
      seed=args.seed,
  )

  with open(args.output, "w", encoding="utf-8") as f:
    for row in records:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")

  print(
      f"Wrote {len(records)} samples to {args.output}. "
      f"Skipped {skipped_for_secrets} prompts containing secret-like tokens."
  )


if __name__ == "__main__":
  main()
