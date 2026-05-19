# coding=utf-8
"""Replace selected ReasonIF rows with AgentReasoningXmlDraftReview."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
  from instruction_following_eval import instructions_registry
except ModuleNotFoundError:
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  from instruction_following_eval import instructions_registry


TARGET_REASONING_INSTRUCTION_ID = "agent_reasoning:xml_draft_review"

DEFAULT_INPUTS = (
    "/Users/sasha/data/s2_ifeval_3cat/train_input_data.jsonl",
    "/Users/sasha/data/s2_ifeval_3cat/val_input_data.jsonl",
)

THINKING_GUIDELINES_RE = re.compile(
    r"<thinking guidelines>\n.*?\n</thinking guidelines>",
    flags=re.DOTALL,
)


def _build_target_instruction() -> tuple[str, dict[str, Any]]:
  instruction_cls = instructions_registry.INSTRUCTION_DICT[
      TARGET_REASONING_INSTRUCTION_ID
  ]
  instruction = instruction_cls(TARGET_REASONING_INSTRUCTION_ID)
  description = instruction.build_description()
  kwargs = instruction.get_instruction_args() or {}
  return description, kwargs


def _format_thinking_guidelines(constraint_text: str) -> str:
  return (
      "<thinking guidelines>\n"
      f"{constraint_text}\n"
      "</thinking guidelines>"
  )


def _replace_thinking_guidelines(prompt: str, constraint_text: str) -> str:
  replacement = _format_thinking_guidelines(constraint_text)
  existing_blocks = list(THINKING_GUIDELINES_RE.finditer(prompt))
  if len(existing_blocks) > 1:
    raise ValueError(
        "Expected at most one <thinking guidelines>...</thinking guidelines> "
        "block in prompt."
    )
  if not existing_blocks:
    return f"{prompt.rstrip()}\n\n{replacement}"

  updated_prompt, num_replacements = THINKING_GUIDELINES_RE.subn(
      replacement, prompt, count=1
  )
  if num_replacements == 1:
    return updated_prompt
  raise ValueError("Unable to replace existing thinking guidelines block.")


def _default_output_path(input_path: Path, suffix: str) -> Path:
  return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def convert_file(input_path: Path, output_path: Path) -> int:
  description, kwargs = _build_target_instruction()
  count = 0

  with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
      "w", encoding="utf-8"
  ) as output_file:
    for line_number, line in enumerate(input_file, start=1):
      if not line.strip():
        continue

      row = json.loads(line)
      prompt = row.get("prompt")
      if not isinstance(prompt, str):
        raise ValueError(f"{input_path}:{line_number}: missing string prompt.")

      row["prompt"] = _replace_thinking_guidelines(prompt, description)
      row["reasoning_instruction_id_list"] = TARGET_REASONING_INSTRUCTION_ID
      row["reasoning_kwargs"] = kwargs

      output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
      count += 1

  return count


def convert_in_place(input_path: Path) -> int:
  with tempfile.NamedTemporaryFile(
      "w",
      encoding="utf-8",
      delete=False,
      dir=input_path.parent,
      prefix=f".{input_path.name}.",
      suffix=".tmp",
  ) as output_file:
    temp_path = Path(output_file.name)

  try:
    count = convert_file(input_path, temp_path)
    temp_path.replace(input_path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise

  return count


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Replace or add the <thinking guidelines> reasoning instruction in "
          "selected ReasonIF JSONL files with "
          "agent_reasoning:xml_draft_review. The task instruction_id_list and "
          "kwargs are preserved."
      )
  )
  parser.add_argument(
      "inputs",
      nargs="*",
      default=DEFAULT_INPUTS,
      help="Input selected JSONL files. Defaults to the train/val selected files.",
  )
  parser.add_argument(
      "--suffix",
      default="_xml_draft_review",
      help="Suffix for output files when not using --in_place.",
  )
  parser.add_argument(
      "--in_place",
      action="store_true",
      help="Rewrite each input file in place instead of creating suffixed copies.",
  )
  args = parser.parse_args()

  for input_name in args.inputs:
    input_path = Path(input_name)
    if args.in_place:
      count = convert_in_place(input_path)
      print(f"Updated {count} rows in place: {input_path}")
    else:
      output_path = _default_output_path(input_path, args.suffix)
      count = convert_file(input_path, output_path)
      print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
  main()
