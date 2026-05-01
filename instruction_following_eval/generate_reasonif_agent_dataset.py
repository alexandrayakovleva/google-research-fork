import argparse
import json
import os
import random
import re
from typing import Any


TARGET_INSTRUCTION_IDS = [
    "agent_reasoning:answer_draft_marker",
    "agent_reasoning:sections",
    "agent_reasoning:two_approaches",
]

AGENT_REASONING_SECTION_HEADING_SETS = {
    "formal_caps": (
        "SECTION: QUESTION ANALYSIS",
        "SECTION: ANSWER DRAFT",
        "SECTION: ANSWER VERIFICATION",
        "SECTION: FINAL ANSWER",
    ),
    "numbered_caps": (
        "SECTION 1: QUESTION ANALYSIS",
        "SECTION 2: ANSWER DRAFT",
        "SECTION 3: ANSWER VERIFICATION",
        "SECTION 4: FINAL ANSWER",
    ),
    "bullet_title": (
        "* Question analysis:",
        "* Answer draft:",
        "* Answer verification:",
        "* Final answer:",
    ),
}

AGENT_REASONING_APPROACH_LABEL_SETS = {
    "approach": (
        "Question Analysis:",
        "Approach 1:",
        "Approach 2:",
        "Approach Selection:",
    ),
    "option": (
        "Question Analysis:",
        "Option 1:",
        "Option 2:",
        "Solution Selection:",
    ),
    "solution_path": (
        "Question Analysis:",
        "Solution path 1:",
        "Solution path 2:",
        "Solution Selection:",
    ),
}

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "reasonif_dataset.jsonl",
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "reasonif_agent_dataset.jsonl",
)

THINKING_GUIDELINES_RE = re.compile(
    r"<thinking guidelines>\n.*?\n</thinking guidelines>",
    flags=re.DOTALL,
)


def _build_instruction(
    inst_id: str,
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
  if inst_id == "agent_reasoning:answer_draft_marker":
    return (
        "When reasoning, provide the answer draft between an "
        "'<answer_draft>' tag and a '</answer_draft>' tag. Put the "
        "'<answer_draft>' tag on its own line, and put the "
        "'</answer_draft>' tag on its own line.",
        {},
    )

  if inst_id == "agent_reasoning:sections":
    heading_style = rng.choice(tuple(AGENT_REASONING_SECTION_HEADING_SETS))
    headings = "\n".join(
        AGENT_REASONING_SECTION_HEADING_SETS[heading_style]
    )
    return (
        "When reasoning, include these exact labeled parts in this "
        f"order:\n{headings}",
        {"heading_style": heading_style},
    )

  if inst_id == "agent_reasoning:two_approaches":
    label_style = rng.choice(tuple(AGENT_REASONING_APPROACH_LABEL_SETS))
    labels = "\n".join(AGENT_REASONING_APPROACH_LABEL_SETS[label_style])
    return (
        "When reasoning, include a structured comparison of two possible "
        "solution approaches and then choose one. Use these exact labels in "
        f"this order:\n{labels}",
        {"label_style": label_style},
    )

  raise ValueError(f"Unsupported instruction id: {inst_id}")


def _replace_thinking_guidelines(prompt: str, constraint_text: str) -> str:
  replacement = (
      "<thinking guidelines>\n"
      f"{constraint_text}\n"
      "</thinking guidelines>"
  )
  updated_prompt, num_replacements = THINKING_GUIDELINES_RE.subn(
      replacement, prompt, count=1
  )
  if num_replacements != 1:
    raise ValueError(
        "Expected exactly one <thinking guidelines>...</thinking guidelines> "
        "block in prompt."
    )
  return updated_prompt


def convert_dataset(
    input_path: str,
    output_path: str,
    seed: int,
) -> int:
  rng = random.Random(seed)
  count = 0

  with open(input_path, "r", encoding="utf-8") as input_file, open(
      output_path, "w", encoding="utf-8"
  ) as output_file:
    for line_number, line in enumerate(input_file, start=1):
      if not line.strip():
        continue

      row = json.loads(line)
      prompt = row.get("prompt")
      if not isinstance(prompt, str):
        raise ValueError(f"Line {line_number}: missing string prompt.")

      inst_id = rng.choice(TARGET_INSTRUCTION_IDS)
      description, kwargs = _build_instruction(inst_id, rng)
      row["prompt"] = _replace_thinking_guidelines(prompt, description)
      row["instruction_id_list"] = [inst_id]
      row["kwargs"] = [kwargs]

      output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
      count += 1

  return count


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Copy a ReasonIF JSONL dataset and replace each thinking guidelines "
          "block with one sampled agent_reasoning instruction."
      )
  )
  parser.add_argument(
      "--input",
      type=str,
      default=DEFAULT_INPUT,
      help="Input ReasonIF JSONL file path.",
  )
  parser.add_argument(
      "--output",
      type=str,
      default=DEFAULT_OUTPUT,
      help="Output JSONL file path.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for instruction sampling.",
  )
  args = parser.parse_args()

  num_rows = convert_dataset(
      input_path=args.input,
      output_path=args.output,
      seed=args.seed,
  )
  print(f"Wrote {num_rows} samples to {args.output}")


if __name__ == "__main__":
  main()
