# coding=utf-8
"""Copy ReasonIF JSONL files while sampling alternate instruction wording."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("/Users/sasha/data/reasonif_ans_num_words_reg_train")
DEFAULT_OUTPUT_DIR = Path(
    "/Users/sasha/data/reasonif_ans_num_words_reg_train_reworded"
)
DEFAULT_FILENAMES = ("train_input_data.jsonl", "val_input_data.jsonl")

THINKING_GUIDELINES_RE = re.compile(
    r"<thinking guidelines>\n.*?\n</thinking guidelines>",
    flags=re.DOTALL,
)

REASONING_LANGUAGE_ID = "reasonif:language:reasoning_language"
NUMBER_WORDS_ID = "reasonif:length_constraint_checkers:number_words"
JSON_FORMAT_ID = "reasonif:detectable_format:json_format"
END_CHECKER_ID = "reasonif:startend:end_checker"
ENGLISH_CAPITAL_ID = "reasonif:change_case:english_capital"
NO_COMMA_ID = "reasonif:punctuation:no_comma"

LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
}


INSTRUCTION_WORDINGS = {
    REASONING_LANGUAGE_ID: (
        "When reasoning, use only {language} language, no other language is allowed.",
        "During reasoning, write exclusively in {language}.",
        "Use {language} only for the reasoning; do not mix in any other language.",
        "The reasoning text must be entirely in {language}.",
        "Keep all reasoning in {language} and avoid every other language.",
        "Reason only in {language}.",
        "Produce the reasoning using {language} alone.",
        "Your reasoning should contain only {language} text.",
        "For the reasoning section, use no language except {language}.",
        "Write the whole reasoning portion in {language}.",
        "Restrict the reasoning to {language} language only.",
    ),
    NUMBER_WORDS_ID: (
        "When reasoning, use less than {num_words} words.",
        "Keep the reasoning under {num_words} words.",
        "The reasoning must contain fewer than {num_words} words.",
        "Use no more than {num_words} words minus one in the reasoning.",
        "Write the reasoning in less than {num_words} words.",
        "Limit your reasoning to fewer than {num_words} words.",
        "Make sure the reasoning word count is below {num_words}.",
        "The reasoning section should be shorter than {num_words} words.",
        "Do not reach {num_words} words in the reasoning.",
        "Keep the reasoning word total strictly below {num_words}.",
        "Your reasoning must stay below the {num_words}-word limit.",
    ),
    JSON_FORMAT_ID: (
        "When reasoning, follow JSON format. You can use markdown ticks such as ```.",
        "Format the reasoning as JSON. Markdown code fences are allowed.",
        "The reasoning must be valid JSON; you may wrap it in markdown ticks.",
        "Provide the reasoning in JSON format, with optional markdown code fences.",
        "Write the reasoning as a JSON object or value. Markdown ticks may be used.",
        "Structure the reasoning using JSON syntax. Code fences are permitted.",
        "Make the reasoning parseable as JSON; markdown ticks such as ``` are okay.",
        "Return the reasoning in JSON form, optionally enclosed in markdown ticks.",
        "Use JSON formatting for the reasoning. Markdown code blocks are acceptable.",
        "The reasoning should follow valid JSON format; markdown ticks may surround it.",
        "Express the reasoning as JSON, with markdown fences allowed if helpful.",
    ),
    END_CHECKER_ID: (
        'When reasoning, put this exact phrase "{end_phrase}" in the end of reasoning. No other reasoning words should follow this phrase.',
        'End the reasoning with the exact phrase "{end_phrase}" and add no reasoning text after it.',
        'The final words of the reasoning must be "{end_phrase}" exactly.',
        'Finish the reasoning by writing "{end_phrase}"; nothing may follow it.',
        'Put "{end_phrase}" at the very end of the reasoning.',
        'Conclude the reasoning with this exact phrase: "{end_phrase}".',
        'The reasoning must terminate with "{end_phrase}" and no later reasoning words.',
        'Make "{end_phrase}" the closing text of the reasoning.',
        'After all reasoning, write exactly "{end_phrase}" as the ending.',
        'The last phrase in the reasoning should be exactly "{end_phrase}".',
        'Close the reasoning with "{end_phrase}" and do not continue afterward.',
    ),
    ENGLISH_CAPITAL_ID: (
        "When reasoning, use only capital letters.",
        "Write the reasoning entirely in uppercase letters.",
        "Use uppercase letters only in the reasoning.",
        "The reasoning text must be all caps.",
        "Make every letter in the reasoning uppercase.",
        "Do not use lowercase letters in the reasoning.",
        "Keep the full reasoning in capital letters.",
        "The reasoning should contain uppercase lettering only.",
        "Use all-capital text for the reasoning.",
        "Render the reasoning with no lowercase alphabetic characters.",
        "Every alphabetic character in the reasoning must be capitalized.",
    ),
    NO_COMMA_ID: (
        "When reasoning, refrain from the use of any commas.",
        "Do not use commas in the reasoning.",
        "The reasoning must contain no comma characters.",
        "Avoid every comma while reasoning.",
        "Write the reasoning without using any commas.",
        "No commas are allowed in the reasoning text.",
        "Keep comma punctuation out of the reasoning.",
        "The reasoning section should not contain any comma.",
        "Use no comma marks in your reasoning.",
        'Ensure the reasoning does not include "," anywhere.',
        "Reason without inserting commas.",
    ),
}


def _single_instruction_id(row: dict[str, Any], line_number: int) -> str:
  instruction_ids = row.get("instruction_id_list")
  if not isinstance(instruction_ids, list) or len(instruction_ids) != 1:
    raise ValueError(
        f"Line {line_number}: expected exactly one instruction_id_list entry."
    )
  return str(instruction_ids[0])


def _single_kwargs(row: dict[str, Any], line_number: int) -> dict[str, Any]:
  kwargs_list = row.get("kwargs")
  if not isinstance(kwargs_list, list) or len(kwargs_list) != 1:
    raise ValueError(f"Line {line_number}: expected exactly one kwargs entry.")
  kwargs = kwargs_list[0]
  if not isinstance(kwargs, dict):
    raise ValueError(f"Line {line_number}: kwargs entry must be an object.")
  return kwargs


def _language_name(language_code: str) -> str:
  try:
    return LANGUAGE_CODES[language_code].capitalize()
  except KeyError as exc:
    raise ValueError(f"Unsupported language code: {language_code}") from exc


def _format_wording(
    instruction_id: str,
    kwargs: dict[str, Any],
    wording: str,
) -> str:
  if instruction_id == REASONING_LANGUAGE_ID:
    return wording.format(language=_language_name(str(kwargs["language"])))
  if instruction_id == NUMBER_WORDS_ID:
    return wording.format(num_words=kwargs["num_words"])
  if instruction_id == END_CHECKER_ID:
    return wording.format(end_phrase=str(kwargs["end_phrase"]).strip())
  return wording


def _replace_thinking_guidelines(prompt: str, instruction_text: str) -> str:
  replacement = (
      "<thinking guidelines>\n"
      f"{instruction_text}\n"
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


def rewrite_file(input_path: Path, output_path: Path, rng: random.Random) -> int:
  output_path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"Line {line_number}: missing string prompt.")

      instruction_id = _single_instruction_id(row, line_number)
      if instruction_id not in INSTRUCTION_WORDINGS:
        raise ValueError(
            f"Line {line_number}: unsupported instruction id {instruction_id!r}."
        )
      kwargs = _single_kwargs(row, line_number)
      wording = rng.choice(INSTRUCTION_WORDINGS[instruction_id])
      instruction_text = _format_wording(instruction_id, kwargs, wording)

      row["prompt"] = _replace_thinking_guidelines(prompt, instruction_text)
      output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
      count += 1

  return count


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Copy train/val ReasonIF JSONL files and uniformly sample each "
          "instruction wording from the original plus 10 alternatives."
      )
  )
  parser.add_argument(
      "--input_dir",
      type=Path,
      default=DEFAULT_INPUT_DIR,
      help="Directory containing train_input_data.jsonl and val_input_data.jsonl.",
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=DEFAULT_OUTPUT_DIR,
      help="Directory where rewritten JSONL copies will be written.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for wording sampling.",
  )
  args = parser.parse_args()

  rng = random.Random(args.seed)
  for filename in DEFAULT_FILENAMES:
    input_path = args.input_dir / filename
    output_path = args.output_dir / filename
    count = rewrite_file(input_path, output_path, rng)
    print(f"Wrote {count} samples to {output_path}")


if __name__ == "__main__":
  main()
