# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import re
from typing import Dict, Optional, Sequence, Union

from instruction_following_eval import instructions_registry


def _normalize_prompt(prompt: str) -> str:
  """Normalizes prompt text for robust key matching."""
  prompt = prompt.replace("\r\n", "\n").strip()
  return re.sub(r"\s+", " ", prompt)


def _get_responses_for_prompt(inp_prompt, prompt_to_response):
  """Fetches all responses for a prompt with normalized fallback."""
  responses = prompt_to_response.get(inp_prompt)
  if responses is not None:
    return responses
  responses = prompt_to_response.get(_normalize_prompt(inp_prompt))
  if responses is not None:
    return responses
  return None


def _get_response_for_prompt(inp_prompt, prompt_to_response):
  """Fetches one response for a prompt with normalized fallback."""
  responses = _get_responses_for_prompt(inp_prompt, prompt_to_response)
  if responses is None:
    return None
  if isinstance(responses, str):
    return responses
  if not responses:
    return None
  return responses[-1]


def _is_missing_response(response) -> bool:
  """Returns True when response should be treated as missing."""
  if response is None or not isinstance(response, str):
    return True

  text = response.strip()
  if not text:
    return True

  return False


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  response_index: int
  follow_all_instructions: bool
  follow_instruction_list: list[bool]
  feedback_list: list[str]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
      )
      f.write("\n")


def _build_missing_output(inp, response_index: int) -> OutputExample:
  """Builds a standard missing-response output record."""
  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response="",
      response_index=response_index,
      follow_all_instructions=False,
      follow_instruction_list=[False] * len(inp.instruction_id_list),
      feedback_list=[
          "Missing response for this prompt in --input_response_data."
      ] * len(inp.instruction_id_list),
  )


def evaluate_instruction_following_strict(
    inp,
    response: Optional[str],
    response_index: int = 0,
):
  """Tests a single response to see if instrutions are followed."""
  if _is_missing_response(response):
    return _build_missing_output(inp, response_index)
  instruction_list = inp.instruction_id_list
  is_following_list = []
  feedback_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    passed, feedback = instruction.check_following(response)
    if response.strip() and passed:
      is_following_list.append(True)
    else:
      is_following_list.append(False)
    feedback_list.append(feedback)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      response_index=response_index,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
      feedback_list=feedback_list,
  )


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Backward-compatible wrapper for evaluating one prompt response."""
  response = _get_response_for_prompt(inp.prompt, prompt_to_response)
  return evaluate_instruction_following_strict(inp, response)


def evaluate_instruction_following_loose(
    inp,
    response: Optional[str],
    response_index: int = 0,
):
  """Tests a single response for an upper bound for following instructions."""
  if _is_missing_response(response):
    return _build_missing_output(inp, response_index)
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []
  feedback_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    final_feedback = None
    first_feedback = None
    for r in all_responses:
      passed, feedback = instruction.check_following(r)
      if first_feedback is None:
        first_feedback = feedback
      if r.strip() and passed:
        is_following = True
        final_feedback = feedback
        break

    feedback = final_feedback if is_following else first_feedback

    is_following_list.append(is_following)
    feedback_list.append(feedback)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      response_index=response_index,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
      feedback_list=feedback_list,
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Backward-compatible wrapper for evaluating one prompt response."""
  response = _get_response_for_prompt(inp.prompt, prompt_to_response)
  return evaluate_instruction_following_loose(inp, response)


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and responses."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      prompt = example["prompt"]
      response = example["response"]
      responses = return_dict.setdefault(prompt, [])
      responses.append(response)
      normalized_prompt = _normalize_prompt(prompt)
      if normalized_prompt not in return_dict:
        return_dict[normalized_prompt] = responses
  return return_dict


def print_report(outputs):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  print(f"prompt-level (per response): {prompt_correct / prompt_total}")
  print(f"instruction-level: {instruction_correct / instruction_total}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
