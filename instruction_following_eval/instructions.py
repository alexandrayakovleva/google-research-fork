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

"""Library of instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

from absl import logging
import langdetect

from instruction_following_eval import instructions_util


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.", 'I think yes.', 'I think no.', 'I think maybe.')

# The options of starter keywords.
_STARTER_OPTIONS = ("I would say", "My answer is", "I believe",
                    "In my opinion", "I think", "I reckon", "I feel",
                    "From my perspective", "As I see it", "According to me",
                    "As far as I'm concerned", "To my understanding",
                    "In my view", "My take on it is", "As per my perception")

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?",
                   "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# phrases
_PHRASES = [
    "Dance like nobody is watching you",
    "The early bird catches the worm",
    "Time flies when having fun",
    "Every cloud has a silver lining",
    "Actions speak louder than words",
    "Don't judge a book by cover",
    "Live each day to the fullest",
    "All that glitters is not gold",
    "Laughter is the best medicine",
    "The pen is mightier than sword",
]


class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
  """Check the language of the entire response."""

  def build_description(self, *, language = None):
    """Build the instruction description.

    Args:
      language: A string representing the expected language of the response. The
        language has to comply to the 97 types defined in
        `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
        ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
        for example, `en` for English, `zh` for Chinese, `fr` for French.

    Returns:
      A string representing the instruction description.
    """
    self._language = language
    if self._language is None:
      self._language = random.choice(list(_LANGUAGES.keys()))
    # TODO(tianjianlu): opens the description generation to more choices.
    self._description_pattern = (
        "Your ENTIRE response should be in {language} language, no other " +
        "language is allowed.")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"language": self._language}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["language"]

  def check_following(self, value):
    """Check if the language of the entire response follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the language of `value` follows instruction; otherwise False.
    """
    assert isinstance(value, str)

    try:
      detected_lang = langdetect.detect(value)
      return detected_lang == self._language, f"Detected language '{detected_lang}', required '{self._language}'."
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True, "Language detection failed; counted as passed."


class NumberOfSentences(Instruction):
  """Check the number of sentences."""

  def build_description(self, *, num_sentences = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_sentences: An integer specifying the number of sentences as a
        threshold.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of sentences < the threshold;
        if 'at least', the actual number of sentences >= the threshold.

    Returns:
      A string representing the instruction description.
    """
    # The number of sentences as a threshold for comparison.
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or
        self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Your response should contain {relation} {num_sentences} sentences.")
    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_sentences", "relation"]

  def check_following(self, value):
    """Check if the number of sentences follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the response follows the instruction.

    Raise:
        ValueError if the string in `instruction_args` is not in
        [`less_than`, `at_least`].
    """
    num_sentences = instructions_util.count_sentences(value)
    feedback = f"Found {num_sentences} sentences, required {self._comparison_relation} {self._num_sentences_threshold}."
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      if num_sentences >= self._num_sentences_threshold:
        feedback += f" Shorten the response by at least {num_sentences - self._num_sentences_threshold + 1} sentence(s)."
      return num_sentences < self._num_sentences_threshold, feedback # pytype: disable=bad-return-type
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      if num_sentences < self._num_sentences_threshold:
         feedback += f" Add at least {self._num_sentences_threshold - num_sentences} sentence(s) more to the response."
      return num_sentences >= self._num_sentences_threshold, feedback  # pytype: disable=bad-return-type


class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders = None):
    """Build the instruction description.

    Args:
      num_placeholders: An integer denoting the minimum number of
        placeholders required in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "The response must contain at least {num_placeholders} placeholders " +
        "represented by square brackets, such as [address].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_placeholders"]

  def check_following(self, value):
    """Check if the number of placeholders follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual number of placeholders in the response is greater than
      or equal to `num_placeholders`; otherwise, False.
    """
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    feedback = f"Found {num_placeholders} placeholders: {placeholders}, required at least {self._num_placeholders}."
    passed = (num_placeholders >= self._num_placeholders)
    if not passed:
      feedback += f" Add at least {self._num_placeholders - num_placeholders}."
    return passed, feedback


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet lists
        that is required to appear in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Your answer must contain exactly {num_bullets} bullet points. " +
        "Use the markdown bullet points such as:\n" +
        "* This is point 1. \n" +
        "* This is point 2")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_bullets"]

  def check_following(self, value):
    r"""Check if the number of bullet lists meets the requirement.

    Args:
      value: A string representing the response. The response is expected to
        contain some bullet lists that start with `\*`.

    Returns:
      True if the actual number of bullet lists in the response meets the
      requirement.
    """
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    passed = (num_bullet_lists == self._num_bullets)
    feedback = f"Found {num_bullet_lists} bullets, required exactly {self._num_bullets}."
    if not passed:
      if num_bullet_lists < self._num_bullets:
        feedback += f" Rewrite the response to add exactly {self._num_bullets - num_bullet_lists}  more bullets."
      else:
        feedback += f" Rewrite the response to remove exactly {num_bullet_lists - self._num_bullets} bullets."
    return passed, feedback


class ConstrainedResponseChecker(Instruction):
  """Checks the constrained response."""

  def build_description(self):
    """Build the instruction description."""
    # A sequence of string(s) representing the options of the expected response.
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = (
        "Answer with one of the following options: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response matches the constrained options.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual response contains one of the options in the constrained
      responses; otherwise False.
    """
    value = value.strip()
    for constrained_response in self._constrained_responses:
      if constrained_response in value:
        return True, f"Response matched constrained option: '{constrained_response}'."
    return False, f"Response did not match any constrained option: {self._constrained_responses}."


class ConstrainedStartChecker(Instruction):
  """Checks the response start."""

  def build_description(self, *, starter = None):
    """Build the instruction description.

    Args:
      starter: A string representing the keyward that the response should start
        with.

    Returns:
      A string representing the instruction description.
    """
    self._starter = starter.strip() if isinstance(starter, str) else starter
    if self._starter is None:
      self._starter = random.choice(_STARTER_OPTIONS)
    self._description_pattern = (
        "During the conversation, when it is your turn, " +
        "please always start with {starter}")
    return self._description_pattern.format(starter=self._starter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"starter": self._starter}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["starter"]

  def check_following(self, value):
    """Checks if the response starts with the constrained keyword or phrase.

    Args:
      value: A string representing the response.

    Returns:
      True if the response starts with the given phrase or keyword that is
      contained in `instruction_args`; otherwise, False.
    """
    response_pattern = r"^\s*" + self._starter + r".*$"
    response_with_constrained_start = re.search(response_pattern, value,
                                                flags=re.MULTILINE)
    if response_with_constrained_start:
        return True, f"Response starts with '{self._starter}': True."
    return False, f"Response starts with '{value.lstrip()[:len(self._starter)]}', expected start: '{self._starter}'. Start the corrected response with '{self._starter}'."


class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights = None):
    """Build the instruction description.

    Args:
      num_highlights: An integer specifying the minimum number of highlighted
        sections.

    Returns:
      A string representing the instruction description.
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "Highlight at least {num_highlights} sections in your answer with " +
        "markdown, i.e. *highlighted section*.")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_highlights": self._num_highlights}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_highlights"]

  def check_following(self, value):
    """Checks if the number of highlighted sections meets the requirement.

    Args:
      value: a string repesenting the response. The response is expected to
        contain highlighted sections in the format of *highlighted*.

    Returns:
      True if the actual number of highlighted sections in the format of
      *highlighed sections* meets the minimum requirement; otherwise False.
    """
    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", value)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
    to_print = []
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
        to_print.append(highlight)
    for highlight in double_highlights:
      if highlight.removeprefix("**").removesuffix("**").strip():
        num_highlights += 1
        to_print.append(highlight)

    feedback = f"Found {num_highlights} highlighted sections, required at least {self._num_highlights}. Highlighted sections: {to_print}."
    passed = num_highlights >= self._num_highlights
    if not passed:
      feedback += f" Add at least {self._num_highlights - num_highlights} highlighted sections."

    return passed, feedback


class SectionChecker(Instruction):
  """Checks the sections."""

  def build_description(self, *, section_spliter = None,
                        num_sections = None):
    """Build the instruction description.

    Args:
      section_spliter: A string represents the section spliter keyword that
        marks a new section, i.e., `Section` or `SECTION`.
      num_sections: An integer specifying the number of sections.

    Returns:
      A string representing the instruction description.
    """
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    self._description_pattern = (
        "Your response must have {num_sections} sections. Mark the beginning " +
        "of each section with {section_spliter} X, such as:\n" +
        "{section_spliter} 1\n" +
        "[content of section 1]\n" +
        "{section_spliter} 2\n" +
        "[content of section 2]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["section_spliter", "num_sections"]

  def check_following(self, value):
    """Checks the response contains multiple sections.

    Args:
      value: A string representing the response. The response is expected
        to contain multiple sections (number of sections is greater than 1).
        A new section starts with `Section 1`, where the number denotes the
        section index.

    Returns:
      True if the number of sections in the response is greater than or equal to
      the minimum number of sections; otherwise, False.
    """
    section_splitter_patten = r"\s?" + self._section_spliter  + r"\s?\d+\s?"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    feedback = f"Found {num_sections} sections using splitter '{self._section_spliter}', required at least {self._num_sections}."
    passed = num_sections >= self._num_sections
    if not passed:
      feedback += f" Add at least {self._num_sections - num_sections} sections using splitter '{self._section_spliter}."
    return passed, feedback


class ParagraphChecker(Instruction):
  """Checks the paragraphs."""

  def build_description(self, *, num_paragraphs = None):
    """Build the instruction description.

    Args:
      num_paragraphs: An integer specifying the number of paragraphs.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "There should be {num_paragraphs} paragraphs. " +
        "Paragraphs are separated with the markdown divider: ***")

    return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs"]

  def check_following(self, value):
    """Checks the response contains required number of paragraphs.

    Args:
      value: A string representing the response. The response may contain
        paragraphs that are separated by the markdown divider: `***`.

    Returns:
      True if the actual number of paragraphs is the same as required;
      otherwise, False.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False, "Invalid paragraph structure: empty paragraph between separators '***'."

    feedback = f"Found {num_paragraphs} paragraphs separated by '***', required exactly {self._num_paragraphs}."
    passed = num_paragraphs == self._num_paragraphs
    if not passed:
      if num_paragraphs > self._num_paragraphs:
        feedback += f" Shorten the response by exactly {num_paragraphs - self._num_paragraphs} paragraph(s)."
      else:
        feedback += f" Add exactly {self._num_paragraphs - num_paragraphs} paragraph(s) to the response."

    return passed, feedback


class PostscriptChecker(Instruction):
  """Checks the postscript."""

  def build_description(self, *, postscript_marker = None
                        ):
    """Build the instruction description.

    Args:
      postscript_marker: A string containing the keyword that marks the start
        of the postscript section.

    Returns:
      A string representing the instruction description.
    """
    self._postscript_marker = postscript_marker.strip() if isinstance(
        postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = (
        "At the end of your response, please explicitly add a postscript " +
        "starting with {postscript}")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["postscript_marker"]

  def check_following(self, value):
    """Checks if the response follows the postscript format.

    Args:
      value: a string representing the response. The response is expected to
        contain a postscript section.

    Returns:
      True if the response contains a postscript section starting with
      the keyword containing in the `instruction_args`; otherwise False.
    """
    value = value.lower()
    if self._postscript_marker == "P.P.S":
      postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif self._postscript_marker == "P.S.":
      postscript_pattern = r"\s*p\.\s?s\..*$"
    else:
      postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
    postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
    if postscript:
      return True, f"Postscript marker '{self._postscript_marker}' found."
    return False, f"No postscript marker '{self._postscript_marker}' found. Add a postscript starting with '{self._postscript_marker}'."


class RephraseChecker(Instruction):
  """Checks the repharse."""

  def build_description(self, *, original_message):
    """Build the instruction description.

    Args:
      original_message: A string representing the original message. The
        rephrased response should only change its words/sentences in between
        its two asterisks, for example, *change me*. Both original and rephrased
        messages should contain the changes in the form of *change me*.

    Returns:
      A string representing the instruction description.
    """
    if not self.is_change(original_message):
      raise ValueError(f"Message {original_message} does not contain changes "
                       "in the form of *change me*.")

    self._reference_without_change = original_message
    self._description = ("Rephrasing: Your rephrased response should only" +
                         "change the words/sentences in between two asterisks" +
                         "such as *change me*.")
    return self._description

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"original_message": self._reference_without_change}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["original_message"]

  def check_following(self, value):
    r"""Checks if the rephrasing follows the instruction.

    Args:
      value: A string representing the response, which is expected to rephras
        the string of `instruction_args`.

    Returns:
      True if `value` and `instruction_args` only differ by the words/sentences
      in between two asterisks such as *change me*; otherwise, False.
    """

    if not self.is_change(value):
      raise ValueError(f"value {value} does not contain "
                       "changes in the form of *change me*.")

    response_without_changes = self.strip_changes(value)
    reference_without_changes = self.strip_changes(
        self._reference_without_change)

    if response_without_changes == reference_without_changes:
      return True, f"Response matches original text outside *changes*."
    return False, f"Response does not match original text outside *changes*. Copy the original text to the corrected response before and after *changes*."

  def is_change(self, response):
    """Check if there is change in the response in the form of *change me*."""
    return re.search(r"\*.*\*", response)

  def strip_changes(self, response):
    """Strips off the changes."""
    return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
  """Check the exisitence of certain keywords."""

  def build_description(self, *, keywords = None
                        ):
    """Build the instruction description.

    Args:
      keywords: A sequence of strings representing the keywords that are
        expected in the response.

    Returns:
      A string representing the instruction description.
    """

    if not keywords:
      self._keywords = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("Include keywords {keywords} in the response.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keywords"]

  def check_following(self, value):
    """Check if the response contain the expected keywords."""
    missing_keywords = []
    for keyword in self._keywords:
      if not re.search(keyword, value, flags=re.IGNORECASE):
        missing_keywords.append(keyword)
    if len(missing_keywords) == 0:
        return True, f"All required keywords present: {self._keywords}."
    return False, f"Missing keywords: {missing_keywords}. Add them to the response."


class KeywordFrequencyChecker(Instruction):
  """Check the keyword frequency."""

  def build_description(self, *, keyword = None,
                        frequency = None,
                        relation = None):
    """Build the instruction description.

    Args:
      keyword: A string representing a keyword that is expected in the response.
      frequency: An integer specifying the number of times `keyword` is expected
        to appear in the response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of occurrences < frequency;
        if 'at least', the actual number of occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if not keyword:
      self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "In your response, the word {keyword} should appear {relation} " +
        "{frequency} times.")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=self._comparison_relation,
        frequency=self._frequency)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    """Checks if the response contain the keyword with required frequency."""
    actual_occurrences = len(re.findall(
        self._keyword, value, flags=re.IGNORECASE))
    feedback = f"Keyword '{self._keyword}' found {actual_occurrences} times, required {self._comparison_relation} {self._frequency}."
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      if actual_occurrences >= self._frequency:
        feedback += f" Remove/replace at least {actual_occurrences - self._frequency + 1} keyword(s)."
      return actual_occurrences < self._frequency, feedback
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      if actual_occurrences < self._frequency:
        feedback += f" Add at least {self._frequency - actual_occurrences} more keyword(s)."
      return actual_occurrences >= self._frequency, feedback  # pytype: disable=bad-return-type


class NumberOfWords(Instruction):
  """Checks the number of words."""

  def build_description(self, *, num_words = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_words: An integer specifying the number of words contained in the
        response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of words < num_words;
        if 'at least', the actual number of words >= num_words.

    Returns:
      A string representing the instruction description.
    """

    self._num_words = num_words
    if self._num_words is None or self._num_words < 0:
      self._num_words = random.randint(
          _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
      )

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Answer with {relation} {num_words} words.")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_words=self._num_words)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_words": self._num_words,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_words", "relation"]

  def check_following(self, value):
    """Checks if the response contains the expected number of words."""
    num_words = instructions_util.count_words(value)
    feedback = f"Found {num_words} words, required {self._comparison_relation} {self._num_words}."
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      if num_words >= self._num_words:
        feedback += f" Shorten the response by at least {num_words - self._num_words + 1} word(s)."
      return num_words < self._num_words, feedback
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      if num_words < self._num_words:
        feedback += f" Add at least {self._num_words - num_words} word(s) to the response."
      return num_words >= self._num_words, feedback  # pytype: disable=bad-return-type


class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):
    self._description_pattern = (
        "Entire output should be wrapped in JSON format. You can use markdown"
        " ticks such as ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as e:
      return False, f"Invalid JSON format. JSON load error: {e}."
    return True, f"Valid JSON format: True."


class ParagraphFirstWordCheck(Instruction):
  """Check the paragraph and the first word of the nth paragraph."""

  def build_description(self, num_paragraphs = None,
                        nth_paragraph = None,
                        first_word = None):
    r"""Build the instruction description.

    Args:
      num_paragraphs: An integer indicating the number of paragraphs expected
        in the response. A paragraph is a subset of the string that is
        expected to be separated by '\n\n'.
      nth_paragraph: An integer indicating the paragraph number that we look at.
        Note that n starts from 1.
      first_word: A string that represent the first word of the bth paragraph.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
    self._first_word = self._first_word.lower()

    self._description_pattern = (
        "There should be {num_paragraphs} paragraphs. " +
        "Paragraphs and only paragraphs are separated with each other by two " +
        "new lines as if it was '\\n\\n' in python. " +
        "Paragraph {nth_paragraph} must start with word {first_word}.")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    """Checks for required number of paragraphs and correct first word.

    Args:
      value: a string representing the response. The response may contain
        paragraphs that are separated by two new lines and the first word of
        the nth paragraph will have to match a specified word.

    Returns:
      True if the number of paragraphs is the same as required and the first
      word of the specified paragraph is the same as required. Otherwise, false.
    """

    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    # check that index doesn't go out of bounds
    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False, f"Paragraph count={num_paragraphs}, required={self._num_paragraphs}. Paragraph {self._nth_paragraph} is invalid."
    else:
      return False, f"Paragraph count={num_paragraphs}, required={self._num_paragraphs}."

    first_word = ""
    punctuation = {".", ",", "?", "!", "'", '"'}

    # get first word and remove punctuation
    word = paragraph.split()[0].strip()
    # TODO(jeffrey): make more complex?
    word = word.lstrip("'")
    word = word.lstrip('"')

    for letter in word:
      if letter in punctuation:
        break
      first_word += letter
    first_word_normalized = first_word.lower()

    feedback = f"Paragraph count={num_paragraphs}, required={self._num_paragraphs}. Paragraph {self._nth_paragraph} starts with '{first_word}', required '{self._first_word}'."
    if num_paragraphs < self._num_paragraphs:
      feedback += f" Extend the response by exactly {self._num_paragraphs - num_paragraphs} paragraph(s)."
    if num_paragraphs > self._num_paragraphs:
      feedback += f" Shorten the response by exactly {num_paragraphs - self._num_paragraphs} paragraph(s)."
    if first_word_normalized != self._first_word:
      feedback += f" Start paragraph {self._nth_paragraph} with '{self._first_word}'."

    return (
        num_paragraphs == self._num_paragraphs
        and first_word_normalized == self._first_word
    ), feedback


# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
  """Check the existence of certain key sentences."""

  def build_description(self, key_sentences = None,
                        num_sentences = None):
    """Build the instruction description.

    Args:
      key_sentences: A sequences of strings representing the key sentences that
        are expected in the response.
      num_sentences: The number of key sentences that are expected to be seen in
        the response.

    Returns:
      A string representing the instruction description.
    """

    if not key_sentences:
      # TODO(jeffrey) make a generate sentences function? wonderwords package
      self._key_sentences = set(["For now, this is fine."])
    else:
      self._key_sentences = key_sentences

    if not num_sentences:
      self._num_sentences = random.randint(1, len(self._key_sentences))
    else:
      self._num_sentences = num_sentences

    self._description_pattern = (
        "Include {num_sentences} of the following sentences {key_sentences}"
    )

    return self._description_pattern.format(
        num_sentences=self._num_sentences, key_sentences=self._key_sentences
    )

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences)}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_sentences", "key_sentences"]

  def check_following(self, value):
    """Checks if the response contains the expected key sentences."""
    count = 0
    sentences = instructions_util.split_into_sentences(value)
    key_sentences_found = []
    for sentence in self._key_sentences:
      if sentence in sentences:
        count += 1
        key_sentences_found.append(sentence)

    feedback = f"Found {count} key sentences, required {self._num_sentences}. Key sentences found: {key_sentences_found}."
    passed = (count == self._num_sentences)
    if not passed:
      if count > self._num_sentences:
        feedback += f" Remove / replace exactly {count - self._num_sentences} key sentence(s)."
      else:
        feedback += f" Add exactly {count - self._num_sentences} more key sentence(s)."
    return passed, feedback


class ForbiddenWords(Instruction):
  """Checks that specified words are not used in response."""

  def build_description(self, forbidden_words = None
                        ):
    """Build the instruction description.

    Args:
      forbidden_words: A sequences of strings respresenting words that are not
        allowed in the response.

    Returns:
      A string representing the instruction description.
    """

    if not forbidden_words:
      self._forbidden_words = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "Do not include keywords {forbidden_words} in the response."
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words
    )

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["forbidden_words"]

  def check_following(self, value):
    """Check if the response does not contain the expected keywords."""
    forbidden_words_used = []
    for word in self._forbidden_words:
      if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
        forbidden_words_used.append(word)
    if len(forbidden_words_used) == 0:
      return True, "No forbidden words detected."
    return False, f"Forbidden words detected: {forbidden_words_used}. Remove / replace them."


class RephraseParagraph(Instruction):
  """Checks that the paragraph is rephrased."""

  def build_description(self, *, original_paragraph, low, high
                        ):
    """Builds the instruction description.

    Args:
      original_paragraph: A string presenting the original paragraph. The
        rephrases response should have betweeb low-high words in common.
      low: An integer presenting the lower bound of similar words.
      high: An integer representing the upper bound of similar words.

    Returns:
      A string representing the instruction description.
    """
    # TODO(jeffrey) make more encompassing
    self._original_paragraph = original_paragraph
    self._low = low
    self._high = high

    self._description = ("Rephrase the following paragraph: " +
                         "{original_paragraph}\nYour response should have " +
                         "between {low} and {high} of the same words. " +
                         "Words are the same if and only if all of the " +
                         "letters, ignoring cases, are the same. For " +
                         "example, 'run' is the same as 'Run' but different " +
                         "to 'ran'.")

    return self._description.format(original_paragraph=original_paragraph,
                                    low=self._low, high=self._high)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["original_paragraph", "low", "high"]

  def check_following(self, value):
    val_words = re.findall(r"\w+", value)
    original_words = re.findall(r"\w+", self._original_paragraph)
    similar_words = 0

    dict_val = collections.Counter(word.lower() for word in val_words)
    dict_original = collections.Counter(word.lower() for word in original_words)
    original_case_by_lower = {}
    for word in original_words:
      lower_word = word.lower()
      if lower_word not in original_case_by_lower:
        original_case_by_lower[lower_word] = word

    shared_words = []
    for word in dict_original:
      similar_words += min(dict_original[word], dict_val[word])
      if min(dict_original[word], dict_val[word]) > 0:
        shared_words.append(original_case_by_lower.get(word, word))

    feedback = f"Found {similar_words} shared words with original, required between {self._low} and {self._high}. Shared words: {shared_words}."
    passed = (similar_words >= self._low and similar_words <= self._high)
    if not passed:
      if similar_words > self._high:
        feedback += f" Remove or replace at least {similar_words - self._high} (and no more than {similar_words - self._low}) shared words."
      else:
        feedback += f" Add at least {self._low - similar_words} (and no more than {self._low - similar_words}) shared words."
    return passed, feedback


class TwoResponsesChecker(Instruction):
  """Check that two responses were given."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Give two different responses. Responses and only responses should"
        " be separated by 6 asterisk symbols: ******."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response has two different answers.

    Args:
      value: A string representing the response.

    Returns:
      True if two responses are detected and false otherwise.
    """
    valid_responses = list()
    responses = value.split("******")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False, f"Invalid response structure: empty segment found at position {index} between '******' separators."
      else:
        valid_responses.append(response)
    feedback = f"Detected {valid_responses} valid responses separated by '******', required 2."
    if len(valid_responses) != 2:
      if len(valid_responses) < 2:
        feedback += f" Add exactly {2 - len(valid_responses)} more response(s) separated by '******'."
      else:
        feedback += f" Shorten by exactly {len(valid_responses) - 2} response(s) separated by '******'."
    if len(valid_responses) == 2 and valid_responses[0].strip() == valid_responses[1].strip():
      feedback += " The responses are identical. Write two different responses."
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    ), feedback


class RepeatPromptThenAnswer(Instruction):
  """Checks that Prompt is first repeated then answered."""

  def build_description(self, *, prompt_to_repeat = None):
    """Build the instruction description.

    Args:
      prompt_to_repeat: The prompt that is meant to be repeated.

    Returns:
      A string representing the instruction description.
    """
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeat must be set.")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "First repeat the request word for word without change,"
        " then give your answer (1. do not say any words or characters"
        " before repeating the request; 2. the request you need to repeat"
        " does not include this sentence)"
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["prompt_to_repeat"]

  def check_following(self, value):
    stripped_value = value.strip()
    expected_prompt = self._prompt_to_repeat.strip()
    if stripped_value.lower().startswith(expected_prompt.lower()):
      return True, "Response starts with prompt."
    actual_start = stripped_value[:len(expected_prompt)]
    return False, (
        f"Response starts with '{actual_start}', expected start: "
        f"'{expected_prompt}'. Start the corrected response with "
        f"'{expected_prompt}'."
    )


class EndChecker(Instruction):
  """Checks that the prompt ends with a given phrase."""

  def build_description(self, *, end_phrase = None):
    """Build the instruction description.

    Args:
      end_phrase: A string representing the phrase the response should end with.

    Returns:
      A string representing the instruction description.
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Finish your response with this exact phrase {ender}. "
        "No other words should follow this phrase.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["end_phrase"]

  def check_following(self, value):
    """Checks if the response ends with the expected phrase."""
    stripped_value = value.strip().strip("\"")
    expected_phrase = self._end_phrase.strip()
    if stripped_value.lower().endswith(expected_phrase.lower()):
      return True, f"Response ends with '{expected_phrase}': True."
    actual_ending = stripped_value[-len(expected_phrase):] if expected_phrase else ""
    return False, (
        f"Response ends with '{actual_ending}', expected ending: "
        f"'{expected_phrase}'. End the corrected response with "
        f"'{expected_phrase}'."
    )


class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your answer must contain a title, wrapped in double angular brackets,"
        " such as <<poem of joy>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response contains a title."""
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True, f"Valid title present: {title}."
    return False, f"None of the titles {titles} found. Use at least one of the titles."


class LetterFrequencyChecker(Instruction):
  """Checks letter frequency."""

  def build_description(self, *, letter = None,
                        let_frequency = None,
                        let_relation = None):
    """Build the instruction description.

    Args:
      letter: A string representing a letter that is expected in the response.
      let_frequency: An integer specifying the number of times `keyword` is
        expected to appear in the response.
      let_relation: A string in (`less than`, `at least`), defining the
        relational operator for comparison. Two relational comparisons are
        supported for now; if 'less than', the actual number of
        occurrences < frequency; if 'at least', the actual number of
        occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if not isinstance(letter, str):
      letter = None

    cleaned_letter = letter.strip() if letter else ""
    if len(cleaned_letter) != 1:
      self._letter = random.choice(list(string.ascii_letters))
    else:
      self._letter = cleaned_letter

    self._letter = self._letter.lower()

    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {let_relation} is given."
      )
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "In your response, the letter {letter} should appear {let_relation}"
        " {let_frequency} times."
    )

    return self._description_pattern.format(
        letter=self._letter,
        let_frequency=self._frequency,
        let_relation=self._comparison_relation,
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    """Checks that the response contains the letter at the right frequency."""
    value = value.lower()
    letters = collections.Counter(value)
    feedback = f"Letter '{self._letter}' appears {letters[self._letter]} times, required {self._comparison_relation} {self._frequency}."
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      if letters[self._letter] >= self._frequency:
        feedback += f" Rewrite the response to remove at least {letters[self._letter] - self._frequency + 1} letter(s) '{self._letter}'."
      return letters[self._letter] < self._frequency, feedback
    else:
      if letters[self._letter] < self._frequency:
        feedback += f" Rewrite the response to add at least {self._frequency - letters[self._letter]} more letter(s) '{self._letter}'."
      return letters[self._letter] >= self._frequency, feedback


class CapitalLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all capital letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your entire response should be in English, and in all capital letters."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in English and in all capital letters."""
    assert isinstance(value, str)

    try:
      detected_lang = langdetect.detect(value)
      feedback = f"The response is in all capital letters: {value.isupper()}, required True. Detected language: {detected_lang}, required en."
      if not value.isupper():
        feedback += " Use only capital letters."
      if detected_lang != "en":
        feedback += " Respond in English."
      return value.isupper() and detected_lang == "en", feedback
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True, "Unable to detect language, returned True."


class LowercaseLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all lowercase letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your entire response should be in English, and in all lowercase"
        " letters. No capital letters are allowed."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in English and in all lowercase letters."""
    assert isinstance(value, str)

    try:
      detected_lang = langdetect.detect(value)
      feedback = f"The response is in all lowercase letters: {value.islower()}, required True. Detected language: {detected_lang}, required en."
      if not value.islower():
        feedback += " Use only lowercase letters."
      if detected_lang != "en":
        feedback += " Respond in English."
      return value.islower() and detected_lang == "en", feedback
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True, "Unable to detect language, returned True."


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "In your entire response, refrain from the use of any commas."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response does not contain commas."""
    if not re.search(r"\,", value):
      return True, "Response does not contain commas as required."
    return False, "Comma detected in the response. The response must not contain commas."


class CapitalWordFrequencyChecker(Instruction):
  """Checks frequency of words with all capital letters."""

  def build_description(
      self,
      capital_frequency = None,
      capital_relation = None,
  ):
    """Build the instruction description.

    Args:
      capital_frequency: An integer that represents the number of words that
        should be in all capital letters.
      capital_relation: A string that is 'at least' or 'at most' that refers to
        the frequency.

    Returns:
      A string representing the instruction description.
    """
    self._frequency = capital_frequency
    if self._frequency is None:
      self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

    self._comparison_relation = capital_relation
    if capital_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif capital_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {capital_relation} is given."
      )

    self._description_pattern = (
        "In your response, words with all capital letters should appear"
        " {relation} {frequency} times."
    )

    return self._description_pattern.format(
        frequency=self._frequency, relation=self._comparison_relation
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {
        "capital_frequency": self._frequency,
        "capital_relation": self._comparison_relation,
    }

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["capital_frequency", "capital_relation"]

  def check_following(self, value):
    """Checks the frequency of words with all capital letters."""
    # Hyphenated words will count as one word
    words = instructions_util.nltk.word_tokenize(value)
    capital_words = [word for word in words if word.isupper()]

    feedback = f"Found {len(capital_words)} ALL-CAPS words: {capital_words}, required {self._comparison_relation} {self._frequency}."
    capital_words = len(capital_words)
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      if capital_words >= self._frequency:
        feedback += f" Turn at least {capital_words - self._frequency + 1} word(s) into lower case."
      return capital_words < self._frequency, feedback
    else:
      if capital_words < self._frequency:
        feedback += f" Turn at least {self._frequency - capital_words} word(s) into upper case."
      return capital_words >= self._frequency, feedback


class QuotationChecker(Instruction):
  """Checks response is wrapped with double quotation marks."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Wrap your entire response with double quotation marks."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response is wrapped with double quotation marks."""
    value = value.strip()
    if len(value) > 1 and value[0] == '"' and value[-1] == '"':
      return True, "Response is wrapped with double quotation marks: True."
    return False, "No double quotation marks detected. The response must be wrapped with double quotation marks."


class RepeatPhraseChecker(Instruction):
    "Repeat the phrase {phrase} exactly {small_n} times, transforming it slightly each time by replacing only one word in the center of the phrase."

    def build_description(self, phrase=None, small_n=None):
        """Build the instruction description.

        Args:
          phrase: A string representing the phrase to be repeated.
          N: An integer representing the number of times to repeat the phrase.
          word_count: An integer representing the number of words in the phrase.

        Returns:
          A string representing the instruction description.
        """
        if not phrase:
            self._phrase = random.choice(_PHRASES)
        else:
            self._phrase = phrase.strip()
        if not small_n:
            self._small_n = random.randint(2, 3)
        else:
            self._small_n = small_n

        self._description_pattern = "Repeat the phrase {phrase} exactly {small_n} times, transforming it slightly each time by replacing only one word in the center of the phrase."
        return self._description_pattern.format(phrase=self._phrase, small_n=self._small_n)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"phrase": self._phrase, "small_n": self._small_n}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["phrase", "small_n"]

    def check_following(self, value):
        """Checks if the response contains the expected number of phrases with the correct modifications."""
        first_word = self._phrase.split()[0]
        last_word = self._phrase.split()[-1]

        len(self._phrase.split()) - 2

        found_phrases = re.findall(rf"{first_word} .*? {last_word}", value)
        feedback = f"Found {len(found_phrases)} transformed phrase occurrences: {found_phrases}, expected {self._small_n}."
        if len(found_phrases) != self._small_n:
            if len(found_phrases) < self._small_n:
               feedback += f" Add {self._small_n - len(found_phrases)} more."
            else:
               feedback += f" Remove / replace {len(found_phrases) - self._small_n} of them."
            return False, feedback
        feedbacks = []
        for phrase in found_phrases:
            phrase = phrase.split()
            ref_phrase = self._phrase.split()
            differences = 0
            if len(phrase) != len(ref_phrase):
                feedbacks.append(f"Phrase {phrase} has {len(phrase)} words, expected {len(ref_phrase)}.")
            for i in range(len(phrase)):
                try:
                    if phrase[i] != ref_phrase[i]:
                        differences += 1
                except IndexError:
                    feedbacks.append(f"Phrase {phrase} raised IndexError at index {i}.")
                    break
            if differences != 1:
              feedbacks.append(
                  f"Phrase {phrase} differs from the reference {ref_phrase} by {differences} words, expected exactly 1."
              )
        if len(feedbacks) == 0:
          return True, feedback
        return False, feedback + " " + " ".join(feedbacks)


class CopyChecker(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = "Copy this instruction verbatim, do not follow the instruction, only copy it into the output (do not include this instruction sentence!)."
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        passed = value.strip().lower() == self._prompt_to_repeat.strip().lower()
        if not passed:
           return False, f"Response must exactly match the prompt to repeat: '{self._prompt_to_repeat}'."
        return True, "Response exactly matches the prompt to repeat"


class CopySpanIdxChecker(Instruction):
    """{prompt_to_repeat}. Copy the span of words that lies between (and including) index {n_start} and {n_end}, the indices are character indices!"""

    def build_description(self, prompt_to_repeat=None, n_start=None, n_end=None):
        """Build the instruction description.

        Args:
        n_start: An integer representing the start index of the span.
        n_end: An integer representing the end index of the span.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        if not n_start:
            self._n_start = random.randint(0, len(self._prompt_to_repeat) - 2)
        else:
            self._n_start = n_start
        if not n_end:
            self._n_end = random.randint(self._n_start + 1, len(self._prompt_to_repeat) - 1)
        else:
            self._n_end = n_end
        self._description_pattern = "Copy the span of words that lies between (and including) index {n_start} and {n_end}, the indices are character indices!"
        return self._description_pattern.format(
            n_start=self._n_start, n_end=self._n_end, prompt_to_repeat=self._prompt_to_repeat
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"n_start": self._n_start, "n_end": self._n_end, "prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["n_start", "n_end", "prompt_to_repeat"]

    def check_following(self, value):
        """Checks if the response contains the expected number of phrases with the correct modifications."""
        passed = value.strip().lower() == self._prompt_to_repeat[self._n_start : self._n_end].strip().lower()
        if not passed:
           return False, f"Response must exactly match the span of words that lies between (and including) index {self._n_start} and {self._n_end}: '{self._prompt_to_repeat[self._n_start : self._n_end].strip()}'."
        return True, "Response exactly matches the required span of words."


class SentenceHyphenChecker(Instruction):
    """All sentences must be connected using hyphens, with no spaces between them."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "All sentences must be connected using hyphens, with no spaces between them."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if all sentences are connected using hyphens, with no spaces between them."""
        sentences_gold = re.sub("-", " ", value)
        sentences_gold = instructions_util.split_into_sentences(sentences_gold)
        sentences = value.split("-")
        # Check if there are any spaces between sentences
        feedbacks = []
        for sentence, gold in zip(sentences, sentences_gold):
            if sentence.strip() != sentence:
              feedbacks.append(f"Sentence '{sentence}' has leading / trailing spaces, not allowed. Remove spaces between sentences.")
            if sentence != gold:
              feedbacks.append(f"Segment '{sentence}' does not match '{gold}'. Make sure hyphens are only used to connect sentences with no spaces.")
        if len(feedbacks) == 0:
          return True, "All sentences are correctly connected using hyphens, with no spaces between them."
        return False, ' '.join(feedbacks)


class AdjacentLetterChecker(Instruction):
    """No two adjacent words can start with consecutive letters of the alphabet."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "No two adjacent words can start with consecutive letters of the alphabet."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if no two adjacent words start with consecutive letters of the alphabet."""
        if not value.strip():
            return False, "Response is empty."
        words = value.split()
        empty_word_feedback = "Found a word without valid letters."
        violation_feedback = "Adjacent words start with consecutive letters: "
        num_empty = 0
        word_pairs = []
        for i in range(len(words) - 1):
            first_letter = words[i][0].lower()
            second_letter = words[i + 1][0].lower()
            if len(first_letter) != 1 or len(second_letter) != 1:
                num_empty += 1
            if ord(second_letter) - ord(first_letter) == 1:
                word_pairs.append(f"'{words[i]}' -> '{words[i+1]}'")
        if len(word_pairs) == 0 and num_empty == 0:
           return True, "No two adjacent words start with consecutive letters of the alphabet: True."
        feedback = ""
        if num_empty > 0:
           feedback += empty_word_feedback + " "
        if len(word_pairs) > 0:
           feedback += violation_feedback + ', '.join(word_pairs) + '.'
        return False, feedback


class SquareBracketChecker(Instruction):
    """Enclose every word in your response within square brackets."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Enclose every word in your response within square brackets."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if every word in the response is enclosed within square brackets."""
        if not value.strip():
            return False, "Response is empty."
        words = value.split()
        words_without_brackets = []
        for w in words:
            if not (w.startswith("[") and w.endswith("]")):
                words_without_brackets.append(w)
        if len(words_without_brackets) > 0:
          return False, f"Found {len(words_without_brackets)} words that are not enclosed within square brackets: {words_without_brackets}. Enclose every word within square brackets."
        return True, "Every word in the response is enclosed within square brackets: True."


class KeywordFrequencyOnceChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = 1

        self._description_pattern = "Include keyword {keyword} in your response."

        return self._description_pattern.format(keyword=self._keyword, frequency=self._frequency)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if actual_occurrences == 1:
           return True, f"Response contains keyword '{self._keyword}' only once: True."
        elif actual_occurrences == 0:
           return False, f"Keyword '{self._keyword}' is not found. Response must contain exactly one word '{self._keyword}'."
        else:
           return False, f"Found {actual_occurrences} keywords '{self._keyword}', 1 expected. Remove / replace {actual_occurrences - 1} words '{self._keyword}'."


class KeywordFrequencyCheckerDifferent(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = "In your response, the word {keyword} should appear {frequency} times."

        return self._description_pattern.format(
            keyword=self._keyword, relation=self._comparison_relation, frequency=self._frequency
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword, "frequency": self._frequency, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        feedback = f"Keyword '{self._keyword}' found {actual_occurrences} times, required {self._comparison_relation} {self._frequency}."
        if self._comparison_relation == _COMPARISON_RELATION[0]:
          if actual_occurrences >= self._frequency:
            feedback += f" Remove/replace at least {actual_occurrences - self._frequency + 1} keyword(s)."
          return actual_occurrences < self._frequency, feedback
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
          if actual_occurrences < self._frequency:
            feedback += f" Add at least {self._frequency - actual_occurrences} more keyword(s)."
          return actual_occurrences >= self._frequency, feedback  # pytype: disable=bad-return-type


class ExcludeWordHarderChecker(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, keyword=None, instruction=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings respresenting words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = random.choice(instruction.split())
        else:
            self._keyword = keyword.strip()

        self._description_pattern = "Do not include keyword {keyword} in the response."

        return self._description_pattern.format(keyword=self._keyword)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        if not value.strip():
            return False, "Response is empty."
        passed = " " + self._keyword + " " not in value
        if not passed:
           return False, f"Forbidden keyword '{self._keyword}' found. Exclude it from the response."
        return True, f"Keyword '{self._keyword}' is not found."


class ParagraphBasicChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = (
            "There should be 2 paragraphs. " + "Paragraphs are separated with the markdown divider: ***"
        )

        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False, "Empty paragraph detected between dividers."

        passed = num_paragraphs == 2
        if not passed:
           return False, f"Found {num_paragraphs} separated by '***', required 2. Split / combine paragraphs so that the output has exactly 2 paragraphs separated by '***'."
        return True, "Found exactly 2 paragraphs separated by '***': True."


class ParagraphBasicChecker2(Instruction):
    """Checks the paragraphs."""

    def build_description(self):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = "There should be 2 paragraphs. Paragraphs and only paragraphs are separated with each other by two line breaks. "

        return self._description_pattern.format()

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False, "Empty paragraph detected (multiple consecutive blank lines)."

        passed = num_paragraphs == 2
        if not passed:
           return False, f"Found {num_paragraphs} separated by 2 line breaks, required 2. Split / combine paragraphs so that the output has exactly 2 paragraphs separated by 2 line breaks."
        return True, "Found exactly 2 paragraphs separated by 2 line breaks: True."


class FirstWordSentChecker(Instruction):
    """The first word of each sentence should be the word {first_word}."""

    def build_description(self, first_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the first word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not first_word:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(first_word, str):
                self._first_word = first_word[0].strip()
            else:
                self._first_word = first_word.strip()

        self._description_pattern = "The first word of each sentence should be the word {first_word}."

        return self._description_pattern.format(first_word=self._first_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["first_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        if not value.strip():
            return False, "Response is empty."
        sentences = instructions_util.split_into_sentences(value)

        # Check if the first word of each sentence matches the expected word
        incorrect_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                return False, "Found empty sentence."
            first_word = sentence.split()[0].strip()
            if first_word.lower() != self._first_word.lower():
                incorrect_sentences.append(sentence)
        if len(incorrect_sentences) == 0:
          return True, f"Each sentence starts with word '{self._first_word}': True."
        else:
          return False, f"Found {len(incorrect_sentences)} sentences with the first word different from '{self._first_word}': {incorrect_sentences}. Edit the sentences so that each of them starts with '{self._first_word}'."


class FirstWordAnswerChecker(Instruction):
    """The first word of each sentence should be the word {first_word}."""

    def build_description(self, first_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the first word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not first_word:
            self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._first_word = first_word.strip()

        self._description_pattern = "The first word of your response should be the word {first_word}."

        return self._description_pattern.format(first_word=self._first_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["first_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        words = value.split()
        if not words:
            return False, "Response is empty. Start your response with the required word."
        first_word = words[0].strip()
        passed = first_word.lower() == self._first_word.lower()
        if not passed:
           return False, f"First word of response: '{first_word}', expected '{self._first_word}'. Start your response with '{self._first_word}'."
        return True, f"Response starts with '{self._first_word}': True."


class LastWordSentChecker(Instruction):
    """The last word of each sentence should be the word {last_word}."""

    def build_description(self, last_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the last word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not last_word:
            self._last_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(last_word, str):
                self._last_word = last_word[0].strip()
            else:
                self._last_word = last_word.strip()

        self._description_pattern = (
            "The last word of each sentence, before punctuation, should be the word {last_word}."
        )

        return self._description_pattern.format(last_word=self._last_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["last_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        if not value.strip():
            return False, "Response is empty."
        sentences = instructions_util.split_into_sentences(value)

        # Check if the first word of each sentence matches the expected word
        incorrect_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                return False, "Found empty sentence."
            last_word = sentence.split()[-1].strip()
            # remove any punctuation from last_word
            last_word = re.sub(r"[^\w\s]", "", last_word)
            if last_word.lower() != self._last_word.lower():
                incorrect_sentences.append(sentence)
        if len(incorrect_sentences) == 0:
          return True, f"Each sentence ends with word '{self._last_word}': True."
        else:
          return False, f"Found {len(incorrect_sentences)} sentences with the last word different from '{self._last_word}': {incorrect_sentences}. Edit the sentences so that each of them ends with '{self._last_word}'."


class LastWordAnswerChecker(Instruction):
    """The last word of your response should be the word {last_word}."""

    def build_description(self, last_word=None):
        """Build the instruction description.

        Args:
        first_word: A string representing the last word of each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not last_word:
            self._last_word = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            self._last_word = last_word.strip()

        self._description_pattern = "The last word of your response should be the word {last_word}."

        return self._description_pattern.format(last_word=self._last_word)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["last_word"]

    def check_following(self, value):
        """Checks if the first word of each sentence is the expected word.

        Args:
          value: A string representing the response.

        Returns:
          True if the first word of each sentence is the expected word;
          otherwise, False.
        """
        words = value.split()
        if not words:
          return False, "Response is empty. Provide a response ending with the required word."
        last_word = words[-1].strip()
        # remove any punctuation from last_word
        last_word = re.sub(r"[^\w\s]", "", last_word)
        passed = last_word.lower() == self._last_word.lower()
        if not passed:
          return False, f"Last word of response: '{last_word}', expected '{self._last_word}'. Rewrite the response so that it ends with '{self._last_word}'."
        return True, f"Response ends with '{self._last_word}': True."


class BiGramWrappingChecker(Instruction):
    "Wrap every word bigram in double angular brackets, such as <<I am>> <<at home>> <<with my>> <<cute dog>>."

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Wrap every word bigram in double angular brackets, such as <<I am>> <<at home>> <<with my>> <<cute dog>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if every word bigram is enclosed within double angular brackets."""
        if not value.strip():
            return False, "Response is empty."
        words = value.split()
        bigrams = []
        for i in range(0, len(words) - 1, 2):
            if i + 1 < len(words) and not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
                bigrams.append((words[i], words[i + 1]))
        if len(bigrams) == 0:
           return True, "Every word bigram is wrapped in double angular brackets: True."
        return False, f"Found {len(bigrams)} word bigram(s) not wrapped in double angular brackets: {bigrams}. Wrap all bigrams in double angular brackets."


class CopyingSimpleChecker(Instruction):
    "Repeat the request without change (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"

    def build_description(self, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
        prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = "Repeat the request without change (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        passed = value.strip().lower() == self._prompt_to_repeat.strip().lower()
        if passed:
           return True, "Response repeats the actual request: True."
        return False, f"Response must exactly repeat the request (case insensitive): '{self._prompt_to_repeat}'."


class CopyingMultipleChecker(Instruction):
    "Repeat the request without change {N} times, separated by 6 asterisk symbols (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"

    def build_description(self, prompt_to_repeat=None, N=None):
        """Build the instruction description.

        Args:
        prompt_to_repeat: The prompt that is meant to be repeated.
        N: An integer representing the number of times to repeat the phrase.

        Returns:
        A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        self._description_pattern = "Repeat the request without change {N} times, separated by 6 asterisk symbols (do not say anything before repeating the request; the request you need to repeat does not include this sentence) and do not answer the actual request!"
        return self._description_pattern.format(N=self._N)

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat, "N": self._N}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat", "N"]

    def check_following(self, value):
        prompts = value.split("******")
        if len(prompts) != self._N:
            feedback = f"Expected {self._N} repeats separated by '******'; found {len(prompts)} parts."
            if len(prompts) > self._N:
               feedback += f" Remove exactly {len(prompts) - self._N} repeats."
            else:
               feedback += f" Add exactly {self._N - len(prompts)} more repeats."
            return False, feedback
        feedbacks = []
        for i, p in enumerate(prompts, start=1):
          if p.strip().lower() != self._prompt_to_repeat.strip().lower():
            feedbacks.append(f"Repeat #{i} does not exactly match the request (case-insensitive): '{p.strip()}' != 'self._prompt_to_repeat.strip()'.")
        if len(feedbacks) == 0:
           return True, f"Response repeats the request {self._N} times, separated by '******': True."
        return False, " ".join(feedbacks)


class PunctuationDotChecker(Instruction):
    "In your entire response, refrain from the use of . (i.e. dots) as punctuation and in general."

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "In your entire response, refrain from the use of . (i.e. dots) as punctuation and in general."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain dots."""
        m = re.search(r"\.", value)
        passed = not m
        if not passed:
           return False, f"Found '.' at position {m.start()}. Remove all dots."
        return True, "No dots found in the response."


class PunctuationExclamationChecker(Instruction):
    "In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general."

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "In your entire response, refrain from the use of ! (i.e. exclamation marks) as punctuation and in general."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain exclamation marks."""
        m = re.search(r"\!", value)
        passed = not m
        if not passed:
           return False, f"Found '!' at position {m.start()}. Remove all exclamation marks."
        return True, "No exclamation marks found in the response."


class LowercaseCountingChecker(Instruction):
    "In your response, all lowercase words should appear at most {N} times."

    def build_description(self, N=None):
        """Build the instruction description.

        Args:
        N: An integer representing the maximum number of lowercase words allowed.

        Returns:
        A string representing the instruction description.
        """
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        self._description_pattern = "In your response, all lowercase words should appear at most {N} times."
        return self._description_pattern.format(N=self._N)

    def get_instruction_args(self):
        return {"N": self._N}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["N"]

    def check_following(self, value):
        """Checks that the response does not contain lowercase words more than N times."""
        lowercase_words = re.findall(r"\b[a-z]+\b", value)
        passed = len(lowercase_words) <= self._N
        if passed:
           return True, f"Lowercase words appear {len(lowercase_words)} <= {self._N} times."
        return False, f"Found {len(lowercase_words)} lowercase words, required at most {self._N}. Make at least one letter capital in at least {len(lowercase_words) - self._N} words."


class LetterCountingChecker(Instruction):
    "Answer with {relation} {N} letters."

    def build_description(self, N=None, relation=None):
        """Build the instruction description.

        Args:
        N: An integer representing the maximum number of letters allowed.

        Returns:
        A string representing the instruction description.
        """
        if not N:
            self._N = random.randint(2, 3)
        else:
            self._N = N
        if not relation:
            self._relation = random.choice(_COMPARISON_RELATION)
        else:
            self._relation = relation
        self._description_pattern = "Answer with {relation} {N} letters."
        return self._description_pattern.format(N=self._N, relation=self._relation)

    def get_instruction_args(self):
        return {"N": self._N, "relation": self._relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["N", "relation"]

    def check_following(self, value):
        """Checks that the response does not contain lowercase words more than N times."""
        letters = re.findall(r"[a-zA-Z]", value)
        feedback = f"Response contains {len(letters)} letters, required {self._relation} {self._N}."
        if self._relation == "at least":
            if len(letters) < self._N:
               feedback += f" Extend the response by at least {self._N - len(letters)} letters."
            return len(letters) >= self._N, feedback
        elif self._relation == "less than":
            if len(letters) >= self._N:
              feedback += f" Shorten the response by at least {len(letters) - self._N + 1} letters."
            return len(letters) < self._N, feedback  # pytype: disable=bad-return-type


class CountingCompositionChecker(Instruction):
    "Write 3 paragraphs, delimited by the markdown divider: * * *, with exactly {n_sent} sentences each, with exactly {n_words} words in each sentence."

    def build_description(self, n_sent=None, n_words=None):
        """Build the instruction description.

        Args:
        n_sent: An integer representing the number of sentences in each paragraph.
        n_words: An integer representing the number of words in each sentence.

        Returns:
        A string representing the instruction description.
        """
        if not n_sent:
            self._n_sent = random.randint(2, 3)
        else:
            self._n_sent = n_sent
        if not n_words:
            self._n_words = random.randint(2, 3)
        else:
            self._n_words = n_words
        self._description_pattern = "Write 3 paragraphs, delimited by the markdown divider: * * *, with exactly {n_sent} sentences each, with exactly {n_words} words in each sentence."
        return self._description_pattern.format(n_sent=self._n_sent, n_words=self._n_words)

    def get_instruction_args(self):
        return {"n_sent": self._n_sent, "n_words": self._n_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["n_sent", "n_words"]

    def check_following(self, value):
        """Checks that the response contains the expected number of paragraphs, sentences, and words.

        Args:
          value: A string representing the response.

        Returns:
          True if the response meets the requirements; otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        feedbacks = []
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False, "Empty paragraph detected (multiple consecutive blank lines)."

            sentences = instructions_util.split_into_sentences(paragraph)
            num_sentences = len(sentences)

            if num_sentences != self._n_sent:
                feedbacks.append(f"Paragraph {index + 1} contains {num_sentences} sentences, required exactly {self._n_sent}.")

            for sentence_index, sentence in enumerate(sentences):
                words = instructions_util.nltk.word_tokenize(sentence)
                num_words = len(words)

                if num_words != self._n_words:
                    feedbacks.append(f"Sentence {sentence_index + 1} in paragraph {index + 1} contains {num_words} words, required exactly {self._n_words}.")

        if num_paragraphs != 3:
            feedbacks.append(f"Found {num_paragraphs} paragraphs, required exactly 3.")
        if len(feedbacks) == 0:
           return True, "Found exactly 3 paragraphs separated by the markdown divider * * * : True."
        return False, " ".join(feedbacks)


class CountUniqueChecker(Instruction):
    "Only use unique words in your response, no word should be repeated!"

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Only use unique words in your response, no word should be repeated!"
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response contains unique words."""
        if not value.strip():
            return False, "Response is empty."
        words = instructions_util.nltk.word_tokenize(value)
        first_seen_form = {}
        word_counts = collections.Counter()
        for w in words:
            lw = w.lower()
            if lw not in first_seen_form:
                first_seen_form[lw] = w
            word_counts[lw] += 1
        repeated_words_cnt = {
            word: count for word, count in word_counts.items() if count > 1
        }
        if len(repeated_words_cnt) == 0:
          return True, "No repeated words found in the response: True."
        repeated_words = ", ".join(
            f"{first_seen_form[word]} -- {cnt} times"
            for word, cnt in repeated_words_cnt.items()
        )
        feedback = (
            f"Found {len(repeated_words_cnt)} repeated words in the response: "
            f"{repeated_words}."
        )
        return False, feedback


class CountIncrementWordChecker(Instruction):
    "Include keyword {keyword1} once in your response, keyword {keyword2} twice in your response."

    def build_description(self, keyword1=None, keyword2=None):
        """Build the instruction description.

        Args:
        keyword1: A string representing a keyword that is expected in the response.
        keyword2: A string representing a keyword that is expected in the response.

        Returns:
        A string representing the instruction description.
        """
        if not keyword1:
            self._keyword1 = instructions_util.generate_keywords(num_keywords=1)
        else:
            self._keyword1 = keyword1.strip()
        if not keyword2:
            self._keyword2 = instructions_util.generate_keywords(num_keywords=1)
        else:
            self._keyword2 = keyword2.strip()

        self._description_pattern = (
            "Include keyword {keyword1} once in your response, keyword {keyword2} twice in your response."
        )

        return self._description_pattern.format(keyword1=self._keyword1, keyword2=self._keyword2)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword1": self._keyword1, "keyword2": self._keyword2}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword1", "keyword2"]

    def check_following(self, value):
        """Checks if the response contains the expected number of keywords.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains the expected number of keywords;
          otherwise, False.
        """
        actual_occurrences1 = len(re.findall(self._keyword1, value, flags=re.IGNORECASE))
        actual_occurrences2 = len(re.findall(self._keyword2, value, flags=re.IGNORECASE))
        feedback = f"Keyword '{self._keyword1}' was used {actual_occurrences1} times, required 1. Keyword '{self._keyword2}' was used {actual_occurrences2} times, required 2."

        if actual_occurrences1 < 1:
          feedback += f" Use keyword '{self._keyword1}' exactly once."
        elif actual_occurrences1 > 1:
          feedback += f" Remove / replace {actual_occurrences1 - 1} keywords '{self._keyword1}'."

        if actual_occurrences1 < 2:
          feedback += f" Use keyword '{self._keyword2}' exactly twice: add {2 - actual_occurrences2} more '{self._keyword2}'."
        elif actual_occurrences1 > 2:
          feedback += f" Remove / replace {actual_occurrences2 - 2} keywords '{self._keyword2}'."

        return bool(actual_occurrences1 == 1 and actual_occurrences2 == 2), feedback


class PalindromeBasicChecker(Instruction):
    "Include a palindrome in your response."

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Include a palindrome in your response."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a palindrome.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains a palindrome; otherwise, False.
        """
        palindromes = [word for word in value.split() if word == word[::-1]]
        return len(palindromes) > 0, f"Found {len(palindromes)} palindromes, required at least 1."


class KeywordSpecificPositionChecker(Instruction):
    "Include keyword {keyword1} in the {n}-th sentence, as the {m}-th word of that sentence."

    def build_description(self, keyword=None, n=None, m=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          n: An integer representing the sentence number.
          m: An integer representing the word number.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            if not isinstance(keyword, str):
                self._keyword = keyword[0].strip()
            else:
                self._keyword = keyword.strip()
        if not n:
            self._n = random.randint(1, 20)
        else:
            self._n = n
        if not m:
            self._m = random.randint(1, 30)
        else:
            self._m = m

        self._description_pattern = (
            "Include keyword {keyword} in the {n}-th sentence, as the {m}-th word of that sentence."
        )

        return self._description_pattern.format(keyword=self._keyword, n=self._n, m=self._m)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keyword": self._keyword, "n": self._n, "m": self._m}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "n", "m"]

    def check_following(self, value):
        """Checks if the response contains the expected number of keywords.

        Args:
          value: A string representing the response.

        Returns:
          True if the response contains the expected number of keywords;
          otherwise, False.
        """
        sentences = instructions_util.split_into_sentences(value)
        if len(sentences) < self._n:
            return False, f"Response contains {len(sentences)}, required at least {self._n}, and the {self._m}th word in the {self._n}th sentence must be '{self._keyword}'."
        words = instructions_util.nltk.word_tokenize(sentences[self._n - 1])
        if len(words) < self._m:
            return False, f"Sentence {self._n} contains {len(words)}, required at least {self._m}, and the {self._m}th word must be '{self._keyword}'."
        return words[self._m - 1] == self._keyword, f"Word {self._m} in Sentence _{self._n}: {words[self._m - 1]}, expected {self._keyword}."


class StartEndChecker(Instruction):
    "Start and end your response with the same word (do not write anything after the last word, not even punctuation)."

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "Start and end your response with the same word (do not write anything after the last word, not even punctuation)."
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response starts and ends with the same word.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts and ends with the same word;
          otherwise, False.
        """
        words = instructions_util.nltk.word_tokenize(value)
        if len(words) < 2:
            return False, "Need at least two words in the response."
        feedback = (
            f"The first word of the response: {words[0]}, the last word of "
            f"the response: {words[-1]}. Expected the first and last words to "
            "match (case-insensitive)."
        )
        passed = words[0].lower() == words[-1].lower()
        if not passed:
           return False, feedback + f" {words[0]} != {words[-1]}."
        return True, feedback + f" {words[0]} == {words[-1]}."
