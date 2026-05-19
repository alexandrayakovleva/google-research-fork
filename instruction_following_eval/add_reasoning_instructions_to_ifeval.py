#!/usr/bin/env python3
"""Add agent-reasoning instructions to an existing IF-Eval JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_GOOGLE_RESEARCH_ROOT = "/home/yakovla2/repos/google-research-fork"

DEFAULT_REASONING_INSTRUCTION_IDS = [
    # "agent_reasoning:answer_draft_marker",
    "agent_reasoning:sections",
    # "agent_reasoning:two_approaches",
]


def import_reasoning_builder(google_research_root: str):
    if google_research_root not in sys.path:
        sys.path.insert(0, google_research_root)
    from instruction_following_eval.generate_reasonif_agent_dataset import (
        TARGET_INSTRUCTION_IDS,
        _build_instruction,
    )

    return TARGET_INSTRUCTION_IDS, _build_instruction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_instruction_ids(raw_ids: str) -> list[str]:
    return [item.strip() for item in raw_ids.split(",") if item.strip()]


def make_instruction_plan(
    num_records: int,
    instruction_ids: list[str],
    sample_mode: str,
    rng: random.Random,
) -> list[str]:
    if sample_mode == "random":
        return [rng.choice(instruction_ids) for _ in range(num_records)]

    plan = [
        instruction_ids[index % len(instruction_ids)]
        for index in range(num_records)
    ]
    rng.shuffle(plan)
    return plan


def combine_prompt(prompt: str, reasoning_description: str) -> str:
    return (
        prompt
        + "\n\n<thinking guidelines>\n"
        + reasoning_description
        + "\n</thinking guidelines>"
    )


def add_reasoning_instructions(
    records: list[dict[str, Any]],
    instruction_ids: list[str],
    seed: int,
    sample_mode: str,
    build_instruction,
    reasoning_id_field: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    random.seed(seed)
    instruction_plan = make_instruction_plan(
        num_records=len(records),
        instruction_ids=instruction_ids,
        sample_mode=sample_mode,
        rng=rng,
    )

    new_records = []
    for record, instruction_id in zip(records, instruction_plan, strict=True):
        prompt = record.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"Row with key={record.get('key')} is missing a string prompt.")

        description, kwargs = build_instruction(instruction_id, rng)
        combined = combine_prompt(prompt, description)

        new_record = dict(record)
        new_record["prompt"] = combined
        new_record[reasoning_id_field] = instruction_id
        new_record["reasoning_kwargs"] = kwargs
        new_records.append(new_record)

    return new_records


def count_reasoning_ids(records: list[dict[str, Any]], reasoning_id_field: str) -> Counter[str]:
    return Counter(
        record[reasoning_id_field]
        for record in records
        if reasoning_id_field in record
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment IF-Eval rows with one agent_reasoning instruction and "
            "replace prompt with a thinking-guidelines prompt."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--reasoning-instruction-ids",
        default=",".join(DEFAULT_REASONING_INSTRUCTION_IDS),
        help="Comma-separated agent reasoning instruction IDs.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "random"],
        default="balanced",
        help="Use balanced counts or reference-style random sampling.",
    )
    parser.add_argument(
        "--reasoning-id-field",
        default="reasoning_instruction_id_list",
        help=(
            "Output field for the reasoning instruction id. The default keeps "
            "the requested spelling."
        ),
    )
    parser.add_argument(
        "--google-research-root",
        default=DEFAULT_GOOGLE_RESEARCH_ROOT,
        help="Path containing the instruction_following_eval package.",
    )
    args = parser.parse_args()

    supported_instruction_ids, build_instruction = import_reasoning_builder(
        args.google_research_root
    )
    instruction_ids = parse_instruction_ids(args.reasoning_instruction_ids)
    if not instruction_ids:
        raise ValueError("Need at least one reasoning instruction ID.")

    missing = [
        instruction_id
        for instruction_id in instruction_ids
        if instruction_id not in supported_instruction_ids
    ]
    if missing:
        raise ValueError(f"Unknown IF-Eval instruction IDs: {missing}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = read_jsonl(input_path)
    new_records = add_reasoning_instructions(
        records=records,
        instruction_ids=instruction_ids,
        seed=args.seed,
        sample_mode=args.sample_mode,
        build_instruction=build_instruction,
        reasoning_id_field=args.reasoning_id_field,
    )
    write_jsonl(output_path, new_records)

    counts = count_reasoning_ids(new_records, args.reasoning_id_field)
    print(f"Read {len(records)} rows from {input_path}")
    print(f"Wrote {len(new_records)} rows to {output_path}")
    print("\nReasoning instruction counts")
    for instruction_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {instruction_id}: {count}")


if __name__ == "__main__":
    main()
