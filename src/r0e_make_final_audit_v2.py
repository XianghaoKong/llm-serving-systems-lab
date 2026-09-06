#!/usr/bin/env python3
"""
R0-E audit patch — rebuild the final 90-row human audit with enough context
to actually review Document QA and Long-context QA.

Why this exists
---------------
The original `r0e_final_audit.csv` kept only the first 800 characters of each
prompt. For document-based requests, the question appears near the END of the
prompt, so the audit file could not verify whether the question/document pair
looked sensible.

This script recreates the SAME deterministic 15-per-category sample used by
R0-E and adds:
- request_id
- explicit question text when available
- prompt head
- prompt tail
- document title-ish preview
- response-median metadata where available

Input
-----
workloads/final/realistic_requests.json

Output
------
workloads/final/r0e_final_audit_v2.csv
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_JSON = PROJECT_ROOT / "workloads" / "final" / "realistic_requests.json"
OUTPUT_CSV = PROJECT_ROOT / "workloads" / "final" / "r0e_final_audit_v2.csv"

# R0-E used args.seed=911, then write_audit_sample(seed + 7777)
AUDIT_SEED = 911 + 7777
PER_CATEGORY = 15

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def observed_response_median(row: Dict[str, Any]):
    metadata = row.get("selection_metadata")
    if isinstance(metadata, dict):
        return metadata.get("response_token_median")
    return None


def question_text(row: Dict[str, Any]) -> str:
    category = row.get("workload_category")

    if category in {"document_qa", "long_context_qa"}:
        return normalize_space(row.get("question"))

    if category == "knowledge_qa":
        return normalize_space(row.get("prompt"))

    return ""


def prompt_head(prompt: str, n: int = 500) -> str:
    return normalize_space(prompt)[:n]


def prompt_tail(prompt: str, n: int = 700) -> str:
    clean = normalize_space(prompt)
    return clean[-n:]


def document_lead(row: Dict[str, Any], n: int = 250) -> str:
    category = row.get("workload_category")
    if category not in {"document_qa", "long_context_qa"}:
        return ""

    prompt = normalize_space(row.get("prompt"))
    marker = "Document:"
    pos = prompt.find(marker)

    if pos == -1:
        return prompt[:n]

    start = pos + len(marker)
    return prompt[start:start + n]


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing canonical workload: {INPUT_JSON}"
        )

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise RuntimeError("Canonical workload JSON has no requests list.")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in requests:
        groups[str(row["workload_category"])].append(row)

    rng = random.Random(AUDIT_SEED)
    audit_rows: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        group = groups[category]
        if len(group) < PER_CATEGORY:
            raise RuntimeError(
                f"{category}: only {len(group)} rows available."
            )

        # This reproduces the same sequential RNG sampling pattern used by R0-E.
        chosen = rng.sample(group, PER_CATEGORY)

        for row in chosen:
            prompt = str(row.get("prompt", ""))

            audit_rows.append({
                "request_id": row.get("request_id"),
                "workload_category": category,
                "source": row.get("source"),
                "source_id": row.get("source_id"),
                "input_tokens": row.get("input_tokens"),
                "max_new_tokens": row.get("max_new_tokens"),
                "observed_response_token_median": observed_response_median(row),
                "question": question_text(row),
                "document_lead": document_lead(row),
                "prompt_head": prompt_head(prompt),
                "prompt_tail": prompt_tail(prompt),
                "human_suitable": "",
                "human_notes": "",
            })

    df = pd.DataFrame(audit_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 76)
    print("R0-E ENHANCED FINAL AUDIT CREATED")
    print("=" * 76)
    print(f"Rows:   {len(df)}")
    print(f"Output: {OUTPUT_CSV}")
    print()
    print("The same 15-per-category final audit sample is now reviewable for")
    print("Document QA and Long-context QA because the question and prompt tail")
    print("are included explicitly.")


if __name__ == "__main__":
    main()
