#!/usr/bin/env python3
"""
R0-E final patch — replace a small set of audit-identified weak samples,
preserve workload counts/distributions, and create a replacement audit.

This script:
1) Reads workloads/final/realistic_requests.json.
2) Reads the six R0-D shortlist JSONL files.
3) Replaces only the manually rejected samples listed below.
4) Chooses replacements from the SAME workload category.
5) Avoids global prompt duplicates and cross-NQ question duplicates.
6) Prefers replacements closest in input length; for long_output it also
   prefers a similar observed response-token median.
7) Creates a backup before updating the canonical workload.
8) Regenerates a metadata-only manifest and a replacement audit.

It does NOT run inference.

Outputs
-------
workloads/final/
├── realistic_requests_pre_freeze.json   # backup
├── realistic_requests.json              # patched canonical workload
├── public_workload_manifest.csv         # regenerated
├── r0e_replacement_audit.csv            # inspect this before freeze
└── r0e_patch_report.json

Run
---
python src/r0e_patch_and_prepare_freeze.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FINAL_DIR = PROJECT_ROOT / "workloads" / "final"
TOKEN_DIR = PROJECT_ROOT / "workloads" / "token_bands"

CANONICAL = FINAL_DIR / "realistic_requests.json"
BACKUP = FINAL_DIR / "realistic_requests_pre_freeze.json"
MANIFEST = FINAL_DIR / "public_workload_manifest.csv"
REPLACEMENT_AUDIT = FINAL_DIR / "r0e_replacement_audit.csv"
PATCH_REPORT = FINAL_DIR / "r0e_patch_report.json"

SHORTLISTS = {
    "short_interactive": TOKEN_DIR / "short_interactive_shortlist.jsonl",
    "knowledge_qa": TOKEN_DIR / "knowledge_qa_shortlist.jsonl",
    "coding_request": TOKEN_DIR / "coding_request_shortlist.jsonl",
    "document_qa": TOKEN_DIR / "document_qa_shortlist.jsonl",
    "long_context_qa": TOKEN_DIR / "long_context_qa_shortlist.jsonl",
    "long_output": TOKEN_DIR / "long_output_shortlist.jsonl",
}

EXPECTED_COUNTS = {
    "short_interactive": 300,
    "knowledge_qa": 200,
    "coding_request": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "long_output": 100,
}

DOCUMENT_BAND = (1024, 2048)
LONG_CONTEXT_BAND = (3072, 4096)

# These were identified in the final 90-row human audit.
# The reasons are about benchmark quality / public suitability, not model safety.
MANUAL_REJECTS = {
    # Short Interactive
    "cad2852472c5420bbebf9438a57a609d":
        "unsafe/implausible fuel-to-set-person-like-subject-on-fire prompt",
    "11a2862bef004d6e8b514245e712ea33":
        "group-targeted joke is unnecessary for a public systems benchmark",
    "a3186685600a47b8bc84b4beee2a4966":
        "high-stakes medication-effect question unnecessary for load testing",

    # Knowledge QA
    "train_8613":
        "depends on missing external figure: 'line A'",

    # Document QA
    "-5517860695160245892":
        "image-search/navigational query rather than document QA",
    "4332328861050610531":
        "travel-time question lacks origin and is incomplete",

    # Long-context QA
    "-7018183719515876184":
        "navigational Wikipedia query rather than a clear QA request",
    "-4062022321288664632":
        "depends on missing answer choices: 'which of these'",
    "3827949212799340147":
        "depends on missing answer choices: 'which of the following'",

    # Long Output
    "1ea482b0b7e44700afbb37add04012c1":
        "high-stakes investment-strategy advice unnecessary for load testing",
}

# Generic completeness filters applied ONLY when selecting replacements.
INCOMPLETE_NQ_PATTERNS = [
    r"\bwhich of (?:the )?(?:following|these)\b",
    r"\bwhat (?:of|among) (?:the )?(?:following|these)\b",
    r"\bline [a-z]\b",
    r"^\s*images?\s+of\b",
    r"^\s*wikipedia\b",
    r"^\s*how long does it take to get to\b",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_for_dedup(text: str) -> str:
    return (
        normalize_space(text)
        .lower()
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def prompt_hash(row: Dict[str, Any]) -> str:
    return stable_hash(normalize_for_dedup(str(row.get("prompt", ""))))


def nq_question_text(row: Dict[str, Any], category: str) -> Optional[str]:
    if category == "knowledge_qa":
        text = normalize_space(row.get("prompt"))
        return text or None

    if category in {"document_qa", "long_context_qa"}:
        text = normalize_space(row.get("question"))
        return text or None

    return None


def nq_question_hash(row: Dict[str, Any], category: str) -> Optional[str]:
    text = nq_question_text(row, category)
    if not text:
        return None
    return stable_hash(normalize_for_dedup(text))


def incomplete_nq_question(row: Dict[str, Any], category: str) -> bool:
    text = nq_question_text(row, category)
    if not text:
        return category in {"knowledge_qa", "document_qa", "long_context_qa"}

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in INCOMPLETE_NQ_PATTERNS
    )


def response_median(row: Dict[str, Any]) -> Optional[float]:
    metadata = row.get("selection_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("response_token_median")
        if value is not None:
            return float(value)
    return None


def source_license(source: str) -> str:
    if source == "lmsys_chatbot_arena_conversations":
        return "CC-BY-4.0 (user prompts)"
    if source in {"google_nq_open", "google_natural_questions"}:
        return "CC-BY-SA-3.0"
    return "see source dataset"


def validate_band(row: Dict[str, Any], category: str) -> bool:
    tokens = int(row["input_tokens"])

    if category == "document_qa":
        return DOCUMENT_BAND[0] <= tokens <= DOCUMENT_BAND[1]

    if category == "long_context_qa":
        return LONG_CONTEXT_BAND[0] <= tokens <= LONG_CONTEXT_BAND[1]

    return tokens > 0


def candidate_distance(
    candidate: Dict[str, Any],
    rejected: Dict[str, Any],
    category: str,
) -> Tuple[float, float, str]:
    input_distance = abs(
        float(candidate["input_tokens"])
        - float(rejected["input_tokens"])
    )

    if category == "long_output":
        cand_resp = response_median(candidate)
        rej_resp = response_median(rejected)

        if cand_resp is not None and rej_resp is not None:
            response_distance = abs(cand_resp - rej_resp)
        else:
            response_distance = 1e9

        # For long-output, preserve observed decode-behavior first.
        return (
            response_distance,
            input_distance,
            prompt_hash(candidate),
        )

    return (
        input_distance,
        0.0,
        prompt_hash(candidate),
    )


def build_used_sets(
    final_rows: Sequence[Dict[str, Any]],
) -> Tuple[set, set, set]:
    prompt_hashes = set()
    source_keys = set()
    nq_hashes = set()

    for row in final_rows:
        category = str(row["workload_category"])
        prompt_hashes.add(prompt_hash(row))
        source_keys.add(
            (
                str(row.get("source", "")),
                str(row.get("source_id", "")),
            )
        )

        q_hash = nq_question_hash(row, category)
        if q_hash:
            nq_hashes.add(q_hash)

    return prompt_hashes, source_keys, nq_hashes


def valid_replacement(
    candidate: Dict[str, Any],
    category: str,
    *,
    used_prompt_hashes: set,
    used_source_keys: set,
    used_nq_hashes: set,
) -> bool:
    if str(candidate.get("workload_category")) != category:
        return False

    source_id = str(candidate.get("source_id", ""))

    if source_id in MANUAL_REJECTS:
        return False

    if not validate_band(candidate, category):
        return False

    p_hash = prompt_hash(candidate)

    if p_hash in used_prompt_hashes:
        return False

    source_key = (
        str(candidate.get("source", "")),
        source_id,
    )

    if source_key in used_source_keys:
        return False

    if category in {
        "knowledge_qa",
        "document_qa",
        "long_context_qa",
    }:
        if incomplete_nq_question(candidate, category):
            return False

        q_hash = nq_question_hash(candidate, category)

        if not q_hash:
            return False

        if q_hash in used_nq_hashes:
            return False

    return True


def regenerate_manifest(
    final_rows: Sequence[Dict[str, Any]],
) -> None:
    rows = []

    for row in final_rows:
        rows.append({
            "request_id": row["request_id"],
            "workload_category": row["workload_category"],
            "source": row["source"],
            "source_id": row["source_id"],
            "source_license": source_license(str(row["source"])),
            "prompt_hash": prompt_hash(row),
            "question_hash": (
                nq_question_hash(
                    row,
                    str(row["workload_category"]),
                )
                or ""
            ),
            "input_tokens": row["input_tokens"],
            "max_new_tokens": row["max_new_tokens"],
            "has_document_context": (
                row["workload_category"]
                in {"document_qa", "long_context_qa"}
            ),
        })

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)


def integrity_check(final_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(final_rows) != 1000:
        raise RuntimeError(
            f"Expected 1000 requests, found {len(final_rows)}."
        )

    counts: Dict[str, int] = {
        category: 0
        for category in EXPECTED_COUNTS
    }

    p_hashes = []
    nq_hashes = []

    for row in final_rows:
        category = str(row["workload_category"])
        counts[category] += 1

        if not validate_band(row, category):
            raise RuntimeError(
                f"Token-band violation in {row['request_id']}"
            )

        p_hashes.append(prompt_hash(row))

        q_hash = nq_question_hash(row, category)
        if q_hash:
            nq_hashes.append(q_hash)

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Category counts changed.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Actual:   {counts}"
        )

    if len(set(p_hashes)) != len(p_hashes):
        raise RuntimeError("Duplicate prompt found after patch.")

    if len(set(nq_hashes)) != len(nq_hashes):
        raise RuntimeError("Cross-NQ overlap found after patch.")

    return {
        "total_requests": len(final_rows),
        "category_counts": counts,
        "unique_prompt_hashes": len(set(p_hashes)),
        "cross_nq_duplicates": 0,
    }


def main() -> None:
    if not CANONICAL.exists():
        raise FileNotFoundError(
            f"Missing canonical workload: {CANONICAL}"
        )

    for category, path in SHORTLISTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing shortlist for {category}: {path}"
            )

    with CANONICAL.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    final_rows = payload.get("requests")

    if not isinstance(final_rows, list):
        raise RuntimeError("Canonical JSON has no requests list.")

    reject_rows = [
        row
        for row in final_rows
        if str(row.get("source_id")) in MANUAL_REJECTS
    ]

    found_reject_ids = {
        str(row.get("source_id"))
        for row in reject_rows
    }

    missing_reject_ids = set(MANUAL_REJECTS) - found_reject_ids

    if missing_reject_ids:
        print(
            "Warning: some audit reject source IDs were not found "
            "in the current canonical workload:"
        )
        for source_id in sorted(missing_reject_ids):
            print(f"  {source_id}")

    if not reject_rows:
        raise RuntimeError(
            "No manual-reject rows found. "
            "The workload may already have been patched."
        )

    # Backup once.
    if not BACKUP.exists():
        shutil.copy2(CANONICAL, BACKUP)

    # Temporarily remove rejected rows before building uniqueness sets.
    rejected_request_ids = {
        str(row["request_id"])
        for row in reject_rows
    }

    working_rows = [
        row
        for row in final_rows
        if str(row["request_id"])
        not in rejected_request_ids
    ]

    used_prompt_hashes, used_source_keys, used_nq_hashes = build_used_sets(
        working_rows
    )

    shortlist_cache = {
        category: read_jsonl(path)
        for category, path in SHORTLISTS.items()
    }

    replacements: List[Dict[str, Any]] = []
    patched_rows = list(working_rows)

    for rejected in sorted(
        reject_rows,
        key=lambda row: str(row["request_id"]),
    ):
        category = str(rejected["workload_category"])

        candidates = [
            candidate
            for candidate in shortlist_cache[category]
            if valid_replacement(
                candidate,
                category,
                used_prompt_hashes=used_prompt_hashes,
                used_source_keys=used_source_keys,
                used_nq_hashes=used_nq_hashes,
            )
        ]

        if not candidates:
            raise RuntimeError(
                f"No valid replacement found for "
                f"{rejected['request_id']} ({category})."
            )

        candidates.sort(
            key=lambda candidate: candidate_distance(
                candidate,
                rejected,
                category,
            )
        )

        replacement = dict(candidates[0])

        # Keep the request ID stable so downstream references remain clear.
        replacement["request_id"] = rejected["request_id"]
        replacement["workload_category"] = category
        replacement["prompt_hash"] = prompt_hash(replacement)

        q_hash = nq_question_hash(replacement, category)
        if q_hash:
            replacement["question_hash"] = q_hash

        patched_rows.append(replacement)

        used_prompt_hashes.add(prompt_hash(replacement))
        used_source_keys.add(
            (
                str(replacement.get("source", "")),
                str(replacement.get("source_id", "")),
            )
        )
        if q_hash:
            used_nq_hashes.add(q_hash)

        replacements.append({
            "request_id": rejected["request_id"],
            "workload_category": category,
            "rejected_source_id": rejected.get("source_id"),
            "rejection_reason": MANUAL_REJECTS[
                str(rejected.get("source_id"))
            ],
            "replacement_source_id": replacement.get("source_id"),
            "old_input_tokens": rejected.get("input_tokens"),
            "new_input_tokens": replacement.get("input_tokens"),
            "old_response_token_median": response_median(rejected),
            "new_response_token_median": response_median(replacement),
            "replacement_question": (
                nq_question_text(replacement, category)
                or ""
            ),
            "replacement_prompt_preview": str(
                replacement.get("prompt", "")
            )[:800],
        })

    # Restore canonical request order by request_id.
    patched_rows.sort(
        key=lambda row: int(
            str(row["request_id"]).split("_")[-1]
        )
    )

    integrity = integrity_check(patched_rows)

    payload["metadata"]["version"] = "R0-E-v1-patched-pre-freeze"
    payload["metadata"]["manual_audit_replacements"] = len(replacements)
    payload["requests"] = patched_rows

    with CANONICAL.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    regenerate_manifest(patched_rows)

    pd.DataFrame(replacements).to_csv(
        REPLACEMENT_AUDIT,
        index=False,
    )

    report = {
        "stage": "R0-E-final-patch",
        "manual_reject_count": len(reject_rows),
        "replacement_count": len(replacements),
        "manual_reject_reasons": MANUAL_REJECTS,
        "integrity": integrity,
        "backup": str(BACKUP),
        "patched_canonical": str(CANONICAL),
        "replacement_audit": str(REPLACEMENT_AUDIT),
    }

    with PATCH_REPORT.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print("R0-E FINAL PATCH COMPLETE")
    print("=" * 80)
    print(f"Replacements:        {len(replacements)}")
    print(f"Total requests:      {integrity['total_requests']}")
    print(f"Unique prompt hashes:{integrity['unique_prompt_hashes']}")
    print(f"Cross-NQ duplicates: {integrity['cross_nq_duplicates']}")
    print()
    print(f"Backup:              {BACKUP}")
    print(f"Patched workload:    {CANONICAL}")
    print(f"Replacement audit:   {REPLACEMENT_AUDIT}")
    print(f"Patch report:        {PATCH_REPORT}")
    print()
    print("Review r0e_replacement_audit.csv before freezing the workload.")


if __name__ == "__main__":
    main()
