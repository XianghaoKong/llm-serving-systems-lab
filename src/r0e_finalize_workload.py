#!/usr/bin/env python3
"""
R0-E — Final 1,000-request corpus construction and quality control.

Inputs
------
workloads/token_bands/
├── short_interactive_shortlist.jsonl   (600)
├── knowledge_qa_shortlist.jsonl        (400)
├── coding_request_shortlist.jsonl      (300)
├── document_qa_shortlist.jsonl         (300)
├── long_context_qa_shortlist.jsonl     (200)
└── long_output_shortlist.jsonl         (200)

Final targets
-------------
- short_interactive: 300
- knowledge_qa:      200
- coding_request:    150
- document_qa:       150
- long_context_qa:   100
- long_output:       100
TOTAL:              1000

What R0-E does
--------------
1) Validates every R0-D shortlist row.
2) Applies a conservative public-repository suitability filter.
3) Removes global duplicate prompts.
4) Prevents the same Natural Questions question from appearing in more than
   one of knowledge_qa / document_qa / long_context_qa.
5) Uses deterministic distribution-preserving sampling rather than a purely
   random 50% cut.
6) Re-validates exact document token bands.
7) Produces a local canonical workload plus a public metadata-only manifest.
8) Produces a final QA sample and a report.

Important
---------
- `realistic_requests.json` contains full prompts and is intended to remain
  LOCAL during the benchmark-development phase.
- `public_workload_manifest.csv` contains metadata/hashes only and is safe to
  review for GitHub publishing.
- The suitability filter is deliberately conservative. It is NOT a general
  safety/moderation classifier.

Outputs
-------
workloads/final/
├── realistic_requests.json
├── public_workload_manifest.csv
├── r0e_final_report.json
├── r0e_final_audit.csv
└── r0e_flagged_candidates.csv

Run
---
python src/r0e_finalize_workload.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SEED = 911

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TOKEN_DIR = PROJECT_ROOT / "workloads" / "token_bands"
OUT_DIR = PROJECT_ROOT / "workloads" / "final"

INPUT_FILES = {
    "short_interactive": TOKEN_DIR / "short_interactive_shortlist.jsonl",
    "knowledge_qa": TOKEN_DIR / "knowledge_qa_shortlist.jsonl",
    "coding_request": TOKEN_DIR / "coding_request_shortlist.jsonl",
    "document_qa": TOKEN_DIR / "document_qa_shortlist.jsonl",
    "long_context_qa": TOKEN_DIR / "long_context_qa_shortlist.jsonl",
    "long_output": TOKEN_DIR / "long_output_shortlist.jsonl",
}

FINAL_TARGETS = {
    "short_interactive": 300,
    "knowledge_qa": 200,
    "coding_request": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "long_output": 100,
}

DOCUMENT_BAND = (1024, 2048)
LONG_CONTEXT_BAND = (3072, 4096)

CANONICAL_JSON = OUT_DIR / "realistic_requests.json"
PUBLIC_MANIFEST = OUT_DIR / "public_workload_manifest.csv"
REPORT_JSON = OUT_DIR / "r0e_final_report.json"
AUDIT_CSV = OUT_DIR / "r0e_final_audit.csv"
FLAGGED_CSV = OUT_DIR / "r0e_flagged_candidates.csv"


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
    text = normalize_space(text).lower()
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    return text


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


REVIEW_PATTERNS: List[Tuple[str, str]] = [
    ("possible_private_email", r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"),
    ("possible_credit_card", r"\b(?:\d[ -]*?){13,19}\b"),
    ("possible_ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    (
        "explicit_self_harm",
        r"\b(?:kill myself|commit suicide|suicide method|how to suicide)\b",
    ),
    (
        "explicit_explosive_instruction",
        r"\b(?:how to make|how do i make|build|construct)\b.{0,40}\b"
        r"(?:bomb|explosive device)\b",
    ),
    (
        "credential_theft_or_malware",
        r"\b(?:steal passwords?|credential theft|ransomware|keylogger|"
        r"password stealer|malware payload)\b",
    ),
    (
        "explicit_sexual_content",
        r"\b(?:explicit pornography|pornographic sex|rape fantasy)\b",
    ),
]

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def suitability_reasons(row: Dict[str, Any]) -> List[str]:
    prompt = str(row.get("prompt", ""))
    reasons: List[str] = []
    if not normalize_space(prompt):
        reasons.append("empty_prompt")
        return reasons
    if CONTROL_CHAR_RE.search(prompt):
        reasons.append("control_characters")
    for reason, pattern in REVIEW_PATTERNS:
        if re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL):
            reasons.append(reason)
    return reasons


def validate_row(row: Dict[str, Any], expected_category: str) -> None:
    required = [
        "workload_category",
        "source",
        "source_id",
        "prompt",
        "input_tokens",
        "max_new_tokens",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise RuntimeError(
            f"{expected_category}: missing fields {missing} "
            f"for source_id={row.get('source_id')}"
        )
    if row["workload_category"] != expected_category:
        raise RuntimeError(
            f"Category mismatch: expected {expected_category}, "
            f"found {row['workload_category']}"
        )
    input_tokens = int(row["input_tokens"])
    if input_tokens <= 0:
        raise RuntimeError(
            f"{expected_category}: invalid input_tokens={input_tokens}"
        )
    if int(row["max_new_tokens"]) <= 0:
        raise RuntimeError(f"{expected_category}: invalid max_new_tokens")
    if expected_category == "document_qa":
        if not (DOCUMENT_BAND[0] <= input_tokens <= DOCUMENT_BAND[1]):
            raise RuntimeError(
                f"document_qa token-band violation: {input_tokens}"
            )
    if expected_category == "long_context_qa":
        if not (LONG_CONTEXT_BAND[0] <= input_tokens <= LONG_CONTEXT_BAND[1]):
            raise RuntimeError(
                f"long_context_qa token-band violation: {input_tokens}"
            )


def prepare_pools(
    raw_pools: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
    flagged: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "raw_counts": {},
        "removed_suitability": Counter(),
        "removed_duplicate_prompt": Counter(),
        "removed_duplicate_source_id": Counter(),
        "removed_nq_question_overlap": Counter(),
    }

    for category, rows in raw_pools.items():
        report["raw_counts"][category] = len(rows)
        for row in rows:
            validate_row(row, category)

    priority = [
        "long_context_qa",
        "document_qa",
        "long_output",
        "coding_request",
        "short_interactive",
        "knowledge_qa",
    ]

    clean: Dict[str, List[Dict[str, Any]]] = {
        category: [] for category in raw_pools
    }
    seen_prompt_hashes = set()
    seen_source_ids = set()
    seen_nq_questions = set()

    for category in priority:
        for row in raw_pools[category]:
            reasons = suitability_reasons(row)
            if reasons:
                report["removed_suitability"][category] += 1
                flagged.append({
                    "workload_category": category,
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                    "reasons": " | ".join(reasons),
                    "input_tokens": row.get("input_tokens"),
                    "prompt_preview": str(row.get("prompt", ""))[:500],
                })
                continue

            p_hash = prompt_hash(row)
            if p_hash in seen_prompt_hashes:
                report["removed_duplicate_prompt"][category] += 1
                continue

            source_id_key = (
                str(row.get("source", "")),
                str(row.get("source_id", "")),
            )
            if source_id_key in seen_source_ids:
                report["removed_duplicate_source_id"][category] += 1
                continue

            q_hash = nq_question_hash(row, category)
            if q_hash is not None and q_hash in seen_nq_questions:
                report["removed_nq_question_overlap"][category] += 1
                continue

            item = dict(row)
            item["prompt_hash"] = p_hash
            if q_hash is not None:
                item["question_hash"] = q_hash
                seen_nq_questions.add(q_hash)

            clean[category].append(item)
            seen_prompt_hashes.add(p_hash)
            seen_source_ids.add(source_id_key)

    report["clean_counts"] = {
        category: len(rows) for category, rows in clean.items()
    }
    for key in [
        "removed_suitability",
        "removed_duplicate_prompt",
        "removed_duplicate_source_id",
        "removed_nq_question_overlap",
    ]:
        report[key] = dict(report[key])

    return clean, flagged, report


def selection_key(
    row: Dict[str, Any],
    category: str,
) -> Tuple[float, float, str]:
    input_tokens = float(row["input_tokens"])
    response_median = 0.0
    metadata = row.get("selection_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("response_token_median")
        if value is not None:
            response_median = float(value)

    if category == "long_output":
        return (response_median, input_tokens, row["prompt_hash"])
    return (input_tokens, response_median, row["prompt_hash"])


def systematic_distribution_sample(
    rows: List[Dict[str, Any]],
    *,
    category: str,
    target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(rows) < target:
        raise RuntimeError(
            f"{category}: only {len(rows)} clean rows available; need {target}"
        )

    ordered = sorted(rows, key=lambda r: selection_key(r, category))
    n = len(ordered)
    chosen_indices = set()

    for i in range(target):
        start = math.floor(i * n / target)
        end = math.floor((i + 1) * n / target) - 1
        if end < start:
            end = start
        idx = rng.randint(start, end)
        if idx in chosen_indices:
            for candidate_idx in range(start, end + 1):
                if candidate_idx not in chosen_indices:
                    idx = candidate_idx
                    break
        chosen_indices.add(idx)

    if len(chosen_indices) != target:
        raise RuntimeError(
            f"{category}: sampler selected {len(chosen_indices)} rows; "
            f"expected {target}"
        )

    return [ordered[i] for i in sorted(chosen_indices)]


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {}
    s = pd.Series(values, dtype="float64")
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def pool_stats(
    pools: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for category, rows in pools.items():
        category_stats: Dict[str, Any] = {
            "input_tokens": distribution([
                float(row["input_tokens"]) for row in rows
            ])
        }

        response_values: List[float] = []
        for row in rows:
            metadata = row.get("selection_metadata")
            if not isinstance(metadata, dict):
                continue
            value = metadata.get("response_token_median")
            if value is not None:
                response_values.append(float(value))

        if response_values:
            category_stats["observed_response_token_median"] = distribution(
                response_values
            )

        result[category] = category_stats
    return result


def integrity_check(
    final_pools: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    counts = {
        category: len(rows)
        for category, rows in final_pools.items()
    }
    if counts != FINAL_TARGETS:
        raise RuntimeError(
            f"Final counts do not match target.\n"
            f"Expected: {FINAL_TARGETS}\nActual: {counts}"
        )

    flat = [
        row
        for category in FINAL_TARGETS
        for row in final_pools[category]
    ]
    if len(flat) != 1000:
        raise RuntimeError(f"Expected 1000 rows, found {len(flat)}")

    p_hashes = [row["prompt_hash"] for row in flat]
    if len(set(p_hashes)) != len(p_hashes):
        raise RuntimeError("Global duplicate prompt hash detected.")

    nq_hashes: List[str] = []
    for category in ["knowledge_qa", "document_qa", "long_context_qa"]:
        for row in final_pools[category]:
            q_hash = row.get("question_hash")
            if q_hash:
                nq_hashes.append(str(q_hash))

    if len(set(nq_hashes)) != len(nq_hashes):
        raise RuntimeError("Cross-category Natural Questions overlap detected.")

    for row in final_pools["document_qa"]:
        value = int(row["input_tokens"])
        if not (DOCUMENT_BAND[0] <= value <= DOCUMENT_BAND[1]):
            raise RuntimeError("Final document_qa token-band violation.")

    for row in final_pools["long_context_qa"]:
        value = int(row["input_tokens"])
        if not (LONG_CONTEXT_BAND[0] <= value <= LONG_CONTEXT_BAND[1]):
            raise RuntimeError("Final long_context_qa token-band violation.")

    return {
        "total_requests": len(flat),
        "category_counts": counts,
        "unique_prompt_hashes": len(set(p_hashes)),
        "unique_nq_question_hashes": len(set(nq_hashes)),
        "document_qa_band_valid": True,
        "long_context_qa_band_valid": True,
        "global_prompt_duplicates": 0,
        "cross_nq_question_duplicates": 0,
    }


def assign_final_ids(
    final_pools: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    final_rows: List[Dict[str, Any]] = []
    counter = 1
    for category in FINAL_TARGETS:
        for row in final_pools[category]:
            item = dict(row)
            item["request_id"] = f"req_{counter:04d}"
            item["workload_category"] = category
            final_rows.append(item)
            counter += 1
    return final_rows


def source_license(source: str) -> str:
    if source == "lmsys_chatbot_arena_conversations":
        return "CC-BY-4.0 (user prompts)"
    if source in {"google_nq_open", "google_natural_questions"}:
        return "CC-BY-SA-3.0"
    return "see source dataset"


def write_canonical_json(
    final_rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    seed: int,
) -> None:
    payload = {
        "metadata": {
            "name": "Production-like Real-World LLM Serving Workload",
            "version": "R0-E-v1",
            "seed": seed,
            "total_requests": 1000,
            "category_counts": FINAL_TARGETS,
            "document_qa_input_band": list(DOCUMENT_BAND),
            "long_context_qa_input_band": list(LONG_CONTEXT_BAND),
            "generation_policy": (
                "Generate until EOS or category-specific max_new_tokens cap. "
                "Record actual output tokens at benchmark runtime."
            ),
            "publishing_note": (
                "This canonical file contains full prompts. Keep local during "
                "benchmark development; use the metadata-only public manifest "
                "for GitHub."
            ),
        },
        "distributions": stats,
        "requests": final_rows,
    }
    with CANONICAL_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_public_manifest(final_rows: List[Dict[str, Any]]) -> None:
    rows = []
    for row in final_rows:
        rows.append({
            "request_id": row["request_id"],
            "workload_category": row["workload_category"],
            "source": row["source"],
            "source_id": row["source_id"],
            "source_license": source_license(row["source"]),
            "prompt_hash": row["prompt_hash"],
            "question_hash": row.get("question_hash", ""),
            "input_tokens": row["input_tokens"],
            "max_new_tokens": row["max_new_tokens"],
            "has_document_context": (
                row["workload_category"]
                in {"document_qa", "long_context_qa"}
            ),
        })
    pd.DataFrame(rows).to_csv(PUBLIC_MANIFEST, index=False)


def write_audit_sample(
    final_pools: Dict[str, List[Dict[str, Any]]],
    *,
    rng: random.Random,
    per_category: int = 15,
) -> None:
    rows: List[Dict[str, Any]] = []
    for category in FINAL_TARGETS:
        group = final_pools[category]
        n = min(per_category, len(group))
        for row in rng.sample(group, n):
            metadata = row.get("selection_metadata")
            response_median = None
            if isinstance(metadata, dict):
                response_median = metadata.get("response_token_median")

            rows.append({
                "workload_category": category,
                "source": row["source"],
                "source_id": row["source_id"],
                "input_tokens": row["input_tokens"],
                "max_new_tokens": row["max_new_tokens"],
                "observed_response_token_median": response_median,
                "prompt_preview": str(row["prompt"])[:800],
                "human_suitable": "",
                "human_notes": "",
            })
    pd.DataFrame(rows).to_csv(AUDIT_CSV, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Deterministic final sampling seed (default: 911)",
    )
    args = parser.parse_args()

    for category, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing R0-D shortlist for {category}: {path}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("R0-E — FINAL 1,000-REQUEST CORPUS & QUALITY CONTROL")
    print("=" * 84)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Seed:         {args.seed}")
    print()
    print(
        "R0-E validates, deduplicates, filters, and "
        "distribution-samples the R0-D 2x shortlist."
    )

    raw_pools = {
        category: read_jsonl(path)
        for category, path in INPUT_FILES.items()
    }

    print("\nRaw R0-D shortlist counts:")
    for category in FINAL_TARGETS:
        print(f"  {category:20s} {len(raw_pools[category])}")

    print(
        "\n[1/4] Structural validation, public-suitability filtering, "
        "and global dedup..."
    )
    clean_pools, flagged, cleaning_report = prepare_pools(raw_pools)

    print("\nClean eligible counts:")
    for category in FINAL_TARGETS:
        target = FINAL_TARGETS[category]
        available = len(clean_pools[category])
        status = "OK" if available >= target else "INSUFFICIENT"
        print(
            f"  {category:20s} available={available:4d} "
            f"target={target:3d} [{status}]"
        )
        if available < target:
            raise RuntimeError(
                f"{category}: insufficient rows after QC. "
                f"Need {target}, have {available}."
            )

    print("\n[2/4] Deterministic distribution-preserving sampling...")
    final_pools: Dict[str, List[Dict[str, Any]]] = {}

    for index, category in enumerate(FINAL_TARGETS):
        final_pools[category] = systematic_distribution_sample(
            clean_pools[category],
            category=category,
            target=FINAL_TARGETS[category],
            rng=random.Random(args.seed + 1009 * (index + 1)),
        )

    print("\n[3/4] Final integrity checks...")
    integrity = integrity_check(final_pools)
    final_stats = pool_stats(final_pools)
    clean_stats = pool_stats(clean_pools)
    final_rows = assign_final_ids(final_pools)

    print("\n[4/4] Writing canonical workload and manifests...")
    write_canonical_json(final_rows, final_stats, args.seed)
    write_public_manifest(final_rows)
    write_audit_sample(
        final_pools,
        rng=random.Random(args.seed + 7777),
        per_category=15,
    )

    pd.DataFrame(flagged).to_csv(FLAGGED_CSV, index=False)

    report = {
        "stage": "R0-E",
        "name": "Final 1,000-request corpus construction and QC",
        "seed": args.seed,
        "final_targets": FINAL_TARGETS,
        "cleaning": cleaning_report,
        "clean_pool_distributions": clean_stats,
        "final_distributions": final_stats,
        "integrity": integrity,
        "flagged_candidates": len(flagged),
        "output_files": {
            "canonical_local_workload": str(CANONICAL_JSON),
            "public_metadata_manifest": str(PUBLIC_MANIFEST),
            "final_report": str(REPORT_JSON),
            "human_audit_sample": str(AUDIT_CSV),
            "flagged_candidates": str(FLAGGED_CSV),
        },
        "publishing_policy": {
            "canonical_json": "keep local during benchmark development",
            "public_manifest": "metadata-only; no prompt or document text",
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print("R0-E COMPLETE")
    print("=" * 84)

    print("\nFinal counts:")
    for category in FINAL_TARGETS:
        print(f"  {category:20s} {len(final_pools[category])}")

    print(f"\nTotal requests: {integrity['total_requests']}")
    print(f"Unique prompt hashes: {integrity['unique_prompt_hashes']}")
    print(
        f"Cross-NQ duplicates: "
        f"{integrity['cross_nq_question_duplicates']}"
    )
    print(f"Suitability-flagged candidates: {len(flagged)}")

    print("\nFinal input-token distributions:")
    for category in FINAL_TARGETS:
        stats = final_stats[category]["input_tokens"]
        print(
            f"  {category:20s} "
            f"min={stats['min']:.0f} "
            f"p50={stats['p50']:.1f} "
            f"p95={stats['p95']:.1f} "
            f"max={stats['max']:.0f}"
        )

    print("\nSaved:")
    for path in (
        CANONICAL_JSON,
        PUBLIC_MANIFEST,
        REPORT_JSON,
        AUDIT_CSV,
        FLAGGED_CSV,
    ):
        print(f"  {path}")

    print()
    print(
        "Next: review r0e_final_audit.csv, then freeze the corpus "
        "before running the sequential baseline."
    )


if __name__ == "__main__":
    main()
