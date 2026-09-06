#!/usr/bin/env python3
"""
R0-C v4 — Response-aware, systems-oriented workload profiling.

Motivation
----------
Three rounds of manual audit showed that semantic keyword classification is
not the best way to build a serving benchmark. For systems experiments, the
most important workload properties are request and response behavior.

This stage therefore uses the *observed first-turn responses already present
in the LMSYS Chatbot Arena dataset* to derive systems-level workload profiles.

Profiles
--------
- short_interactive
    Short real user request with short/medium observed responses.
- coding_request
    Explicit coding task plus code-like observed model response.
- long_output
    Observed responses are consistently long.
- unassigned
    Everything else.

The profiles are intentionally disjoint. Long-output takes priority over
coding because output length is the dominant serving characteristic.

Important
---------
The script DOES NOT save LMSYS model responses into the repository.
It stores only derived response-token counts and code-like flags.

Outputs
-------
workloads/profiled_v4/
├── lmsys_profiled_v4.jsonl
├── r0c_v4_profile_report.json
└── r0c_v4_blind_audit.csv

Run
---
python src/r0c_v4_response_aware_profiles.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 251

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "workloads"
    / "candidates"
    / "lmsys_candidates.jsonl"
)

OUT_DIR = PROJECT_ROOT / "workloads" / "profiled_v4"
PROFILED_OUT = OUT_DIR / "lmsys_profiled_v4.jsonl"
REPORT_OUT = OUT_DIR / "r0c_v4_profile_report.json"
AUDIT_OUT = OUT_DIR / "r0c_v4_blind_audit.csv"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Conversation extraction
# ---------------------------------------------------------------------------

def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def first_assistant_message(conversation: Any) -> Optional[str]:
    if not isinstance(conversation, list):
        return None

    for msg in conversation:
        if not isinstance(msg, dict):
            continue

        role = normalize_space(
            msg.get("role", msg.get("from", msg.get("speaker", "")))
        ).lower()

        if role in {"assistant", "gpt", "bot"}:
            content = msg.get(
                "content",
                msg.get("value", msg.get("text")),
            )
            text = normalize_space(content)
            if text:
                return text

    return None


def output_token_count(tokenizer, text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


# ---------------------------------------------------------------------------
# Coding-intent signals
# ---------------------------------------------------------------------------

CODE_TASK_PATTERNS = [
    r"\bwrite\s+(?:me\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,3}"
    r"(?:function|script|program|query|regex|class|method|code)\b",
    r"\bcreate\s+(?:me\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,3}"
    r"(?:function|script|program|query|regex|class|method)\b",
    r"\bimplement\b",
    r"\bdebug\b",
    r"\brefactor\b",
    r"\bfix\s+(?:this|my|the)\s+(?:code|script|function|program|bug)\b",
    r"\bhow\s+(?:do|can|would)\s+i\s+"
    r"(?:implement|code|program|write|debug|fix)\b",
    r"\bedit\s+this\s+(?:file|code|script)\b",
    r"\btranslate\s+.+\s+to\s+(?:python|java|c\+\+|c#|rust|fortran|javascript)\b",
]

CODE_RESPONSE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+\s*[:{]",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"#include\s*<",
    r"\bfunction\s+\w+\s*\(",
    r"\bconst\s+\w+\s*=",
    r"\blet\s+\w+\s*=",
    r"\bSELECT\b.+\bFROM\b",
    r"\bpublic\s+static\s+void\b",
]


def explicit_coding_task(prompt: str) -> bool:
    return any(
        re.search(
            pattern,
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in CODE_TASK_PATTERNS
    )


def code_like_response(text: Optional[str]) -> bool:
    if not text:
        return False

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in CODE_RESPONSE_PATTERNS
    )


# ---------------------------------------------------------------------------
# Profile logic
# ---------------------------------------------------------------------------

def response_stats(
    a_tokens: Optional[int],
    b_tokens: Optional[int],
) -> Dict[str, Optional[float]]:
    values = [
        int(x)
        for x in (a_tokens, b_tokens)
        if x is not None
    ]

    if not values:
        return {
            "response_token_min": None,
            "response_token_median": None,
            "response_token_max": None,
        }

    return {
        "response_token_min": min(values),
        "response_token_median": float(median(values)),
        "response_token_max": max(values),
    }


def assign_profile(
    *,
    prompt: str,
    input_tokens: int,
    response_a_tokens: Optional[int],
    response_b_tokens: Optional[int],
    response_a_code_like: bool,
    response_b_code_like: bool,
) -> Dict[str, Any]:
    stats = response_stats(
        response_a_tokens,
        response_b_tokens,
    )

    rmin = stats["response_token_min"]
    rmed = stats["response_token_median"]
    rmax = stats["response_token_max"]

    if rmed is None or rmin is None or rmax is None:
        return {
            "profile": "unassigned",
            "profile_reason": "missing_observed_response",
            **stats,
        }

    # Long-output is a systems property. Require both observed responses to
    # be substantial, or a very large median with a still substantial minimum.
    long_output = (
        rmin >= 384
        or (rmed >= 512 and rmin >= 256)
    )

    if long_output:
        return {
            "profile": "long_output",
            "profile_reason": "consistently_long_observed_responses",
            **stats,
        }

    coding = (
        explicit_coding_task(prompt)
        and (response_a_code_like or response_b_code_like)
    )

    if coding:
        return {
            "profile": "coding_request",
            "profile_reason": "explicit_code_task_plus_code_like_response",
            **stats,
        }

    # Short-interactive is intentionally a systems-level bucket, not a
    # semantic "casual chat" label.
    short_interactive = (
        input_tokens <= 128
        and rmax <= 256
        and rmin >= 4
    )

    if short_interactive:
        return {
            "profile": "short_interactive",
            "profile_reason": "short_input_and_short_to_medium_observed_response",
            **stats,
        }

    return {
        "profile": "unassigned",
        "profile_reason": "outside_high_precision_profile_rules",
        **stats,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def numeric_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}

    s = pd.Series(values, dtype="float64")
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def build_blind_audit(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    per_group: int,
) -> pd.DataFrame:
    groups = defaultdict(list)

    for row in rows:
        groups[row["profile"]].append(row)

    audit_rows: List[Dict[str, Any]] = []

    for profile in [
        "short_interactive",
        "coding_request",
        "long_output",
        "unassigned",
    ]:
        group = groups.get(profile, [])
        if not group:
            continue

        n = min(per_group, len(group))

        for row in rng.sample(group, n):
            audit_rows.append({
                "source_id": row["source_id"],
                "profile": row["profile"],
                "profile_reason": row["profile_reason"],
                "input_tokens_estimate": row["input_tokens_estimate"],
                "response_a_tokens": row["response_a_tokens"],
                "response_b_tokens": row["response_b_tokens"],
                "response_token_median": row["response_token_median"],
                "response_a_code_like": row["response_a_code_like"],
                "response_b_code_like": row["response_b_code_like"],
                "prompt": row["prompt"],
                "human_profile_sensible": "",
                "human_notes": "",
            })

    return pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-per-group",
        type=int,
        default=25,
        help="Blind-audit sample per profile (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Blind-audit sampling seed (default: 251)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing candidate pool: {INPUT_FILE}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("R0-C v4 — RESPONSE-AWARE SYSTEMS WORKLOAD PROFILING")
    print("=" * 82)
    print(f"Candidates:  {INPUT_FILE}")
    print(f"Tokenizer:   {MODEL_NAME}")
    print(f"Audit seed:  {args.seed}")
    print()
    print("This stage derives workload profiles from observed LMSYS response behavior.")
    print("Raw model responses are NOT written to the output files.")

    candidates = read_jsonl(INPUT_FILE)
    print(f"\nLoaded {len(candidates)} clean LMSYS candidates.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading LMSYS source dataset...")
    ds = load_dataset(
        "lmsys/chatbot_arena_conversations",
        split="train",
    )

    profiled: List[Dict[str, Any]] = []
    counts = Counter()
    missing_source_rows = 0

    for idx, candidate in enumerate(candidates, start=1):
        source_row = candidate.get("source_row")

        if not isinstance(source_row, int):
            missing_source_rows += 1
            continue

        rec = ds[source_row]

        response_a = first_assistant_message(
            rec.get("conversation_a")
        )
        response_b = first_assistant_message(
            rec.get("conversation_b")
        )

        a_tokens = output_token_count(
            tokenizer,
            response_a,
        )
        b_tokens = output_token_count(
            tokenizer,
            response_b,
        )

        a_code = code_like_response(response_a)
        b_code = code_like_response(response_b)

        decision = assign_profile(
            prompt=candidate["prompt"],
            input_tokens=int(
                candidate["input_tokens_estimate"]
            ),
            response_a_tokens=a_tokens,
            response_b_tokens=b_tokens,
            response_a_code_like=a_code,
            response_b_code_like=b_code,
        )

        out = dict(candidate)
        out.update({
            "response_a_tokens": a_tokens,
            "response_b_tokens": b_tokens,
            "response_a_code_like": a_code,
            "response_b_code_like": b_code,
        })
        out.update(decision)

        profiled.append(out)
        counts[out["profile"]] += 1

        if idx % 5000 == 0:
            print(
                f"  processed {idx}/{len(candidates)}"
            )

    write_jsonl(PROFILED_OUT, profiled)

    report: Dict[str, Any] = {
        "stage": "R0-C-v4",
        "name": "Response-aware systems workload profiling",
        "tokenizer": MODEL_NAME,
        "input_candidates": len(candidates),
        "profiled_candidates": len(profiled),
        "missing_source_rows": missing_source_rows,
        "profile_counts": dict(counts),
        "profile_definitions": {
            "short_interactive": (
                "input <=128 tokens and both observed responses <=256 tokens"
            ),
            "coding_request": (
                "explicit coding task plus code-like observed response"
            ),
            "long_output": (
                "both observed responses >=384 tokens, or median >=512 "
                "with minimum >=256"
            ),
            "unassigned": (
                "outside the high-precision systems-profile rules"
            ),
        },
        "response_token_distributions": {},
    }

    for profile in [
        "short_interactive",
        "coding_request",
        "long_output",
        "unassigned",
    ]:
        group = [
            row
            for row in profiled
            if row["profile"] == profile
        ]

        medians = [
            float(row["response_token_median"])
            for row in group
            if row["response_token_median"] is not None
        ]

        report["response_token_distributions"][profile] = (
            numeric_distribution(medians)
        )

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    audit_df = build_blind_audit(
        profiled,
        rng=random.Random(args.seed),
        per_group=args.audit_per_group,
    )
    audit_df.to_csv(AUDIT_OUT, index=False)

    print("\nProfile counts:")
    for profile in [
        "short_interactive",
        "coding_request",
        "long_output",
        "unassigned",
    ]:
        print(
            f"  {profile:20s} "
            f"{counts[profile]}"
        )

    print("\nEligibility check:")
    targets = {
        "short_interactive": 300,
        "coding_request": 150,
        "long_output": 100,
    }

    for profile, target in targets.items():
        available = counts[profile]
        status = (
            "OK"
            if available >= target
            else "INSUFFICIENT"
        )
        print(
            f"  {profile:20s} "
            f"available={available:5d} "
            f"target={target:3d} "
            f"[{status}]"
        )

    print("\n" + "=" * 82)
    print("R0-C v4 COMPLETE")
    print("=" * 82)
    print(f"Profiled pool: {PROFILED_OUT}")
    print(f"Report:        {REPORT_OUT}")
    print(f"Blind audit:   {AUDIT_OUT}")
    print()
    print("Review the v4 audit before building the final token-band workload.")


if __name__ == "__main__":
    main()
