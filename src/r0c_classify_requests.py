#!/usr/bin/env python3
"""
R0-C — High-precision request classification for LMSYS candidates.

Purpose
-------
Assign auditable, high-confidence workload labels to cleaned LMSYS prompts
without forcing every prompt into a category.

Categories produced
-------------------
- short_chat
- coding_reasoning
- generation_heavy
- ambiguous

Important
---------
This is NOT semantic ground truth. It is an auditable workload-labeling layer
for systems benchmarking. Only high-confidence examples will be eligible for
the final R0-E workload.

The classifier uses multiple independent signals:
- input length
- code / programming structure
- mathematical / reasoning structure
- explicit long-form generation intent
- exclusion/conflict checks

Outputs
-------
workloads/classified/
├── lmsys_classified.jsonl
├── r0c_classification_report.json
└── r0c_manual_audit.csv

Run
---
python src/r0c_classify_requests.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "workloads" / "candidates" / "lmsys_candidates.jsonl"
OUT_DIR = PROJECT_ROOT / "workloads" / "classified"

CLASSIFIED_OUT = OUT_DIR / "lmsys_classified.jsonl"
REPORT_OUT = OUT_DIR / "r0c_classification_report.json"
AUDIT_OUT = OUT_DIR / "r0c_manual_audit.csv"


# ---------------------------------------------------------------------------
# Signal dictionaries
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c++", "c#", "golang",
    "rust", "swift", "kotlin", "php", "ruby", "matlab", "r language",
    "sql", "bash", "powershell", "html", "css", "react", "node.js",
    "pytorch", "tensorflow", "numpy", "pandas",
]

PROGRAMMING_ACTIONS = [
    "debug", "fix", "implement", "refactor", "optimize", "compile",
    "write code", "write a function", "write a script", "code for",
    "program", "function", "class", "method", "api", "endpoint",
    "stack trace", "exception", "error message", "syntax error",
    "unit test", "algorithm", "data structure", "regex", "query",
]

REASONING_DOMAIN_TERMS = [
    "equation", "probability", "matrix", "integral", "derivative",
    "theorem", "proof", "geometry", "algebra", "calculus", "statistics",
    "expected value", "variance", "combinatorics", "logic puzzle",
    "optimization problem", "linear programming",
]

REASONING_ACTIONS = [
    "solve", "prove", "derive", "calculate", "compute", "show that",
    "find the value", "find x", "evaluate", "step by step",
]

LONG_FORM_NOUNS = [
    "essay", "article", "report", "story", "chapter", "screenplay",
    "script", "speech", "blog post", "newsletter", "proposal",
    "case study", "white paper", "guide", "tutorial", "review",
    "press release", "cover letter", "business plan",
]

GENERATION_ACTIONS = [
    "write", "draft", "compose", "create", "generate", "rewrite",
    "expand", "develop",
]

LONG_FORM_QUALIFIERS = [
    "detailed", "comprehensive", "in-depth", "long-form", "thorough",
    "extensive", "elaborate", "full-length",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def count_terms(text: str, terms: List[str]) -> Tuple[int, List[str]]:
    hits = []
    lower = text.lower()
    for term in terms:
        if term in lower:
            hits.append(term)
    return len(hits), hits


def has_code_fence(text: str) -> bool:
    return "```" in text


def has_stacktrace_shape(text: str) -> bool:
    patterns = [
        r"\btraceback \(most recent call last\)",
        r"\bexception\b.*\bat\b",
        r"\bsyntaxerror\b",
        r"\btypeerror\b",
        r"\bvalueerror\b",
        r"\bnullpointerexception\b",
        r"\bsegmentation fault\b",
    ]
    lower = text.lower()
    return any(re.search(p, lower, flags=re.DOTALL) for p in patterns)


def has_code_shape(text: str) -> bool:
    patterns = [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[:{]",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bSELECT\b.+\bFROM\b",
        r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bpublic\s+static\s+void\b",
        r"#include\s*<",
        r"\bfor\s*\(.+;.+;.+\)",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in patterns)


def has_math_shape(text: str) -> bool:
    patterns = [
        r"\b\d+\s*[\+\-\*/=]\s*\d+",
        r"\b[a-zA-Z]\s*=\s*[-+]?\d",
        r"\b\d+(\.\d+)?\s*%",
        r"\bP\(.+\)",
        r"\bE\(.+\)",
        r"\bsin\(",
        r"\bcos\(",
        r"\blog\(",
        r"\bsqrt\(",
    ]
    return any(re.search(p, text) for p in patterns)


def explicit_word_count_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{3,5}\s*[- ]?(?:word|words)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def explicit_multi_section_request(text: str) -> bool:
    lower = text.lower()
    section_markers = [
        "introduction", "conclusion", "executive summary",
        "background section", "methodology", "recommendations",
    ]
    hit_count = sum(marker in lower for marker in section_markers)
    return hit_count >= 2


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_coding_reasoning(prompt: str) -> Dict[str, Any]:
    lower = prompt.lower()
    score = 0
    evidence: List[str] = []
    subtype = None

    lang_n, lang_hits = count_terms(prompt, PROGRAMMING_LANGUAGES)
    action_n, action_hits = count_terms(prompt, PROGRAMMING_ACTIONS)
    reasoning_domain_n, reasoning_domain_hits = count_terms(
        prompt, REASONING_DOMAIN_TERMS
    )
    reasoning_action_n, reasoning_action_hits = count_terms(
        prompt, REASONING_ACTIONS
    )

    programming_score = 0

    if has_code_fence(prompt):
        programming_score += 4
        evidence.append("code_fence")

    if has_code_shape(prompt):
        programming_score += 4
        evidence.append("code_structure")

    if has_stacktrace_shape(prompt):
        programming_score += 4
        evidence.append("stacktrace_or_error_structure")

    if lang_n >= 1 and action_n >= 1:
        programming_score += 3
        evidence.append(
            f"programming_language+action:{lang_hits[:2]}+{action_hits[:2]}"
        )
    elif lang_n >= 2:
        programming_score += 2
        evidence.append(f"multiple_programming_terms:{lang_hits[:3]}")
    elif action_n >= 2:
        programming_score += 2
        evidence.append(f"multiple_programming_actions:{action_hits[:3]}")

    reasoning_score = 0

    if reasoning_domain_n >= 1 and reasoning_action_n >= 1:
        reasoning_score += 3
        evidence.append(
            f"reasoning_domain+action:"
            f"{reasoning_domain_hits[:2]}+{reasoning_action_hits[:2]}"
        )

    if has_math_shape(prompt) and reasoning_action_n >= 1:
        reasoning_score += 3
        evidence.append("math_structure+reasoning_action")

    if reasoning_domain_n >= 2:
        reasoning_score += 2
        evidence.append(f"multiple_reasoning_terms:{reasoning_domain_hits[:3]}")

    if "step by step" in lower and (
        reasoning_domain_n >= 1 or has_math_shape(prompt)
    ):
        reasoning_score += 1
        evidence.append("explicit_step_by_step_reasoning")

    if programming_score >= reasoning_score and programming_score > 0:
        subtype = "coding"
        score = programming_score
    elif reasoning_score > 0:
        subtype = "reasoning"
        score = reasoning_score

    return {
        "score": score,
        "subtype": subtype,
        "evidence": evidence,
        "programming_score": programming_score,
        "reasoning_score": reasoning_score,
    }


def score_generation_heavy(prompt: str, token_count: int) -> Dict[str, Any]:
    score = 0
    evidence: List[str] = []

    action_n, action_hits = count_terms(prompt, GENERATION_ACTIONS)
    noun_n, noun_hits = count_terms(prompt, LONG_FORM_NOUNS)
    qual_n, qual_hits = count_terms(prompt, LONG_FORM_QUALIFIERS)

    if action_n >= 1 and noun_n >= 1:
        score += 3
        evidence.append(f"generation_action+long_form:{action_hits[:2]}+{noun_hits[:2]}")

    if explicit_word_count_request(prompt):
        score += 4
        evidence.append("explicit_3plus_digit_word_count")

    if qual_n >= 1 and action_n >= 1:
        score += 2
        evidence.append(f"long_form_qualifier:{qual_hits[:2]}")

    if explicit_multi_section_request(prompt):
        score += 3
        evidence.append("explicit_multi_section_structure")

    # Long prompt plus explicit generation intent is additional evidence,
    # but length alone can never make something generation-heavy.
    if token_count >= 160 and action_n >= 1:
        score += 1
        evidence.append("long_input_with_generation_intent")

    return {
        "score": score,
        "evidence": evidence,
    }


def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(row["prompt"])
    token_count = int(row["input_tokens_estimate"])

    cr = score_coding_reasoning(prompt)
    gh = score_generation_heavy(prompt, token_count)

    cr_score = cr["score"]
    gh_score = gh["score"]

    # High-confidence coding/reasoning requires >=3 and a clear margin over
    # competing generation-heavy intent.
    if cr_score >= 3 and cr_score >= gh_score + 2:
        return {
            "r0c_label": "coding_reasoning",
            "r0c_subtype": cr["subtype"],
            "r0c_confidence": "high",
            "r0c_score": cr_score,
            "r0c_evidence": cr["evidence"],
            "competing_score": gh_score,
        }

    # High-confidence generation-heavy uses the same margin rule.
    if gh_score >= 3 and gh_score >= cr_score + 2:
        return {
            "r0c_label": "generation_heavy",
            "r0c_subtype": "long_form_generation",
            "r0c_confidence": "high",
            "r0c_score": gh_score,
            "r0c_evidence": gh["evidence"],
            "competing_score": cr_score,
        }

    # Short-chat is intentionally conservative:
    # - short input,
    # - no high-confidence coding/reasoning,
    # - no explicit long-form generation intent,
    # - no code structure.
    if (
        token_count <= 128
        and cr_score < 3
        and gh_score < 3
        and not has_code_fence(prompt)
        and not has_code_shape(prompt)
        and not has_stacktrace_shape(prompt)
    ):
        return {
            "r0c_label": "short_chat",
            "r0c_subtype": "short_general_request",
            "r0c_confidence": "high",
            "r0c_score": 1,
            "r0c_evidence": ["input_tokens<=128_and_no_specialized_high_confidence_signal"],
            "competing_score": max(cr_score, gh_score),
        }

    return {
        "r0c_label": "ambiguous",
        "r0c_subtype": None,
        "r0c_confidence": "unassigned",
        "r0c_score": max(cr_score, gh_score),
        "r0c_evidence": (
            ["conflicting_or_insufficient_signals"]
            + cr["evidence"][:3]
            + gh["evidence"][:3]
        ),
        "competing_score": min(cr_score, gh_score),
    }


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def token_distribution(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}

    s = pd.Series(values, dtype="float64")
    return {
        "count": int(len(s)),
        "min": int(s.min()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
        "mean": float(s.mean()),
    }


def build_manual_audit(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    per_group: int,
) -> pd.DataFrame:
    groups = defaultdict(list)

    for row in rows:
        groups[row["r0c_label"]].append(row)

    audit_rows = []

    group_order = [
        "short_chat",
        "coding_reasoning",
        "generation_heavy",
        "ambiguous",
    ]

    for label in group_order:
        group = groups.get(label, [])
        if not group:
            continue

        sample_n = min(per_group, len(group))
        chosen = rng.sample(group, sample_n)

        for row in chosen:
            audit_rows.append({
                "source_id": row["source_id"],
                "r0c_label": row["r0c_label"],
                "r0c_subtype": row["r0c_subtype"],
                "r0c_confidence": row["r0c_confidence"],
                "r0c_score": row["r0c_score"],
                "input_tokens_estimate": row["input_tokens_estimate"],
                "evidence": " | ".join(row["r0c_evidence"]),
                "prompt": row["prompt"],
                # Human-audit columns intentionally left blank.
                "human_correct": "",
                "human_label_if_wrong": "",
                "human_notes": "",
            })

    return pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-per-group",
        type=int,
        default=25,
        help="Manual-audit sample per label (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Sampling seed (default: 42)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing R0-B input: {INPUT_FILE}\n"
            "Run src/r0b_build_candidate_pools.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("R0-C — HIGH-PRECISION LMSYS REQUEST CLASSIFICATION")
    print("=" * 78)
    print(f"Input:       {INPUT_FILE}")
    print(f"Seed:        {args.seed}")
    print(f"Audit/group: {args.audit_per_group}")
    print()
    print("Important: ambiguous prompts are intentionally left unassigned.")

    rows = read_jsonl(INPUT_FILE)
    print(f"\nLoaded {len(rows)} clean LMSYS candidates.")

    classified: List[Dict[str, Any]] = []
    counts = Counter()
    subtype_counts = Counter()
    token_by_label = defaultdict(list)
    evidence_counts = Counter()

    for row in rows:
        decision = classify(row)
        out = dict(row)
        out.update(decision)
        classified.append(out)

        label = out["r0c_label"]
        counts[label] += 1
        if out["r0c_subtype"]:
            subtype_counts[out["r0c_subtype"]] += 1
        token_by_label[label].append(int(out["input_tokens_estimate"]))

        for ev in out["r0c_evidence"]:
            evidence_counts[ev] += 1

    write_jsonl(CLASSIFIED_OUT, classified)

    report = {
        "stage": "R0-C",
        "name": "High-precision LMSYS request classification",
        "input_candidates": len(rows),
        "policy": {
            "principle": (
                "Only assign specialized categories when multiple/structural "
                "signals provide high confidence; otherwise preserve ambiguity."
            ),
            "labels": [
                "short_chat",
                "coding_reasoning",
                "generation_heavy",
                "ambiguous",
            ],
            "short_chat_max_input_tokens": 128,
            "specialized_min_score": 3,
            "specialized_required_margin": 2,
            "note": (
                "These are workload labels for performance experiments, not "
                "semantic ground-truth annotations."
            ),
        },
        "label_counts": dict(counts),
        "subtype_counts": dict(subtype_counts),
        "token_distributions": {
            label: token_distribution(values)
            for label, values in token_by_label.items()
        },
        "top_evidence": evidence_counts.most_common(30),
    }

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    rng = random.Random(args.seed)
    audit_df = build_manual_audit(
        classified,
        rng=rng,
        per_group=args.audit_per_group,
    )
    audit_df.to_csv(AUDIT_OUT, index=False)

    print("\nClassification counts:")
    for label in [
        "short_chat",
        "coding_reasoning",
        "generation_heavy",
        "ambiguous",
    ]:
        print(f"  {label:20s} {counts[label]}")

    print("\nSubtype counts:")
    for subtype, n in subtype_counts.most_common():
        print(f"  {subtype:24s} {n}")

    print("\nEligibility check for planned final workload:")
    targets = {
        "short_chat": 300,
        "coding_reasoning": 150,
        "generation_heavy": 100,
    }
    for label, target in targets.items():
        available = counts[label]
        status = "OK" if available >= target else "INSUFFICIENT"
        print(
            f"  {label:20s} available={available:5d} "
            f"target={target:3d}  [{status}]"
        )

    print("\n" + "=" * 78)
    print("R0-C COMPLETE")
    print("=" * 78)
    print(f"Classified pool: {CLASSIFIED_OUT}")
    print(f"Report:          {REPORT_OUT}")
    print(f"Manual audit:    {AUDIT_OUT}")
    print()
    print("Next:")
    print("1. Inspect r0c_manual_audit.csv.")
    print("2. Review classification precision before R0-D token-band construction.")


if __name__ == "__main__":
    main()
