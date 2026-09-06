#!/usr/bin/env python3
"""
R0-C v3 — Intent-first, high-precision LMSYS workload classification.

Why v3?
-------
Two manual audits showed that keyword-only labeling was still too permissive:
- factual / technical QA could leak into General Assistant,
- technical vocabulary could create false Coding labels,
- ordinary writing tasks could be mistaken for decode-heavy generation.

v3 therefore classifies by actual task intent rather than topic vocabulary.

Labels
------
- general_assistant
- coding_reasoning
- long_output_generation
- ambiguous

Design principle
----------------
Precision > recall.

The final workload needs only:
- 300 General Assistant
- 150 Coding / Reasoning
- 100 Long-output Generation

so v3 intentionally leaves uncertain prompts unassigned.

Outputs
-------
workloads/classified_v3/
├── lmsys_classified_v3.jsonl
├── r0c_v3_classification_report.json
└── r0c_v3_blind_audit.csv

Run
---
python src/r0c_v3_classify_requests.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


SEED = 173  # new blind-audit seed, different from v1/v2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "workloads" / "candidates" / "lmsys_candidates.jsonl"
OUT_DIR = PROJECT_ROOT / "workloads" / "classified_v3"

CLASSIFIED_OUT = OUT_DIR / "lmsys_classified_v3.jsonl"
REPORT_OUT = OUT_DIR / "r0c_v3_classification_report.json"
AUDIT_OUT = OUT_DIR / "r0c_v3_blind_audit.csv"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

GREETING_PREFIX = re.compile(
    r"""^\s*
    (?:
        hello|hi|hey|hiya|good\s+morning|good\s+afternoon|good\s+evening
    )
    (?:\s+there)?
    [!,.:\-\s]*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def strip_greeting_prefix(text: str) -> str:
    """
    Remove only a leading greeting, then classify the remaining task.

    Example:
        "Hi! Can you explain transformers?"
        -> "Can you explain transformers?"
    """
    stripped = GREETING_PREFIX.sub("", text, count=1).strip()
    return stripped if stripped else text.strip()


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


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
# Generic helpers
# ---------------------------------------------------------------------------

def rx(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))


def term_present(text: str, term: str) -> bool:
    # Word boundaries for ordinary alphanumeric terms reduce substring errors
    # such as "api" accidentally matching inside unrelated words.
    if re.fullmatch(r"[A-Za-z0-9_]+", term):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return term.lower() in text.lower()


def hit_terms(text: str, terms: List[str]) -> List[str]:
    return [t for t in terms if term_present(text, t)]


def explicit_word_count(text: str) -> Optional[int]:
    m = re.search(
        r"\b(\d{2,5})\s*[- ]?(?:word|words)\b",
        text,
        flags=re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Intent signals: Coding / Reasoning
# ---------------------------------------------------------------------------

PROGRAMMING_OBJECTS = [
    "python", "javascript", "typescript", "java", "c++", "c#", "rust",
    "golang", "go", "swift", "kotlin", "php", "ruby", "sql", "bash",
    "powershell", "shell", "html", "css", "react", "node.js", "unity",
    "hlsl", "pytorch", "tensorflow", "numpy", "pandas", "docker",
    "kubernetes", "regex", "git", "api", "endpoint", "function", "class",
    "method", "script", "program", "code", "query", "server", "client",
    "database",
]

CODE_TASK_PATTERNS = [
    r"\bwrite\s+(?:me\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,3}(?:function|script|program|query|regex|class|method|api|endpoint|server|client|code)\b",
    r"\bcreate\s+(?:me\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,3}(?:function|script|program|query|regex|class|method|api|endpoint|server|client)\b",
    r"\bimplement\b",
    r"\bdebug\b",
    r"\bfix\s+(?:this|my|the)\s+(?:code|script|function|program|query|bug|error)\b",
    r"\brefactor\b",
    r"\boptimi[sz]e\s+(?:this|my|the)\s+(?:code|function|program|query|implementation)\b",
    r"\bconvert\s+(?:this|the)\s+(?:code|script|function)\b",
    r"\bhow\s+(?:do|can|would)\s+i\s+(?:implement|code|program|write|debug|fix)\b",
]

CODE_STRUCTURE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+\s*[:{]",
    r"\bfrom\s+\w+\s+import\b",
    r"\bimport\s+\w+",
    r"#include\s*<",
    r"\bSELECT\b.+\bFROM\b",
    r"\bfunction\s+\w+\s*\(",
    r"\bconst\s+\w+\s*=",
    r"\blet\s+\w+\s*=",
    r"\bpublic\s+static\s+void\b",
    r"\bInvoke-[A-Za-z]+\b",
]

ERROR_PATTERNS = [
    r"\btraceback \(most recent call last\)",
    r"\bsyntaxerror\b",
    r"\btypeerror\b",
    r"\bvalueerror\b",
    r"\bnullpointerexception\b",
    r"\bsegmentation fault\b",
]

REASONING_TASK_PATTERNS = [
    r"\bsolve\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bderive\b",
    r"\bprove\b",
    r"\bfind\s+(?:the\s+)?(?:value|probability|percentage|ratio|distance|speed|rate|area|volume|x|y)\b",
    r"\bwork\s+out\b",
]

REASONING_DOMAIN_TERMS = [
    "equation", "probability", "percentage", "ratio", "algebra",
    "calculus", "integral", "derivative", "matrix", "geometry",
    "combinatorics", "expected value", "variance", "interest rate",
]

MATH_STRUCTURE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*%",
    r"\b\d+(?:\.\d+)?\s*[\+\-\*/=]\s*\d+(?:\.\d+)?",
    r"\b[a-zA-Z]\s*=\s*[-+]?\d",
    r"\bP\(.+\)",
    r"\bsqrt\(",
    r"\bintegral\b",
]


def code_intent(task_text: str) -> Dict[str, Any]:
    evidence: List[str] = []

    task_hits = [p for p in CODE_TASK_PATTERNS if rx(task_text, p)]
    structure_hits = [p for p in CODE_STRUCTURE_PATTERNS if rx(task_text, p)]
    error_hits = [p for p in ERROR_PATTERNS if rx(task_text, p)]
    programming_terms = hit_terms(task_text, PROGRAMMING_OBJECTS)

    score = 0

    if task_hits:
        score += 6
        evidence.append("explicit_code_task")

    if structure_hits:
        score += 6
        evidence.append("code_structure")

    if error_hits:
        # Error text alone is not enough; require an action around it unless
        # the prompt also contains code.
        if rx(task_text, r"\b(?:debug|fix|solve|help)\b") or structure_hits:
            score += 5
            evidence.append("actionable_error_debugging")

    # Explicit action + a programming object is strong, but a technical
    # explanation mentioning Python/API alone is not.
    if rx(
        task_text,
        r"\b(?:write|create|build|implement|debug|fix|refactor|modify|convert|generate)\b",
    ) and programming_terms:
        score += 4
        evidence.append(
            f"programming_action+object:{programming_terms[:3]}"
        )

    # Exclude explanatory / factual technical questions unless code-task
    # evidence is already strong.
    if rx(
        task_text,
        r"^\s*(?:what|who|when|where|why|which|explain|describe|compare)\b"
        r"|^\s*how\s+(?:does|do|is|are|was|were|can)\b",
    ) and not task_hits and not structure_hits:
        score = min(score, 1)
        evidence.append("technical_qa_exclusion")

    return {
        "score": score,
        "evidence": evidence,
        "programming_terms": programming_terms,
    }


def reasoning_intent(task_text: str) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    task_hits = [p for p in REASONING_TASK_PATTERNS if rx(task_text, p)]
    math_hits = [p for p in MATH_STRUCTURE_PATTERNS if rx(task_text, p)]
    domain_hits = hit_terms(task_text, REASONING_DOMAIN_TERMS)

    if task_hits and (math_hits or domain_hits):
        score += 6
        evidence.append("explicit_quantitative_reasoning_task")

    if len(math_hits) >= 2 and task_hits:
        score += 2
        evidence.append("multiple_math_structures")

    if domain_hits and task_hits:
        score += 2
        evidence.append(f"reasoning_domain:{domain_hits[:3]}")

    # "think step by step" is supporting evidence only.
    if "step by step" in task_text.lower() and score >= 6:
        score += 1
        evidence.append("step_by_step_support")

    # Summarization / extraction should never become reasoning.
    if rx(
        task_text,
        r"\b(?:summari[sz]e|extract|categorize|classify|list|rewrite|rephrase)\b",
    ):
        score = 0
        evidence.append("non_reasoning_task_exclusion")

    return {
        "score": score,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Intent signals: Long-output generation
# ---------------------------------------------------------------------------

LONG_FORM_OBJECTS = [
    "essay", "article", "report", "story", "chapter", "screenplay",
    "speech", "blog post", "newsletter", "proposal", "case study",
    "white paper", "guide", "tutorial", "review", "press release",
    "cover letter", "business plan", "sermon", "background story",
    "narrative",
]

LONG_QUALIFIERS = [
    "detailed", "comprehensive", "in-depth", "thorough", "extensive",
    "elaborate", "full-length", "long", "complete",
]

SHORT_OUTPUT_MARKERS = [
    "short", "brief", "concise", "one paragraph", "single paragraph",
    "10 sentences", "ten sentences", "5 sentences", "five sentences",
    "few sentences", "100 words", "150 words", "200 words", "250 words",
    "bullet points", "summary", "summarize", "summarise", "tl;dr",
]


def long_output_intent(task_text: str, input_tokens: int) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    lower = task_text.lower()
    wc = explicit_word_count(task_text)

    if any(marker in lower for marker in SHORT_OUTPUT_MARKERS):
        return {
            "score": 0,
            "evidence": ["explicit_short_output_exclusion"],
        }

    # Coding requests must not enter long-output writing.
    c = code_intent(task_text)
    if c["score"] >= 6:
        return {
            "score": 0,
            "evidence": ["coding_intent_exclusion"],
        }

    if rx(
        task_text,
        r"\b(?:summari[sz]e|rewrite|rephrase|extract|categorize|classify)\b",
    ):
        return {
            "score": 0,
            "evidence": ["transformation_task_exclusion"],
        }

    if wc is not None:
        if wc >= 1000:
            score += 10
            evidence.append(f"explicit_word_count_1000plus:{wc}")
        elif wc >= 800:
            score += 9
            evidence.append(f"explicit_word_count_800plus:{wc}")
        elif wc >= 500:
            score += 8
            evidence.append(f"explicit_word_count_500plus:{wc}")
        elif wc >= 400:
            score += 6
            evidence.append(f"explicit_word_count_400plus:{wc}")
        else:
            evidence.append(f"word_count_below_long_threshold:{wc}")

    object_hits = hit_terms(task_text, LONG_FORM_OBJECTS)
    qualifier_hits = hit_terms(task_text, LONG_QUALIFIERS)

    has_write_action = rx(
        task_text,
        r"\b(?:write|draft|compose|create|develop|generate)\b",
    )

    if has_write_action and object_hits and qualifier_hits:
        score += 7
        evidence.append(
            f"qualified_long_form:{object_hits[:2]}+{qualifier_hits[:2]}"
        )

    # Multi-section report / proposal / plan is a strong systems-level
    # long-output signal.
    section_markers = [
        "introduction", "executive summary", "background", "methodology",
        "analysis", "discussion", "recommendations", "conclusion",
    ]
    section_count = sum(term_present(task_text, s) for s in section_markers)

    if has_write_action and object_hits and section_count >= 3:
        score += 8
        evidence.append(f"multi_section_long_form:{section_count}")

    # Certain inherently long objects can qualify if explicitly asked for
    # full/complete output, but plain "write a story" remains ambiguous.
    if (
        has_write_action
        and object_hits
        and any(x in lower for x in ["full-length", "complete", "full chapter"])
    ):
        score += 7
        evidence.append("explicit_full_length_object")

    # Input length is never a deciding factor; only note it.
    if score >= 6 and input_tokens >= 160:
        evidence.append("long_input_support_only")

    return {
        "score": score,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Intent signals: Knowledge exclusion and General Assistant
# ---------------------------------------------------------------------------

KNOWLEDGE_START_PATTERNS = [
    r"^\s*who\b",
    r"^\s*what\b",
    r"^\s*when\b",
    r"^\s*where\b",
    r"^\s*which\b",
    r"^\s*why\b",
    r"^\s*how\s+(?:does|do|did|is|are|was|were|can|could|would)\b",
    r"^\s*explain\b",
    r"^\s*define\b",
    r"^\s*describe\b",
    r"^\s*compare\b",
    r"^\s*tell\s+me\s+about\b",
]

GENERAL_ASSISTANT_PATTERNS = [
    r"^\s*(?:hi|hello|hey)[!. ]*$",
    r"\bhow are you\b",
    r"\bwhat do you think\b",
    r"\bwhat would you do\b",
    r"\bwhat would you recommend\b",
    r"\bgive me advice\b",
    r"\bany advice\b",
    r"\bhelp me decide\b",
    r"\bhelp me choose\b",
    r"\bhelp me plan\b",
    r"\brecommend (?:me )?\b",
    r"\bbrainstorm\b",
    r"\bgive me ideas\b",
    r"\btranslate\b",
    r"\brewrite\b",
    r"\brephrase\b",
    r"\bmake this sound\b",
    r"\bwrite (?:me )?(?:a )?(?:short )?(?:email|message|text)\b",
]


def knowledge_like(task_text: str) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    if any(rx(task_text, p) for p in KNOWLEDGE_START_PATTERNS):
        score += 4
        evidence.append("knowledge_question_shape")

    if task_text.rstrip().endswith("?") and len(task_text.split()) >= 5:
        score += 1
        evidence.append("standalone_question")

    return {"score": score, "evidence": evidence}


def general_assistant_intent(
    original_text: str,
    task_text: str,
    input_tokens: int,
) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    hits = [p for p in GENERAL_ASSISTANT_PATTERNS if rx(task_text, p)]

    if hits:
        score += 6
        evidence.append("explicit_general_assistant_task")

    # A pure greeting can still use the original text after greeting stripping.
    if rx(
        original_text,
        r"^\s*(?:hi|hello|hey|hiya|good morning|good afternoon|good evening)[!. ]*$",
    ):
        score += 7
        evidence.append("pure_greeting")

    if input_tokens <= 128 and score >= 6:
        score += 1
        evidence.append("short_request_support")

    return {"score": score, "evidence": evidence}


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    original = str(row["prompt"])
    task_text = strip_greeting_prefix(original)
    input_tokens = int(row["input_tokens_estimate"])

    c = code_intent(task_text)
    r = reasoning_intent(task_text)
    lo = long_output_intent(task_text, input_tokens)
    k = knowledge_like(task_text)
    ga = general_assistant_intent(original, task_text, input_tokens)

    # 1. Explicit coding task
    if c["score"] >= 6 and c["score"] >= lo["score"] + 2:
        return {
            "r0c_v3_label": "coding_reasoning",
            "r0c_v3_subtype": "coding",
            "r0c_v3_confidence": "high",
            "r0c_v3_score": c["score"],
            "r0c_v3_evidence": c["evidence"],
            "task_text_after_greeting_strip": task_text,
        }

    # 2. Explicit quantitative/math reasoning
    if r["score"] >= 6:
        return {
            "r0c_v3_label": "coding_reasoning",
            "r0c_v3_subtype": "reasoning",
            "r0c_v3_confidence": "high",
            "r0c_v3_score": r["score"],
            "r0c_v3_evidence": r["evidence"],
            "task_text_after_greeting_strip": task_text,
        }

    # 3. Strong long-output generation intent
    if lo["score"] >= 6 and lo["score"] >= c["score"] + 2:
        return {
            "r0c_v3_label": "long_output_generation",
            "r0c_v3_subtype": "long_output_generation",
            "r0c_v3_confidence": "high",
            "r0c_v3_score": lo["score"],
            "r0c_v3_evidence": lo["evidence"],
            "task_text_after_greeting_strip": task_text,
        }

    # 4. Factual / explanatory QA is deliberately excluded from General
    # Assistant because Knowledge QA has its own Natural Questions source.
    if k["score"] >= 4:
        return {
            "r0c_v3_label": "ambiguous",
            "r0c_v3_subtype": "knowledge_like_excluded",
            "r0c_v3_confidence": "unassigned",
            "r0c_v3_score": k["score"],
            "r0c_v3_evidence": ["knowledge_like_exclusion"] + k["evidence"],
            "task_text_after_greeting_strip": task_text,
        }

    # 5. Positive General Assistant intent only.
    if (
        input_tokens <= 128
        and ga["score"] >= 6
        and c["score"] < 6
        and r["score"] < 6
        and lo["score"] < 6
    ):
        return {
            "r0c_v3_label": "general_assistant",
            "r0c_v3_subtype": "general_assistant",
            "r0c_v3_confidence": "high",
            "r0c_v3_score": ga["score"],
            "r0c_v3_evidence": ga["evidence"],
            "task_text_after_greeting_strip": task_text,
        }

    # 6. Preserve ambiguity.
    evidence = ["unassigned_for_precision"]

    if c["score"] > 0:
        evidence.extend(c["evidence"][:3])
    if r["score"] > 0:
        evidence.extend(r["evidence"][:3])
    if lo["score"] > 0:
        evidence.extend(lo["evidence"][:3])
    if ga["score"] > 0:
        evidence.extend(ga["evidence"][:2])

    return {
        "r0c_v3_label": "ambiguous",
        "r0c_v3_subtype": None,
        "r0c_v3_confidence": "unassigned",
        "r0c_v3_score": max(
            c["score"], r["score"], lo["score"], ga["score"], k["score"]
        ),
        "r0c_v3_evidence": evidence,
        "task_text_after_greeting_strip": task_text,
    }


# ---------------------------------------------------------------------------
# Reporting and blind audit
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


def build_blind_audit(
    rows: List[Dict[str, Any]],
    rng: random.Random,
    per_group: int,
) -> pd.DataFrame:
    groups = defaultdict(list)
    for row in rows:
        groups[row["r0c_v3_label"]].append(row)

    audit_rows: List[Dict[str, Any]] = []

    for label in [
        "general_assistant",
        "coding_reasoning",
        "long_output_generation",
        "ambiguous",
    ]:
        group = groups.get(label, [])
        if not group:
            continue

        n = min(per_group, len(group))
        chosen = rng.sample(group, n)

        for row in chosen:
            audit_rows.append({
                "source_id": row["source_id"],
                "r0c_v3_label": row["r0c_v3_label"],
                "r0c_v3_subtype": row["r0c_v3_subtype"],
                "r0c_v3_confidence": row["r0c_v3_confidence"],
                "r0c_v3_score": row["r0c_v3_score"],
                "input_tokens_estimate": row["input_tokens_estimate"],
                "evidence": " | ".join(row["r0c_v3_evidence"]),
                "prompt": row["prompt"],
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
        help="New blind-audit sample per label (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Blind-audit seed (default: 173)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}\n"
            "Run R0-B first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("R0-C v3 — INTENT-FIRST HIGH-PRECISION LMSYS CLASSIFICATION")
    print("=" * 82)
    print(f"Input:       {INPUT_FILE}")
    print(f"Audit seed:  {args.seed}")
    print(f"Audit/group: {args.audit_per_group}")
    print()
    print("v3 uses task intent rather than topic vocabulary.")
    print("Uncertain prompts are intentionally left ambiguous.")

    rows = read_jsonl(INPUT_FILE)
    print(f"\nLoaded {len(rows)} clean LMSYS candidates.")

    classified: List[Dict[str, Any]] = []
    counts = Counter()
    subtypes = Counter()
    token_by_label = defaultdict(list)

    for row in rows:
        decision = classify(row)
        out = dict(row)
        out.update(decision)
        classified.append(out)

        label = out["r0c_v3_label"]
        counts[label] += 1
        token_by_label[label].append(int(out["input_tokens_estimate"]))

        if out["r0c_v3_subtype"]:
            subtypes[out["r0c_v3_subtype"]] += 1

    write_jsonl(CLASSIFIED_OUT, classified)

    report = {
        "stage": "R0-C-v3",
        "name": "Intent-first high-precision LMSYS workload classification",
        "input_candidates": len(rows),
        "audit_seed": args.seed,
        "policy": {
            "precision_over_recall": True,
            "greeting_prefix_stripped_before_intent_detection": True,
            "technical_vocabulary_alone_does_not_imply_coding": True,
            "long_input_alone_does_not_imply_long_output": True,
            "long_output_requires_explicit_output_length_or_long_form_structure": True,
            "knowledge_like_prompts_excluded_from_general_assistant": True,
            "labels": [
                "general_assistant",
                "coding_reasoning",
                "long_output_generation",
                "ambiguous",
            ],
        },
        "label_counts": dict(counts),
        "subtype_counts": dict(subtypes),
        "token_distributions": {
            label: token_distribution(values)
            for label, values in token_by_label.items()
        },
    }

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    audit_df = build_blind_audit(
        classified,
        rng=random.Random(args.seed),
        per_group=args.audit_per_group,
    )
    audit_df.to_csv(AUDIT_OUT, index=False)

    print("\nClassification counts:")
    for label in [
        "general_assistant",
        "coding_reasoning",
        "long_output_generation",
        "ambiguous",
    ]:
        print(f"  {label:24s} {counts[label]}")

    print("\nSubtype counts:")
    for subtype, n in subtypes.most_common():
        print(f"  {subtype:24s} {n}")

    print("\nEligibility check for planned final workload:")
    targets = {
        "general_assistant": 300,
        "coding_reasoning": 150,
        "long_output_generation": 100,
    }

    for label, target in targets.items():
        available = counts[label]
        status = "OK" if available >= target else "INSUFFICIENT"
        print(
            f"  {label:24s} available={available:5d} "
            f"target={target:3d} [{status}]"
        )

    print("\n" + "=" * 82)
    print("R0-C v3 COMPLETE")
    print("=" * 82)
    print(f"Classified pool: {CLASSIFIED_OUT}")
    print(f"Report:          {REPORT_OUT}")
    print(f"Blind audit:     {AUDIT_OUT}")
    print()
    print("Do not proceed to R0-D until this new blind audit is reviewed.")


if __name__ == "__main__":
    main()
