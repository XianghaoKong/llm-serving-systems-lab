#!/usr/bin/env python3
"""
R0-C v2 — High-precision LMSYS workload classification.

This revision addresses issues found in the first 100-sample manual audit:
1) factual/technical QA leaking into short/general chat,
2) programming "script" requests leaking into generation-heavy,
3) some coding requests being left ambiguous,
4) generation-heavy labels being assigned to long-input / short-output prompts.

Principle
---------
Prefer precision over recall. Specialized labels are assigned only when
high-confidence intent is present. Knowledge-like prompts are excluded from
General Assistant and remain ambiguous/unused because Knowledge QA will come
from Google Natural Questions.

Labels
------
- general_assistant
- coding_reasoning
- generation_heavy
- ambiguous

Outputs
-------
workloads/classified_v2/
├── lmsys_classified_v2.jsonl
├── r0c_v2_classification_report.json
└── r0c_v2_blind_audit.csv

Run
---
python src/r0c_v2_classify_requests.py
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


SEED = 97  # intentionally different from v1 audit seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "workloads" / "candidates" / "lmsys_candidates.jsonl"
OUT_DIR = PROJECT_ROOT / "workloads" / "classified_v2"

CLASSIFIED_OUT = OUT_DIR / "lmsys_classified_v2.jsonl"
REPORT_OUT = OUT_DIR / "r0c_v2_classification_report.json"
AUDIT_OUT = OUT_DIR / "r0c_v2_blind_audit.csv"


# ---------------------------------------------------------------------------
# Term sets
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c++", "c#", "golang",
    "rust", "swift", "kotlin", "php", "ruby", "matlab", "sql", "bash",
    "powershell", "html", "css", "react", "node.js", "unity", "hlsl",
    "pytorch", "tensorflow", "numpy", "pandas", "docker", "kubernetes",
]

PROGRAMMING_ACTIONS = [
    "debug", "fix", "implement", "refactor", "optimize", "compile",
    "write code", "write a function", "write a script", "create a script",
    "make a script", "script that", "code for", "program", "function",
    "class", "method", "api", "endpoint", "stack trace", "exception",
    "syntax error", "unit test", "regex", "query", "server", "client",
    "package", "library", "framework", "repository", "git", "terminal",
]

REASONING_DOMAINS = [
    "equation", "probability", "matrix", "integral", "derivative",
    "theorem", "proof", "geometry", "algebra", "calculus", "statistics",
    "expected value", "variance", "combinatorics", "logic puzzle",
    "optimization problem", "linear programming", "percentage",
    "ratio", "distance", "speed", "rate", "interest rate",
]

REASONING_ACTIONS = [
    "solve", "prove", "derive", "calculate", "compute", "show that",
    "find the value", "find x", "evaluate", "work out",
]

LONG_FORM_TYPES = [
    "essay", "article", "report", "story", "chapter", "screenplay",
    "speech", "blog post", "newsletter", "proposal", "case study",
    "white paper", "guide", "tutorial", "review", "press release",
    "cover letter", "business plan", "sermon", "background story",
]

WRITING_ACTIONS = [
    "write", "draft", "compose", "create", "generate", "develop",
]

LONG_FORM_QUALIFIERS = [
    "detailed", "comprehensive", "in-depth", "long-form", "thorough",
    "extensive", "elaborate", "full-length",
]

# Explicitly conversational / assistant-like intents.
GENERAL_ASSISTANT_PATTERNS = [
    r"^(hi|hello|hey|good morning|good evening)\b",
    r"\bhow are you\b",
    r"\bcan we talk\b",
    r"\blet'?s talk\b",
    r"\bwhat do you think\b",
    r"\bwhat would you do\b",
    r"\bwhat would you recommend\b",
    r"\bhelp me decide\b",
    r"\bgive me advice\b",
    r"\bany advice\b",
    r"\bhelp me plan\b",
    r"\bhelp me choose\b",
    r"\brewrite\b",
    r"\brephrase\b",
    r"\bmake this sound\b",
    r"\btranslate\b",
    r"\bsummarize this message\b",
    r"\bwrite a short message\b",
    r"\bwrite a text\b",
    r"\bwrite an email\b",
    r"\bbrainstorm\b",
    r"\bgive me ideas\b",
]

# Strong signals that a prompt is primarily factual / explanatory QA.
KNOWLEDGE_QA_PATTERNS = [
    r"^\s*who\b",
    r"^\s*what\b",
    r"^\s*when\b",
    r"^\s*where\b",
    r"^\s*which\b",
    r"^\s*why\b",
    r"^\s*how (does|do|did|is|are|was|were|can|could|would)\b",
    r"^\s*explain\b",
    r"^\s*define\b",
    r"^\s*describe\b",
    r"^\s*compare\b",
]

TECHNICAL_KNOWLEDGE_TERMS = [
    "cpu", "gpu", "x86", "reinforcement learning", "neural network",
    "transformer", "database", "operating system", "compiler", "algorithm",
    "physics", "chemistry", "biology", "economics", "inflation",
    "history", "politics", "philosophy", "knn", "vector-jacobian",
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

def term_hits(text: str, terms: List[str]) -> List[str]:
    lower = text.lower()
    return [term for term in terms if term in lower]


def regex_hits(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            hits.append(pattern)
    return hits


def has_code_fence(text: str) -> bool:
    return "```" in text


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
        r"\bInvoke-[A-Za-z]+\b",
        r"\bGet-[A-Za-z]+\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in patterns)


def has_error_shape(text: str) -> bool:
    patterns = [
        r"\btraceback \(most recent call last\)",
        r"\bsyntaxerror\b",
        r"\btypeerror\b",
        r"\bvalueerror\b",
        r"\bnullpointerexception\b",
        r"\bsegmentation fault\b",
        r"\bexception\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def has_math_shape(text: str) -> bool:
    patterns = [
        r"\b\d+(?:\.\d+)?\s*[\+\-\*/=]\s*\d+(?:\.\d+)?",
        r"\b[a-zA-Z]\s*=\s*[-+]?\d",
        r"\b\d+(?:\.\d+)?\s*%",
        r"\bP\(.+\)",
        r"\bE\(.+\)",
        r"\bsin\(",
        r"\bcos\(",
        r"\blog\(",
        r"\bsqrt\(",
    ]
    return any(re.search(p, text) for p in patterns)


def explicit_word_count(text: str) -> int | None:
    match = re.search(
        r"\b(\d{2,5})\s*[- ]?(?:word|words)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1))


def explicit_section_count(text: str) -> int:
    lower = text.lower()
    markers = [
        "introduction", "conclusion", "executive summary",
        "background", "methodology", "recommendations",
        "discussion", "analysis", "literature review",
    ]
    return sum(marker in lower for marker in markers)


def looks_like_programming_script(text: str) -> bool:
    lower = text.lower()
    if "script" not in lower:
        return False

    programming_context = [
        "python", "bash", "powershell", "node.js", "javascript", "unity",
        "linux", "windows", "shell", "terminal", "api", "file", "folder",
        "server", "command", "automate", "automation", "code",
    ]
    return any(term in lower for term in programming_context)


def looks_like_media_script(text: str) -> bool:
    lower = text.lower()
    media_context = [
        "screenplay", "movie script", "film script", "video script",
        "podcast script", "youtube script", "dialogue script",
        "commercial script", "advertisement script", "radio script",
    ]
    return any(term in lower for term in media_context)


def is_summarization_prompt(text: str) -> bool:
    lower = text.lower()
    return (
        "summarize" in lower
        or "summarise" in lower
        or "tl;dr" in lower
        or "bullet points" in lower
    )


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def coding_reasoning_score(prompt: str) -> Dict[str, Any]:
    evidence: List[str] = []
    coding_score = 0
    reasoning_score = 0

    lang_hits = term_hits(prompt, PROGRAMMING_LANGUAGES)
    action_hits = term_hits(prompt, PROGRAMMING_ACTIONS)
    domain_hits = term_hits(prompt, REASONING_DOMAINS)
    reasoning_hits = term_hits(prompt, REASONING_ACTIONS)

    if has_code_fence(prompt):
        coding_score += 5
        evidence.append("code_fence")

    if has_code_shape(prompt):
        coding_score += 5
        evidence.append("code_structure")

    if has_error_shape(prompt):
        coding_score += 4
        evidence.append("error_or_exception_structure")

    if looks_like_programming_script(prompt):
        coding_score += 5
        evidence.append("programming_script_context")

    if lang_hits and action_hits:
        coding_score += 4
        evidence.append(
            f"programming_language+action:{lang_hits[:2]}+{action_hits[:2]}"
        )
    elif len(lang_hits) >= 2:
        coding_score += 3
        evidence.append(f"multiple_programming_terms:{lang_hits[:3]}")
    elif len(action_hits) >= 2:
        coding_score += 2
        evidence.append(f"multiple_programming_actions:{action_hits[:3]}")

    # Technical explanation alone is not coding.
    explanation_only = bool(
        regex_hits(prompt, [r"^\s*explain\b", r"^\s*what is\b", r"^\s*how does\b"])
    )
    if explanation_only and not (
        has_code_shape(prompt)
        or has_code_fence(prompt)
        or looks_like_programming_script(prompt)
        or (lang_hits and action_hits)
    ):
        coding_score = max(0, coding_score - 3)
        evidence.append("technical_explanation_penalty")

    if domain_hits and reasoning_hits:
        reasoning_score += 4
        evidence.append(
            f"reasoning_domain+action:{domain_hits[:2]}+{reasoning_hits[:2]}"
        )

    if has_math_shape(prompt) and reasoning_hits:
        reasoning_score += 4
        evidence.append("math_structure+reasoning_action")

    if len(domain_hits) >= 2:
        reasoning_score += 2
        evidence.append(f"multiple_reasoning_terms:{domain_hits[:3]}")

    # "Think step by step" alone is not enough.
    if "step by step" in prompt.lower() and (
        domain_hits or has_math_shape(prompt)
    ):
        reasoning_score += 1
        evidence.append("step_by_step_with_reasoning_signal")

    if is_summarization_prompt(prompt):
        reasoning_score = max(0, reasoning_score - 3)
        coding_score = max(0, coding_score - 2)
        evidence.append("summarization_penalty")

    if coding_score >= reasoning_score and coding_score > 0:
        subtype = "coding"
        score = coding_score
    elif reasoning_score > 0:
        subtype = "reasoning"
        score = reasoning_score
    else:
        subtype = None
        score = 0

    return {
        "score": score,
        "subtype": subtype,
        "coding_score": coding_score,
        "reasoning_score": reasoning_score,
        "evidence": evidence,
    }


def generation_score(prompt: str, input_tokens: int) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    action_hits = term_hits(prompt, WRITING_ACTIONS)
    type_hits = term_hits(prompt, LONG_FORM_TYPES)
    qualifier_hits = term_hits(prompt, LONG_FORM_QUALIFIERS)

    if looks_like_programming_script(prompt):
        return {
            "score": 0,
            "evidence": ["programming_script_exclusion"],
        }

    if is_summarization_prompt(prompt):
        return {
            "score": 0,
            "evidence": ["summarization_exclusion"],
        }

    wc = explicit_word_count(prompt)
    if wc is not None:
        if wc >= 800:
            score += 7
            evidence.append(f"explicit_word_count_very_long:{wc}")
        elif wc >= 500:
            score += 5
            evidence.append(f"explicit_word_count_long:{wc}")
        elif wc >= 300:
            score += 3
            evidence.append(f"explicit_word_count_medium:{wc}")
        else:
            evidence.append(f"explicit_word_count_short:{wc}")

    section_count = explicit_section_count(prompt)
    if section_count >= 3:
        score += 4
        evidence.append(f"multi_section_request:{section_count}")
    elif section_count == 2:
        score += 2
        evidence.append("two_section_request")

    if action_hits and type_hits:
        score += 4
        evidence.append(
            f"writing_action+long_form_type:{action_hits[:2]}+{type_hits[:2]}"
        )

    if qualifier_hits and action_hits and type_hits:
        score += 3
        evidence.append(f"detailed_long_form:{qualifier_hits[:2]}")

    if looks_like_media_script(prompt):
        score += 4
        evidence.append("media_script_generation")

    # Input length is only weak supporting evidence. Never sufficient by itself.
    if input_tokens >= 160 and score >= 4:
        score += 1
        evidence.append("long_input_supporting_signal")

    return {
        "score": score,
        "evidence": evidence,
    }


def knowledge_like_score(prompt: str) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    qa_hits = regex_hits(prompt, KNOWLEDGE_QA_PATTERNS)
    technical_hits = term_hits(prompt, TECHNICAL_KNOWLEDGE_TERMS)

    if qa_hits:
        score += 2
        evidence.append("factual_or_explanatory_question_shape")

    if technical_hits:
        score += 2
        evidence.append(f"technical_knowledge_terms:{technical_hits[:3]}")

    if (
        prompt.strip().endswith("?")
        and len(prompt.split()) >= 4
        and not regex_hits(prompt, GENERAL_ASSISTANT_PATTERNS)
    ):
        score += 1
        evidence.append("standalone_question")

    return {
        "score": score,
        "evidence": evidence,
    }


def general_assistant_score(prompt: str, input_tokens: int) -> Dict[str, Any]:
    evidence: List[str] = []
    score = 0

    conversational_hits = regex_hits(prompt, GENERAL_ASSISTANT_PATTERNS)

    if conversational_hits:
        score += 4
        evidence.append("explicit_general_assistant_intent")

    if input_tokens <= 64 and conversational_hits:
        score += 1
        evidence.append("short_conversational_request")

    # Very short greetings are allowed even without another pattern.
    if re.search(
        r"^\s*(hi|hello|hey|good morning|good evening)[!. ]*$",
        prompt,
        flags=re.IGNORECASE,
    ):
        score += 5
        evidence.append("simple_greeting")

    return {
        "score": score,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(row["prompt"])
    input_tokens = int(row["input_tokens_estimate"])

    cr = coding_reasoning_score(prompt)
    gh = generation_score(prompt, input_tokens)
    kq = knowledge_like_score(prompt)
    ga = general_assistant_score(prompt, input_tokens)

    # Priority 1: explicit coding/reasoning.
    if cr["score"] >= 4 and cr["score"] >= gh["score"] + 2:
        return {
            "r0c_v2_label": "coding_reasoning",
            "r0c_v2_subtype": cr["subtype"],
            "r0c_v2_confidence": "high",
            "r0c_v2_score": cr["score"],
            "r0c_v2_evidence": cr["evidence"],
            "knowledge_like_score": kq["score"],
        }

    # Priority 2: genuine long-output generation intent.
    if gh["score"] >= 4 and gh["score"] >= cr["score"] + 2:
        return {
            "r0c_v2_label": "generation_heavy",
            "r0c_v2_subtype": "long_form_generation",
            "r0c_v2_confidence": "high",
            "r0c_v2_score": gh["score"],
            "r0c_v2_evidence": gh["evidence"],
            "knowledge_like_score": kq["score"],
        }

    # Priority 3: General Assistant only when there is positive conversational /
    # assistant-task evidence AND no strong factual/technical QA signal.
    if (
        input_tokens <= 128
        and ga["score"] >= 4
        and kq["score"] < 3
        and cr["score"] < 4
        and gh["score"] < 4
    ):
        return {
            "r0c_v2_label": "general_assistant",
            "r0c_v2_subtype": "general_assistant",
            "r0c_v2_confidence": "high",
            "r0c_v2_score": ga["score"],
            "r0c_v2_evidence": ga["evidence"],
            "knowledge_like_score": kq["score"],
        }

    # Explicit factual / explanatory QA is deliberately not mapped into General
    # Assistant, because Knowledge QA comes from NQ.
    evidence = ["unassigned_for_precision"]

    if kq["score"] >= 3:
        evidence.extend(["knowledge_like_exclusion"] + kq["evidence"])

    if cr["score"] > 0:
        evidence.extend(cr["evidence"][:3])

    if gh["score"] > 0:
        evidence.extend(gh["evidence"][:3])

    if ga["score"] > 0:
        evidence.extend(ga["evidence"][:2])

    return {
        "r0c_v2_label": "ambiguous",
        "r0c_v2_subtype": None,
        "r0c_v2_confidence": "unassigned",
        "r0c_v2_score": max(cr["score"], gh["score"], ga["score"], kq["score"]),
        "r0c_v2_evidence": evidence,
        "knowledge_like_score": kq["score"],
    }


# ---------------------------------------------------------------------------
# Reporting / blind audit
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
        groups[row["r0c_v2_label"]].append(row)

    audit_rows = []

    for label in [
        "general_assistant",
        "coding_reasoning",
        "generation_heavy",
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
                "r0c_v2_label": row["r0c_v2_label"],
                "r0c_v2_subtype": row["r0c_v2_subtype"],
                "r0c_v2_confidence": row["r0c_v2_confidence"],
                "r0c_v2_score": row["r0c_v2_score"],
                "knowledge_like_score": row["knowledge_like_score"],
                "input_tokens_estimate": row["input_tokens_estimate"],
                "evidence": " | ".join(row["r0c_v2_evidence"]),
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
        help="Blind manual-audit sample per label (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Blind-audit sampling seed (default: 97)",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing R0-B candidate file: {INPUT_FILE}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("R0-C v2 — HIGH-PRECISION LMSYS WORKLOAD CLASSIFICATION")
    print("=" * 80)
    print(f"Input:       {INPUT_FILE}")
    print(f"Audit seed:  {args.seed}")
    print(f"Audit/group: {args.audit_per_group}")
    print()
    print("This revision prioritizes precision and prevents factual QA leakage into")
    print("General Assistant. The blind audit uses a different seed from v1.")

    rows = read_jsonl(INPUT_FILE)
    print(f"\nLoaded {len(rows)} clean LMSYS candidates.")

    classified: List[Dict[str, Any]] = []
    counts = Counter()
    subtype_counts = Counter()
    token_by_label = defaultdict(list)
    knowledge_excluded = 0

    for row in rows:
        result = classify(row)
        out = dict(row)
        out.update(result)
        classified.append(out)

        label = out["r0c_v2_label"]
        counts[label] += 1
        token_by_label[label].append(int(out["input_tokens_estimate"]))

        if out["r0c_v2_subtype"]:
            subtype_counts[out["r0c_v2_subtype"]] += 1

        if (
            label == "ambiguous"
            and "knowledge_like_exclusion" in out["r0c_v2_evidence"]
        ):
            knowledge_excluded += 1

    write_jsonl(CLASSIFIED_OUT, classified)

    report = {
        "stage": "R0-C-v2",
        "name": "High-precision LMSYS workload classification",
        "input_candidates": len(rows),
        "audit_seed": args.seed,
        "policy": {
            "precision_over_recall": True,
            "labels": [
                "general_assistant",
                "coding_reasoning",
                "generation_heavy",
                "ambiguous",
            ],
            "general_assistant_max_input_tokens": 128,
            "knowledge_like_prompts_excluded_from_general_assistant": True,
            "programming_script_priority_over_generation": True,
            "generation_heavy_requires_output_intent": True,
            "note": (
                "Labels are auditable workload labels for systems experiments, "
                "not semantic ground truth."
            ),
        },
        "label_counts": dict(counts),
        "subtype_counts": dict(subtype_counts),
        "knowledge_like_excluded_to_ambiguous": knowledge_excluded,
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
        "generation_heavy",
        "ambiguous",
    ]:
        print(f"  {label:22s} {counts[label]}")

    print(f"\nKnowledge-like prompts excluded to ambiguous: {knowledge_excluded}")

    print("\nEligibility check for planned final workload:")
    targets = {
        "general_assistant": 300,
        "coding_reasoning": 150,
        "generation_heavy": 100,
    }

    for label, target in targets.items():
        available = counts[label]
        status = "OK" if available >= target else "INSUFFICIENT"
        print(
            f"  {label:22s} available={available:5d} "
            f"target={target:3d} [{status}]"
        )

    print("\n" + "=" * 80)
    print("R0-C v2 COMPLETE")
    print("=" * 80)
    print(f"Classified pool: {CLASSIFIED_OUT}")
    print(f"Report:          {REPORT_OUT}")
    print(f"Blind audit:     {AUDIT_OUT}")
    print()
    print("Do not proceed to R0-D until the new blind audit is reviewed.")


if __name__ == "__main__":
    main()
