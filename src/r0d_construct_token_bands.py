#!/usr/bin/env python3
"""
R0-D — Construct and audit systems-oriented token bands.

Inputs
------
1) workloads/profiled_v4/lmsys_profiled_v4.jsonl
2) workloads/candidates/nq_open_candidates.jsonl
3) workloads/candidates/nq_document_candidates.jsonl

This stage creates an OVERSAMPLED shortlist for the final 1,000-request corpus.
It does not yet perform the final public-suitability audit or final sampling.

Target shortlist sizes (2x the eventual final workload)
--------------------------------------------------------
- short_interactive:     600   (final target 300)
- knowledge_qa:          400   (final target 200)
- coding_request:        300   (final target 150)
- document_qa:           300   (final target 150)
- long_context_qa:       200   (final target 100)
- long_output:           200   (final target 100)

Exact Qwen input bands
----------------------
- document_qa:      1024..2048 input tokens
- long_context_qa:  3072..4096 input tokens

The token count includes:
    chat template + instruction + document + question + generation prompt

Outputs
-------
workloads/token_bands/
├── short_interactive_shortlist.jsonl
├── knowledge_qa_shortlist.jsonl
├── coding_request_shortlist.jsonl
├── document_qa_shortlist.jsonl
├── long_context_qa_shortlist.jsonl
├── long_output_shortlist.jsonl
├── r0d_token_band_report.json
└── r0d_preview.csv

Run
---
python src/r0d_construct_token_bands.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 314

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LMSYS_PROFILED = (
    PROJECT_ROOT / "workloads" / "profiled_v4" / "lmsys_profiled_v4.jsonl"
)
NQ_OPEN = (
    PROJECT_ROOT / "workloads" / "candidates" / "nq_open_candidates.jsonl"
)
NQ_DOCUMENTS = (
    PROJECT_ROOT / "workloads" / "candidates" / "nq_document_candidates.jsonl"
)

OUT_DIR = PROJECT_ROOT / "workloads" / "token_bands"

SHORT_OUT = OUT_DIR / "short_interactive_shortlist.jsonl"
KNOWLEDGE_OUT = OUT_DIR / "knowledge_qa_shortlist.jsonl"
CODING_OUT = OUT_DIR / "coding_request_shortlist.jsonl"
DOCUMENT_OUT = OUT_DIR / "document_qa_shortlist.jsonl"
LONG_CONTEXT_OUT = OUT_DIR / "long_context_qa_shortlist.jsonl"
LONG_OUTPUT_OUT = OUT_DIR / "long_output_shortlist.jsonl"
REPORT_OUT = OUT_DIR / "r0d_token_band_report.json"
PREVIEW_OUT = OUT_DIR / "r0d_preview.csv"


SHORTLIST_TARGETS = {
    "short_interactive": 600,
    "knowledge_qa": 400,
    "coding_request": 300,
    "document_qa": 300,
    "long_context_qa": 200,
    "long_output": 200,
}

FINAL_TARGETS = {
    "short_interactive": 300,
    "knowledge_qa": 200,
    "coding_request": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "long_output": 100,
}

MAX_NEW_TOKENS = {
    "short_interactive": 256,
    "knowledge_qa": 128,
    "coding_request": 512,
    "document_qa": 256,
    "long_context_qa": 256,
    "long_output": 1024,
}

DOCUMENT_BAND = (1024, 2048)
LONG_CONTEXT_BAND = (3072, 4096)

QA_INSTRUCTION = (
    "Use the document below to answer the question. Base your answer on the "
    "document. If the answer cannot be determined from the document, say so."
)


# ---------------------------------------------------------------------------
# I/O helpers
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def chat_input_tokens(tokenizer, prompt: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    return len(encoded["input_ids"])


def build_document_prompt(question: str, document_text: str) -> str:
    return (
        f"{QA_INSTRUCTION}\n\n"
        f"Document:\n{document_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def construct_prompt_in_band(
    tokenizer,
    *,
    question: str,
    document_text: str,
    band: Tuple[int, int],
    target_tokens: int,
) -> Optional[Tuple[str, int, int]]:
    """
    Construct a document-QA prompt whose FINAL Qwen chat-template token count
    lies inside `band` and is as close as possible to target_tokens without
    exceeding it.

    Returns:
        (prompt, actual_input_tokens, document_qwen_tokens_used)
    """
    low_band, high_band = band
    if not (low_band <= target_tokens <= high_band):
        raise ValueError("target_tokens must lie inside band")

    doc_ids = tokenizer.encode(document_text, add_special_tokens=False)
    if not doc_ids:
        return None

    # Verify the available document can reach the lower edge of the band.
    full_prompt = build_document_prompt(
        question,
        tokenizer.decode(doc_ids, skip_special_tokens=True),
    )
    full_tokens = chat_input_tokens(tokenizer, full_prompt)
    if full_tokens < low_band:
        return None

    lo = 0
    hi = len(doc_ids)
    best_prompt: Optional[str] = None
    best_count = -1
    best_doc_tokens = 0

    # Largest document prefix whose complete prompt is <= target_tokens.
    while lo <= hi:
        mid = (lo + hi) // 2
        doc_text = tokenizer.decode(
            doc_ids[:mid],
            skip_special_tokens=True,
        )
        prompt = build_document_prompt(question, doc_text)
        count = chat_input_tokens(tokenizer, prompt)

        if count <= target_tokens:
            if count > best_count:
                best_prompt = prompt
                best_count = count
                best_doc_tokens = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_prompt is None:
        return None

    if best_count < low_band or best_count > high_band:
        return None

    return best_prompt, best_count, best_doc_tokens


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LMSYS shortlist construction
# ---------------------------------------------------------------------------


def make_lmsys_shortlist(
    rows: List[Dict[str, Any]],
    *,
    profile: str,
    target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    eligible = [r for r in rows if r.get("profile") == profile]
    if len(eligible) < target:
        raise RuntimeError(
            f"Only {len(eligible)} rows available for {profile}; need {target}."
        )

    chosen = rng.sample(eligible, target)
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(chosen, start=1):
        item = {
            "workload_category": profile,
            "source": "lmsys_chatbot_arena_conversations",
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "input_tokens": int(row["input_tokens_estimate"]),
            "max_new_tokens": MAX_NEW_TOKENS[profile],
            "selection_metadata": {
                "response_a_tokens": row.get("response_a_tokens"),
                "response_b_tokens": row.get("response_b_tokens"),
                "response_token_median": row.get("response_token_median"),
                "response_a_code_like": row.get("response_a_code_like"),
                "response_b_code_like": row.get("response_b_code_like"),
                "profile_reason": row.get("profile_reason"),
            },
        }
        out.append(item)

    return out


# ---------------------------------------------------------------------------
# NQ shortlist construction
# ---------------------------------------------------------------------------


def make_knowledge_shortlist(
    rows: List[Dict[str, Any]],
    *,
    target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(rows) < target:
        raise RuntimeError(
            f"Only {len(rows)} NQ Open rows available; need {target}."
        )

    chosen = rng.sample(rows, target)
    out: List[Dict[str, Any]] = []

    for row in chosen:
        out.append({
            "workload_category": "knowledge_qa",
            "source": "google_nq_open",
            "source_id": row["source_id"],
            "prompt": row["question"],
            "input_tokens": int(row["input_tokens_estimate"]),
            "max_new_tokens": MAX_NEW_TOKENS["knowledge_qa"],
            "selection_metadata": {
                "answer_count": row.get("answer_count"),
            },
        })

    return out


def make_document_shortlist(
    tokenizer,
    rows: List[Dict[str, Any]],
    *,
    category: str,
    band: Tuple[int, int],
    target: int,
    rng: random.Random,
    excluded_source_ids: Optional[Set[str]] = None,
    require_long_eligible: bool = False,
) -> Tuple[List[Dict[str, Any]], Set[str], Counter]:
    excluded_source_ids = set(excluded_source_ids or set())

    candidates = [
        r for r in rows
        if str(r.get("source_id")) not in excluded_source_ids
        and (
            not require_long_eligible
            or bool(r.get("long_context_eligible"))
        )
    ]
    rng.shuffle(candidates)

    out: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    counters = Counter()

    for row in candidates:
        if len(out) >= target:
            break

        counters["considered"] += 1

        source_id = str(row["source_id"])
        question = str(row["question"])
        document_text = str(row["document_text"])

        target_tokens = rng.randint(band[0], band[1])

        built = construct_prompt_in_band(
            tokenizer,
            question=question,
            document_text=document_text,
            band=band,
            target_tokens=target_tokens,
        )

        if built is None:
            counters["could_not_reach_band"] += 1
            continue

        prompt, actual_tokens, doc_qwen_tokens = built

        if not (band[0] <= actual_tokens <= band[1]):
            counters["band_validation_failed"] += 1
            continue

        item = {
            "workload_category": category,
            "source": "google_natural_questions",
            "source_id": source_id,
            "prompt": prompt,
            "question": question,
            "input_tokens": int(actual_tokens),
            "target_input_tokens": int(target_tokens),
            "document_qwen_tokens_used": int(doc_qwen_tokens),
            "max_new_tokens": MAX_NEW_TOKENS[category],
            "selection_metadata": {
                "source_document_tokens": row.get("source_document_tokens"),
                "stored_document_tokens": row.get("stored_document_tokens"),
                "long_context_eligible": row.get("long_context_eligible"),
            },
        }

        out.append(item)
        used_ids.add(source_id)
        counters["kept"] += 1

    if len(out) < target:
        raise RuntimeError(
            f"Could only build {len(out)}/{target} {category} prompts. "
            f"Counters: {dict(counters)}"
        )

    return out, used_ids, counters


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def build_preview(
    pools: Dict[str, List[Dict[str, Any]]],
    *,
    rng: random.Random,
    per_category: int = 10,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for category, pool in pools.items():
        sample_n = min(per_category, len(pool))
        for item in rng.sample(pool, sample_n):
            rows.append({
                "workload_category": category,
                "source": item["source"],
                "source_id": item["source_id"],
                "input_tokens": item["input_tokens"],
                "max_new_tokens": item["max_new_tokens"],
                "prompt_preview": item["prompt"][:800],
                "human_suitable": "",
                "human_notes": "",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Sampling/target-token seed (default: 314)",
    )
    args = parser.parse_args()

    for path in (LMSYS_PROFILED, NQ_OPEN, NQ_DOCUMENTS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("=" * 84)
    print("R0-D — TOKEN-BAND CONSTRUCTION & OVERSAMPLED SHORTLIST")
    print("=" * 84)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tokenizer:    {MODEL_NAME}")
    print(f"Seed:         {args.seed}")
    print()
    print("R0-D creates a 2x oversampled shortlist for R0-E final QA/sampling.")
    print("Document bands are validated on the complete Qwen chat input.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading R0-C v4 LMSYS profiles...")
    lmsys_rows = read_jsonl(LMSYS_PROFILED)

    print("Loading NQ Open candidates...")
    nq_open_rows = read_jsonl(NQ_OPEN)

    print("Loading full-NQ document candidates...")
    nq_doc_rows = read_jsonl(NQ_DOCUMENTS)

    print("\n[1/6] Short Interactive shortlist...")
    short_pool = make_lmsys_shortlist(
        lmsys_rows,
        profile="short_interactive",
        target=SHORTLIST_TARGETS["short_interactive"],
        rng=rng,
    )

    print("[2/6] Knowledge QA shortlist...")
    knowledge_pool = make_knowledge_shortlist(
        nq_open_rows,
        target=SHORTLIST_TARGETS["knowledge_qa"],
        rng=rng,
    )

    print("[3/6] Coding Request shortlist...")
    coding_pool = make_lmsys_shortlist(
        lmsys_rows,
        profile="coding_request",
        target=SHORTLIST_TARGETS["coding_request"],
        rng=rng,
    )

    print("[4/6] Long Output shortlist...")
    long_output_pool = make_lmsys_shortlist(
        lmsys_rows,
        profile="long_output",
        target=SHORTLIST_TARGETS["long_output"],
        rng=rng,
    )

    # Build the stricter long-context pool first so it gets first choice of the
    # long-eligible NQ documents. Document QA is then selected from disjoint IDs.
    print("[5/6] Long-context QA shortlist (3072..4096 exact Qwen input tokens)...")
    long_context_pool, long_ids, long_counters = make_document_shortlist(
        tokenizer,
        nq_doc_rows,
        category="long_context_qa",
        band=LONG_CONTEXT_BAND,
        target=SHORTLIST_TARGETS["long_context_qa"],
        rng=rng,
        require_long_eligible=True,
    )

    print("[6/6] Document QA shortlist (1024..2048 exact Qwen input tokens)...")
    document_pool, document_ids, document_counters = make_document_shortlist(
        tokenizer,
        nq_doc_rows,
        category="document_qa",
        band=DOCUMENT_BAND,
        target=SHORTLIST_TARGETS["document_qa"],
        rng=rng,
        excluded_source_ids=long_ids,
        require_long_eligible=False,
    )

    pools = {
        "short_interactive": short_pool,
        "knowledge_qa": knowledge_pool,
        "coding_request": coding_pool,
        "document_qa": document_pool,
        "long_context_qa": long_context_pool,
        "long_output": long_output_pool,
    }

    print("\nWriting shortlist files...")
    written = {
        "short_interactive": write_jsonl(SHORT_OUT, short_pool),
        "knowledge_qa": write_jsonl(KNOWLEDGE_OUT, knowledge_pool),
        "coding_request": write_jsonl(CODING_OUT, coding_pool),
        "document_qa": write_jsonl(DOCUMENT_OUT, document_pool),
        "long_context_qa": write_jsonl(LONG_CONTEXT_OUT, long_context_pool),
        "long_output": write_jsonl(LONG_OUTPUT_OUT, long_output_pool),
    }

    preview_df = build_preview(
        pools,
        rng=random.Random(args.seed + 1),
        per_category=10,
    )
    preview_df.to_csv(PREVIEW_OUT, index=False)

    input_distributions = {
        category: distribution([
            float(x["input_tokens"])
            for x in pool
        ])
        for category, pool in pools.items()
    }

    observed_response_distributions: Dict[str, Dict[str, Any]] = {}
    for category in ("short_interactive", "coding_request", "long_output"):
        vals = [
            x["selection_metadata"].get("response_token_median")
            for x in pools[category]
            if x["selection_metadata"].get("response_token_median") is not None
        ]
        observed_response_distributions[category] = distribution(
            [float(v) for v in vals]
        )

    report = {
        "stage": "R0-D",
        "name": "Token-band construction and oversampled shortlist",
        "tokenizer": MODEL_NAME,
        "seed": args.seed,
        "final_target_counts": FINAL_TARGETS,
        "shortlist_target_counts": SHORTLIST_TARGETS,
        "written_counts": written,
        "document_qa_band": list(DOCUMENT_BAND),
        "long_context_qa_band": list(LONG_CONTEXT_BAND),
        "input_token_distributions": input_distributions,
        "observed_response_token_distributions": observed_response_distributions,
        "document_build_counters": dict(document_counters),
        "long_context_build_counters": dict(long_counters),
        "nq_source_overlap_between_document_and_long_context": len(
            document_ids.intersection(long_ids)
        ),
        "notes": [
            "Document/long-context token counts include the Qwen chat template, instruction, document, question, and generation prompt.",
            "The R0-D shortlist is intentionally 2x larger than the final R0-E workload.",
            "R0-E will perform final public-suitability QA and deterministic sampling to exactly 1,000 unique requests.",
            "Arena response text is not stored; only derived response-token metadata is retained.",
        ],
    }

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print("R0-D COMPLETE")
    print("=" * 84)

    print("\nShortlist counts:")
    for category in SHORTLIST_TARGETS:
        print(
            f"  {category:20s} "
            f"{written[category]:4d} "
            f"(final target {FINAL_TARGETS[category]})"
        )

    print("\nInput-token distributions:")
    for category, stats in input_distributions.items():
        print(
            f"  {category:20s} "
            f"min={stats['min']:.0f} "
            f"p50={stats['p50']:.1f} "
            f"p95={stats['p95']:.1f} "
            f"max={stats['max']:.0f}"
        )

    print("\nDocument-band validation:")
    print(
        f"  document_qa:     expected={DOCUMENT_BAND[0]}..{DOCUMENT_BAND[1]}"
    )
    print(
        f"  long_context_qa: expected={LONG_CONTEXT_BAND[0]}..{LONG_CONTEXT_BAND[1]}"
    )
    print(
        f"  shared NQ source IDs across the two pools: "
        f"{len(document_ids.intersection(long_ids))}"
    )

    print("\nSaved:")
    for path in (
        SHORT_OUT,
        KNOWLEDGE_OUT,
        CODING_OUT,
        DOCUMENT_OUT,
        LONG_CONTEXT_OUT,
        LONG_OUTPUT_OUT,
        REPORT_OUT,
        PREVIEW_OUT,
    ):
        print(f"  {path}")

    print("\nNext: review r0d_preview.csv and the token-band report before R0-E.")


if __name__ == "__main__":
    main()
