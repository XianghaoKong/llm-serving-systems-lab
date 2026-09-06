#!/usr/bin/env python3
"""
R0-B — Cleaning, deduplication, and candidate-pool construction.

This stage consumes the same public sources audited in R0-A and creates clean,
reproducible candidate pools for later workload classification and sampling.

It DOES NOT:
- build the final 1,000-request workload,
- assign final request categories to LMSYS prompts,
- run model inference.

Outputs
-------
workloads/candidates/
├── lmsys_candidates.jsonl
├── nq_open_candidates.jsonl
├── nq_document_candidates.jsonl
├── r0b_cleaning_report.json
└── r0b_preview.csv

Run from repository root:
    python src/r0b_build_candidate_pools.py

Optional:
    python src/r0b_build_candidate_pools.py --nq-doc-candidates 1500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = PROJECT_ROOT / "workloads" / "candidates"

LMSYS_OUT = OUT_DIR / "lmsys_candidates.jsonl"
NQ_OPEN_OUT = OUT_DIR / "nq_open_candidates.jsonl"
NQ_DOC_OUT = OUT_DIR / "nq_document_candidates.jsonl"
REPORT_OUT = OUT_DIR / "r0b_cleaning_report.json"
PREVIEW_OUT = OUT_DIR / "r0b_preview.csv"

# R0-B filter bounds are intentionally broad. R0-C/R0-D will perform
# category-specific sampling and token-band validation.
LMSYS_MIN_TOKENS = 5
LMSYS_MAX_TOKENS = 2048

NQ_OPEN_MIN_TOKENS = 5
NQ_OPEN_MAX_TOKENS = 256

# Source-document token count (Wikipedia tokens, not Qwen tokens).
# 1,500 gives enough room for later 1K-2K document-QA prompts while still
# allowing a separate long-context eligibility flag for 4K-class prompts.
NQ_DOC_MIN_SOURCE_TOKENS = 1500
NQ_DOC_LONG_ELIGIBLE_SOURCE_TOKENS = 5000

# Store only the first N non-HTML source tokens to keep the candidate file
# manageable while retaining enough context for the later 3K-4K Qwen prompts.
NQ_DOC_MAX_STORED_SOURCE_TOKENS = 7000


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_dedup(text: str) -> str:
    text = normalize_space(text).lower()
    # Keep punctuation because it can matter in code/questions, but normalize
    # quotation variants and surrounding whitespace.
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    return text


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def qwen_chat_tokens(tokenizer, text: str) -> int:
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
    )
    return len(encoded["input_ids"])


def first_user_message(conversation: Any) -> Optional[str]:
    if not isinstance(conversation, list):
        return None

    for msg in conversation:
        if not isinstance(msg, dict):
            continue

        role = normalize_space(
            msg.get("role", msg.get("from", msg.get("speaker", "")))
        ).lower()

        if role in {"user", "human"}:
            content = msg.get(
                "content",
                msg.get("value", msg.get("text")),
            )
            text = normalize_space(content)
            if text:
                return text

    return None


def is_english(record: Dict[str, Any]) -> bool:
    language = normalize_space(record.get("language")).lower()
    return language == "en" or language.startswith("english")


def openai_moderation_flagged(record: Dict[str, Any]) -> bool:
    obj = record.get("openai_moderation")

    if isinstance(obj, dict):
        if bool(obj.get("flagged", False)):
            return True

        results = obj.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict) and bool(item.get("flagged", False)):
                    return True

    return False


def toxic_tag_flagged(record: Dict[str, Any]) -> bool:
    obj = record.get("toxic_chat_tag")

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, str):
        return obj.strip().lower() in {"true", "yes", "1", "toxic"}

    if isinstance(obj, dict):
        # LMSYS may expose one or more classifier-specific nested objects.
        for value in obj.values():
            if isinstance(value, dict) and bool(value.get("flagged", False)):
                return True
            if isinstance(value, bool) and value:
                return True

    return False


def safety_flagged(record: Dict[str, Any]) -> bool:
    return openai_moderation_flagged(record) or toxic_tag_flagged(record)


def extract_nq_open_question(record: Dict[str, Any]) -> str:
    q = record.get("question")

    if isinstance(q, str):
        return normalize_space(q)

    if isinstance(q, dict):
        return normalize_space(q.get("text", q.get("question", "")))

    for key in ("query", "prompt"):
        if isinstance(record.get(key), str):
            return normalize_space(record[key])

    return ""


def extract_full_nq_question(record: Dict[str, Any]) -> str:
    q = record.get("question")

    if isinstance(q, dict):
        return normalize_space(q.get("text", ""))

    if isinstance(q, str):
        return normalize_space(q)

    return ""


def extract_non_html_document_tokens(record: Dict[str, Any]) -> List[str]:
    document = record.get("document")
    if not isinstance(document, dict):
        return []

    tokens = document.get("tokens")

    # Hugging Face Natural Questions commonly stores dict-of-lists.
    if isinstance(tokens, dict):
        token_values = tokens.get("token")
        html_flags = tokens.get("is_html")

        if not isinstance(token_values, list):
            return []

        if isinstance(html_flags, list) and len(html_flags) == len(token_values):
            return [
                str(tok)
                for tok, is_html in zip(token_values, html_flags)
                if not bool(is_html) and normalize_space(tok)
            ]

        return [str(tok) for tok in token_values if normalize_space(tok)]

    # Alternate representation.
    if isinstance(tokens, list):
        clean: List[str] = []
        for item in tokens:
            if isinstance(item, dict):
                if not bool(item.get("is_html", False)):
                    tok = item.get("token")
                    if normalize_space(tok):
                        clean.append(str(tok))
            elif isinstance(item, str) and normalize_space(item):
                clean.append(item)
        return clean

    return []


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def distribution(values: List[int]) -> Dict[str, Any]:
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


def build_lmsys_candidates(
    tokenizer,
    preview_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[R0-B 1/3] Cleaning LMSYS candidate pool...")

    ds = load_dataset(
        "lmsys/chatbot_arena_conversations",
        split="train",
    )

    counters = Counter()
    seen_question_ids = set()
    seen_prompts = set()
    candidates: List[Dict[str, Any]] = []
    lengths: List[int] = []

    for row_index, rec in enumerate(ds):
        counters["raw_rows"] += 1

        if not is_english(rec):
            counters["removed_non_english"] += 1
            continue

        if safety_flagged(rec):
            counters["removed_safety_flagged"] += 1
            continue

        question_id = normalize_space(rec.get("question_id"))

        if question_id and question_id in seen_question_ids:
            counters["removed_duplicate_question_id"] += 1
            continue

        prompt = first_user_message(rec.get("conversation_a"))
        if not prompt:
            counters["removed_missing_prompt"] += 1
            continue

        normalized = normalize_for_dedup(prompt)
        prompt_hash = stable_hash(normalized)

        if prompt_hash in seen_prompts:
            counters["removed_duplicate_prompt"] += 1
            continue

        token_count = qwen_chat_tokens(tokenizer, prompt)

        if token_count < LMSYS_MIN_TOKENS:
            counters["removed_too_short"] += 1
            continue

        if token_count > LMSYS_MAX_TOKENS:
            counters["removed_too_long"] += 1
            continue

        candidate = {
            "source": "lmsys_chatbot_arena_conversations",
            "source_row": row_index,
            "source_id": question_id or f"row_{row_index}",
            "prompt_hash": prompt_hash,
            "language": normalize_space(rec.get("language")) or "unknown",
            "prompt": prompt,
            "input_tokens_estimate": token_count,
            "turn": rec.get("turn"),
        }

        candidates.append(candidate)
        lengths.append(token_count)

        if question_id:
            seen_question_ids.add(question_id)
        seen_prompts.add(prompt_hash)

        if len(preview_rows) < 8:
            preview_rows.append({
                "source": "lmsys",
                "source_id": candidate["source_id"],
                "tokens": token_count,
                "long_context_eligible": None,
                "preview": prompt[:500],
            })

    counters["kept_candidates"] = len(candidates)

    report = {
        "dataset": "lmsys/chatbot_arena_conversations",
        "filters": {
            "english_only": True,
            "remove_safety_flagged": True,
            "deduplicate_by_question_id": True,
            "deduplicate_by_normalized_prompt": True,
            "min_qwen_input_tokens": LMSYS_MIN_TOKENS,
            "max_qwen_input_tokens": LMSYS_MAX_TOKENS,
        },
        "counts": dict(counters),
        "qwen_input_token_distribution": distribution(lengths),
    }

    print(f"Raw rows:          {counters['raw_rows']}")
    print(f"Non-English:       {counters['removed_non_english']}")
    print(f"Safety flagged:    {counters['removed_safety_flagged']}")
    print(f"Duplicate qid:     {counters['removed_duplicate_question_id']}")
    print(f"Duplicate prompt:  {counters['removed_duplicate_prompt']}")
    print(f"Missing prompt:    {counters['removed_missing_prompt']}")
    print(f"Too short/long:    "
          f"{counters['removed_too_short']} / {counters['removed_too_long']}")
    print(f"Kept candidates:   {len(candidates)}")

    return candidates, report


def build_nq_open_candidates(
    tokenizer,
    preview_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[R0-B 2/3] Cleaning NQ Open candidate pool...")

    ds = load_dataset(
        "google-research-datasets/nq_open",
        split="train",
    )

    counters = Counter()
    seen_questions = set()
    candidates: List[Dict[str, Any]] = []
    lengths: List[int] = []

    base_preview_count = sum(1 for x in preview_rows if x["source"] == "nq_open")

    for row_index, rec in enumerate(ds):
        counters["raw_rows"] += 1

        question = extract_nq_open_question(rec)
        if not question:
            counters["removed_missing_question"] += 1
            continue

        norm = normalize_for_dedup(question)
        q_hash = stable_hash(norm)

        if q_hash in seen_questions:
            counters["removed_duplicate_question"] += 1
            continue

        token_count = qwen_chat_tokens(tokenizer, question)

        if token_count < NQ_OPEN_MIN_TOKENS:
            counters["removed_too_short"] += 1
            continue

        if token_count > NQ_OPEN_MAX_TOKENS:
            counters["removed_too_long"] += 1
            continue

        answers = rec.get("answer", rec.get("answers"))
        answer_count = len(answers) if isinstance(answers, list) else int(answers is not None)

        candidate = {
            "source": "google_nq_open",
            "source_row": row_index,
            "source_id": f"train_{row_index}",
            "question_hash": q_hash,
            "question": question,
            "input_tokens_estimate": token_count,
            "answer_count": answer_count,
        }

        candidates.append(candidate)
        lengths.append(token_count)
        seen_questions.add(q_hash)

        if sum(1 for x in preview_rows if x["source"] == "nq_open") < 8:
            preview_rows.append({
                "source": "nq_open",
                "source_id": candidate["source_id"],
                "tokens": token_count,
                "long_context_eligible": None,
                "preview": question[:500],
            })

    counters["kept_candidates"] = len(candidates)

    report = {
        "dataset": "google-research-datasets/nq_open",
        "filters": {
            "deduplicate_by_normalized_question": True,
            "min_qwen_input_tokens": NQ_OPEN_MIN_TOKENS,
            "max_qwen_input_tokens": NQ_OPEN_MAX_TOKENS,
        },
        "counts": dict(counters),
        "qwen_input_token_distribution": distribution(lengths),
    }

    print(f"Raw rows:          {counters['raw_rows']}")
    print(f"Duplicates:        {counters['removed_duplicate_question']}")
    print(f"Missing question:  {counters['removed_missing_question']}")
    print(f"Too short/long:    "
          f"{counters['removed_too_short']} / {counters['removed_too_long']}")
    print(f"Kept candidates:   {len(candidates)}")

    return candidates, report


def build_nq_document_candidates(
    preview_rows: List[Dict[str, Any]],
    target_candidates: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("\n[R0-B 3/3] Building full-NQ document candidate pool...")
    print(
        f"Streaming until {target_candidates} clean document candidates are kept..."
    )

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    counters = Counter()
    seen_questions = set()
    seen_source_ids = set()
    candidates: List[Dict[str, Any]] = []
    document_lengths: List[int] = []

    for rec in ds:
        if len(candidates) >= target_candidates:
            break

        counters["streamed_rows"] += 1

        question = extract_full_nq_question(rec)
        if not question:
            counters["removed_missing_question"] += 1
            continue

        question_hash = stable_hash(normalize_for_dedup(question))
        if question_hash in seen_questions:
            counters["removed_duplicate_question"] += 1
            continue

        source_id = normalize_space(
            rec.get("id", rec.get("example_id", ""))
        )
        if source_id and source_id in seen_source_ids:
            counters["removed_duplicate_source_id"] += 1
            continue

        doc_tokens = extract_non_html_document_tokens(rec)
        if not doc_tokens:
            counters["removed_missing_document"] += 1
            continue

        source_doc_len = len(doc_tokens)

        if source_doc_len < NQ_DOC_MIN_SOURCE_TOKENS:
            counters["removed_document_too_short"] += 1
            continue

        stored_tokens = doc_tokens[:NQ_DOC_MAX_STORED_SOURCE_TOKENS]
        document_text = normalize_space(" ".join(stored_tokens))

        if not document_text:
            counters["removed_empty_document_text"] += 1
            continue

        long_eligible = source_doc_len >= NQ_DOC_LONG_ELIGIBLE_SOURCE_TOKENS

        candidate = {
            "source": "google_natural_questions",
            "source_id": source_id or f"stream_{counters['streamed_rows']}",
            "question_hash": question_hash,
            "question": question,
            "source_document_tokens": source_doc_len,
            "stored_document_tokens": len(stored_tokens),
            "long_context_eligible": long_eligible,
            "document_text": document_text,
        }

        candidates.append(candidate)
        document_lengths.append(source_doc_len)
        seen_questions.add(question_hash)
        if source_id:
            seen_source_ids.add(source_id)

        if sum(1 for x in preview_rows if x["source"] == "nq_document") < 8:
            preview_rows.append({
                "source": "nq_document",
                "source_id": candidate["source_id"],
                "tokens": source_doc_len,
                "long_context_eligible": long_eligible,
                "preview": question[:250] + " | DOC: " + document_text[:250],
            })

        if len(candidates) % 250 == 0:
            print(
                f"  kept {len(candidates)}/{target_candidates} "
                f"after streaming {counters['streamed_rows']} rows"
            )

    counters["kept_candidates"] = len(candidates)
    counters["long_context_eligible_candidates"] = sum(
        1 for x in candidates if x["long_context_eligible"]
    )

    report = {
        "dataset": "google-research-datasets/natural_questions",
        "mode": "streaming candidate acquisition",
        "target_candidates": target_candidates,
        "filters": {
            "deduplicate_by_normalized_question": True,
            "deduplicate_by_source_id": True,
            "min_source_document_tokens": NQ_DOC_MIN_SOURCE_TOKENS,
            "long_context_eligible_source_tokens": NQ_DOC_LONG_ELIGIBLE_SOURCE_TOKENS,
            "max_stored_source_document_tokens": NQ_DOC_MAX_STORED_SOURCE_TOKENS,
        },
        "counts": dict(counters),
        "source_document_token_distribution": distribution(document_lengths),
        "note": (
            "source_document_tokens are Natural Questions source tokens, not "
            "final Qwen prompt tokens. Final document-QA prompts will be "
            "constructed and re-tokenized in R0-D."
        ),
    }

    print(f"Streamed rows:       {counters['streamed_rows']}")
    print(f"Missing question:    {counters['removed_missing_question']}")
    print(f"Duplicate question:  {counters['removed_duplicate_question']}")
    print(f"Missing document:    {counters['removed_missing_document']}")
    print(f"Document too short:  {counters['removed_document_too_short']}")
    print(f"Kept candidates:     {len(candidates)}")
    print(
        f"Long eligible:       "
        f"{counters['long_context_eligible_candidates']}"
    )

    if len(candidates) < target_candidates:
        raise RuntimeError(
            f"Streaming ended after only {len(candidates)} document candidates; "
            f"target was {target_candidates}."
        )

    return candidates, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nq-doc-candidates",
        type=int,
        default=1500,
        help="Number of clean full-NQ document candidates to retain (default: 1500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Streaming shuffle seed (default: 42)",
    )
    args = parser.parse_args()

    if args.nq_doc_candidates < 300:
        raise ValueError(
            "--nq-doc-candidates should be at least 300 so R0-C/D have "
            "enough document candidates."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("R0-B — CLEANING, DEDUPLICATION & CANDIDATE-POOL CONSTRUCTION")
    print("=" * 78)
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Tokenizer:          {MODEL_NAME}")
    print(f"NQ doc candidates:  {args.nq_doc_candidates}")
    print(f"Seed:               {args.seed}")
    print()
    print("R0-B creates clean candidate pools only.")
    print("It does NOT build the final 1,000-request workload.")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    preview_rows: List[Dict[str, Any]] = []

    lmsys_candidates, lmsys_report = build_lmsys_candidates(
        tokenizer,
        preview_rows,
    )

    nq_open_candidates, nq_open_report = build_nq_open_candidates(
        tokenizer,
        preview_rows,
    )

    nq_doc_candidates, nq_doc_report = build_nq_document_candidates(
        preview_rows,
        target_candidates=args.nq_doc_candidates,
        seed=args.seed,
    )

    lmsys_written = write_jsonl(LMSYS_OUT, lmsys_candidates)
    nq_open_written = write_jsonl(NQ_OPEN_OUT, nq_open_candidates)
    nq_doc_written = write_jsonl(NQ_DOC_OUT, nq_doc_candidates)

    report = {
        "stage": "R0-B",
        "name": "Cleaning, deduplication, and candidate-pool construction",
        "tokenizer": MODEL_NAME,
        "seed": args.seed,
        "outputs": {
            "lmsys_candidates": str(LMSYS_OUT),
            "nq_open_candidates": str(NQ_OPEN_OUT),
            "nq_document_candidates": str(NQ_DOC_OUT),
        },
        "sources": {
            "lmsys": lmsys_report,
            "nq_open": nq_open_report,
            "natural_questions_documents": nq_doc_report,
        },
        "written_counts": {
            "lmsys_candidates": lmsys_written,
            "nq_open_candidates": nq_open_written,
            "nq_document_candidates": nq_doc_written,
        },
    }

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pd.DataFrame(preview_rows).to_csv(PREVIEW_OUT, index=False)

    print("\n" + "=" * 78)
    print("R0-B COMPLETE")
    print("=" * 78)
    print(f"LMSYS candidates:   {lmsys_written}")
    print(f"NQ Open candidates: {nq_open_written}")
    print(f"NQ doc candidates:  {nq_doc_written}")
    print()
    print("Saved:")
    print(f"  {LMSYS_OUT}")
    print(f"  {NQ_OPEN_OUT}")
    print(f"  {NQ_DOC_OUT}")
    print(f"  {REPORT_OUT}")
    print(f"  {PREVIEW_OUT}")
    print()
    print("Next stage after audit: R0-C request classification.")


if __name__ == "__main__":
    main()
