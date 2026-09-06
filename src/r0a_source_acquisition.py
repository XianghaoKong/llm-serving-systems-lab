#!/usr/bin/env python3
"""
R0-A — Real-world source acquisition and audit.

Purpose
-------
Validate access, schema, size, safety metadata, language distribution, basic
token-length distributions, and document availability for the real-world
datasets that will feed the production-like LLM serving workload.

This stage DOES NOT build the final 1,000-request corpus and DOES NOT run GPU
inference.

Sources
-------
1. lmsys/chatbot_arena_conversations
2. google-research-datasets/nq_open
3. google-research-datasets/natural_questions (streaming audit only)

Outputs
-------
workloads/source_audit/
├── r0a_source_audit.json
└── r0a_preview.csv

Run from repository root:
    python src/r0a_source_acquisition.py

Optional:
    python src/r0a_source_acquisition.py --nq-stream-samples 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = PROJECT_ROOT / "workloads" / "source_audit"
AUDIT_JSON = OUT_DIR / "r0a_source_audit.json"
PREVIEW_CSV = OUT_DIR / "r0a_preview.csv"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


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

def distribution(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}

    s = pd.Series(values, dtype="float64")
    return {
        "count": int(len(s)),
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
        "mean": float(s.mean()),
    }


def first_user_message(conversation: Any) -> Optional[str]:
    if not isinstance(conversation, list):
        return None

    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = normalize_text(msg.get("role", msg.get("from", ""))).lower()
        if role in {"user", "human"}:
            content = msg.get("content", msg.get("value", msg.get("text")))
            text = normalize_text(content)
            if text:
                return text

    return None


def moderation_flagged(record: Dict[str, Any]) -> bool:
    moderation = record.get("openai_moderation")
    if isinstance(moderation, dict) and bool(moderation.get("flagged", False)):
        return True
    return False


def toxic_flagged(record: Dict[str, Any]) -> bool:
    toxic = record.get("toxic_chat_tag")
    if not isinstance(toxic, dict):
        return False

    for classifier in ("roberta-large", "t5-large"):
        obj = toxic.get(classifier)
        if isinstance(obj, dict) and bool(obj.get("flagged", False)):
            return True

    return False


def extract_nq_open_question(record: Dict[str, Any]) -> str:
    q = record.get("question")
    if isinstance(q, str):
        return normalize_text(q)
    if isinstance(q, dict):
        return normalize_text(q.get("text", q.get("question", "")))
    return ""


def extract_full_nq_question(record: Dict[str, Any]) -> str:
    q = record.get("question")
    if isinstance(q, dict):
        return normalize_text(q.get("text", ""))
    if isinstance(q, str):
        return normalize_text(q)
    return ""


def count_non_html_document_tokens(record: Dict[str, Any]) -> Optional[int]:
    document = record.get("document")
    if not isinstance(document, dict):
        return None

    tokens = document.get("tokens")

    # HF representation can be dict-of-lists.
    if isinstance(tokens, dict):
        token_values = tokens.get("token")
        is_html = tokens.get("is_html")
        if isinstance(token_values, list):
            if isinstance(is_html, list) and len(is_html) == len(token_values):
                return sum(1 for flag in is_html if not bool(flag))
            return len(token_values)

    # Alternate list-of-dicts representation.
    if isinstance(tokens, list):
        count = 0
        for item in tokens:
            if isinstance(item, dict):
                if not bool(item.get("is_html", False)):
                    count += 1
            elif isinstance(item, str):
                count += 1
        return count

    return None


def audit_lmsys(tokenizer, preview_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n[R0-A 1/3] LMSYS Chatbot Arena")
    print("Loading gated dataset...")

    ds = load_dataset(
        "lmsys/chatbot_arena_conversations",
        split="train",
    )

    languages = Counter()
    prompt_tokens: List[int] = []
    prompt_chars: List[int] = []
    turns: List[int] = []
    unique_question_ids = set()

    moderation_count = 0
    toxic_count = 0
    either_flagged_count = 0
    missing_prompt_count = 0
    english_count = 0

    samples_added = 0

    for rec in ds:
        qid = normalize_text(rec.get("question_id"))
        if qid:
            unique_question_ids.add(qid)

        language = normalize_text(rec.get("language")) or "unknown"
        languages[language] += 1
        if language.lower().startswith("english") or language.lower() == "en":
            english_count += 1

        if isinstance(rec.get("turn"), (int, float)):
            turns.append(int(rec["turn"]))

        mod = moderation_flagged(rec)
        tox = toxic_flagged(rec)

        if mod:
            moderation_count += 1
        if tox:
            toxic_count += 1
        if mod or tox:
            either_flagged_count += 1

        prompt = first_user_message(rec.get("conversation_a"))
        if not prompt:
            missing_prompt_count += 1
            continue

        token_count = qwen_chat_tokens(tokenizer, prompt)
        prompt_tokens.append(token_count)
        prompt_chars.append(len(prompt))

        if samples_added < 8:
            preview_rows.append({
                "source": "lmsys_chatbot_arena_conversations",
                "source_id": qid,
                "language": language,
                "flagged": mod or tox,
                "question_tokens": token_count,
                "document_tokens": None,
                "text_preview": prompt[:500],
            })
            samples_added += 1

    result = {
        "dataset": "lmsys/chatbot_arena_conversations",
        "loaded_rows": len(ds),
        "columns": list(ds.column_names),
        "unique_question_ids": len(unique_question_ids),
        "missing_first_user_prompt": missing_prompt_count,
        "english_rows": english_count,
        "english_fraction": english_count / len(ds) if len(ds) else None,
        "openai_moderation_flagged_rows": moderation_count,
        "toxic_tag_flagged_rows": toxic_count,
        "either_safety_flagged_rows": either_flagged_count,
        "either_safety_flagged_fraction": (
            either_flagged_count / len(ds) if len(ds) else None
        ),
        "top_languages": languages.most_common(15),
        "turn_distribution": distribution(turns),
        "qwen_input_token_distribution": distribution(prompt_tokens),
        "prompt_character_distribution": distribution(prompt_chars),
    }

    print(f"Rows:             {result['loaded_rows']}")
    print(f"Unique questions: {result['unique_question_ids']}")
    print(f"English rows:     {english_count}")
    print(f"Safety flagged:   {either_flagged_count}")
    print(f"Prompt P50/P95:   "
          f"{result['qwen_input_token_distribution'].get('p50', 'n/a'):.1f} / "
          f"{result['qwen_input_token_distribution'].get('p95', 'n/a'):.1f} tokens")

    return result


def audit_nq_open(tokenizer, preview_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n[R0-A 2/3] Google NQ Open")
    print("Loading dataset...")

    ds = load_dataset(
        "google-research-datasets/nq_open",
        split="train",
    )

    question_tokens: List[int] = []
    question_chars: List[int] = []
    answer_counts: List[int] = []
    unique_questions = set()
    missing_questions = 0

    samples_added = 0

    for i, rec in enumerate(ds):
        question = extract_nq_open_question(rec)
        if not question:
            missing_questions += 1
            continue

        unique_questions.add(question.lower())
        t = qwen_chat_tokens(tokenizer, question)
        question_tokens.append(t)
        question_chars.append(len(question))

        answers = rec.get("answer", rec.get("answers"))
        if isinstance(answers, list):
            answer_counts.append(len(answers))
        elif answers is not None:
            answer_counts.append(1)

        if samples_added < 8:
            preview_rows.append({
                "source": "google_nq_open",
                "source_id": f"train_{i}",
                "language": "English",
                "flagged": None,
                "question_tokens": t,
                "document_tokens": None,
                "text_preview": question[:500],
            })
            samples_added += 1

    result = {
        "dataset": "google-research-datasets/nq_open",
        "loaded_rows": len(ds),
        "columns": list(ds.column_names),
        "unique_normalized_questions": len(unique_questions),
        "missing_questions": missing_questions,
        "qwen_input_token_distribution": distribution(question_tokens),
        "question_character_distribution": distribution(question_chars),
        "answer_count_distribution": distribution(answer_counts),
    }

    print(f"Rows:             {result['loaded_rows']}")
    print(f"Unique questions: {result['unique_normalized_questions']}")
    print(f"Question P50/P95: "
          f"{result['qwen_input_token_distribution'].get('p50', 'n/a'):.1f} / "
          f"{result['qwen_input_token_distribution'].get('p95', 'n/a'):.1f} tokens")

    return result


def audit_full_nq_stream(
    tokenizer,
    preview_rows: List[Dict[str, Any]],
    sample_count: int,
    seed: int,
) -> Dict[str, Any]:
    print("\n[R0-A 3/3] Full Google Natural Questions")
    print("Streaming a small shuffled audit sample; full dataset is NOT downloaded...")

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )

    # Buffer shuffle gives a useful non-prefix audit sample without materializing
    # the full dataset.
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    question_tokens: List[int] = []
    document_token_counts: List[int] = []
    ids = set()
    missing_question = 0
    missing_document = 0

    samples_added = 0
    actual = 0

    for rec in ds:
        if actual >= sample_count:
            break

        actual += 1

        rid = normalize_text(rec.get("id", rec.get("example_id", "")))
        if rid:
            ids.add(rid)

        question = extract_full_nq_question(rec)
        if not question:
            missing_question += 1
        else:
            question_tokens.append(qwen_chat_tokens(tokenizer, question))

        doc_tokens = count_non_html_document_tokens(rec)
        if doc_tokens is None:
            missing_document += 1
        else:
            document_token_counts.append(doc_tokens)

        if samples_added < 8:
            preview_rows.append({
                "source": "google_natural_questions",
                "source_id": rid,
                "language": "English",
                "flagged": None,
                "question_tokens": (
                    qwen_chat_tokens(tokenizer, question)
                    if question else None
                ),
                "document_tokens": doc_tokens,
                "text_preview": question[:500] if question else "",
            })
            samples_added += 1

    result = {
        "dataset": "google-research-datasets/natural_questions",
        "mode": "streaming audit sample only",
        "requested_sample_rows": sample_count,
        "audited_rows": actual,
        "unique_ids_in_sample": len(ids),
        "missing_questions": missing_question,
        "missing_documents": missing_document,
        "qwen_question_token_distribution": distribution(question_tokens),
        "non_html_document_token_distribution": distribution(document_token_counts),
        "note": (
            "Document token counts are source-document tokens, not final Qwen "
            "prompt lengths. R0-B will construct/truncate document prompts."
        ),
    }

    print(f"Audited rows:      {actual}")
    if result["qwen_question_token_distribution"]:
        print(
            f"Question P50/P95: "
            f"{result['qwen_question_token_distribution']['p50']:.1f} / "
            f"{result['qwen_question_token_distribution']['p95']:.1f} tokens"
        )
    if result["non_html_document_token_distribution"]:
        print(
            f"Document P50/P95: "
            f"{result['non_html_document_token_distribution']['p50']:.1f} / "
            f"{result['non_html_document_token_distribution']['p95']:.1f} source tokens"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nq-stream-samples",
        type=int,
        default=200,
        help="Number of full-NQ streaming rows to audit (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Shuffle/random seed (default: 42)",
    )
    args = parser.parse_args()

    if args.nq_stream_samples <= 0:
        raise ValueError("--nq-stream-samples must be > 0")

    print("=" * 76)
    print("R0-A — REAL-WORLD SOURCE ACQUISITION & AUDIT")
    print("=" * 76)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tokenizer:    {MODEL_NAME}")
    print(f"Seed:         {args.seed}")
    print()
    print("This stage validates source access and distributions.")
    print("It does NOT build the final 1,000-request workload.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    preview_rows: List[Dict[str, Any]] = []

    audit = {
        "stage": "R0-A",
        "name": "Real-world source acquisition and audit",
        "tokenizer": MODEL_NAME,
        "seed": args.seed,
        "sources": {},
    }

    audit["sources"]["lmsys"] = audit_lmsys(tokenizer, preview_rows)
    audit["sources"]["nq_open"] = audit_nq_open(tokenizer, preview_rows)
    audit["sources"]["natural_questions"] = audit_full_nq_stream(
        tokenizer,
        preview_rows,
        sample_count=args.nq_stream_samples,
        seed=args.seed,
    )

    with AUDIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    pd.DataFrame(preview_rows).to_csv(PREVIEW_CSV, index=False)

    print("\n" + "=" * 76)
    print("R0-A COMPLETE")
    print("=" * 76)
    print(f"Audit report: {AUDIT_JSON}")
    print(f"Preview:      {PREVIEW_CSV}")
    print()
    print("Next stage after review: R0-B cleaning, deduplication, and source filtering.")


if __name__ == "__main__":
    main()
