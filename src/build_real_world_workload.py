#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

TARGET_COUNTS = {
    "short_chat": 300,
    "knowledge_qa": 200,
    "coding_reasoning": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "generation_heavy": 100,
}

MAX_NEW_TOKENS = {
    "short_chat": 128,
    "knowledge_qa": 128,
    "coding_reasoning": 384,
    "document_qa": 256,
    "long_context_qa": 256,
    "generation_heavy": 512,
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "workloads"
OUTPUT_JSON = OUTPUT_DIR / "realistic_requests.json"
STATS_JSON = OUTPUT_DIR / "workload_stats.json"
PREVIEW_CSV = OUTPUT_DIR / "workload_preview.csv"

CODING_TERMS = [
    "python", "javascript", "typescript", "java ", " c++", " c#", "golang",
    "rust ", "sql", "bash", "shell", "regex", "function", "class ", "api ",
    "debug", "bug", "stack trace", "exception", "compile", "algorithm",
    "leetcode", "program", "coding", "code ", "html", "css", "react",
    "pytorch", "tensorflow", "numpy", "pandas", "git ", "docker",
]

REASONING_TERMS = [
    "solve", "prove", "derive", "calculate", "equation", "probability",
    "mathematical", "math problem", "logic puzzle", "step by step",
    "reasoning", "theorem", "integral", "derivative", "matrix",
]

GENERATION_TERMS = [
    "write an essay", "write a story", "write a report", "write an article",
    "write a blog", "draft a", "compose a", "create a story", "create a report",
    "generate a story", "generate an essay", "screenplay", "script for",
    "cover letter", "press release", "marketing copy", "detailed guide",
    "comprehensive guide", "in-depth", "long-form", "newsletter",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def chat_token_count(tokenizer, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        return len(ids)
    except Exception:
        return len(tokenizer.encode(prompt, add_special_tokens=True))


def is_english_record(record: Dict[str, Any]) -> bool:
    lang = record.get("language")
    if lang is None:
        return True
    s = str(lang).strip().lower()
    return s.startswith("en") or s == "english"


def looks_flagged(record: Dict[str, Any]) -> bool:
    toxic = record.get("toxic_chat_tag")
    if toxic is True:
        return True
    if isinstance(toxic, str) and toxic.lower() in {"true", "1", "yes", "toxic"}:
        return True

    moderation = record.get("openai_moderation")
    if isinstance(moderation, dict):
        if moderation.get("flagged") is True:
            return True
        results = moderation.get("results")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and r.get("flagged") is True:
                    return True
    return False


def first_user_message(conversation: Any) -> Optional[str]:
    if conversation is None:
        return None
    if isinstance(conversation, str):
        s = conversation.strip()
        try:
            conversation = json.loads(s)
        except Exception:
            return normalize_text(s) if s else None
    if isinstance(conversation, dict):
        for key in ("messages", "conversation", "turns"):
            if key in conversation:
                conversation = conversation[key]
                break
    if not isinstance(conversation, list):
        return None
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", msg.get("from", msg.get("speaker", "")))).lower()
        if role in {"user", "human"}:
            content = msg.get("content", msg.get("value", msg.get("text")))
            if content:
                return normalize_text(content)
    for msg in conversation:
        if isinstance(msg, dict):
            content = msg.get("content", msg.get("value", msg.get("text")))
            if content:
                return normalize_text(content)
    return None


def extract_lmsys_prompt(record: Dict[str, Any]) -> Optional[str]:
    for key in ("conversation_a", "conversation", "messages", "turns"):
        if key in record:
            prompt = first_user_message(record[key])
            if prompt:
                return prompt
    for key in ("prompt", "question", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return None


def classify_lmsys_prompt(prompt: str, token_count: int) -> Optional[str]:
    p = prompt.lower()
    if "```" in prompt or any(term in p for term in CODING_TERMS):
        return "coding_reasoning"
    if any(term in p for term in REASONING_TERMS):
        return "coding_reasoning"
    if any(term in p for term in GENERATION_TERMS):
        return "generation_heavy"
    if token_count <= 128:
        return "short_chat"
    return None


def question_text(record: Dict[str, Any]) -> Optional[str]:
    q = record.get("question")
    if isinstance(q, str):
        return normalize_text(q)
    if isinstance(q, dict):
        for key in ("text", "question"):
            value = q.get(key)
            if isinstance(value, str):
                return normalize_text(value)
    for key in ("query", "prompt"):
        value = record.get(key)
        if isinstance(value, str):
            return normalize_text(value)
    return None


def extract_nq_document_text(record: Dict[str, Any]) -> Optional[str]:
    doc = record.get("document")
    if not isinstance(doc, dict):
        return None
    tokens = doc.get("tokens")
    if isinstance(tokens, dict):
        token_values = tokens.get("token")
        is_html_values = tokens.get("is_html")
        if isinstance(token_values, list):
            if isinstance(is_html_values, list) and len(is_html_values) == len(token_values):
                clean = [
                    str(tok)
                    for tok, is_html in zip(token_values, is_html_values)
                    if not bool(is_html)
                ]
            else:
                clean = [str(tok) for tok in token_values]
            return normalize_text(" ".join(clean))
    if isinstance(tokens, list):
        clean = []
        for item in tokens:
            if isinstance(item, dict):
                if not item.get("is_html", False):
                    tok = item.get("token")
                    if tok:
                        clean.append(str(tok))
            elif isinstance(item, str):
                clean.append(item)
        if clean:
            return normalize_text(" ".join(clean))
    return None


def truncate_context_to_band(
    tokenizer,
    question: str,
    context: str,
    min_input_tokens: int,
    max_input_tokens: int,
    rng: random.Random,
) -> Optional[Tuple[str, int, int]]:
    target_cap = rng.randint(min_input_tokens, max_input_tokens)
    prefix = (
        "Use the following document to answer the question. "
        "Base the answer on the document. If the answer cannot be determined "
        "from the document, say so.\n\nDocument:\n"
    )
    suffix = f"\n\nQuestion: {question}\nAnswer:"
    overhead = chat_token_count(tokenizer, prefix + suffix)
    if overhead >= target_cap - 64:
        return None
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) < min_input_tokens - overhead:
        return None
    context_budget = max(64, target_cap - overhead - 8)
    context_ids = context_ids[:context_budget]
    for _ in range(8):
        trimmed_context = tokenizer.decode(context_ids, skip_special_tokens=True)
        prompt = prefix + trimmed_context + suffix
        n_tokens = chat_token_count(tokenizer, prompt)
        if n_tokens > max_input_tokens:
            context_ids = context_ids[:-32]
            if len(context_ids) < 64:
                return None
            continue
        if n_tokens < min_input_tokens:
            return None
        return prompt, n_tokens, target_cap
    return None


def add_request(
    requests: List[Dict[str, Any]],
    seen_prompt_hashes: set,
    *,
    category: str,
    prompt: str,
    source: str,
    source_id: str,
    input_tokens: int,
    source_license: str,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    prompt = prompt.strip()
    h = stable_hash(prompt)
    if h in seen_prompt_hashes:
        return False
    idx = len(requests) + 1
    item = {
        "id": f"req_{idx:04d}",
        "category": category,
        "prompt": prompt,
        "source": source,
        "source_id": str(source_id),
        "source_license": source_license,
        "input_tokens_estimate": int(input_tokens),
        "max_new_tokens": MAX_NEW_TOKENS[category],
    }
    if extra:
        item.update(extra)
    requests.append(item)
    seen_prompt_hashes.add(h)
    return True


def build_lmsys_requests(tokenizer, requests, seen_prompt_hashes, rng):
    needed = {
        "short_chat": TARGET_COUNTS["short_chat"],
        "coding_reasoning": TARGET_COUNTS["coding_reasoning"],
        "generation_heavy": TARGET_COUNTS["generation_heavy"],
    }
    print("\n[1/3] Loading LMSYS Chatbot Arena Conversations...")
    print("      Make sure you accepted the gated dataset terms on Hugging Face.")
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    counts = Counter()
    for i in indices:
        if all(counts[k] >= v for k, v in needed.items()):
            break
        rec = ds[i]
        if not is_english_record(rec) or looks_flagged(rec):
            continue
        prompt = extract_lmsys_prompt(rec)
        if not prompt or len(prompt) < 5:
            continue
        n_tokens = chat_token_count(tokenizer, prompt)
        if n_tokens > 1024:
            continue
        category = classify_lmsys_prompt(prompt, n_tokens)
        if category not in needed or counts[category] >= needed[category]:
            continue
        source_id = rec.get("question_id", rec.get("id", f"row_{i}"))
        ok = add_request(
            requests,
            seen_prompt_hashes,
            category=category,
            prompt=prompt,
            source="lmsys_chatbot_arena_conversations",
            source_id=str(source_id),
            input_tokens=n_tokens,
            source_license="CC-BY-4.0 (user prompts)",
            extra={
                "language": rec.get("language", "unknown"),
                "selection_rule": "heuristic_category_filter",
            },
        )
        if ok:
            counts[category] += 1
    print("      LMSYS selected:", dict(counts))
    missing = {k: needed[k] - counts[k] for k in needed if counts[k] < needed[k]}
    if missing:
        raise RuntimeError(f"Could not fill LMSYS categories: {missing}")


def build_nq_open_requests(tokenizer, requests, seen_prompt_hashes, rng):
    target = TARGET_COUNTS["knowledge_qa"]
    print("\n[2/3] Loading Google NQ Open for knowledge QA...")
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    count = 0
    for i in indices:
        if count >= target:
            break
        rec = ds[i]
        question = question_text(rec)
        if not question or len(question) < 5:
            continue
        n_tokens = chat_token_count(tokenizer, question)
        if n_tokens > 256:
            continue
        source_id = rec.get("id", f"row_{i}")
        ok = add_request(
            requests,
            seen_prompt_hashes,
            category="knowledge_qa",
            prompt=question,
            source="google_nq_open",
            source_id=str(source_id),
            input_tokens=n_tokens,
            source_license="CC-BY-SA-3.0",
            extra={"selection_rule": "real_search_question"},
        )
        if ok:
            count += 1
    print(f"      NQ Open selected: {count}")
    if count < target:
        raise RuntimeError(f"Could only select {count}/{target} NQ Open requests.")


def build_nq_document_requests(tokenizer, requests, seen_prompt_hashes, rng):
    needed = {
        "document_qa": TARGET_COUNTS["document_qa"],
        "long_context_qa": TARGET_COUNTS["long_context_qa"],
    }
    print("\n[3/3] Streaming full Google Natural Questions for document workloads...")
    print("      Streaming avoids downloading the complete source dataset.")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=SEED, buffer_size=5000)
    counts = Counter()
    source_questions_seen = set()
    for rec in ds:
        if all(counts[k] >= v for k, v in needed.items()):
            break
        q = question_text(rec)
        context = extract_nq_document_text(rec)
        if not q or not context or len(q) < 5 or len(context) < 1000:
            continue
        q_hash = stable_hash(q)
        if q_hash in source_questions_seen:
            continue
        if counts["document_qa"] < needed["document_qa"] and (
            counts["long_context_qa"] >= needed["long_context_qa"]
            or (counts["document_qa"] + counts["long_context_qa"]) % 2 == 0
        ):
            category = "document_qa"
            band = (1024, 2048)
        elif counts["long_context_qa"] < needed["long_context_qa"]:
            category = "long_context_qa"
            band = (3072, 4096)
        else:
            category = "document_qa"
            band = (1024, 2048)
        built = truncate_context_to_band(
            tokenizer,
            question=q,
            context=context,
            min_input_tokens=band[0],
            max_input_tokens=band[1],
            rng=rng,
        )
        if built is None:
            continue
        prompt, n_tokens, target_cap = built
        source_id = rec.get("id", rec.get("example_id", stable_hash(q)))
        ok = add_request(
            requests,
            seen_prompt_hashes,
            category=category,
            prompt=prompt,
            source="google_natural_questions",
            source_id=str(source_id),
            input_tokens=n_tokens,
            source_license="CC-BY-SA-3.0",
            extra={
                "question": q,
                "input_length_band": [band[0], band[1]],
                "target_input_cap": target_cap,
                "selection_rule": "real_question_plus_source_document_context",
            },
        )
        if ok:
            counts[category] += 1
            source_questions_seen.add(q_hash)
            total = counts["document_qa"] + counts["long_context_qa"]
            if total % 25 == 0:
                print(
                    f"      Document workloads selected: "
                    f"{counts['document_qa']} document / {counts['long_context_qa']} long"
                )
    print("      NQ document selected:", dict(counts))
    missing = {k: needed[k] - counts[k] for k in needed if counts[k] < needed[k]}
    if missing:
        raise RuntimeError(f"Could not fill NQ document categories: {missing}")


def write_outputs(requests: List[Dict[str, Any]]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    category_order = {k: i for i, k in enumerate(TARGET_COUNTS)}
    requests.sort(key=lambda x: (category_order[x["category"]], x["source_id"]))
    for i, item in enumerate(requests, start=1):
        item["id"] = f"req_{i:04d}"

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "name": "Real-World LLM Inference Workload",
                    "version": 1,
                    "seed": SEED,
                    "tokenizer": MODEL_NAME,
                    "total_requests": len(requests),
                    "target_counts": TARGET_COUNTS,
                    "notes": [
                        "LMSYS prompts are real user prompts from Chatbot Arena.",
                        "NQ questions originate from real Google search queries.",
                        "Document/long-context prompts wrap real NQ questions and Wikipedia document context in a fixed QA instruction.",
                        "input_tokens_estimate includes the Qwen chat template when available.",
                        "Actual token counts must be recorded again during benchmark execution.",
                    ],
                },
                "requests": requests,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    df = pd.DataFrame(requests)
    stats = {
        "total_requests": len(df),
        "by_category": df["category"].value_counts().to_dict(),
        "input_tokens": {
            "min": int(df["input_tokens_estimate"].min()),
            "p50": float(df["input_tokens_estimate"].quantile(0.50)),
            "p90": float(df["input_tokens_estimate"].quantile(0.90)),
            "p95": float(df["input_tokens_estimate"].quantile(0.95)),
            "p99": float(df["input_tokens_estimate"].quantile(0.99)),
            "max": int(df["input_tokens_estimate"].max()),
            "mean": float(df["input_tokens_estimate"].mean()),
        },
        "category_input_tokens": {},
    }
    for category, g in df.groupby("category"):
        stats["category_input_tokens"][category] = {
            "count": int(len(g)),
            "min": int(g["input_tokens_estimate"].min()),
            "p50": float(g["input_tokens_estimate"].quantile(0.50)),
            "p95": float(g["input_tokens_estimate"].quantile(0.95)),
            "max": int(g["input_tokens_estimate"].max()),
            "mean": float(g["input_tokens_estimate"].mean()),
        }

    with STATS_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    preview_cols = [
        "id", "category", "source", "source_id",
        "input_tokens_estimate", "max_new_tokens", "prompt",
    ]
    df[preview_cols].to_csv(PREVIEW_CSV, index=False)

    print("\n" + "=" * 72)
    print("WORKLOAD BUILD COMPLETE")
    print("=" * 72)
    print(f"Total requests: {len(requests)}")
    for category in TARGET_COUNTS:
        n = sum(r["category"] == category for r in requests)
        print(f"  {category:20s} {n}")
    print("\nSaved:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {STATS_JSON}")
    print(f"  {PREVIEW_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print("=" * 72)
    print("R0 — Build 1,000-Request Real-World Workload Corpus")
    print("=" * 72)
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Seed:      {args.seed}")
    print(f"Target:    {sum(TARGET_COUNTS.values())} unique requests")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    requests: List[Dict[str, Any]] = []
    seen_prompt_hashes = set()

    build_lmsys_requests(tokenizer, requests, seen_prompt_hashes, rng)
    build_nq_open_requests(tokenizer, requests, seen_prompt_hashes, rng)
    build_nq_document_requests(tokenizer, requests, seen_prompt_hashes, rng)

    expected = sum(TARGET_COUNTS.values())
    if len(requests) != expected:
        raise RuntimeError(f"Expected {expected} requests, got {len(requests)}")

    write_outputs(requests)


if __name__ == "__main__":
    main()
