#!/usr/bin/env python3
"""
S3-A vLLM Sequential Validation v2

Goals
-----
- Reuse the frozen R0 workload exactly.
- Reproduce the R1 pilot selection: deterministic shuffle with seed 2026,
  first 10 requests/category (60 total) in the fixed mixed order.
- Reuse R1 per-request sampling seeds: 42026000 + numeric request-id suffix.
- Reuse each frozen request's category-specific max_new_tokens cap.
- Warm up with six representative NON-MEASURED requests (one/category), capped at
  32 output tokens, to avoid exact-prompt prefix-cache contamination.
- Natural generation: stop at EOS or the frozen category cap.
- Record client TTFT, visible-text TTFT, E2E, SSE event timing, usage/token counts,
  finish reason, hashes, and strict prompt-token validation.

This client is intended for the compiled vLLM server on 127.0.0.1:8001.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx


# -----------------------------------------------------------------------------
# Frozen protocol
# -----------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8001"

WORKLOAD_FILE = Path("workloads/final/realistic_requests.json")
EXPECTED_CORPUS_SHA256 = (
    "7593429c095064a7f375e12d40db43ebf174d0f3a57f49bc51db030a0619d014"
)

RESULT_ROOT = Path("results/s3/raw/sequential/pilot60")

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1
SAMPLING_SEED_BASE = 42026000

EXECUTION_SEED = 2026
PILOT_PER_CATEGORY = 10
WARMUP_MAX_TOKENS = 32

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]

EXPECTED_COUNTS = {
    "short_interactive": 300,
    "knowledge_qa": 200,
    "coding_request": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "long_output": 100,
}

REQUEST_FIELDS = [
    "request_id",
    "workload_category",
    "status",
    "http_status",
    "error_type",
    "error_message",
    "sampling_seed",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "frozen_input_tokens",
    "server_prompt_tokens",
    "prompt_token_match",
    "max_tokens",
    "output_tokens",
    "finish_reason",
    "hit_max_tokens",
    "output_chars",
    "output_sha256",
    "client_first_choice_event_ttft_ms",
    "client_first_token_event_ttft_ms",
    "client_first_visible_text_ttft_ms",
    "client_e2e_ms",
    "sse_choice_event_count",
    "sse_content_event_count",
    "sse_visible_event_count",
    "mean_content_event_itl_ms",
    "p50_content_event_itl_ms",
    "p95_content_event_itl_ms",
    "content_event_count_matches_output_tokens",
    "request_payload_bytes",
]

EVENT_FIELDS = [
    "request_id",
    "workload_category",
    "choice_event_index",
    "content_event_index",
    "arrival_ms_from_request_start",
    "choice_inter_event_ms",
    "content_inter_event_ms",
    "content_present",
    "content_visible",
    "payload_chars",
    "finish_reason",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def fmt_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}ms"


def request_sampling_seed(request_id: str) -> int:
    match = re.search(r"(\d+)$", str(request_id))
    if not match:
        raise RuntimeError(
            f"Could not derive sampling seed from request_id={request_id!r}"
        )
    return SAMPLING_SEED_BASE + int(match.group(1))


def load_workload(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen workload not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        requests = payload
        metadata: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        requests = payload.get("requests")
        metadata = payload.get("metadata", {})
    else:
        raise RuntimeError("Unsupported workload JSON structure.")

    if not isinstance(requests, list):
        raise RuntimeError("Workload JSON does not contain a requests list.")

    if len(requests) != 1000:
        raise RuntimeError(
            f"Frozen corpus must contain exactly 1000 requests; found {len(requests)}."
        )

    ids = [str(row.get("request_id")) for row in requests]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate request_id detected in frozen corpus.")

    counts = {category: 0 for category in CATEGORY_ORDER}
    required_fields = {
        "request_id",
        "workload_category",
        "prompt",
        "input_tokens",
        "max_new_tokens",
    }

    for row in requests:
        missing = sorted(required_fields - set(row))
        if missing:
            raise RuntimeError(
                f"Request {row.get('request_id')} missing required fields: {missing}"
            )
        category = str(row["workload_category"])
        if category not in counts:
            raise RuntimeError(f"Unknown workload category: {category}")
        counts[category] += 1

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "Frozen category counts do not match R0 definition.\n"
            f"Expected: {EXPECTED_COUNTS}\nActual:   {counts}"
        )

    return metadata, requests


def verify_corpus(path: Path) -> str:
    actual = sha256_file(path)
    if actual != EXPECTED_CORPUS_SHA256:
        raise RuntimeError(
            "FROZEN CORPUS SHA-256 MISMATCH.\n"
            f"Expected: {EXPECTED_CORPUS_SHA256}\n"
            f"Actual:   {actual}\n"
            "Do not run S3 until the canonical workload is restored."
        )
    return actual


def create_pilot_order(requests: Sequence[Dict[str, Any]]) -> List[str]:
    request_by_id = {str(row["request_id"]): row for row in requests}
    all_ids = list(request_by_id)

    rng = random.Random(EXECUTION_SEED)
    rng.shuffle(all_ids)

    counts = {category: 0 for category in CATEGORY_ORDER}
    pilot_ids: List[str] = []

    for request_id in all_ids:
        category = str(request_by_id[request_id]["workload_category"])
        if counts[category] < PILOT_PER_CATEGORY:
            pilot_ids.append(request_id)
            counts[category] += 1

        if all(counts[c] == PILOT_PER_CATEGORY for c in CATEGORY_ORDER):
            break

    expected = PILOT_PER_CATEGORY * len(CATEGORY_ORDER)
    if len(pilot_ids) != expected:
        raise RuntimeError(
            f"Pilot selection produced {len(pilot_ids)} requests; expected {expected}."
        )

    return pilot_ids


def representative_warmups_excluding_measured(
    requests: Sequence[Dict[str, Any]], measured_ids: Iterable[str]
) -> List[Dict[str, Any]]:
    measured = set(str(x) for x in measured_ids)
    chosen: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        group = [
            row
            for row in requests
            if row["workload_category"] == category
            and str(row["request_id"]) not in measured
        ]
        if not group:
            raise RuntimeError(f"No non-measured warmup candidate for {category}")
        ordered = sorted(group, key=lambda r: int(r["input_tokens"]))
        chosen.append(ordered[len(ordered) // 2])

    return chosen


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Request execution
# -----------------------------------------------------------------------------


def make_payload(row: Dict[str, Any], max_tokens_override: Optional[int] = None) -> Dict[str, Any]:
    request_id = str(row["request_id"])
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": str(row["prompt"])}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": int(
            max_tokens_override
            if max_tokens_override is not None
            else row["max_new_tokens"]
        ),
        "seed": request_sampling_seed(request_id),
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def run_stream_request(
    *,
    client: httpx.Client,
    url: str,
    row: Dict[str, Any],
    max_tokens_override: Optional[int] = None,
    capture_events: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    request_id = str(row["request_id"])
    category = str(row["workload_category"])
    payload = make_payload(row, max_tokens_override=max_tokens_override)
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    start_ns = time.perf_counter_ns()
    first_choice_ns: Optional[int] = None
    first_content_ns: Optional[int] = None
    first_visible_ns: Optional[int] = None
    prev_choice_ns: Optional[int] = None
    prev_content_ns: Optional[int] = None

    choice_event_count = 0
    content_event_count = 0
    visible_event_count = 0
    content_event_itls: List[float] = []
    output_parts: List[str] = []
    event_rows: List[Dict[str, Any]] = []

    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    http_status: Optional[int] = None
    error_type = ""
    error_message = ""

    try:
        with client.stream("POST", url, json=payload) as response:
            http_status = response.status_code
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                now_ns = time.perf_counter_ns()

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if not choices:
                    continue

                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                content_present = "content" in delta and delta.get("content") is not None
                content = delta.get("content") if content_present else None
                content_visible = isinstance(content, str) and len(content) > 0

                choice_event_count += 1
                if first_choice_ns is None:
                    first_choice_ns = now_ns

                choice_inter_ms = (
                    None
                    if prev_choice_ns is None
                    else (now_ns - prev_choice_ns) / 1e6
                )
                prev_choice_ns = now_ns

                content_inter_ms: Optional[float] = None
                content_index: Optional[int] = None

                if content_present:
                    content_event_count += 1
                    content_index = content_event_count
                    if first_content_ns is None:
                        first_content_ns = now_ns

                    if prev_content_ns is not None:
                        content_inter_ms = (now_ns - prev_content_ns) / 1e6
                        content_event_itls.append(content_inter_ms)
                    prev_content_ns = now_ns

                    if isinstance(content, str):
                        output_parts.append(content)

                    if content_visible:
                        visible_event_count += 1
                        if first_visible_ns is None:
                            first_visible_ns = now_ns

                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])

                if capture_events:
                    event_rows.append(
                        {
                            "request_id": request_id,
                            "workload_category": category,
                            "choice_event_index": choice_event_count,
                            "content_event_index": content_index,
                            "arrival_ms_from_request_start": (now_ns - start_ns) / 1e6,
                            "choice_inter_event_ms": choice_inter_ms,
                            "content_inter_event_ms": content_inter_ms,
                            "content_present": int(content_present),
                            "content_visible": int(content_visible),
                            "payload_chars": len(content) if isinstance(content, str) else 0,
                            "finish_reason": choice.get("finish_reason") or "",
                        }
                    )

        end_ns = time.perf_counter_ns()

    except Exception as exc:
        end_ns = time.perf_counter_ns()
        error_type = type(exc).__name__
        error_message = str(exc)[:1000]

    output_text = "".join(output_parts)

    server_prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    prompt_token_match: Optional[bool] = None
    if server_prompt_tokens is not None:
        prompt_token_match = int(server_prompt_tokens) == int(row["input_tokens"])

    max_tokens = int(payload["max_tokens"])
    hit_max_tokens = (
        output_tokens is not None
        and int(output_tokens) >= max_tokens
        and finish_reason == "length"
    )

    content_event_match: Optional[bool] = None
    if output_tokens is not None:
        content_event_match = content_event_count == int(output_tokens)

    status = "ok" if not error_type else "error"

    result = {
        "request_id": request_id,
        "workload_category": category,
        "status": status,
        "http_status": http_status,
        "error_type": error_type,
        "error_message": error_message,
        "sampling_seed": payload["seed"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "frozen_input_tokens": int(row["input_tokens"]),
        "server_prompt_tokens": server_prompt_tokens,
        "prompt_token_match": prompt_token_match,
        "max_tokens": max_tokens,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "hit_max_tokens": int(bool(hit_max_tokens)),
        "output_chars": len(output_text),
        "output_sha256": sha256_text(output_text),
        "client_first_choice_event_ttft_ms": (
            None if first_choice_ns is None else (first_choice_ns - start_ns) / 1e6
        ),
        # Defined as first SSE choice event carrying a `delta.content` field.
        # This is not assumed to be exactly one generated token until audited.
        "client_first_token_event_ttft_ms": (
            None if first_content_ns is None else (first_content_ns - start_ns) / 1e6
        ),
        "client_first_visible_text_ttft_ms": (
            None if first_visible_ns is None else (first_visible_ns - start_ns) / 1e6
        ),
        "client_e2e_ms": (end_ns - start_ns) / 1e6,
        "sse_choice_event_count": choice_event_count,
        "sse_content_event_count": content_event_count,
        "sse_visible_event_count": visible_event_count,
        "mean_content_event_itl_ms": (
            statistics.mean(content_event_itls) if content_event_itls else None
        ),
        "p50_content_event_itl_ms": percentile(content_event_itls, 0.50),
        "p95_content_event_itl_ms": percentile(content_event_itls, 0.95),
        "content_event_count_matches_output_tokens": content_event_match,
        "request_payload_bytes": payload_bytes,
        # JSON-only convenience field; omitted from CSV by extrasaction=ignore.
        "output_text": output_text,
    }

    return result, event_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workload", type=Path, default=WORKLOAD_FILE)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    url = base_url + "/v1/chat/completions"

    corpus_sha256 = verify_corpus(args.workload)
    workload_metadata, requests = load_workload(args.workload)
    request_by_id = {str(row["request_id"]): row for row in requests}

    pilot_ids = create_pilot_order(requests)
    measured = [request_by_id[rid] for rid in pilot_ids]
    warmups = representative_warmups_excluding_measured(requests, pilot_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_csv = args.output_dir / "requests.csv"
    requests_json = args.output_dir / "requests.json"
    events_csv_gz = args.output_dir / "sse_events.csv.gz"
    metadata_json = args.output_dir / "metadata.json"
    order_json = args.output_dir / "pilot_order.json"

    existing = [p for p in [requests_csv, requests_json, events_csv_gz, metadata_json] if p.exists()]
    if existing:
        raise RuntimeError(
            "S3-A pilot output already exists. Move/delete it intentionally before rerun:\n"
            + "\n".join(f"  {p}" for p in existing)
        )

    print("=" * 88)
    print("S3-A v2 — vLLM NATURAL-GENERATION SEQUENTIAL VALIDATION")
    print("=" * 88)
    print(f"Workload:       {args.workload}")
    print(f"Corpus SHA256:  {corpus_sha256}")
    print(f"Server:         {base_url}")
    print(f"Model:          {MODEL_ID}")
    print(f"Warmups:        6 (one/category; excluded from measured set)")
    print(f"Measured:       {len(measured)} (10/category; R1 execution seed={EXECUTION_SEED})")
    print(f"Output dir:     {args.output_dir}")

    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)

    request_rows: List[Dict[str, Any]] = []
    request_json_rows: List[Dict[str, Any]] = []
    all_event_rows: List[Dict[str, Any]] = []

    with httpx.Client(timeout=timeout, limits=limits, http2=False) as client:
        # Health/model checks.
        health = client.get(base_url + "/health")
        health.raise_for_status()
        models = client.get(base_url + "/v1/models")
        models.raise_for_status()

        print("\n[1/2] Warmup")
        for i, row in enumerate(warmups, start=1):
            result, _ = run_stream_request(
                client=client,
                url=url,
                row=row,
                max_tokens_override=min(int(row["max_new_tokens"]), WARMUP_MAX_TOKENS),
                capture_events=False,
            )
            if result["status"] != "ok":
                raise RuntimeError(
                    f"Warmup failed for {row['workload_category']}: "
                    f"{result['error_type']}: {result['error_message']}"
                )
            if result["prompt_token_match"] is False:
                raise RuntimeError(
                    f"Warmup prompt-token mismatch for {row['request_id']}: "
                    f"frozen={row['input_tokens']} server={result['server_prompt_tokens']}"
                )
            print(
                f"  [{i}/6] {row['workload_category']:20s} "
                f"id={row['request_id']} in={int(row['input_tokens']):4d} "
                f"out={int(result['output_tokens']):3d} "
                f"TTFT={fmt_ms(result['client_first_token_event_ttft_ms'])}"
            )

        print("\n[2/2] Measured pilot60")
        run_start = time.perf_counter()

        for i, row in enumerate(measured, start=1):
            result, event_rows = run_stream_request(
                client=client,
                url=url,
                row=row,
                capture_events=True,
            )

            request_rows.append(result)
            request_json_rows.append(result)
            all_event_rows.extend(event_rows)

            print(
                f"  [{i:2d}/60] {row['request_id']} "
                f"{row['workload_category']:20s} "
                f"in={int(row['input_tokens']):4d} "
                f"cap={int(row['max_new_tokens']):4d}",
                end="",
                flush=True,
            )

            if result["status"] != "ok":
                print(f" -> ERROR {result['error_type']}: {result['error_message']}")
                raise RuntimeError(
                    f"Measured request failed: {row['request_id']}"
                )

            if result["prompt_token_match"] is False:
                print(
                    f" -> TOKEN MISMATCH frozen={row['input_tokens']} "
                    f"server={result['server_prompt_tokens']}"
                )
                raise RuntimeError(
                    f"Prompt-token mismatch for {row['request_id']}"
                )

            print(
                f" -> out={int(result['output_tokens']):4d} "
                f"TTFT={fmt_ms(result['client_first_token_event_ttft_ms']):>10s} "
                f"visible={fmt_ms(result['client_first_visible_text_ttft_ms']):>10s} "
                f"E2E={float(result['client_e2e_ms'])/1000.0:6.2f}s "
                f"[{result['finish_reason']}]"
            )

        elapsed = time.perf_counter() - run_start

    # Write outputs only after complete successful run.
    write_csv(requests_csv, REQUEST_FIELDS, request_rows)
    write_gzip_csv(events_csv_gz, EVENT_FIELDS, all_event_rows)

    with requests_json.open("w", encoding="utf-8") as f:
        json.dump(request_json_rows, f, ensure_ascii=False, indent=2)

    with order_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "execution_seed": EXECUTION_SEED,
                "pilot_per_category": PILOT_PER_CATEGORY,
                "pilot_request_ids": pilot_ids,
            },
            f,
            indent=2,
        )

    successful = [r for r in request_rows if r["status"] == "ok"]
    token_matches = [r for r in successful if r["prompt_token_match"] is True]
    event_matches = [
        r
        for r in successful
        if r["content_event_count_matches_output_tokens"] is True
    ]

    ttft = [
        float(r["client_first_token_event_ttft_ms"])
        for r in successful
        if r["client_first_token_event_ttft_ms"] is not None
    ]
    visible_ttft = [
        float(r["client_first_visible_text_ttft_ms"])
        for r in successful
        if r["client_first_visible_text_ttft_ms"] is not None
    ]
    e2e = [float(r["client_e2e_ms"]) for r in successful]
    output_tokens_total = sum(int(r["output_tokens"] or 0) for r in successful)

    finish_counts: Dict[str, int] = {}
    for r in successful:
        key = str(r["finish_reason"])
        finish_counts[key] = finish_counts.get(key, 0) + 1

    metadata = {
        "phase": "S3-A",
        "protocol_version": 2,
        "mode": "natural_generation_sequential_pilot60",
        "status": "complete",
        "corpus_sha256": corpus_sha256,
        "workload_metadata": workload_metadata,
        "execution_seed": EXECUTION_SEED,
        "pilot_per_category": PILOT_PER_CATEGORY,
        "measured_request_count": len(request_rows),
        "successful_request_count": len(successful),
        "warmup_count": 6,
        "warmup_policy": (
            "one median-input request/category selected outside measured pilot; "
            "max_tokens capped at 32"
        ),
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "per_request_seed_base": SAMPLING_SEED_BASE,
            "seed_formula": "42026000 + integer suffix of request_id",
            "natural_eos": True,
            "max_tokens_policy": "frozen row.max_new_tokens",
        },
        "client": {
            "http2": False,
            "max_connections": 1,
            "max_keepalive_connections": 1,
            "endpoint": url,
        },
        "server_expected_config": {
            "vllm_version": "0.28.0",
            "dtype": "bfloat16",
            "attention_backend": "FlashAttention 2",
            "flashinfer_sampler": False,
            "torch_compile": True,
            "cuda_graphs": True,
            "prefix_caching": True,
            "chunked_prefill": True,
            "gpu_memory_utilization": 0.85,
            "wsl2_pin_memory": True,
        },
        "streaming_metric_semantics": {
            "client_first_token_event_ttft_ms": (
                "first SSE choice event carrying delta.content; not assumed to map "
                "1:1 to generated tokens unless event-count audit passes"
            ),
            "client_first_visible_text_ttft_ms": (
                "first SSE delta.content with non-empty visible text"
            ),
        },
        "audit": {
            "prompt_token_matches": len(token_matches),
            "prompt_token_total": len(successful),
            "content_event_count_matches_output_tokens": len(event_matches),
            "content_event_count_total": len(successful),
            "finish_reason_counts": finish_counts,
            "total_output_tokens": output_tokens_total,
        },
        "summary": {
            "p50_first_token_event_ttft_ms": percentile(ttft, 0.50),
            "p95_first_token_event_ttft_ms": percentile(ttft, 0.95),
            "p50_first_visible_text_ttft_ms": percentile(visible_ttft, 0.50),
            "p95_first_visible_text_ttft_ms": percentile(visible_ttft, 0.95),
            "p50_e2e_ms": percentile(e2e, 0.50),
            "p95_e2e_ms": percentile(e2e, 0.95),
            "elapsed_seconds_measured": elapsed,
        },
    }

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("S3-A v2 PILOT COMPLETE")
    print("=" * 88)
    print(f"Success:                  {len(successful)}/60")
    print(f"Prompt-token matches:     {len(token_matches)}/{len(successful)}")
    print(
        "SSE content-event = tokens: "
        f"{len(event_matches)}/{len(successful)} requests"
    )
    print(f"Finish reasons:           {finish_counts}")
    print(f"Total output tokens:      {output_tokens_total}")
    print(f"P50 token-event TTFT:     {fmt_ms(percentile(ttft, 0.50))}")
    print(f"P95 token-event TTFT:     {fmt_ms(percentile(ttft, 0.95))}")
    print(f"P50 visible-text TTFT:    {fmt_ms(percentile(visible_ttft, 0.50))}")
    print(f"P95 visible-text TTFT:    {fmt_ms(percentile(visible_ttft, 0.95))}")
    print(f"P50 E2E:                  {fmt_ms(percentile(e2e, 0.50))}")
    print(f"P95 E2E:                  {fmt_ms(percentile(e2e, 0.95))}")
    print(f"Measured elapsed:         {elapsed:.2f}s")
    print()
    print(f"Saved: {requests_csv}")
    print(f"Saved: {requests_json}")
    print(f"Saved: {events_csv_gz}")
    print(f"Saved: {metadata_json}")
    print(f"Saved: {order_json}")


if __name__ == "__main__":
    main()
