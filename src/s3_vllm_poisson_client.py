#!/usr/bin/env python3
"""
S3-B / S3-C — vLLM Poisson arrivals + continuous batching benchmark client.

Purpose
-------
Drive the validated vLLM server with concurrent HTTP/SSE requests arriving
according to a deterministic Poisson schedule.

Protocol
--------
- Natural generation is the primary experiment.
- Reuse the exact S2 master schedule (request IDs, order, and unit-rate
  exponential arrival draws) for direct arrival-process comparability.
- Sampling: temperature=0.7, top_p=0.8, top_k=20,
  repetition_penalty=1.1.
- Per-request seed: 42026000 + integer suffix of request_id.
- EOS is natural; max_tokens comes from frozen row.max_new_tokens.
- Six representative warmups complete before t=0 and are excluded from
  the measured schedule.
- No resume: if a load point is interrupted, delete/move that output
  directory and rerun the entire point.
- SSE content-event count is NOT assumed to equal generated token count.
  Generated token quantity comes from usage.completion_tokens.
- Optional /metrics polling records vLLM scheduler/cache gauges when exposed.

Default S3-B pilot
------------------
Uses the exact S2 pilot schedule at 0.25 req/s:

    python src/s3_vllm_poisson_client.py --phase pilot --arrival-rate 0.25

Later S3-C sweeps can reuse the same script with the S2 coarse 120-request
master schedule and different arrival rates.

Outputs
-------
results/s3/raw/poisson/<phase>/lambda_<rate>/
├── requests.csv
├── requests.json
├── sse_events.csv.gz
├── scheduler_metrics.csv.gz
├── run_metadata.json
└── schedule_used.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKLOAD_FILE = (
    PROJECT_ROOT / "workloads" / "final" / "realistic_requests.json"
)
CHECKSUM_FILE = (
    PROJECT_ROOT / "workloads" / "final" / "SHA256SUMS.txt"
)

S2_SCHEDULE_ROOT = PROJECT_ROOT / "results" / "s2" / "schedules"
S3_SEQ_PILOT = (
    PROJECT_ROOT
    / "results"
    / "s3"
    / "raw"
    / "sequential"
    / "pilot60"
    / "requests.csv"
)
RAW_ROOT = PROJECT_ROOT / "results" / "s3" / "raw" / "poisson"

FROZEN_CORPUS_SHA256 = (
    "7593429c095064a7f375e12d40db43ebf174d0f3a57f49bc51db030a0619d014"
)

SAMPLING_SEED_BASE = 42026000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1

EXPECTED_S2_SCHEDULE_SEEDS = {
    "workload_sample_seed": 2027,
    "arrival_draw_seed": 2028,
    "order_shuffle_seed": 2029,
}

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

S2_PHASE_COUNTS = {
    "pilot": {
        "short_interactive": 18,
        "knowledge_qa": 12,
        "coding_request": 9,
        "document_qa": 9,
        "long_context_qa": 6,
        "long_output": 6,
    },
    "coarse": {
        "short_interactive": 36,
        "knowledge_qa": 24,
        "coding_request": 18,
        "document_qa": 18,
        "long_context_qa": 12,
        "long_output": 12,
    },
}

REQUEST_FIELDS = [
    "schedule_index",
    "request_id",
    "workload_category",
    "status",
    "http_status",
    "error_type",
    "error_message",
    "scheduled_arrival_ms",
    "dispatch_start_ms",
    "dispatch_lag_ms",
    "inflight_at_dispatch",
    "frozen_input_tokens",
    "server_prompt_tokens",
    "prompt_token_match",
    "max_tokens",
    "sampling_seed",
    "finish_reason",
    "output_tokens",
    "output_chars",
    "output_sha256",
    "client_first_token_event_ttft_ms",
    "client_first_visible_text_ttft_ms",
    "scheduled_to_first_token_ms",
    "scheduled_to_first_visible_ms",
    "client_e2e_ms",
    "scheduled_to_completion_ms",
    "content_event_count",
    "visible_content_event_count",
    "content_event_minus_output_tokens",
    "mean_content_event_itl_ms",
    "p50_content_event_itl_ms",
    "p95_content_event_itl_ms",
    "sequential_category_p50_ttft_ms",
    "ttft_inflation_vs_seq_category_p50_ms",
    "request_payload_bytes",
]

EVENT_FIELDS = [
    "schedule_index",
    "request_id",
    "workload_category",
    "content_event_index",
    "arrival_from_request_start_ms",
    "arrival_from_experiment_start_ms",
    "content_chars",
    "content_nonempty",
    "finish_reason",
]

METRIC_FIELDS = [
    "sample_index",
    "time_from_experiment_start_ms",
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "gpu_cache_usage_perc",
    "prefix_cache_hit_rate",
    "http_status",
    "error",
]


def pct(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(xs) if xs else None


def rate_slug(rate: float) -> str:
    text = f"{rate:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_sampling_seed(request_id: str) -> int:
    match = re.search(r"(\d+)$", str(request_id))
    if not match:
        raise RuntimeError(
            f"Could not derive sampling seed from request_id={request_id!r}"
        )
    return SAMPLING_SEED_BASE + int(match.group(1))


def load_workload() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if not WORKLOAD_FILE.exists():
        raise FileNotFoundError(f"Frozen workload not found: {WORKLOAD_FILE}")

    actual = sha256_file(WORKLOAD_FILE)
    if actual != FROZEN_CORPUS_SHA256:
        raise RuntimeError(
            "Frozen workload SHA-256 mismatch.\n"
            f"Expected: {FROZEN_CORPUS_SHA256}\n"
            f"Actual:   {actual}"
        )

    payload = json.loads(WORKLOAD_FILE.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        requests = payload.get("requests")
        metadata = payload.get("metadata", {})
    elif isinstance(payload, list):
        requests = payload
        metadata = {}
    else:
        raise RuntimeError("Unsupported workload JSON structure.")

    if not isinstance(requests, list) or len(requests) != 1000:
        raise RuntimeError("Frozen workload must contain exactly 1000 requests.")

    by_id: Dict[str, Dict[str, Any]] = {}
    counts = Counter()

    for row in requests:
        request_id = str(row["request_id"])
        if request_id in by_id:
            raise RuntimeError(f"Duplicate request_id: {request_id}")
        category = str(row["workload_category"])
        counts[category] += 1
        by_id[request_id] = row

    if dict(counts) != EXPECTED_COUNTS:
        # Counter order is not relevant; compare key-by-key.
        if any(counts[k] != v for k, v in EXPECTED_COUNTS.items()):
            raise RuntimeError(
                f"Frozen category counts mismatch: {dict(counts)}"
            )

    return metadata, by_id


def s2_schedule_path(phase: str) -> Path:
    return S2_SCHEDULE_ROOT / f"{phase}_master_schedule.json"


def load_s2_master_schedule(
    phase: str,
    workload: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    path = s2_schedule_path(phase)
    if not path.exists():
        raise FileNotFoundError(
            "Required S2 master schedule is missing:\n"
            f"  {path}\n"
            "S3 intentionally reuses the exact S2 schedule."
        )

    master = json.loads(path.read_text(encoding="utf-8"))

    if master.get("phase") != phase:
        raise RuntimeError(f"S2 schedule phase mismatch: {master.get('phase')}")

    expected_n = sum(S2_PHASE_COUNTS[phase].values())
    request_ids = [str(x) for x in master.get("request_ids", [])]
    unit_interarrivals = master.get("unit_interarrivals", [])

    if len(request_ids) != expected_n:
        raise RuntimeError(
            f"S2 {phase} schedule request count mismatch: "
            f"{len(request_ids)} != {expected_n}"
        )

    if len(unit_interarrivals) != expected_n:
        raise RuntimeError(
            f"S2 {phase} interarrival count mismatch: "
            f"{len(unit_interarrivals)} != {expected_n}"
        )

    if len(set(request_ids)) != len(request_ids):
        raise RuntimeError("S2 schedule contains duplicate request IDs.")

    missing = [rid for rid in request_ids if rid not in workload]
    if missing:
        raise RuntimeError(f"S2 schedule references missing workload IDs: {missing[:5]}")

    for key, expected in EXPECTED_S2_SCHEDULE_SEEDS.items():
        if int(master.get(key, -1)) != expected:
            raise RuntimeError(
                f"S2 schedule seed mismatch for {key}: "
                f"{master.get(key)} != {expected}"
            )

    counts = Counter(workload[rid]["workload_category"] for rid in request_ids)
    expected_counts = S2_PHASE_COUNTS[phase]
    if any(counts[k] != v for k, v in expected_counts.items()):
        raise RuntimeError(
            f"S2 schedule category mix mismatch: {dict(counts)}"
        )

    return master


def scaled_schedule(
    master: Dict[str, Any],
    arrival_rate: float,
) -> List[Dict[str, Any]]:
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be > 0")

    elapsed = 0.0
    rows: List[Dict[str, Any]] = []

    for i, (request_id, unit_gap) in enumerate(
        zip(master["request_ids"], master["unit_interarrivals"]),
        start=1,
    ):
        elapsed += float(unit_gap) / arrival_rate
        rows.append(
            {
                "schedule_index": i,
                "request_id": str(request_id),
                "unit_interarrival": float(unit_gap),
                "scheduled_offset_s": elapsed,
            }
        )

    return rows


def representative_warmups(
    workload: Dict[str, Dict[str, Any]],
    measured_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    measured = set(measured_ids)
    chosen = []

    for category in CATEGORY_ORDER:
        group = sorted(
            [
                row
                for rid, row in workload.items()
                if row["workload_category"] == category and rid not in measured
            ],
            key=lambda row: int(row["input_tokens"]),
        )
        if not group:
            raise RuntimeError(
                f"No non-measured warmup candidate for category {category}"
            )
        chosen.append(group[len(group) // 2])

    return chosen


def load_sequential_category_baseline() -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not S3_SEQ_PILOT.exists():
        raise FileNotFoundError(
            "S3-A sequential pilot60 result is required:\n"
            f"  {S3_SEQ_PILOT}"
        )

    by_cat: Dict[str, List[float]] = defaultdict(list)
    e2e_values: List[float] = []

    with S3_SEQ_PILOT.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            cat = row["workload_category"]
            ttft = float(row["client_first_token_event_ttft_ms"])
            e2e = float(row["client_e2e_ms"])
            by_cat[cat].append(ttft)
            e2e_values.append(e2e)

    if any(not by_cat.get(cat) for cat in CATEGORY_ORDER):
        raise RuntimeError("S3-A baseline is missing one or more categories.")

    category_p50 = {
        cat: float(pct(values, 0.50))
        for cat, values in by_cat.items()
    }

    mean_service_s = statistics.mean(e2e_values) / 1000.0
    seq_mu = 1.0 / mean_service_s

    return category_p50, {
        "mean_sequential_service_s": mean_service_s,
        "sequential_implied_mu_req_s": seq_mu,
        "n": len(e2e_values),
    }


def make_payload(row: Dict[str, Any], max_tokens_override: Optional[int] = None) -> Dict[str, Any]:
    request_id = str(row["request_id"])
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(row["max_new_tokens"])
    )

    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": str(row["prompt"]),
            }
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": max_tokens,
        "seed": request_sampling_seed(request_id),
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def wait_until(target: float) -> None:
    loop = asyncio.get_running_loop()
    while True:
        remaining = target - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


async def run_warmup(
    client: httpx.AsyncClient,
    url: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    payload = make_payload(
        row,
        max_tokens_override=min(int(row["max_new_tokens"]), 32),
    )

    start = time.perf_counter()
    usage: Dict[str, Any] = {}
    finish_reason = None

    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break

            event = json.loads(body)
            choices = event.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                finish_reason = choices[0]["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not usage:
        raise RuntimeError(f"Warmup {row['request_id']} missing usage block.")

    if int(usage["prompt_tokens"]) != int(row["input_tokens"]):
        raise RuntimeError(
            f"Warmup prompt-token mismatch for {row['request_id']}: "
            f"frozen={row['input_tokens']} server={usage['prompt_tokens']}"
        )

    return {
        "request_id": row["request_id"],
        "category": row["workload_category"],
        "input_tokens": int(row["input_tokens"]),
        "output_tokens": int(usage["completion_tokens"]),
        "finish_reason": finish_reason,
        "client_e2e_ms": elapsed_ms,
    }


class ClientState:
    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self.dispatched = 0
        self.completed = 0
        self.lock = asyncio.Lock()


async def run_one_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    experiment_start_loop: float,
    experiment_start_perf: float,
    scheduled: Dict[str, Any],
    row: Dict[str, Any],
    state: ClientState,
    seq_category_p50_ttft: Dict[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    loop = asyncio.get_running_loop()

    schedule_index = int(scheduled["schedule_index"])
    request_id = str(scheduled["request_id"])
    category = str(row["workload_category"])
    scheduled_offset_s = float(scheduled["scheduled_offset_s"])
    scheduled_loop_time = experiment_start_loop + scheduled_offset_s

    await wait_until(scheduled_loop_time)

    actual_dispatch_loop = loop.time()
    dispatch_lag_ms = (actual_dispatch_loop - scheduled_loop_time) * 1000.0

    async with state.lock:
        inflight_at_dispatch = state.inflight
        state.inflight += 1
        state.dispatched += 1
        state.max_inflight = max(state.max_inflight, state.inflight)

    payload = make_payload(row)
    payload_bytes = len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    request_start_perf = time.perf_counter()

    result: Dict[str, Any] = {
        "schedule_index": schedule_index,
        "request_id": request_id,
        "workload_category": category,
        "status": "error",
        "http_status": None,
        "error_type": "",
        "error_message": "",
        "scheduled_arrival_ms": scheduled_offset_s * 1000.0,
        "dispatch_start_ms": (
            actual_dispatch_loop - experiment_start_loop
        ) * 1000.0,
        "dispatch_lag_ms": dispatch_lag_ms,
        "inflight_at_dispatch": inflight_at_dispatch,
        "frozen_input_tokens": int(row["input_tokens"]),
        "server_prompt_tokens": None,
        "prompt_token_match": None,
        "max_tokens": int(row["max_new_tokens"]),
        "sampling_seed": request_sampling_seed(request_id),
        "finish_reason": None,
        "output_tokens": None,
        "output_chars": None,
        "output_sha256": None,
        "client_first_token_event_ttft_ms": None,
        "client_first_visible_text_ttft_ms": None,
        "scheduled_to_first_token_ms": None,
        "scheduled_to_first_visible_ms": None,
        "client_e2e_ms": None,
        "scheduled_to_completion_ms": None,
        "content_event_count": 0,
        "visible_content_event_count": 0,
        "content_event_minus_output_tokens": None,
        "mean_content_event_itl_ms": None,
        "p50_content_event_itl_ms": None,
        "p95_content_event_itl_ms": None,
        "sequential_category_p50_ttft_ms": seq_category_p50_ttft[category],
        "ttft_inflation_vs_seq_category_p50_ms": None,
        "request_payload_bytes": payload_bytes,
    }

    event_rows: List[Dict[str, Any]] = []
    content_times: List[float] = []
    visible_times: List[float] = []
    output_parts: List[str] = []

    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None

    try:
        async with client.stream("POST", url, json=payload) as response:
            result["http_status"] = response.status_code
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                body = line[5:].strip()
                if body == "[DONE]":
                    break

                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue

                now_perf = time.perf_counter()
                choices = event.get("choices") or []

                if choices:
                    choice = choices[0] or {}
                    delta = choice.get("delta") or {}

                    if "content" in delta:
                        content = delta.get("content")
                        if content is None:
                            content = ""
                        if not isinstance(content, str):
                            content = str(content)

                        content_times.append(now_perf)
                        output_parts.append(content)

                        if content:
                            visible_times.append(now_perf)

                        event_rows.append(
                            {
                                "schedule_index": schedule_index,
                                "request_id": request_id,
                                "workload_category": category,
                                "content_event_index": len(content_times),
                                "arrival_from_request_start_ms": (
                                    now_perf - request_start_perf
                                ) * 1000.0,
                                "arrival_from_experiment_start_ms": (
                                    now_perf - experiment_start_perf
                                ) * 1000.0,
                                "content_chars": len(content),
                                "content_nonempty": int(bool(content)),
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )

                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])

                if event.get("usage"):
                    usage = event["usage"]

        end_perf = time.perf_counter()

        if not usage:
            raise RuntimeError("Streaming response completed without usage block.")

        prompt_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])
        output_text = "".join(output_parts)

        first_content = content_times[0] if content_times else None
        first_visible = visible_times[0] if visible_times else None

        content_itls = [
            (b - a) * 1000.0
            for a, b in zip(content_times, content_times[1:])
        ]

        client_ttft = (
            (first_content - request_start_perf) * 1000.0
            if first_content is not None
            else None
        )
        visible_ttft = (
            (first_visible - request_start_perf) * 1000.0
            if first_visible is not None
            else None
        )
        client_e2e = (end_perf - request_start_perf) * 1000.0

        result.update(
            {
                "status": "ok",
                "finish_reason": finish_reason,
                "server_prompt_tokens": prompt_tokens,
                "prompt_token_match": int(
                    prompt_tokens == int(row["input_tokens"])
                ),
                "output_tokens": output_tokens,
                "output_chars": len(output_text),
                "output_sha256": sha256_text(output_text),
                "client_first_token_event_ttft_ms": client_ttft,
                "client_first_visible_text_ttft_ms": visible_ttft,
                "scheduled_to_first_token_ms": (
                    dispatch_lag_ms + client_ttft
                    if client_ttft is not None
                    else None
                ),
                "scheduled_to_first_visible_ms": (
                    dispatch_lag_ms + visible_ttft
                    if visible_ttft is not None
                    else None
                ),
                "client_e2e_ms": client_e2e,
                "scheduled_to_completion_ms": dispatch_lag_ms + client_e2e,
                "content_event_count": len(content_times),
                "visible_content_event_count": len(visible_times),
                "content_event_minus_output_tokens": (
                    len(content_times) - output_tokens
                ),
                "mean_content_event_itl_ms": mean_or_none(content_itls),
                "p50_content_event_itl_ms": pct(content_itls, 0.50),
                "p95_content_event_itl_ms": pct(content_itls, 0.95),
                "ttft_inflation_vs_seq_category_p50_ms": (
                    client_ttft - seq_category_p50_ttft[category]
                    if client_ttft is not None
                    else None
                ),
            }
        )

        if not result["prompt_token_match"]:
            raise RuntimeError(
                f"Prompt-token mismatch: frozen={row['input_tokens']} "
                f"server={prompt_tokens}"
            )

    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:1000]

    finally:
        async with state.lock:
            state.inflight -= 1
            state.completed += 1

    return result, event_rows


def parse_prometheus_value(
    text: str,
    candidate_suffixes: Sequence[str],
) -> Optional[float]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        metric_part, sep, value_part = line.rpartition(" ")
        if not sep:
            continue

        metric_name = metric_part.split("{", 1)[0]

        if any(
            metric_name == suffix
            or metric_name.endswith(suffix)
            for suffix in candidate_suffixes
        ):
            try:
                return float(value_part.strip())
            except ValueError:
                continue

    return None


async def poll_metrics(
    *,
    client: httpx.AsyncClient,
    metrics_url: str,
    experiment_start_perf: float,
    stop_event: asyncio.Event,
    interval_s: float,
    out_rows: List[Dict[str, Any]],
) -> None:
    sample_index = 0

    while not stop_event.is_set():
        sample_index += 1
        row = {
            "sample_index": sample_index,
            "time_from_experiment_start_ms": (
                time.perf_counter() - experiment_start_perf
            ) * 1000.0,
            "num_requests_running": None,
            "num_requests_waiting": None,
            "kv_cache_usage_perc": None,
            "gpu_cache_usage_perc": None,
            "prefix_cache_hit_rate": None,
            "http_status": None,
            "error": "",
        }

        try:
            response = await client.get(metrics_url)
            row["http_status"] = response.status_code
            response.raise_for_status()
            text = response.text

            row["num_requests_running"] = parse_prometheus_value(
                text,
                [
                    "vllm:num_requests_running",
                    "num_requests_running",
                ],
            )
            row["num_requests_waiting"] = parse_prometheus_value(
                text,
                [
                    "vllm:num_requests_waiting",
                    "num_requests_waiting",
                ],
            )
            row["kv_cache_usage_perc"] = parse_prometheus_value(
                text,
                [
                    "vllm:kv_cache_usage_perc",
                    "kv_cache_usage_perc",
                ],
            )
            row["gpu_cache_usage_perc"] = parse_prometheus_value(
                text,
                [
                    "vllm:gpu_cache_usage_perc",
                    "gpu_cache_usage_perc",
                ],
            )
            row["prefix_cache_hit_rate"] = parse_prometheus_value(
                text,
                [
                    "vllm:prefix_cache_hit_rate",
                    "prefix_cache_hit_rate",
                ],
            )

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]

        out_rows.append(row)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def output_paths(phase: str, arrival_rate: float) -> Dict[str, Path]:
    directory = (
        RAW_ROOT / phase / f"lambda_{rate_slug(arrival_rate)}"
    )
    return {
        "dir": directory,
        "requests_csv": directory / "requests.csv",
        "requests_json": directory / "requests.json",
        "events": directory / "sse_events.csv.gz",
        "metrics": directory / "scheduler_metrics.csv.gz",
        "metadata": directory / "run_metadata.json",
        "schedule": directory / "schedule_used.json",
    }


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def finite_max(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            vals.append(fv)
    return max(vals) if vals else None


async def execute(
    *,
    phase: str,
    arrival_rate: float,
    base_url: str,
    metrics_interval_s: float,
    dry_run: bool,
) -> None:
    workload_metadata, workload = load_workload()
    master = load_s2_master_schedule(phase, workload)
    schedule = scaled_schedule(master, arrival_rate)
    measured_ids = [row["request_id"] for row in schedule]

    seq_cat_p50, seq_ref = load_sequential_category_baseline()

    paths = output_paths(phase, arrival_rate)

    scheduled_duration_s = float(schedule[-1]["scheduled_offset_s"])
    realized_arrival_rate = (
        (len(schedule) - 1) / scheduled_duration_s
        if len(schedule) > 1 and scheduled_duration_s > 0
        else arrival_rate
    )

    seq_mean_service = float(seq_ref["mean_sequential_service_s"])
    seq_mu = float(seq_ref["sequential_implied_mu_req_s"])
    configured_load_vs_seq_mu = arrival_rate / seq_mu
    realized_load_vs_seq_mu = realized_arrival_rate / seq_mu

    print("=" * 92)
    print("S3 — vLLM POISSON ARRIVALS + CONTINUOUS BATCHING")
    print("=" * 92)
    print(f"Phase:                         {phase}")
    print(f"Configured lambda:             {arrival_rate:.3f} req/s")
    print(f"Realized schedule rate:        {realized_arrival_rate:.3f} req/s")
    print(f"Requests:                      {len(schedule)}")
    print(f"Scheduled arrival window:      {scheduled_duration_s/60.0:.2f} min")
    print(f"S3-A mean sequential E2E:      {seq_mean_service:.4f} s")
    print(f"S3-A implied sequential mu:    {seq_mu:.4f} req/s")
    print(f"Configured lambda / seq mu:    {configured_load_vs_seq_mu:.3f}")
    print(f"Realized lambda / seq mu:      {realized_load_vs_seq_mu:.3f}")
    print(f"Reused S2 schedule:            {s2_schedule_path(phase)}")
    print(f"Output directory:              {paths['dir']}")
    print(
        "Important: lambda/mu above is only a sequential reference; "
        "it is NOT vLLM continuous-batching utilization."
    )

    if dry_run:
        print("\nDRY RUN: no requests sent.")
        return

    if paths["dir"].exists():
        raise RuntimeError(
            "This S3 load-point directory already exists:\n"
            f"  {paths['dir']}\n\n"
            "S3 Poisson load points are NOT resumable because arrival/queue "
            "history matters. Move/delete the directory intentionally before rerunning."
        )

    paths["dir"].mkdir(parents=True, exist_ok=False)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    health_url = base_url.rstrip("/") + "/health"
    metrics_url = base_url.rstrip("/") + "/metrics"

    timeout = httpx.Timeout(
        connect=10.0,
        read=None,
        write=30.0,
        pool=None,
    )

    limits = httpx.Limits(
        max_connections=256,
        max_keepalive_connections=256,
    )

    warmup_records: List[Dict[str, Any]] = []
    request_rows: List[Dict[str, Any]] = []
    all_event_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=False,
    ) as client:
        health = await client.get(health_url)
        health.raise_for_status()

        warmups = representative_warmups(workload, measured_ids)

        print("\nWarmup: one non-measured representative request/category...")
        for i, row in enumerate(warmups, start=1):
            rec = await run_warmup(client, url, row)
            warmup_records.append(rec)
            print(
                f"  [{i}/6] {rec['category']:20s} "
                f"id={rec['request_id']} "
                f"in={rec['input_tokens']:4d} "
                f"out={rec['output_tokens']:3d} "
                f"E2E={rec['client_e2e_ms']:8.2f}ms"
            )

        print("\nStarting Poisson clock...")

        state = ClientState()
        stop_metrics = asyncio.Event()

        experiment_start_loop = asyncio.get_running_loop().time()
        experiment_start_perf = time.perf_counter()

        metrics_task = None
        if metrics_interval_s > 0:
            metrics_task = asyncio.create_task(
                poll_metrics(
                    client=client,
                    metrics_url=metrics_url,
                    experiment_start_perf=experiment_start_perf,
                    stop_event=stop_metrics,
                    interval_s=metrics_interval_s,
                    out_rows=metric_rows,
                )
            )

        tasks = []
        for scheduled in schedule:
            rid = str(scheduled["request_id"])
            tasks.append(
                asyncio.create_task(
                    run_one_request(
                        client=client,
                        url=url,
                        experiment_start_loop=experiment_start_loop,
                        experiment_start_perf=experiment_start_perf,
                        scheduled=scheduled,
                        row=workload[rid],
                        state=state,
                        seq_category_p50_ttft=seq_cat_p50,
                    )
                )
            )

        results = await asyncio.gather(*tasks)

        experiment_end_perf = time.perf_counter()

        stop_metrics.set()
        if metrics_task is not None:
            await metrics_task

    for result, events in results:
        request_rows.append(result)
        all_event_rows.extend(events)

    request_rows.sort(key=lambda r: int(r["schedule_index"]))
    all_event_rows.sort(
        key=lambda r: (
            int(r["schedule_index"]),
            int(r["content_event_index"]),
        )
    )

    elapsed_s = experiment_end_perf - experiment_start_perf
    drain_s = max(0.0, elapsed_s - scheduled_duration_s)

    successful = [r for r in request_rows if r["status"] == "ok"]
    failed = [r for r in request_rows if r["status"] != "ok"]

    if failed:
        # Persist diagnostics before raising.
        status = "failed"
    else:
        status = "complete"

    prompt_matches = sum(
        int(r.get("prompt_token_match") == 1)
        for r in successful
    )

    finish_counts = Counter(
        str(r.get("finish_reason"))
        for r in successful
    )

    total_output_tokens = sum(
        int(r["output_tokens"])
        for r in successful
        if r["output_tokens"] is not None
    )
    total_input_tokens = sum(
        int(r["server_prompt_tokens"])
        for r in successful
        if r["server_prompt_tokens"] is not None
    )

    ttfts = [
        float(r["client_first_token_event_ttft_ms"])
        for r in successful
        if r["client_first_token_event_ttft_ms"] is not None
    ]
    visible_ttfts = [
        float(r["client_first_visible_text_ttft_ms"])
        for r in successful
        if r["client_first_visible_text_ttft_ms"] is not None
    ]
    e2es = [
        float(r["client_e2e_ms"])
        for r in successful
        if r["client_e2e_ms"] is not None
    ]
    dispatch_lags = [
        float(r["dispatch_lag_ms"])
        for r in successful
        if r["dispatch_lag_ms"] is not None
    ]
    content_itls = [
        float(r["mean_content_event_itl_ms"])
        for r in successful
        if r["mean_content_event_itl_ms"] is not None
    ]
    inflation = [
        float(r["ttft_inflation_vs_seq_category_p50_ms"])
        for r in successful
        if r["ttft_inflation_vs_seq_category_p50_ms"] is not None
    ]

    metric_max_running = finite_max(metric_rows, "num_requests_running")
    metric_max_waiting = finite_max(metric_rows, "num_requests_waiting")
    metric_max_kv = finite_max(metric_rows, "kv_cache_usage_perc")
    metric_max_gpu_cache = finite_max(metric_rows, "gpu_cache_usage_perc")

    schedule_used = {
        "source": str(s2_schedule_path(phase)),
        "phase": phase,
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_arrival_rate,
        "scheduled_duration_s": scheduled_duration_s,
        "request_ids": [r["request_id"] for r in schedule],
        "unit_interarrivals": [
            r["unit_interarrival"] for r in schedule
        ],
        "scheduled_offsets_s": [
            r["scheduled_offset_s"] for r in schedule
        ],
        "source_schedule_seeds": EXPECTED_S2_SCHEDULE_SEEDS,
    }

    metadata = {
        "phase": "S3-B" if phase == "pilot" else "S3-C",
        "protocol_version": 1,
        "mode": f"natural_generation_poisson_{phase}",
        "status": status,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "workload_metadata": workload_metadata,
        "s2_schedule_reused_exactly": True,
        "schedule_source": str(s2_schedule_path(phase)),
        "schedule_seeds": EXPECTED_S2_SCHEDULE_SEEDS,
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_arrival_rate,
        "scheduled_arrival_window_s": scheduled_duration_s,
        "measured_makespan_s": elapsed_s,
        "drain_after_last_scheduled_arrival_s": drain_s,
        "request_count": len(schedule),
        "successful_request_count": len(successful),
        "failed_request_count": len(failed),
        "max_client_inflight": state.max_inflight,
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
        "warmup": {
            "count": len(warmup_records),
            "policy": (
                "one median-input request/category selected outside measured "
                "schedule; max_tokens capped at 32; completed before t=0"
            ),
            "records": warmup_records,
        },
        "client": {
            "http2": False,
            "max_connections": 256,
            "max_keepalive_connections": 256,
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
        "sequential_reference": {
            **seq_ref,
            "category_p50_ttft_ms": seq_cat_p50,
            "configured_lambda_over_seq_mu": configured_load_vs_seq_mu,
            "realized_lambda_over_seq_mu": realized_load_vs_seq_mu,
            "interpretation": (
                "Sequential mu is a reference only; continuous batching means "
                "this ratio is not vLLM utilization rho."
            ),
        },
        "streaming_semantics": {
            "output_tokens_source": "usage.completion_tokens",
            "content_event_count_is_not_token_count": True,
            "first_token_event_metric": (
                "first SSE choice event containing delta.content; transport/"
                "application event, not assumed one-to-one with tokens"
            ),
            "first_visible_text_metric": (
                "first SSE delta.content with non-empty visible text"
            ),
        },
        "metrics_polling": {
            "enabled": metrics_interval_s > 0,
            "interval_s": metrics_interval_s,
            "sample_count": len(metric_rows),
            "max_num_requests_running": metric_max_running,
            "max_num_requests_waiting": metric_max_waiting,
            "max_kv_cache_usage_perc": metric_max_kv,
            "max_gpu_cache_usage_perc": metric_max_gpu_cache,
            "note": (
                "Gauge names are parsed opportunistically from /metrics; "
                "None means the installed vLLM did not expose the expected name."
            ),
        },
        "audit": {
            "prompt_token_matches": prompt_matches,
            "prompt_token_total": len(successful),
            "finish_reason_counts": dict(finish_counts),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "content_event_minus_output_tokens_values": sorted(
                set(
                    int(r["content_event_minus_output_tokens"])
                    for r in successful
                    if r["content_event_minus_output_tokens"] is not None
                )
            ),
        },
        "summary": {
            "p50_client_ttft_ms": pct(ttfts, 0.50),
            "p95_client_ttft_ms": pct(ttfts, 0.95),
            "p99_client_ttft_ms": pct(ttfts, 0.99),
            "p50_visible_ttft_ms": pct(visible_ttfts, 0.50),
            "p95_visible_ttft_ms": pct(visible_ttfts, 0.95),
            "p50_e2e_ms": pct(e2es, 0.50),
            "p95_e2e_ms": pct(e2es, 0.95),
            "p99_e2e_ms": pct(e2es, 0.99),
            "p50_dispatch_lag_ms": pct(dispatch_lags, 0.50),
            "p95_dispatch_lag_ms": pct(dispatch_lags, 0.95),
            "p50_request_mean_content_event_itl_ms": pct(content_itls, 0.50),
            "p95_request_mean_content_event_itl_ms": pct(content_itls, 0.95),
            "p50_ttft_inflation_vs_seq_category_ms": pct(inflation, 0.50),
            "p95_ttft_inflation_vs_seq_category_ms": pct(inflation, 0.95),
            "achieved_request_throughput_req_s": (
                len(successful) / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_output_tokens_per_s": (
                total_output_tokens / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_input_tokens_per_s": (
                total_input_tokens / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_total_tokens_per_s": (
                (total_input_tokens + total_output_tokens) / elapsed_s
                if elapsed_s > 0
                else None
            ),
        },
    }

    write_csv(paths["requests_csv"], REQUEST_FIELDS, request_rows)
    with paths["requests_json"].open("w", encoding="utf-8") as f:
        json.dump(request_rows, f, ensure_ascii=False, indent=2)
    write_gzip_csv(paths["events"], EVENT_FIELDS, all_event_rows)
    write_gzip_csv(paths["metrics"], METRIC_FIELDS, metric_rows)

    with paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with paths["schedule"].open("w", encoding="utf-8") as f:
        json.dump(schedule_used, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("S3 POISSON RUN COMPLETE")
    print("=" * 92)
    print(f"Status:                       {status}")
    print(f"Success:                      {len(successful)}/{len(schedule)}")
    print(f"Prompt-token matches:         {prompt_matches}/{len(successful)}")
    print(f"Configured lambda:            {arrival_rate:.3f} req/s")
    print(f"Realized schedule lambda:     {realized_arrival_rate:.3f} req/s")
    print(f"Measured makespan:            {elapsed_s:.2f}s")
    print(f"Drain after last arrival:     {drain_s:.2f}s")
    print(f"Max client inflight:          {state.max_inflight}")
    print(f"Finish reasons:               {dict(finish_counts)}")
    print(f"Total output tokens:          {total_output_tokens}")
    print(f"P50 client TTFT:              {pct(ttfts, 0.50):.2f}ms")
    print(f"P95 client TTFT:              {pct(ttfts, 0.95):.2f}ms")
    print(f"P50 E2E:                      {pct(e2es, 0.50):.2f}ms")
    print(f"P95 E2E:                      {pct(e2es, 0.95):.2f}ms")
    print(
        f"Aggregate output throughput:  "
        f"{total_output_tokens / elapsed_s:.2f} tok/s"
    )
    print(
        f"Achieved request throughput:  "
        f"{len(successful) / elapsed_s:.3f} req/s"
    )
    print(
        f"P50 TTFT inflation vs S3-A:   "
        f"{pct(inflation, 0.50):.2f}ms"
    )
    print(
        f"P95 TTFT inflation vs S3-A:   "
        f"{pct(inflation, 0.95):.2f}ms"
    )
    print(f"Metrics samples:              {len(metric_rows)}")
    print(f"Max vLLM running gauge:       {metric_max_running}")
    print(f"Max vLLM waiting gauge:       {metric_max_waiting}")
    print()
    print(f"requests: {paths['requests_csv']}")
    print(f"events:   {paths['events']}")
    print(f"metrics:  {paths['metrics']}")
    print(f"metadata: {paths['metadata']}")
    print(f"schedule: {paths['schedule']}")

    if failed:
        first = failed[0]
        raise RuntimeError(
            f"S3 load point had {len(failed)} failed requests. "
            f"First failure: {first['request_id']} "
            f"{first['error_type']}: {first['error_message']}"
        )

    if prompt_matches != len(successful):
        raise RuntimeError("Prompt-token audit failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["pilot", "coarse"],
        default="pilot",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=0.20,
        help="Seconds between /metrics polls; use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.metrics_interval < 0:
        raise ValueError("--metrics-interval must be >= 0")

    asyncio.run(
        execute(
            phase=args.phase,
            arrival_rate=args.arrival_rate,
            base_url=args.base_url,
            metrics_interval_s=args.metrics_interval,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
