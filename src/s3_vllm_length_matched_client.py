#!/usr/bin/env python3
"""
S3-D — Length-Matched Replay for vLLM continuous-batching validation.

Goal
----
Control generated work quantity so scheduler/concurrency comparisons are not
confounded by natural EOS/output-length differences.

Protocol
--------
- Reuse the exact frozen R0 workload and exact S2 coarse Poisson master schedule.
- Default arrival rate: 4.0 req/s (the observed latency-knee region).
- Per-request target output length comes from the validated S1 Full Flash run:
      results/s1/raw/full/flash_api_requests.csv
- For every measured request:
      max_tokens = S1 Flash output_tokens
      ignore_eos = True
  Therefore the request is expected to generate exactly the target number of
  completion tokens and terminate by length.
- Sampling policy and per-request seed remain unchanged.
- Six non-measured warmups complete before t=0.
- Fresh vLLM process is required for every compared server configuration.
- Prefix caching remains enabled, but fresh-process startup equalizes initial
  cache state.
- SSE content-event count is NOT treated as token count.
  usage.completion_tokens is authoritative.
- No resume: if interrupted, rerun the whole load point.

Recommended comparison
----------------------
1) default server:
   python src/s3_vllm_length_matched_client.py \
       --arrival-rate 4.0 --label default

2) fresh server with --max-num-seqs 12:
   python src/s3_vllm_length_matched_client.py \
       --arrival-rate 4.0 --label maxseq12 --server-max-num-seqs 12

Outputs
-------
results/s3/raw/length_matched/coarse/lambda_<rate>/<label>/
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
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKLOAD_FILE = PROJECT_ROOT / "workloads" / "final" / "realistic_requests.json"
S2_COARSE_SCHEDULE = PROJECT_ROOT / "results" / "s2" / "schedules" / "coarse_master_schedule.json"
S1_FULL_REQUESTS = PROJECT_ROOT / "results" / "s1" / "raw" / "full" / "flash_api_requests.csv"
RAW_ROOT = PROJECT_ROOT / "results" / "s3" / "raw" / "length_matched" / "coarse"

FROZEN_CORPUS_SHA256 = (
    "7593429c095064a7f375e12d40db43ebf174d0f3a57f49bc51db030a0619d014"
)

SAMPLING_SEED_BASE = 42026000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1

EXPECTED_SCHEDULE_SEEDS = {
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

COARSE_COUNTS = {
    "short_interactive": 36,
    "knowledge_qa": 24,
    "coding_request": 18,
    "document_qa": 18,
    "long_context_qa": 12,
    "long_output": 12,
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
    "frozen_natural_max_new_tokens",
    "matched_target_output_tokens",
    "sampling_seed",
    "ignore_eos",
    "finish_reason",
    "output_tokens",
    "length_match",
    "output_chars",
    "output_sha256",
    "client_first_content_event_ttft_ms",
    "client_first_visible_text_ttft_ms",
    "scheduled_to_first_content_ms",
    "client_e2e_ms",
    "scheduled_to_completion_ms",
    "content_event_count",
    "visible_content_event_count",
    "content_event_minus_output_tokens",
    "mean_content_event_itl_ms",
    "p50_content_event_itl_ms",
    "p95_content_event_itl_ms",
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
    xs = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )
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
    xs = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return statistics.mean(xs) if xs else None


def rate_slug(rate: float) -> str:
    text = f"{rate:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def safe_label(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(
            "--label may contain only letters, digits, underscore, hyphen, dot."
        )
    return text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_sampling_seed(request_id: str) -> int:
    m = re.search(r"(\d+)$", str(request_id))
    if not m:
        raise RuntimeError(
            f"Could not derive sampling seed from request_id={request_id!r}"
        )
    return SAMPLING_SEED_BASE + int(m.group(1))


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
        rid = str(row["request_id"])
        if rid in by_id:
            raise RuntimeError(f"Duplicate request_id: {rid}")
        cat = str(row["workload_category"])
        counts[cat] += 1
        by_id[rid] = row

    for cat, expected in EXPECTED_COUNTS.items():
        if counts[cat] != expected:
            raise RuntimeError(
                f"Frozen category count mismatch for {cat}: "
                f"{counts[cat]} != {expected}"
            )

    return metadata, by_id


def load_s2_coarse_master(
    workload: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    if not S2_COARSE_SCHEDULE.exists():
        raise FileNotFoundError(
            f"Required S2 coarse master schedule missing: {S2_COARSE_SCHEDULE}"
        )

    master = json.loads(S2_COARSE_SCHEDULE.read_text(encoding="utf-8"))

    if master.get("phase") != "coarse":
        raise RuntimeError("S2 master schedule is not phase=coarse.")

    ids = [str(x) for x in master.get("request_ids", [])]
    gaps = master.get("unit_interarrivals", [])

    if len(ids) != 120 or len(gaps) != 120:
        raise RuntimeError("S2 coarse schedule must contain exactly 120 entries.")

    if len(set(ids)) != len(ids):
        raise RuntimeError("S2 coarse schedule contains duplicate request IDs.")

    for key, expected in EXPECTED_SCHEDULE_SEEDS.items():
        if int(master.get(key, -1)) != expected:
            raise RuntimeError(
                f"S2 schedule seed mismatch: {key}={master.get(key)} "
                f"expected={expected}"
            )

    missing = [rid for rid in ids if rid not in workload]
    if missing:
        raise RuntimeError(
            f"S2 schedule references IDs missing from workload: {missing[:5]}"
        )

    counts = Counter(workload[rid]["workload_category"] for rid in ids)
    for cat, expected in COARSE_COUNTS.items():
        if counts[cat] != expected:
            raise RuntimeError(
                f"Coarse category mix mismatch for {cat}: "
                f"{counts[cat]} != {expected}"
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

    for i, (rid, unit_gap) in enumerate(
        zip(master["request_ids"], master["unit_interarrivals"]),
        start=1,
    ):
        elapsed += float(unit_gap) / arrival_rate
        rows.append(
            {
                "schedule_index": i,
                "request_id": str(rid),
                "unit_interarrival": float(unit_gap),
                "scheduled_offset_s": elapsed,
            }
        )

    return rows


def load_s1_length_targets() -> Dict[str, int]:
    if not S1_FULL_REQUESTS.exists():
        raise FileNotFoundError(
            "S1 Full request results are required for matched replay:\n"
            f"  {S1_FULL_REQUESTS}"
        )

    targets: Dict[str, int] = {}

    with S1_FULL_REQUESTS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"request_id", "status", "output_tokens"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"S1 file missing required columns {sorted(required)}"
            )

        for row in reader:
            if row.get("status") != "ok":
                continue
            rid = str(row["request_id"])
            n = int(row["output_tokens"])
            if n <= 0:
                raise RuntimeError(
                    f"Invalid S1 output_tokens for {rid}: {n}"
                )
            if rid in targets:
                raise RuntimeError(f"Duplicate S1 request_id: {rid}")
            targets[rid] = n

    if len(targets) != 1000:
        raise RuntimeError(
            f"Expected 1000 successful S1 Full targets, found {len(targets)}."
        )

    return targets


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
                f"No non-measured warmup request for category {category}"
            )
        chosen.append(group[len(group) // 2])

    return chosen


def make_payload(
    row: Dict[str, Any],
    *,
    target_tokens: int,
    ignore_eos: bool,
) -> Dict[str, Any]:
    rid = str(row["request_id"])

    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": str(row["prompt"])}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": int(target_tokens),
        "ignore_eos": bool(ignore_eos),
        "seed": request_sampling_seed(rid),
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


async def warmup_request(
    client: httpx.AsyncClient,
    url: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    # Warmup is deliberately NOT part of matched replay.
    # It only establishes a warm runtime before t=0.
    target = min(int(row["max_new_tokens"]), 32)
    payload = make_payload(
        row,
        target_tokens=target,
        ignore_eos=False,
    )

    start = time.perf_counter()
    usage = None
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

    if not usage:
        raise RuntimeError(f"Warmup {row['request_id']} missing usage.")

    if int(usage["prompt_tokens"]) != int(row["input_tokens"]):
        raise RuntimeError(
            f"Warmup prompt-token mismatch for {row['request_id']}"
        )

    return {
        "request_id": row["request_id"],
        "category": row["workload_category"],
        "input_tokens": int(row["input_tokens"]),
        "output_tokens": int(usage["completion_tokens"]),
        "finish_reason": finish_reason,
        "client_e2e_ms": (time.perf_counter() - start) * 1000.0,
    }


class ClientState:
    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self.lock = asyncio.Lock()


async def run_one_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    experiment_start_loop: float,
    experiment_start_perf: float,
    scheduled: Dict[str, Any],
    row: Dict[str, Any],
    target_tokens: int,
    state: ClientState,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    loop = asyncio.get_running_loop()

    schedule_index = int(scheduled["schedule_index"])
    rid = str(scheduled["request_id"])
    category = str(row["workload_category"])
    scheduled_offset_s = float(scheduled["scheduled_offset_s"])
    scheduled_loop_time = experiment_start_loop + scheduled_offset_s

    await wait_until(scheduled_loop_time)

    actual_dispatch_loop = loop.time()
    dispatch_lag_ms = (
        actual_dispatch_loop - scheduled_loop_time
    ) * 1000.0

    async with state.lock:
        inflight_at_dispatch = state.inflight
        state.inflight += 1
        state.max_inflight = max(state.max_inflight, state.inflight)

    payload = make_payload(
        row,
        target_tokens=target_tokens,
        ignore_eos=True,
    )
    payload_bytes = len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    request_start = time.perf_counter()

    result: Dict[str, Any] = {
        "schedule_index": schedule_index,
        "request_id": rid,
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
        "frozen_natural_max_new_tokens": int(row["max_new_tokens"]),
        "matched_target_output_tokens": int(target_tokens),
        "sampling_seed": request_sampling_seed(rid),
        "ignore_eos": 1,
        "finish_reason": None,
        "output_tokens": None,
        "length_match": None,
        "output_chars": None,
        "output_sha256": None,
        "client_first_content_event_ttft_ms": None,
        "client_first_visible_text_ttft_ms": None,
        "scheduled_to_first_content_ms": None,
        "client_e2e_ms": None,
        "scheduled_to_completion_ms": None,
        "content_event_count": 0,
        "visible_content_event_count": 0,
        "content_event_minus_output_tokens": None,
        "mean_content_event_itl_ms": None,
        "p50_content_event_itl_ms": None,
        "p95_content_event_itl_ms": None,
        "request_payload_bytes": payload_bytes,
    }

    event_rows: List[Dict[str, Any]] = []
    content_times: List[float] = []
    visible_times: List[float] = []
    output_parts: List[str] = []

    usage = None
    finish_reason = None

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

                event = json.loads(body)
                now = time.perf_counter()
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

                        content_times.append(now)
                        output_parts.append(content)

                        if content:
                            visible_times.append(now)

                        event_rows.append(
                            {
                                "schedule_index": schedule_index,
                                "request_id": rid,
                                "workload_category": category,
                                "content_event_index": len(content_times),
                                "arrival_from_request_start_ms": (
                                    now - request_start
                                ) * 1000.0,
                                "arrival_from_experiment_start_ms": (
                                    now - experiment_start_perf
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

        end = time.perf_counter()

        if not usage:
            raise RuntimeError("Streaming response completed without usage.")

        prompt_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])
        output_text = "".join(output_parts)

        first_content = content_times[0] if content_times else None
        first_visible = visible_times[0] if visible_times else None

        content_itls = [
            (b - a) * 1000.0
            for a, b in zip(content_times, content_times[1:])
        ]

        ttft = (
            (first_content - request_start) * 1000.0
            if first_content is not None
            else None
        )
        visible_ttft = (
            (first_visible - request_start) * 1000.0
            if first_visible is not None
            else None
        )
        e2e = (end - request_start) * 1000.0

        result.update(
            {
                "status": "ok",
                "finish_reason": finish_reason,
                "server_prompt_tokens": prompt_tokens,
                "prompt_token_match": int(
                    prompt_tokens == int(row["input_tokens"])
                ),
                "output_tokens": output_tokens,
                "length_match": int(output_tokens == int(target_tokens)),
                "output_chars": len(output_text),
                "output_sha256": sha256_text(output_text),
                "client_first_content_event_ttft_ms": ttft,
                "client_first_visible_text_ttft_ms": visible_ttft,
                "scheduled_to_first_content_ms": (
                    dispatch_lag_ms + ttft
                    if ttft is not None
                    else None
                ),
                "client_e2e_ms": e2e,
                "scheduled_to_completion_ms": dispatch_lag_ms + e2e,
                "content_event_count": len(content_times),
                "visible_content_event_count": len(visible_times),
                "content_event_minus_output_tokens": (
                    len(content_times) - output_tokens
                ),
                "mean_content_event_itl_ms": mean_or_none(content_itls),
                "p50_content_event_itl_ms": pct(content_itls, 0.50),
                "p95_content_event_itl_ms": pct(content_itls, 0.95),
            }
        )

        if not result["prompt_token_match"]:
            raise RuntimeError(
                f"Prompt-token mismatch for {rid}: "
                f"frozen={row['input_tokens']} server={prompt_tokens}"
            )

        if not result["length_match"]:
            raise RuntimeError(
                f"Length-match failure for {rid}: "
                f"target={target_tokens} actual={output_tokens} "
                f"finish_reason={finish_reason}"
            )

    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:1000]

    finally:
        async with state.lock:
            state.inflight -= 1

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
            metric_name == suffix or metric_name.endswith(suffix)
            for suffix in candidate_suffixes
        ):
            try:
                return float(value_part.strip())
            except ValueError:
                pass
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
                text, ["vllm:num_requests_running", "num_requests_running"]
            )
            row["num_requests_waiting"] = parse_prometheus_value(
                text, ["vllm:num_requests_waiting", "num_requests_waiting"]
            )
            row["kv_cache_usage_perc"] = parse_prometheus_value(
                text, ["vllm:kv_cache_usage_perc", "kv_cache_usage_perc"]
            )
            row["gpu_cache_usage_perc"] = parse_prometheus_value(
                text, ["vllm:gpu_cache_usage_perc", "gpu_cache_usage_perc"]
            )
            row["prefix_cache_hit_rate"] = parse_prometheus_value(
                text, ["vllm:prefix_cache_hit_rate", "prefix_cache_hit_rate"]
            )

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]

        out_rows.append(row)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def finite_max(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            vals.append(fv)
    return max(vals) if vals else None


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> None:
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


def output_paths(
    arrival_rate: float,
    label: str,
) -> Dict[str, Path]:
    directory = (
        RAW_ROOT
        / f"lambda_{rate_slug(arrival_rate)}"
        / safe_label(label)
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


async def execute(
    *,
    arrival_rate: float,
    label: str,
    server_max_num_seqs: Optional[int],
    base_url: str,
    metrics_interval_s: float,
    dry_run: bool,
) -> None:

    workload_metadata, workload = load_workload()
    master = load_s2_coarse_master(workload)
    schedule = scaled_schedule(master, arrival_rate)
    targets = load_s1_length_targets()

    measured_ids = [x["request_id"] for x in schedule]
    missing_targets = [rid for rid in measured_ids if rid not in targets]
    if missing_targets:
        raise RuntimeError(
            f"Missing S1 matched-length targets: {missing_targets[:5]}"
        )

    target_total = sum(targets[rid] for rid in measured_ids)
    target_min = min(targets[rid] for rid in measured_ids)
    target_max = max(targets[rid] for rid in measured_ids)
    target_p50 = pct([targets[rid] for rid in measured_ids], 0.50)

    paths = output_paths(arrival_rate, label)

    scheduled_duration_s = float(schedule[-1]["scheduled_offset_s"])
    realized_rate = (
        (len(schedule) - 1) / scheduled_duration_s
        if len(schedule) > 1 and scheduled_duration_s > 0
        else arrival_rate
    )

    print("=" * 94)
    print("S3-D — vLLM LENGTH-MATCHED REPLAY")
    print("=" * 94)
    print(f"Label:                         {label}")
    print(f"Expected server max_num_seqs: {server_max_num_seqs}")
    print(f"Configured lambda:             {arrival_rate:.3f} req/s")
    print(f"Realized schedule rate:        {realized_rate:.3f} req/s")
    print(f"Requests:                      {len(schedule)}")
    print(f"Scheduled arrival window:      {scheduled_duration_s:.2f}s")
    print(f"Matched target source:         {S1_FULL_REQUESTS}")
    print(f"Total matched output tokens:   {target_total}")
    print(f"Target output P50/min/max:      {target_p50:.1f}/{target_min}/{target_max}")
    print(f"ignore_eos:                    True")
    print(f"Output directory:              {paths['dir']}")

    if dry_run:
        print("\nDRY RUN: no requests sent.")
        return

    if paths["dir"].exists():
        raise RuntimeError(
            "S3-D output directory already exists:\n"
            f"  {paths['dir']}\n"
            "Move/delete it intentionally before rerunning this configuration."
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
            rec = await warmup_request(client, url, row)
            warmup_records.append(rec)
            print(
                f"  [{i}/6] {rec['category']:20s} "
                f"id={rec['request_id']} "
                f"in={rec['input_tokens']:4d} "
                f"out={rec['output_tokens']:3d} "
                f"E2E={rec['client_e2e_ms']:8.2f}ms"
            )

        print("\nStarting matched-work Poisson clock...")

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
                        target_tokens=targets[rid],
                        state=state,
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

    prompt_matches = sum(
        int(r.get("prompt_token_match") == 1)
        for r in successful
    )
    length_matches = sum(
        int(r.get("length_match") == 1)
        for r in successful
    )

    finish_counts = Counter(
        str(r.get("finish_reason"))
        for r in successful
    )

    output_total = sum(
        int(r["output_tokens"])
        for r in successful
        if r["output_tokens"] is not None
    )
    input_total = sum(
        int(r["server_prompt_tokens"])
        for r in successful
        if r["server_prompt_tokens"] is not None
    )

    ttfts = [
        float(r["client_first_content_event_ttft_ms"])
        for r in successful
        if r["client_first_content_event_ttft_ms"] is not None
    ]
    e2es = [
        float(r["client_e2e_ms"])
        for r in successful
        if r["client_e2e_ms"] is not None
    ]
    itls = [
        float(r["mean_content_event_itl_ms"])
        for r in successful
        if r["mean_content_event_itl_ms"] is not None
    ]
    dispatch_lags = [
        float(r["dispatch_lag_ms"])
        for r in successful
        if r["dispatch_lag_ms"] is not None
    ]

    metric_max_running = finite_max(metric_rows, "num_requests_running")
    metric_max_waiting = finite_max(metric_rows, "num_requests_waiting")
    metric_max_kv = finite_max(metric_rows, "kv_cache_usage_perc")

    schedule_used = {
        "source": str(S2_COARSE_SCHEDULE),
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_rate,
        "scheduled_duration_s": scheduled_duration_s,
        "request_ids": measured_ids,
        "unit_interarrivals": [
            x["unit_interarrival"] for x in schedule
        ],
        "scheduled_offsets_s": [
            x["scheduled_offset_s"] for x in schedule
        ],
        "source_schedule_seeds": EXPECTED_SCHEDULE_SEEDS,
    }

    status = "complete" if not failed else "failed"

    metadata = {
        "phase": "S3-D",
        "protocol_version": 1,
        "mode": "length_matched_replay_poisson_coarse",
        "status": status,
        "label": label,
        "corpus_sha256": FROZEN_CORPUS_SHA256,
        "workload_metadata": workload_metadata,
        "s2_schedule_reused_exactly": True,
        "schedule_source": str(S2_COARSE_SCHEDULE),
        "schedule_seeds": EXPECTED_SCHEDULE_SEEDS,
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_rate,
        "scheduled_arrival_window_s": scheduled_duration_s,
        "measured_makespan_s": elapsed_s,
        "drain_after_last_scheduled_arrival_s": drain_s,
        "request_count": len(schedule),
        "successful_request_count": len(successful),
        "failed_request_count": len(failed),
        "max_client_inflight": state.max_inflight,
        "controlled_work": {
            "target_source": str(S1_FULL_REQUESTS),
            "target_field": "output_tokens",
            "ignore_eos": True,
            "max_tokens_policy": "per-request S1 Flash output_tokens",
            "expected_total_output_tokens": target_total,
            "observed_total_output_tokens": output_total,
            "target_output_tokens_p50": target_p50,
            "target_output_tokens_min": target_min,
            "target_output_tokens_max": target_max,
            "important_interpretation": (
                "This intentionally changes natural termination semantics. "
                "Use it only as controlled scheduler/batching validation, "
                "not as the production-like UX result."
            ),
        },
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "per_request_seed_base": SAMPLING_SEED_BASE,
            "seed_formula": "42026000 + integer suffix of request_id",
        },
        "warmup": {
            "count": len(warmup_records),
            "policy": (
                "one median-input request/category outside measured schedule; "
                "max_tokens capped at 32; natural EOS allowed; before t=0"
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
            "max_num_seqs": server_max_num_seqs,
            "note": (
                "max_num_seqs is supplied by the benchmark operator as expected "
                "server configuration; the client does not introspect the CLI."
            ),
        },
        "streaming_semantics": {
            "output_tokens_source": "usage.completion_tokens",
            "content_event_count_is_not_token_count": True,
            "first_content_event_metric": (
                "first SSE choice event containing delta.content; not assumed "
                "one-to-one with generated tokens"
            ),
        },
        "metrics_polling": {
            "enabled": metrics_interval_s > 0,
            "interval_s": metrics_interval_s,
            "sample_count": len(metric_rows),
            "max_num_requests_running": metric_max_running,
            "max_num_requests_waiting": metric_max_waiting,
            "max_kv_cache_usage_perc": metric_max_kv,
        },
        "audit": {
            "prompt_token_matches": prompt_matches,
            "prompt_token_total": len(successful),
            "length_matches": length_matches,
            "length_match_total": len(successful),
            "finish_reason_counts": dict(finish_counts),
            "expected_total_output_tokens": target_total,
            "observed_total_output_tokens": output_total,
            "exact_total_work_match": output_total == target_total,
            "total_input_tokens": input_total,
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
            "p50_e2e_ms": pct(e2es, 0.50),
            "p95_e2e_ms": pct(e2es, 0.95),
            "p99_e2e_ms": pct(e2es, 0.99),
            "p50_request_mean_content_event_itl_ms": pct(itls, 0.50),
            "p95_request_mean_content_event_itl_ms": pct(itls, 0.95),
            "p50_dispatch_lag_ms": pct(dispatch_lags, 0.50),
            "p95_dispatch_lag_ms": pct(dispatch_lags, 0.95),
            "achieved_request_throughput_req_s": (
                len(successful) / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_output_tokens_per_s": (
                output_total / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_input_tokens_per_s": (
                input_total / elapsed_s if elapsed_s > 0 else None
            ),
            "aggregate_total_tokens_per_s": (
                (input_total + output_total) / elapsed_s
                if elapsed_s > 0 else None
            ),
        },
    }

    write_csv(paths["requests_csv"], REQUEST_FIELDS, request_rows)
    with paths["requests_json"].open("w", encoding="utf-8") as f:
        json.dump(request_rows, f, ensure_ascii=False, indent=2)
    write_gzip_csv(paths["events"], EVENT_FIELDS, all_event_rows)
    write_gzip_csv(paths["metrics"], METRIC_FIELDS, metric_rows)

    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["schedule"].write_text(
        json.dumps(schedule_used, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 94)
    print("S3-D LENGTH-MATCHED RUN COMPLETE")
    print("=" * 94)
    print(f"Status:                       {status}")
    print(f"Label:                        {label}")
    print(f"Success:                      {len(successful)}/{len(schedule)}")
    print(f"Prompt-token matches:         {prompt_matches}/{len(successful)}")
    print(f"Length matches:               {length_matches}/{len(successful)}")
    print(f"Expected output tokens:       {target_total}")
    print(f"Observed output tokens:       {output_total}")
    print(f"Finish reasons:               {dict(finish_counts)}")
    print(f"Max client inflight:          {state.max_inflight}")
    print(f"Max vLLM running:             {metric_max_running}")
    print(f"Max vLLM waiting:             {metric_max_waiting}")
    print(f"Max KV-cache usage:           {metric_max_kv}")
    print(f"Drain after last arrival:     {drain_s:.2f}s")
    print(f"P50 TTFT:                     {pct(ttfts, 0.50):.2f}ms")
    print(f"P95 TTFT:                     {pct(ttfts, 0.95):.2f}ms")
    print(f"P99 TTFT:                     {pct(ttfts, 0.99):.2f}ms")
    print(f"P50 mean content ITL:         {pct(itls, 0.50):.2f}ms")
    print(f"P95 mean content ITL:         {pct(itls, 0.95):.2f}ms")
    print(f"P50 E2E:                      {pct(e2es, 0.50):.2f}ms")
    print(f"P95 E2E:                      {pct(e2es, 0.95):.2f}ms")
    print(
        f"Aggregate output throughput:  "
        f"{output_total / elapsed_s:.2f} tok/s"
    )
    print(
        f"Achieved request throughput:  "
        f"{len(successful) / elapsed_s:.3f} req/s"
    )
    print(f"Output directory:             {paths['dir']}")

    if failed:
        first = failed[0]
        raise RuntimeError(
            f"S3-D had {len(failed)} failed requests. "
            f"First: {first['request_id']} "
            f"{first['error_type']}: {first['error_message']}"
        )

    if prompt_matches != len(successful):
        raise RuntimeError("Prompt-token audit failed.")

    if length_matches != len(successful):
        raise RuntimeError("Matched output-length audit failed.")

    if output_total != target_total:
        raise RuntimeError(
            f"Total matched work mismatch: expected={target_total} "
            f"observed={output_total}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3-D vLLM length-matched Poisson replay"
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Output/config label, e.g. default or maxseq12.",
    )
    parser.add_argument(
        "--server-max-num-seqs",
        type=int,
        default=None,
        help=(
            "Expected max_num_seqs used to start the server. "
            "Metadata only; the client cannot enforce/introspect the server CLI."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be > 0")
    if args.metrics_interval < 0:
        raise ValueError("--metrics-interval must be >= 0")
    if args.server_max_num_seqs is not None and args.server_max_num_seqs <= 0:
        raise ValueError("--server-max-num-seqs must be > 0")

    asyncio.run(
        execute(
            arrival_rate=args.arrival_rate,
            label=safe_label(args.label),
            server_max_num_seqs=args.server_max_num_seqs,
            base_url=args.base_url,
            metrics_interval_s=args.metrics_interval,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
