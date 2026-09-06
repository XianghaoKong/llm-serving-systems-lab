#!/usr/bin/env python3
"""
S2 — Poisson arrivals + queueing benchmark client.

This stage keeps the validated S1 Flash/BF16 server unchanged and changes only
the arrival process.

System model
------------
- One persistent GPU worker.
- FCFS-like serialization through the S1 server's asyncio lock.
- No batching / continuous batching yet.
- Concurrent HTTP/SSE client requests arrive according to a Poisson process.
- Six short API warm-ups complete before t=0; natural idle gaps after t=0 remain measured.
- This is intentionally a single-server queueing experiment, not yet a
  concurrency-throughput optimization experiment.

Why S2 exists
-------------
R1/S1 sent the next request only after the previous one completed. S2 allows
new requests to arrive while the GPU worker is busy, so waiting time and TTFT
inflation emerge naturally.

Important
---------
DO NOT resume an interrupted S2 load point. Queueing history affects every
later observation. If a load point is interrupted, move/delete that load
point's output directory and rerun the entire rate from the beginning.

Pilot
-----
Use 60 unique, stratified requests at 0.25 req/s:
    python src/s2_poisson_client.py --phase pilot --arrival-rate 0.25

Coarse sweep (run only after pilot audit)
-----------------------------------------
Use the same 120 unique stratified requests and the same unit-rate exponential
arrival draws at every load point:
    0.17, 0.25, 0.31, 0.35, 0.40 req/s

Each arrival schedule is generated deterministically and persisted under
results/s2/schedules/.

Outputs
-------
results/s2/raw/<phase>/lambda_<rate>/
├── requests.csv
├── token_events.csv.gz
├── run_metadata.json
└── schedule_used.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKLOAD_FILE = (
    PROJECT_ROOT
    / "workloads"
    / "final"
    / "realistic_requests.json"
)

S1_FULL_REQUESTS = (
    PROJECT_ROOT
    / "results"
    / "s1"
    / "raw"
    / "full"
    / "flash_api_requests.csv"
)

SCHEDULE_ROOT = PROJECT_ROOT / "results" / "s2" / "schedules"
RAW_ROOT = PROJECT_ROOT / "results" / "s2" / "raw"

# Separate from the R1 execution seed. These are now part of the S2 protocol.
WORKLOAD_SAMPLE_SEED = 2027
ARRIVAL_DRAW_SEED = 2028
ORDER_SHUFFLE_SEED = 2029

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]

PHASE_COUNTS = {
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

COARSE_PROTOCOL_RATES = [0.17, 0.25, 0.31, 0.35, 0.40]

REQUEST_FIELDS = [
    "schedule_index",
    "request_id",
    "workload_category",
    "status",
    "finish_reason",
    "scheduled_arrival_ms",
    "dispatch_start_ms",
    "dispatch_lag_ms",
    "inflight_at_dispatch",
    "request_payload_bytes",
    "response_headers_ms",
    "client_first_token_event_ttft_ms",
    "client_first_visible_text_ttft_ms",
    "client_e2e_ms",
    "client_mean_token_event_itl_ms",
    "client_p50_token_event_itl_ms",
    "client_p95_token_event_itl_ms",
    "server_queue_ms",
    "server_service_ttft_ms",
    "server_model_ttft_ms",
    "server_mean_tpot_ms",
    "server_p95_tpot_ms",
    "server_mean_stream_itl_ms",
    "server_e2e_ms",
    "server_service_e2e_ms",
    "s1_client_ttft_ms",
    "s1_client_e2e_ms",
    "s1_server_model_ttft_ms",
    "s1_server_mean_tpot_ms",
    "client_ttft_inflation_vs_s1_ms",
    "client_e2e_delta_vs_s1_ms",
    "server_model_ttft_delta_vs_s1_ms",
    "server_mean_tpot_delta_vs_s1_ms",
    "frozen_input_tokens",
    "server_prompt_tokens",
    "max_tokens",
    "output_tokens",
    "output_hash",
    "s1_output_hash",
    "s1_hash_match",
    "s1_output_tokens",
    "s1_output_tokens_match",
    "sampling_seed",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "error_type",
    "error_message",
]

TOKEN_EVENT_FIELDS = [
    "schedule_index",
    "request_id",
    "workload_category",
    "token_index",
    "client_arrival_from_dispatch_ms",
    "client_token_event_itl_ms",
    "server_gpu_step_ms",
    "server_stream_itl_ms",
    "payload_chars",
]


@dataclass
class ClientState:
    inflight: int = 0
    max_inflight: int = 0
    dispatched: int = 0
    completed: int = 0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None

    vals = sorted(float(v) for v in values)

    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return vals[lower]

    weight = pos - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(float(v) for v in values)


def rate_slug(rate: float) -> str:
    return f"{rate:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def load_workload() -> Dict[str, Dict[str, Any]]:
    if not WORKLOAD_FILE.exists():
        raise FileNotFoundError(f"Missing workload: {WORKLOAD_FILE}")

    payload = json.loads(WORKLOAD_FILE.read_text(encoding="utf-8"))

    requests = payload["requests"]

    return {
        str(row["request_id"]): row
        for row in requests
    }


def load_s1_baseline() -> Dict[str, Dict[str, Any]]:
    if not S1_FULL_REQUESTS.exists():
        raise FileNotFoundError(
            "S2 requires the completed S1 Full request CSV:\n"
            f"  {S1_FULL_REQUESTS}"
        )

    rows: Dict[str, Dict[str, Any]] = {}

    with S1_FULL_REQUESTS.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") != "ok":
                raise RuntimeError(
                    "S1 baseline contains a non-ok request."
                )
            rows[str(row["request_id"])] = row

    if len(rows) != 1000:
        raise RuntimeError(
            f"Expected 1000 S1 baseline requests, found {len(rows)}"
        )

    return rows


def s1_service_distribution(
    baseline: Dict[str, Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    """
    Return E[S], E[S^2], service rate mu, and CV.

    S is approximated by S1 server_e2e_ms - server_queue_ms because S1 was
    sequential and its queue time was near zero. This is used only as a
    queueing-theory reference, not as ground truth for each S2 load point.
    """
    values = []

    for row in baseline.values():
        service_s = (
            float(row["server_e2e_ms"])
            - float(row["server_queue_ms"])
        ) / 1000.0

        values.append(service_s)

    mean_s = statistics.fmean(values)
    mean_s2 = statistics.fmean(x * x for x in values)
    std_s = statistics.stdev(values)
    cv = std_s / mean_s
    mu = 1.0 / mean_s

    return mean_s, mean_s2, mu, cv


def mg1_reference(
    arrival_rate: float,
    mean_service_s: float,
    mean_service_sq_s2: float,
) -> Dict[str, Optional[float]]:
    rho = arrival_rate * mean_service_s

    if rho < 1.0:
        mean_wait_s = (
            arrival_rate
            * mean_service_sq_s2
            / (2.0 * (1.0 - rho))
        )
    else:
        mean_wait_s = None

    return {
        "rho_from_s1_mean_service": rho,
        "theoretical_mg1_mean_wait_s": mean_wait_s,
    }


def choose_stratified_requests(
    workload: Dict[str, Dict[str, Any]],
    phase: str,
) -> List[str]:
    counts = PHASE_COUNTS[phase]
    rng = random.Random(WORKLOAD_SAMPLE_SEED)

    selected: List[str] = []

    for category in CATEGORY_ORDER:
        candidates = sorted(
            [
                request_id
                for request_id, row in workload.items()
                if row["workload_category"] == category
            ]
        )

        count = counts[category]

        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough {category} requests: "
                f"need {count}, have {len(candidates)}"
            )

        selected.extend(rng.sample(candidates, count))

    order_rng = random.Random(ORDER_SHUFFLE_SEED)
    order_rng.shuffle(selected)

    return selected


def schedule_path(phase: str) -> Path:
    return SCHEDULE_ROOT / f"{phase}_master_schedule.json"


def build_or_load_master_schedule(
    workload: Dict[str, Dict[str, Any]],
    phase: str,
) -> Dict[str, Any]:
    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    path = schedule_path(phase)

    expected_count = sum(PHASE_COUNTS[phase].values())

    if path.exists():
        schedule = json.loads(path.read_text(encoding="utf-8"))

        if schedule.get("phase") != phase:
            raise RuntimeError(
                f"Schedule phase mismatch in {path}"
            )

        if len(schedule["request_ids"]) != expected_count:
            raise RuntimeError(
                f"Schedule request count mismatch in {path}"
            )

        if len(schedule["unit_interarrivals"]) != expected_count:
            raise RuntimeError(
                f"Schedule interarrival count mismatch in {path}"
            )

        return schedule

    request_ids = choose_stratified_requests(workload, phase)

    arrival_rng = random.Random(ARRIVAL_DRAW_SEED)

    # First request arrives at t=0. Subsequent gaps are iid Exp(rate=1).
    unit_interarrivals = [0.0]

    for _ in range(1, expected_count):
        unit_interarrivals.append(
            arrival_rng.expovariate(1.0)
        )

    schedule = {
        "protocol_version": 1,
        "phase": phase,
        "request_count": expected_count,
        "category_counts": PHASE_COUNTS[phase],
        "workload_sample_seed": WORKLOAD_SAMPLE_SEED,
        "order_shuffle_seed": ORDER_SHUFFLE_SEED,
        "arrival_draw_seed": ARRIVAL_DRAW_SEED,
        "request_ids": request_ids,
        "unit_interarrivals": unit_interarrivals,
        "note": (
            "For arrival rate lambda, actual interarrival seconds are "
            "unit_interarrival / lambda. The same request sequence and "
            "same unit-rate exponential draws are reused across load points."
        ),
    }

    path.write_text(
        json.dumps(schedule, indent=2),
        encoding="utf-8",
    )

    return schedule


def scaled_schedule(
    master: Dict[str, Any],
    arrival_rate: float,
) -> List[Dict[str, Any]]:
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be > 0")

    elapsed_s = 0.0
    rows = []

    for index, (request_id, unit_gap) in enumerate(
        zip(
            master["request_ids"],
            master["unit_interarrivals"],
        ),
        start=1,
    ):
        elapsed_s += float(unit_gap) / arrival_rate

        rows.append({
            "schedule_index": index,
            "request_id": str(request_id),
            "unit_interarrival": float(unit_gap),
            "scheduled_offset_s": elapsed_s,
        })

    return rows



def representative_warmup_requests(
    workload: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pick one median-input request per category.

    S1 Full used a warm persistent worker. S2 explicitly warms the API worker
    before the Poisson clock starts so the first measured arrival is not a
    startup/cold-allocator artifact. Natural idle gaps *after* the experiment
    starts are preserved and remain part of the production-like workload.
    """
    chosen: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        group = sorted(
            [
                row
                for row in workload.values()
                if row["workload_category"] == category
            ],
            key=lambda row: int(row["input_tokens"]),
        )

        if not group:
            raise RuntimeError(
                f"No workload rows for warm-up category {category}"
            )

        chosen.append(group[len(group) // 2])

    return chosen


async def warmup_server(
    *,
    client: httpx.AsyncClient,
    url: str,
    workload: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sequentially send one short warm-up request per category.

    Warm-up requests are not part of the measured arrival schedule and use a
    32-token cap. The experiment clock starts only after all six complete.
    """
    print("\nS2 warm-up: one representative request/category...")

    records: List[Dict[str, Any]] = []

    for index, row in enumerate(
        representative_warmup_requests(workload),
        start=1,
    ):
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": str(row["prompt"]),
                }
            ],
            "stream": True,
            "max_tokens": min(
                int(row["max_new_tokens"]),
                32,
            ),
            "request_id": str(row["request_id"]),
        }

        start = time.perf_counter()
        token_events = 0
        final_event: Optional[Dict[str, Any]] = None

        async with client.stream(
            "POST",
            url,
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                body = line[6:]

                if body == "[DONE]":
                    break

                event = json.loads(body)

                if "_server_metrics" in event:
                    final_event = event
                    continue

                metrics = event.get("_metrics", {})

                if metrics.get("flush"):
                    continue

                if metrics.get("token_index") is not None:
                    token_events += 1

        if final_event is None:
            raise RuntimeError(
                f"Warm-up {row['request_id']} ended without final metrics."
            )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        server_metrics = final_event["_server_metrics"]

        record = {
            "category": row["workload_category"],
            "request_id": row["request_id"],
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(
                final_event["usage"]["completion_tokens"]
            ),
            "token_events": token_events,
            "client_e2e_ms": elapsed_ms,
            "server_model_ttft_ms": float(
                server_metrics["model_ttft_ms"]
            ),
        }
        records.append(record)

        print(
            f"  [{index}/6] "
            f"{row['workload_category']:20s} "
            f"in={record['input_tokens']:4d} "
            f"out={record['output_tokens']:3d} "
            f"modelTTFT={record['server_model_ttft_ms']:8.2f} ms"
        )

    print("S2 warm-up PASS.")
    return records



def output_paths(phase: str, arrival_rate: float) -> Dict[str, Path]:
    directory = (
        RAW_ROOT
        / phase
        / f"lambda_{rate_slug(arrival_rate)}"
    )

    return {
        "dir": directory,
        "requests": directory / "requests.csv",
        "tokens": directory / "token_events.csv.gz",
        "metadata": directory / "run_metadata.json",
        "schedule": directory / "schedule_used.json",
    }


def make_payload(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": str(row["prompt"]),
            }
        ],
        "stream": True,
        "max_tokens": int(row["max_new_tokens"]),
        "request_id": str(row["request_id"]),
    }


async def wait_until(
    target_loop_time: float,
) -> None:
    while True:
        now = asyncio.get_running_loop().time()
        remaining = target_loop_time - now

        if remaining <= 0:
            return

        await asyncio.sleep(remaining)


async def run_one_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    experiment_start: float,
    scheduled: Dict[str, Any],
    workload_row: Dict[str, Any],
    s1_row: Dict[str, Any],
    state: ClientState,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    schedule_index = int(scheduled["schedule_index"])
    request_id = str(scheduled["request_id"])
    scheduled_offset_s = float(scheduled["scheduled_offset_s"])
    scheduled_time = experiment_start + scheduled_offset_s

    await wait_until(scheduled_time)

    actual_dispatch = asyncio.get_running_loop().time()
    dispatch_lag_ms = (
        actual_dispatch - scheduled_time
    ) * 1000.0

    inflight_at_dispatch = state.inflight
    state.inflight += 1
    state.dispatched += 1
    state.max_inflight = max(
        state.max_inflight,
        state.inflight,
    )

    payload = make_payload(workload_row)
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
        "request_id": request_id,
        "workload_category": workload_row[
            "workload_category"
        ],
        "status": "error",
        "scheduled_arrival_ms": (
            scheduled_offset_s * 1000.0
        ),
        "dispatch_start_ms": (
            actual_dispatch - experiment_start
        ) * 1000.0,
        "dispatch_lag_ms": dispatch_lag_ms,
        "inflight_at_dispatch": inflight_at_dispatch,
        "request_payload_bytes": payload_bytes,
        "frozen_input_tokens": int(
            workload_row["input_tokens"]
        ),
        "max_tokens": int(
            workload_row["max_new_tokens"]
        ),
        "s1_client_ttft_ms": float(
            s1_row["client_ttft_ms"]
        ),
        "s1_client_e2e_ms": float(
            s1_row["client_e2e_ms"]
        ),
        "s1_server_model_ttft_ms": float(
            s1_row["server_model_ttft_ms"]
        ),
        "s1_server_mean_tpot_ms": float(
            s1_row["server_mean_tpot_ms"]
        ),
        "s1_output_hash": str(
            s1_row["output_hash"]
        ),
        "s1_output_tokens": int(
            s1_row["output_tokens"]
        ),
    }

    token_rows: List[Dict[str, Any]] = []
    token_arrivals: List[float] = []

    first_token_arrival: Optional[float] = None
    first_visible_text_arrival: Optional[float] = None
    done_arrival: Optional[float] = None
    response_headers_ms: Optional[float] = None
    final_event: Optional[Dict[str, Any]] = None

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
        ) as response:
            response_headers_ms = (
                time.perf_counter() - request_start
            ) * 1000.0

            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                body = line[6:]

                if body == "[DONE]":
                    done_arrival = time.perf_counter()
                    break

                event = json.loads(body)

                if "_server_metrics" in event:
                    final_event = event
                    continue

                metrics = event.get("_metrics", {})

                content = (
                    event.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )

                if content is None:
                    content = ""

                arrival = time.perf_counter()

                # Record visible text even when it appears only in the final
                # TextStreamer flush event.
                if (
                    first_visible_text_arrival is None
                    and len(content) > 0
                ):
                    first_visible_text_arrival = arrival

                if metrics.get("flush"):
                    continue

                token_index = metrics.get("token_index")

                if token_index is None:
                    continue

                if first_token_arrival is None:
                    first_token_arrival = arrival

                token_arrivals.append(arrival)

                previous_arrival = (
                    token_arrivals[-2]
                    if len(token_arrivals) >= 2
                    else None
                )

                token_event_itl_ms = (
                    (arrival - previous_arrival)
                    * 1000.0
                    if previous_arrival is not None
                    else None
                )

                token_rows.append({
                    "schedule_index": schedule_index,
                    "request_id": request_id,
                    "workload_category": workload_row[
                        "workload_category"
                    ],
                    "token_index": int(token_index),
                    "client_arrival_from_dispatch_ms": (
                        (arrival - request_start)
                        * 1000.0
                    ),
                    "client_token_event_itl_ms": (
                        token_event_itl_ms
                    ),
                    "server_gpu_step_ms": metrics.get(
                        "gpu_step_ms"
                    ),
                    "server_stream_itl_ms": metrics.get(
                        "server_stream_itl_ms"
                    ),
                    "payload_chars": len(content),
                })

        if final_event is None:
            raise RuntimeError(
                "Stream ended without final server metrics."
            )

        if first_token_arrival is None:
            raise RuntimeError(
                "Stream completed without a token event."
            )

        if done_arrival is None:
            done_arrival = time.perf_counter()

        if first_visible_text_arrival is None:
            # This should no longer happen because flush chunks are observed.
            raise RuntimeError(
                "Stream completed without visible text."
            )

        usage = final_event["usage"]
        server = final_event["_server_metrics"]
        finish_reason = (
            final_event["choices"][0]["finish_reason"]
        )

        output_tokens = int(
            usage["completion_tokens"]
        )

        if output_tokens != len(token_arrivals):
            raise RuntimeError(
                f"Token event mismatch: client={len(token_arrivals)} "
                f"server={output_tokens}"
            )

        server_prompt_tokens = int(
            usage["prompt_tokens"]
        )

        frozen_input_tokens = int(
            workload_row["input_tokens"]
        )

        if server_prompt_tokens != frozen_input_tokens:
            raise RuntimeError(
                f"Prompt-token mismatch: "
                f"frozen={frozen_input_tokens} "
                f"server={server_prompt_tokens}"
            )

        client_first_token_ttft_ms = (
            first_token_arrival - request_start
        ) * 1000.0

        client_first_visible_text_ttft_ms = (
            first_visible_text_arrival
            - request_start
        ) * 1000.0

        client_e2e_ms = (
            done_arrival - request_start
        ) * 1000.0

        token_event_itls = [
            (
                token_arrivals[i]
                - token_arrivals[i - 1]
            )
            * 1000.0
            for i in range(1, len(token_arrivals))
        ]

        server_queue_ms = float(
            server["queue_ms"]
        )

        server_request_ttft_ms = float(
            server["server_request_ttft_ms"]
        )

        server_e2e_ms = float(
            server["server_e2e_ms"]
        )

        server_model_ttft_ms = float(
            server["model_ttft_ms"]
        )

        server_mean_tpot_ms = (
            float(server["mean_tpot_ms"])
            if server["mean_tpot_ms"] is not None
            else None
        )

        output_hash = str(
            server["output_hash"]
        )

        s1_hash = str(
            s1_row["output_hash"]
        )

        s1_output_tokens = int(
            s1_row["output_tokens"]
        )

        result.update({
            "status": "ok",
            "finish_reason": finish_reason,
            "response_headers_ms": response_headers_ms,
            "client_first_token_event_ttft_ms": (
                client_first_token_ttft_ms
            ),
            "client_first_visible_text_ttft_ms": (
                client_first_visible_text_ttft_ms
            ),
            "client_e2e_ms": client_e2e_ms,
            "client_mean_token_event_itl_ms": (
                mean_or_none(token_event_itls)
            ),
            "client_p50_token_event_itl_ms": (
                percentile(token_event_itls, 0.50)
            ),
            "client_p95_token_event_itl_ms": (
                percentile(token_event_itls, 0.95)
            ),
            "server_queue_ms": server_queue_ms,
            "server_service_ttft_ms": (
                server_request_ttft_ms
                - server_queue_ms
            ),
            "server_model_ttft_ms": (
                server_model_ttft_ms
            ),
            "server_mean_tpot_ms": (
                server_mean_tpot_ms
            ),
            "server_p95_tpot_ms": server[
                "p95_tpot_ms"
            ],
            "server_mean_stream_itl_ms": server[
                "mean_server_stream_itl_ms"
            ],
            "server_e2e_ms": server_e2e_ms,
            "server_service_e2e_ms": (
                server_e2e_ms
                - server_queue_ms
            ),
            "client_ttft_inflation_vs_s1_ms": (
                client_first_token_ttft_ms
                - float(s1_row["client_ttft_ms"])
            ),
            "client_e2e_delta_vs_s1_ms": (
                client_e2e_ms
                - float(s1_row["client_e2e_ms"])
            ),
            "server_model_ttft_delta_vs_s1_ms": (
                server_model_ttft_ms
                - float(
                    s1_row["server_model_ttft_ms"]
                )
            ),
            "server_mean_tpot_delta_vs_s1_ms": (
                (
                    server_mean_tpot_ms
                    - float(
                        s1_row["server_mean_tpot_ms"]
                    )
                )
                if server_mean_tpot_ms is not None
                else None
            ),
            "server_prompt_tokens": (
                server_prompt_tokens
            ),
            "output_tokens": output_tokens,
            "output_hash": output_hash,
            "s1_hash_match": int(
                output_hash == s1_hash
            ),
            "s1_output_tokens_match": int(
                output_tokens
                == s1_output_tokens
            ),
            "sampling_seed": server[
                "sampling_seed"
            ],
            "peak_allocated_gib": server[
                "peak_allocated_gib"
            ],
            "peak_reserved_gib": server[
                "peak_reserved_gib"
            ],
            "error_type": "",
            "error_message": "",
        })

    except Exception as exc:
        result.update({
            "response_headers_ms": (
                response_headers_ms
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        })

    finally:
        state.inflight -= 1
        state.completed += 1

    return result, token_rows


def write_request_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=REQUEST_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    ""
                    if row.get(key) is None
                    else row.get(key)
                )
                for key in REQUEST_FIELDS
            })


def write_token_csv_gz(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TOKEN_EVENT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    ""
                    if row.get(key) is None
                    else row.get(key)
                )
                for key in TOKEN_EVENT_FIELDS
            })


async def execute(
    *,
    phase: str,
    arrival_rate: float,
    base_url: str,
    dry_run: bool,
) -> None:
    workload = load_workload()
    baseline = load_s1_baseline()

    mean_s, mean_s2, mu, service_cv = (
        s1_service_distribution(baseline)
    )

    configured_reference = mg1_reference(
        arrival_rate,
        mean_s,
        mean_s2,
    )

    master = build_or_load_master_schedule(
        workload,
        phase,
    )

    schedule = scaled_schedule(
        master,
        arrival_rate,
    )

    paths = output_paths(
        phase,
        arrival_rate,
    )

    scheduled_duration_s = float(
        schedule[-1]["scheduled_offset_s"]
    )

    # A finite exponential trace will not have exactly its configured lambda.
    # Keep the exact Poisson draw; report both configured and realized rates.
    realized_arrival_rate = (
        (len(schedule) - 1) / scheduled_duration_s
        if len(schedule) > 1 and scheduled_duration_s > 0
        else arrival_rate
    )

    realized_reference = mg1_reference(
        realized_arrival_rate,
        mean_s,
        mean_s2,
    )

    print("=" * 92)
    print("S2 — POISSON ARRIVALS + QUEUEING")
    print("=" * 92)
    print(f"Phase:                     {phase}")
    print(f"Configured lambda:         {arrival_rate:.3f} req/s")
    print(f"Realized schedule rate:    {realized_arrival_rate:.3f} req/s")
    print(f"Requests:                  {len(schedule)}")
    print(f"Scheduled arrival window:  {scheduled_duration_s/60.0:.2f} min")
    print(f"S1 mean service time:      {mean_s:.4f} s")
    print(f"S1 implied capacity mu:    {mu:.4f} req/s")
    print(f"S1 service-time CV:        {service_cv:.3f}")
    print(
        f"Configured rho:            "
        f"{configured_reference['rho_from_s1_mean_service']:.3f}"
    )
    print(
        f"Realized rho:              "
        f"{realized_reference['rho_from_s1_mean_service']:.3f}"
    )

    if realized_reference["theoretical_mg1_mean_wait_s"] is None:
        print("M/G/1 wait (realized):     unstable (rho >= 1)")
    else:
        print(
            "M/G/1 wait (realized):     "
            f"{realized_reference['theoretical_mg1_mean_wait_s']:.2f} s"
        )

    print(f"Master schedule:           {schedule_path(phase)}")
    print(f"Output directory:          {paths['dir']}")

    if phase == "coarse":
        print(
            "Coarse protocol rates:      "
            + ", ".join(
                f"{rate:.2f}"
                for rate in COARSE_PROTOCOL_RATES
            )
        )

    if dry_run:
        print("\nDRY RUN: no requests sent.")
        return

    if paths["dir"].exists():
        raise RuntimeError(
            "This S2 load-point directory already exists:\n"
            f"  {paths['dir']}\n\n"
            "S2 load points are NOT resumable because queue history matters. "
            "Move/delete the old directory intentionally before rerunning "
            "this exact rate."
        )

    paths["dir"].mkdir(
        parents=True,
        exist_ok=False,
    )

    paths["schedule"].write_text(
        json.dumps(
            {
                "phase": phase,
                "configured_arrival_rate_req_s": arrival_rate,
                "realized_schedule_arrival_rate_req_s": realized_arrival_rate,
                "master_schedule_file": str(
                    schedule_path(phase)
                ),
                "rows": schedule,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    url = (
        base_url.rstrip("/")
        + "/v1/chat/completions"
    )

    timeout = httpx.Timeout(
        connect=10.0,
        read=None,
        write=30.0,
        pool=None,
    )

    # Streaming HTTP/1.1 requests each occupy a connection while queued.
    # 256 is intentionally much larger than the expected pilot/coarse queue.
    limits = httpx.Limits(
        max_keepalive_connections=64,
        max_connections=256,
    )

    metadata: Dict[str, Any] = {
        "protocol_version": 2,
        "stage": "S2",
        "phase": phase,
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_arrival_rate,
        "transport": "HTTP/1.1 + SSE",
        "client_arrival_process": (
            "Poisson: iid exponential inter-arrivals; finite trace is not "
            "renormalized, so configured and realized rates are both reported"
        ),
        "client_connection_limit": 256,
        "request_count": len(schedule),
        "scheduled_arrival_window_s": (
            scheduled_duration_s
        ),
        "workload_sample_seed": (
            WORKLOAD_SAMPLE_SEED
        ),
        "arrival_draw_seed": (
            ARRIVAL_DRAW_SEED
        ),
        "order_shuffle_seed": (
            ORDER_SHUFFLE_SEED
        ),
        "s1_reference": {
            "mean_service_s": mean_s,
            "mean_service_sq_s2": mean_s2,
            "service_rate_mu_req_s": mu,
            "service_time_cv": service_cv,
            "configured": configured_reference,
            "realized_schedule": realized_reference,
        },
        "startup_warmup": {
            "enabled": True,
            "requests": 6,
            "max_tokens_per_request": 32,
            "note": (
                "Warm-up completes before the Poisson experiment clock starts. "
                "Natural GPU idle gaps after t=0 remain part of S2."
            ),
        },
        "status": "running",
        "started_at_epoch_s": time.time(),
    }

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = ClientState()

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=False,
    ) as client:
        health_url = (
            base_url.rstrip("/")
            + "/health"
        )

        health = await client.get(
            health_url
        )
        health.raise_for_status()

        print(f"\nHealth: {health.json()}")

        warmup_records = await warmup_server(
            client=client,
            url=url,
            workload=workload,
        )

        metadata["startup_warmup"]["records"] = warmup_records
        paths["metadata"].write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nStarting arrivals. "
            "Do not interrupt this load point.\n"
        )

        loop = asyncio.get_running_loop()
        experiment_start = loop.time()

        tasks = []

        for scheduled in schedule:
            request_id = scheduled["request_id"]

            tasks.append(
                asyncio.create_task(
                    run_one_request(
                        client=client,
                        url=url,
                        experiment_start=experiment_start,
                        scheduled=scheduled,
                        workload_row=workload[request_id],
                        s1_row=baseline[request_id],
                        state=state,
                    )
                )
            )

        completed_count = 0
        results: List[Dict[str, Any]] = []
        all_token_rows: List[Dict[str, Any]] = []

        for future in asyncio.as_completed(tasks):
            result, token_rows = await future
            results.append(result)
            all_token_rows.extend(token_rows)

            completed_count += 1

            if (
                completed_count % 10 == 0
                or completed_count == len(tasks)
            ):
                ok_count = sum(
                    row["status"] == "ok"
                    for row in results
                )

                print(
                    f"  completed {completed_count:3d}/{len(tasks):3d} "
                    f"ok={ok_count:3d} "
                    f"inflight={state.inflight:3d} "
                    f"max_inflight={state.max_inflight:3d}"
                )

        experiment_end = loop.time()

    results.sort(
        key=lambda row: int(
            row["schedule_index"]
        )
    )

    all_token_rows.sort(
        key=lambda row: (
            int(row["schedule_index"]),
            int(row["token_index"]),
        )
    )

    write_request_csv(
        paths["requests"],
        results,
    )

    write_token_csv_gz(
        paths["tokens"],
        all_token_rows,
    )

    errors = [
        row
        for row in results
        if row["status"] != "ok"
    ]

    successful = len(results) - len(errors)
    hash_matches = sum(
        int(row.get("s1_hash_match", 0) or 0)
        for row in results
        if row["status"] == "ok"
    )

    dispatch_lags = [
        float(row["dispatch_lag_ms"])
        for row in results
    ]

    queue_values = [
        float(row["server_queue_ms"])
        for row in results
        if row["status"] == "ok"
    ]

    metadata.update({
        "status": (
            "complete"
            if not errors
            else "complete_with_errors"
        ),
        "finished_at_epoch_s": time.time(),
        "elapsed_seconds": (
            experiment_end
            - experiment_start
        ),
        "successful_requests": successful,
        "failed_requests": len(errors),
        "configured_arrival_rate_req_s": arrival_rate,
        "realized_schedule_arrival_rate_req_s": realized_arrival_rate,
        "realized_rho_from_s1_mean_service": (
            realized_reference["rho_from_s1_mean_service"]
        ),
        "s1_output_hash_matches": (
            hash_matches
        ),
        "max_client_inflight": (
            state.max_inflight
        ),
        "dispatch_lag_ms": {
            "p50": percentile(
                dispatch_lags,
                0.50,
            ),
            "p95": percentile(
                dispatch_lags,
                0.95,
            ),
            "p99": percentile(
                dispatch_lags,
                0.99,
            ),
            "max": max(dispatch_lags),
        },
        "server_queue_ms": {
            "mean": mean_or_none(
                queue_values
            ),
            "p50": percentile(
                queue_values,
                0.50,
            ),
            "p95": percentile(
                queue_values,
                0.95,
            ),
            "p99": percentile(
                queue_values,
                0.99,
            ),
            "max": max(queue_values),
        },
    })

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("S2 LOAD POINT COMPLETE")
    print("=" * 92)
    print(f"Successful requests:       {successful}/{len(results)}")
    print(
        f"Configured/realized rate:  "
        f"{arrival_rate:.3f} / {realized_arrival_rate:.3f} req/s"
    )
    print(f"S1 output hash matches:    {hash_matches}/{successful}")
    print(f"Max client inflight:       {state.max_inflight}")
    print(
        f"Dispatch lag P95:          "
        f"{percentile(dispatch_lags, 0.95):.2f} ms"
    )
    print(
        f"Server queue P50/P95:      "
        f"{percentile(queue_values, 0.50):.2f} / "
        f"{percentile(queue_values, 0.95):.2f} ms"
    )
    print(
        f"Total wall time:           "
        f"{(experiment_end-experiment_start)/60.0:.2f} min"
    )
    print(f"Request results:           {paths['requests']}")
    print(f"Token events:              {paths['tokens']}")
    print(f"Metadata:                  {paths['metadata']}")
    print()
    print(
        "For S2, do not infer saturation from one load point alone. "
        "Audit the pilot first, then run the locked coarse sweep."
    )

    if errors:
        print(
            "\nWARNING: one or more requests failed. "
            "This load point is not valid for final analysis."
        )


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
        required=True,
        help="Poisson arrival rate lambda in requests/second.",
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    asyncio.run(
        execute(
            phase=args.phase,
            arrival_rate=args.arrival_rate,
            base_url=args.base_url,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
