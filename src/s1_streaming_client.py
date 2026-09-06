#!/usr/bin/env python3
"""
S1 — Streaming API benchmark client.

This client sends the frozen R0 workload over real HTTP/SSE to the persistent
S1 server and measures client-observed TTFT, streaming chunk ITL, and E2E.

It reuses the exact R1 execution order:
- pilot: same 60 requests used by R1-P
- full: same 1,000 requests used by R1-F

The client also compares output hashes against the R1 Flash baseline when
`results/r1/raw/full/flash_requests.csv` is present.

Run
---
Pilot:
    python src/s1_streaming_client.py --mode pilot

Resume:
    python src/s1_streaming_client.py --mode pilot --resume

Full (only after pilot audit):
    python src/s1_streaming_client.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

EXECUTION_ORDER_FILE = (
    PROJECT_ROOT
    / "results"
    / "r1"
    / "raw"
    / "execution_order.json"
)

R1_FLASH_REQUESTS = (
    PROJECT_ROOT
    / "results"
    / "r1"
    / "raw"
    / "full"
    / "flash_requests.csv"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "s1" / "raw"

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]

REQUEST_FIELDS = [
    "request_id",
    "workload_category",
    "status",
    "finish_reason",
    "frozen_input_tokens",
    "server_prompt_tokens",
    "max_tokens",
    "output_tokens",
    "output_hash",
    "r1_flash_output_hash",
    "r1_flash_hash_match",
    "r1_flash_output_tokens",
    "r1_flash_output_tokens_match",
    "request_payload_bytes",
    "response_headers_ms",
    "client_ttft_ms",
    "client_e2e_ms",
    "client_minus_server_ttft_ms",
    "client_minus_server_e2e_ms",
    "client_mean_chunk_itl_ms",
    "client_p50_chunk_itl_ms",
    "client_p95_chunk_itl_ms",
    "server_queue_ms",
    "server_preprocess_ms",
    "server_h2d_ms",
    "server_model_ttft_ms",
    "server_request_ttft_ms",
    "server_mean_tpot_ms",
    "server_p95_tpot_ms",
    "server_mean_stream_itl_ms",
    "server_e2e_ms",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "sampling_seed",
    "error_type",
    "error_message",
]

CHUNK_FIELDS = [
    "request_id",
    "workload_category",
    "token_index",
    "client_arrival_from_start_ms",
    "client_chunk_itl_ms",
    "server_gpu_step_ms",
    "server_stream_itl_ms",
    "payload_chars",
]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None

    vals = sorted(float(v) for v in values)

    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(vals) - 1)
    weight = pos - lower

    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def load_workload():
    payload = json.loads(
        WORKLOAD_FILE.read_text(encoding="utf-8")
    )
    requests = payload["requests"]
    return {
        str(row["request_id"]): row
        for row in requests
    }


def load_execution_order(mode: str) -> List[str]:
    order = json.loads(
        EXECUTION_ORDER_FILE.read_text(encoding="utf-8")
    )

    if mode == "pilot":
        return list(order["pilot_request_ids"])

    return list(order["full_request_ids"])


def load_r1_flash_reference() -> Dict[str, Dict[str, Any]]:
    if not R1_FLASH_REQUESTS.exists():
        return {}

    reference = {}

    with R1_FLASH_REQUESTS.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            reference[str(row["request_id"])] = row

    return reference


def representative_warmup_requests(
    workload: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chosen = []

    for category in CATEGORY_ORDER:
        group = [
            row
            for row in workload.values()
            if row["workload_category"] == category
        ]

        ordered = sorted(
            group,
            key=lambda row: int(row["input_tokens"]),
        )

        chosen.append(
            ordered[len(ordered) // 2]
        )

    return chosen


def result_paths(mode: str):
    directory = RESULT_ROOT / mode
    directory.mkdir(parents=True, exist_ok=True)

    return {
        "requests": directory / "flash_api_requests.csv",
        "chunks": directory / "flash_api_chunk_latency.csv.gz",
        "metadata": directory / "flash_api_run_metadata.json",
    }


def completed_request_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed = set()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") == "ok":
                completed.add(str(row["request_id"]))

    return completed


def append_csv_row(
    path: Path,
    fields: Sequence[str],
    row: Dict[str, Any],
):
    exists = path.exists() and path.stat().st_size > 0

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            key: "" if row.get(key) is None else row.get(key)
            for key in fields
        })

        f.flush()
        os.fsync(f.fileno())


def append_chunk_rows(
    path: Path,
    rows: Sequence[Dict[str, Any]],
):
    if not rows:
        return

    exists = path.exists() and path.stat().st_size > 0

    with gzip.open(
        path,
        "at",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CHUNK_FIELDS,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


def make_payload(
    row: Dict[str, Any],
    max_tokens_override: Optional[int] = None,
):
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": str(row["prompt"]),
            }
        ],
        "stream": True,
        "max_tokens": int(
            max_tokens_override
            if max_tokens_override is not None
            else row["max_new_tokens"]
        ),
        "request_id": str(row["request_id"]),
    }


def run_stream_request(
    *,
    client: httpx.Client,
    url: str,
    row: Dict[str, Any],
    r1_ref: Optional[Dict[str, Any]],
    max_tokens_override: Optional[int] = None,
    capture_chunks: bool = True,
):
    payload = make_payload(
        row,
        max_tokens_override=max_tokens_override,
    )

    payload_bytes = len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    result = {
        field: ""
        for field in REQUEST_FIELDS
    }

    result.update({
        "request_id": row["request_id"],
        "workload_category": row["workload_category"],
        "status": "error",
        "frozen_input_tokens": row["input_tokens"],
        "max_tokens": payload["max_tokens"],
        "request_payload_bytes": payload_bytes,
    })

    chunk_rows: List[Dict[str, Any]] = []
    token_arrivals: List[float] = []

    request_start = time.perf_counter()
    first_token_arrival = None
    done_arrival = None
    response_headers_ms = None
    final_event = None

    try:
        with client.stream(
            "POST",
            url,
            json=payload,
        ) as response:
            response_headers_ms = (
                time.perf_counter() - request_start
            ) * 1000.0

            response.raise_for_status()

            for line in response.iter_lines():
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

                if metrics.get("flush"):
                    continue

                token_index = metrics.get("token_index")

                if token_index is None:
                    continue

                arrival = time.perf_counter()

                if first_token_arrival is None:
                    first_token_arrival = arrival

                token_arrivals.append(arrival)

                previous_arrival = (
                    token_arrivals[-2]
                    if len(token_arrivals) >= 2
                    else None
                )

                chunk_itl_ms = (
                    (arrival - previous_arrival) * 1000.0
                    if previous_arrival is not None
                    else None
                )

                content = (
                    event.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )

                if capture_chunks:
                    chunk_rows.append({
                        "request_id": row["request_id"],
                        "workload_category": row["workload_category"],
                        "token_index": token_index,
                        "client_arrival_from_start_ms": (
                            (arrival - request_start) * 1000.0
                        ),
                        "client_chunk_itl_ms": chunk_itl_ms,
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
                "Stream ended without final server metrics event."
            )

        if first_token_arrival is None:
            raise RuntimeError(
                "Stream completed without any token event."
            )

        if done_arrival is None:
            done_arrival = time.perf_counter()

        usage = final_event["usage"]
        server = final_event["_server_metrics"]
        finish_reason = (
            final_event["choices"][0]["finish_reason"]
        )

        client_ttft_ms = (
            first_token_arrival - request_start
        ) * 1000.0

        client_e2e_ms = (
            done_arrival - request_start
        ) * 1000.0

        chunk_itls = [
            (
                token_arrivals[i]
                - token_arrivals[i - 1]
            )
            * 1000.0
            for i in range(1, len(token_arrivals))
        ]

        output_tokens = int(
            usage["completion_tokens"]
        )

        if output_tokens != len(token_arrivals):
            raise RuntimeError(
                f"Client observed {len(token_arrivals)} token events, "
                f"server reported {output_tokens} completion tokens."
            )

        output_hash = str(server["output_hash"])

        r1_hash = ""
        r1_hash_match: Any = ""
        r1_output_tokens: Any = ""
        r1_output_tokens_match: Any = ""

        if r1_ref is not None:
            r1_hash = str(
                r1_ref.get("output_hash", "")
            )
            r1_output_tokens = int(
                r1_ref["output_tokens"]
            )
            r1_hash_match = int(
                output_hash == r1_hash
            )
            r1_output_tokens_match = int(
                output_tokens == r1_output_tokens
            )

        result.update({
            "status": "ok",
            "finish_reason": finish_reason,
            "server_prompt_tokens": usage["prompt_tokens"],
            "output_tokens": output_tokens,
            "output_hash": output_hash,
            "r1_flash_output_hash": r1_hash,
            "r1_flash_hash_match": r1_hash_match,
            "r1_flash_output_tokens": r1_output_tokens,
            "r1_flash_output_tokens_match": r1_output_tokens_match,
            "response_headers_ms": response_headers_ms,
            "client_ttft_ms": client_ttft_ms,
            "client_e2e_ms": client_e2e_ms,
            "client_minus_server_ttft_ms": (
                client_ttft_ms
                - float(server["server_request_ttft_ms"])
            ),
            "client_minus_server_e2e_ms": (
                client_e2e_ms
                - float(server["server_e2e_ms"])
            ),
            "client_mean_chunk_itl_ms": mean_or_none(chunk_itls),
            "client_p50_chunk_itl_ms": percentile(chunk_itls, 0.50),
            "client_p95_chunk_itl_ms": percentile(chunk_itls, 0.95),
            "server_queue_ms": server["queue_ms"],
            "server_preprocess_ms": server["preprocess_ms"],
            "server_h2d_ms": server["h2d_ms"],
            "server_model_ttft_ms": server["model_ttft_ms"],
            "server_request_ttft_ms": server["server_request_ttft_ms"],
            "server_mean_tpot_ms": server["mean_tpot_ms"],
            "server_p95_tpot_ms": server["p95_tpot_ms"],
            "server_mean_stream_itl_ms": server[
                "mean_server_stream_itl_ms"
            ],
            "server_e2e_ms": server["server_e2e_ms"],
            "peak_allocated_gib": server["peak_allocated_gib"],
            "peak_reserved_gib": server["peak_reserved_gib"],
            "sampling_seed": server["sampling_seed"],
            "error_type": "",
            "error_message": "",
        })

        if int(usage["prompt_tokens"]) != int(row["input_tokens"]):
            raise RuntimeError(
                f"Prompt token mismatch for {row['request_id']}: "
                f"frozen={row['input_tokens']} "
                f"server={usage['prompt_tokens']}"
            )

        return result, chunk_rows

    except Exception as exc:
        result.update({
            "response_headers_ms": response_headers_ms,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        })

        return result, []


def warmup(
    client: httpx.Client,
    url: str,
    workload: Dict[str, Dict[str, Any]],
):
    print("\nAPI warm-up: one representative request/category...")

    for index, row in enumerate(
        representative_warmup_requests(workload),
        start=1,
    ):
        result, _ = run_stream_request(
            client=client,
            url=url,
            row=row,
            r1_ref=None,
            max_tokens_override=min(
                int(row["max_new_tokens"]),
                32,
            ),
            capture_chunks=False,
        )

        if result["status"] != "ok":
            raise RuntimeError(
                f"Warm-up failed for {row['workload_category']}: "
                f"{result['error_type']}: "
                f"{result['error_message']}"
            )

        print(
            f"  [{index}/6] "
            f"{row['workload_category']:20s} "
            f"input={int(row['input_tokens']):4d} "
            f"out={int(result['output_tokens']):3d} "
            f"client_ttft={float(result['client_ttft_ms']):7.2f}ms"
        )

    print("API warm-up PASS.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    args = parser.parse_args()

    workload = load_workload()
    selected_ids = load_execution_order(args.mode)
    r1_reference = load_r1_flash_reference()
    paths = result_paths(args.mode)

    if paths["requests"].exists() and not args.resume:
        raise RuntimeError(
            f"Result file already exists: {paths['requests']}\n"
            "Use --resume or intentionally move/delete the old S1 result."
        )

    completed = (
        completed_request_ids(paths["requests"])
        if args.resume
        else set()
    )

    remaining_ids = [
        request_id
        for request_id in selected_ids
        if request_id not in completed
    ]

    url = (
        args.base_url.rstrip("/")
        + "/v1/chat/completions"
    )

    print("=" * 88)
    print("S1 — STREAMING API BENCHMARK CLIENT")
    print("=" * 88)
    print(f"Mode:          {args.mode}")
    print(f"URL:           {url}")
    print(f"Requests:      {len(selected_ids)}")
    print(f"Completed:     {len(completed)}")
    print(f"Remaining:     {len(remaining_ids)}")
    print(f"R1 reference:  {'yes' if r1_reference else 'no'}")

    timeout = httpx.Timeout(
        connect=10.0,
        read=None,
        write=30.0,
        pool=None,
    )

    limits = httpx.Limits(
        max_keepalive_connections=1,
        max_connections=1,
    )

    with httpx.Client(
        timeout=timeout,
        limits=limits,
        http2=False,
    ) as client:
        health = client.get(
            args.base_url.rstrip("/")
            + "/health"
        )
        health.raise_for_status()

        print(f"Health:        {health.json()}")

        warmup(
            client,
            url,
            workload,
        )

        metadata = {
            "protocol_version": 1,
            "mode": args.mode,
            "base_url": args.base_url,
            "model": MODEL_ID,
            "transport": "HTTP/1.1 + SSE",
            "connection_reuse": True,
            "client_concurrency": 1,
            "selected_request_count": len(selected_ids),
            "already_completed_on_resume": len(completed),
            "remaining_at_start": len(remaining_ids),
            "r1_execution_order_file": str(
                EXECUTION_ORDER_FILE
            ),
            "r1_flash_reference_file": (
                str(R1_FLASH_REQUESTS)
                if r1_reference
                else None
            ),
            "started_at_epoch_s": time.time(),
            "status": "running",
        }

        paths["metadata"].write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        run_start = time.perf_counter()

        for local_index, request_id in enumerate(
            remaining_ids,
            start=1,
        ):
            row = workload[request_id]
            absolute_position = (
                selected_ids.index(request_id) + 1
            )

            print(
                f"  [{absolute_position:4d}/{len(selected_ids):4d}] "
                f"{request_id} "
                f"{row['workload_category']:20s} "
                f"in={int(row['input_tokens']):4d} "
                f"cap={int(row['max_new_tokens']):4d}",
                end="",
                flush=True,
            )

            result, chunks = run_stream_request(
                client=client,
                url=url,
                row=row,
                r1_ref=r1_reference.get(request_id),
                capture_chunks=True,
            )

            append_csv_row(
                paths["requests"],
                REQUEST_FIELDS,
                result,
            )

            if result["status"] == "ok":
                append_chunk_rows(
                    paths["chunks"],
                    chunks,
                )

                print(
                    f" -> out={int(result['output_tokens']):4d} "
                    f"clientTTFT={float(result['client_ttft_ms']):8.2f}ms "
                    f"serverTTFT={float(result['server_request_ttft_ms']):8.2f}ms "
                    f"E2E={float(result['client_e2e_ms'])/1000.0:7.2f}s "
                    f"[{result['finish_reason']}]"
                )
            else:
                metadata.update({
                    "status": "failed",
                    "failure_request_id": request_id,
                    "failure_error_type": result["error_type"],
                    "failure_error_message": result["error_message"],
                    "elapsed_seconds": (
                        time.perf_counter() - run_start
                    ),
                })

                paths["metadata"].write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    f" -> ERROR {result['error_type']}: "
                    f"{result['error_message']}"
                )

                raise RuntimeError(
                    "S1 stopped after request failure. "
                    "Investigate, then rerun with --resume."
                )

        elapsed = time.perf_counter() - run_start

    final_completed = completed_request_ids(
        paths["requests"]
    )

    metadata.update({
        "status": "complete",
        "finished_at_epoch_s": time.time(),
        "elapsed_seconds_this_invocation": elapsed,
        "completed_request_count": len(
            final_completed
        ),
        "expected_request_count": len(
            selected_ids
        ),
    })

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("S1 RUN COMPLETE")
    print("=" * 88)
    print(
        f"Completed requests: "
        f"{len(final_completed)}/{len(selected_ids)}"
    )
    print(
        f"This invocation:    {elapsed/60.0:.2f} min"
    )
    print(f"Request results:    {paths['requests']}")
    print(f"Chunk trace:        {paths['chunks']}")
    print(f"Run metadata:       {paths['metadata']}")
    print()
    print(
        "Do not run S1 Full until the 60-request pilot has been audited "
        "against the R1 Flash baseline."
    )


if __name__ == "__main__":
    main()
