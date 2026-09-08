#!/usr/bin/env python3
"""S8 causal prefill/decode interference client.

Maintains a closed-loop population of long-running decode requests, then either
injects one or more long-prefill requests or observes an equally timed control
window. All timestamps use one client-side monotonic clock.

Raw outputs are intentionally written below results/s8/raw and excluded from
version control. The analyzer consumes trial.json, requests.jsonl,
content_events.csv.gz, and metrics.jsonl.
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
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8002"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def prometheus_values(text: str, names: Sequence[str]) -> List[float]:
    for name in names:
        pattern = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$"
        values = [
            float(value)
            for value in re.findall(pattern, text, flags=re.MULTILINE)
        ]
        if values:
            return values
    return []


def prometheus_sum(
    text: str,
    names: Sequence[str],
    default: float = 0.0,
) -> float:
    values = prometheus_values(text, names)
    return sum(values) if values else default


async def fetch_metrics(client: httpx.AsyncClient, base_url: str) -> Dict[str, float]:
    response = await client.get(f"{base_url}/metrics")
    response.raise_for_status()
    text = response.text
    return {
        "running": prometheus_sum(text, ["vllm:num_requests_running"]),
        "waiting": prometheus_sum(text, ["vllm:num_requests_waiting"]),
        "kv_usage": prometheus_sum(text, ["vllm:kv_cache_usage_perc"]),
        "preemptions": prometheus_sum(
            text,
            ["vllm:num_preemptions_total", "vllm:num_preemptions"],
        ),
        "prompt_tokens_total": prometheus_sum(
            text,
            ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
        ),
        "generation_tokens_total": prometheus_sum(
            text,
            ["vllm:generation_tokens_total", "vllm:generation_tokens"],
        ),
    }


async def wait_idle(
    client: httpx.AsyncClient,
    base_url: str,
    timeout_s: float,
) -> Dict[str, float]:
    deadline = time.perf_counter() + timeout_s
    last: Dict[str, float] = {}
    while time.perf_counter() < deadline:
        last = await fetch_metrics(client, base_url)
        if (
            last["running"] == 0
            and last["waiting"] == 0
            and last["kv_usage"] <= 1e-9
        ):
            return last
        await asyncio.sleep(0.2)
    raise RuntimeError(f"server did not become idle; last metrics={last}")


def build_synthetic_body(seed: int, paragraphs: int = 2200) -> str:
    rng = random.Random(seed)
    subjects = [
        "satellite", "battery", "compiler", "database", "sensor",
        "railway", "hospital", "warehouse", "network", "turbine",
        "robot", "observatory", "datacenter", "laboratory", "vehicle",
    ]
    actions = [
        "recorded", "validated", "transmitted", "estimated", "compared",
        "recalibrated", "scheduled", "inspected", "aggregated", "classified",
        "measured", "reconstructed", "synchronized", "processed", "reported",
    ]
    properties = [
        "latency", "temperature", "throughput", "pressure", "utilization",
        "voltage", "frequency", "capacity", "bandwidth", "accuracy",
        "load", "duration", "distance", "variance", "memory usage",
    ]
    lines = []
    for index in range(paragraphs):
        lines.append(
            f"Record {index:05d}: The {rng.choice(subjects)} "
            f"{rng.choice(actions)} {rng.choice(properties)} value "
            f"{rng.randint(10, 9999)} and {rng.choice(properties)} value "
            f"{rng.randint(10, 9999)}. Observation code "
            f"{rng.randint(10, 9999)} was retained for experiment item "
            f"{index:05d}."
        )
    return "\n".join(lines)


def exact_prompt(
    body_ids: Sequence[int],
    tokenizer: Any,
    target_tokens: int,
    role: str,
    request_index: int,
    seed: int,
) -> List[int]:
    suffix = tokenizer.encode(
        "\n\nFinal task: summarize the records concisely. "
        f"Request role {role}; request {request_index}; seed {seed}.",
        add_special_tokens=False,
    )
    body_needed = target_tokens - len(suffix)
    if body_needed <= 0:
        raise ValueError(
            f"target length {target_tokens} is too short for suffix "
            f"of {len(suffix)} tokens"
        )
    if body_needed > len(body_ids):
        raise ValueError(
            f"synthetic body has {len(body_ids)} tokens; need {body_needed}"
        )
    prompt = list(body_ids[:body_needed]) + list(suffix)
    if len(prompt) != target_tokens:
        raise AssertionError("exact prompt construction failed")
    return prompt


async def poll_metrics(
    client: httpx.AsyncClient,
    base_url: str,
    trial_t0: float,
    interval_s: float,
    stop_event: asyncio.Event,
    samples: List[Dict[str, Any]],
) -> None:
    while not stop_event.is_set():
        sample: Dict[str, Any] = {
            "t_s": time.perf_counter() - trial_t0,
        }
        try:
            sample.update(await fetch_metrics(client, base_url))
        except Exception as exc:
            sample["metrics_error"] = repr(exc)
        samples.append(sample)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def streaming_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt_ids: Sequence[int],
    output_tokens: int,
    request_id: str,
    role: str,
    sampling_seed: int,
    trial_t0: float,
    traces: List[Dict[str, Any]],
    content_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    start = time.perf_counter()
    trace: Dict[str, Any] = {
        "request_id": request_id,
        "role": role,
        "status": "running",
        "target_input_tokens": len(prompt_ids),
        "target_output_tokens": output_tokens,
        "start_t_s": start - trial_t0,
        "first_event_t_s": None,
        "first_content_t_s": None,
        "end_t_s": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "content_event_count": 0,
        "content_event_minus_output_tokens": None,
        "mean_content_event_gap_ms": None,
        "p95_content_event_gap_ms": None,
        "finish_reason": None,
        "error": None,
    }
    traces.append(trace)
    event_times: List[float] = []
    usage: Optional[Dict[str, Any]] = None
    payload = {
        "model": model,
        "prompt": list(prompt_ids),
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "seed": sampling_seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{body.decode(errors='replace')}"
                )

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                now = time.perf_counter()
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                if trace["first_event_t_s"] is None:
                    trace["first_event_t_s"] = now - trial_t0
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                text_delta = choice.get("text") or ""
                if text_delta:
                    event_t = now - trial_t0
                    event_times.append(event_t)
                    content_events.append(
                        {
                            "t_s": event_t,
                            "role": role,
                            "request_id": request_id,
                            "content_event_index": len(event_times),
                            "content_chars": len(text_delta),
                        }
                    )
                    if trace["first_content_t_s"] is None:
                        trace["first_content_t_s"] = event_t
                if choice.get("finish_reason") is not None:
                    trace["finish_reason"] = choice["finish_reason"]

        trace["status"] = "ok"
        if usage is None:
            raise RuntimeError("stream completed without usage information")
        trace["actual_input_tokens"] = usage.get("prompt_tokens")
        trace["actual_output_tokens"] = usage.get("completion_tokens")
        if trace["actual_input_tokens"] != len(prompt_ids):
            raise RuntimeError(
                f"input token mismatch: expected {len(prompt_ids)}, "
                f"got {trace['actual_input_tokens']}"
            )
        if trace["actual_output_tokens"] != output_tokens:
            raise RuntimeError(
                f"output token mismatch: expected {output_tokens}, "
                f"got {trace['actual_output_tokens']}"
            )
    except asyncio.CancelledError:
        trace["status"] = "cancelled"
        raise
    except Exception as exc:
        trace["status"] = "error"
        trace["error"] = repr(exc)
    finally:
        trace["end_t_s"] = time.perf_counter() - trial_t0
        trace["content_event_count"] = len(event_times)
        if trace["actual_output_tokens"] is not None:
            trace["content_event_minus_output_tokens"] = (
                len(event_times) - int(trace["actual_output_tokens"])
            )
        gaps_ms = [
            (right - left) * 1000.0
            for left, right in zip(
                event_times[:-1], event_times[1:], strict=True
            )
        ]
        if gaps_ms:
            trace["mean_content_event_gap_ms"] = statistics.mean(gaps_ms)
            trace["p95_content_event_gap_ms"] = percentile(gaps_ms, 0.95)
    return trace


async def background_worker(
    worker_index: int,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    body_ids: Sequence[int],
    tokenizer: Any,
    trial_t0: float,
    stop_event: asyncio.Event,
    traces: List[Dict[str, Any]],
    content_events: List[Dict[str, Any]],
) -> None:
    generation = 0
    while not stop_event.is_set():
        request_index = worker_index * 100000 + generation
        prompt_ids = exact_prompt(
            body_ids,
            tokenizer,
            args.background_input_tokens,
            "background",
            request_index,
            args.seed,
        )
        result = await streaming_request(
            client=client,
            base_url=args.base_url,
            model=args.model,
            prompt_ids=prompt_ids,
            output_tokens=args.background_output_tokens,
            request_id=f"bg-{worker_index:03d}-{generation:03d}",
            role="background",
            sampling_seed=args.seed + request_index,
            trial_t0=trial_t0,
            traces=traces,
            content_events=content_events,
        )
        if result["status"] != "ok":
            raise RuntimeError(
                f"background request {result['request_id']} failed: "
                f"{result['error']}"
            )
        generation += 1


def write_outputs(
    out_dir: Path,
    metadata: Dict[str, Any],
    traces: Sequence[Dict[str, Any]],
    content_events: Sequence[Dict[str, Any]],
    metric_samples: Sequence[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trial.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")
    with gzip.open(
        out_dir / "content_events.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "t_s", "role", "request_id", "content_event_index", "content_chars"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(content_events, key=lambda row: float(row["t_s"])))
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for sample in metric_samples:
            handle.write(json.dumps(sample) + "\n")


async def execute(args: argparse.Namespace) -> Path:
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path("results/s8/raw/interference") / run_stamp
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=os.environ.get("HF_HOME", "/workspace/hf-cache"),
    )
    body_ids = tokenizer.encode(
        build_synthetic_body(args.seed), add_special_tokens=False
    )
    max_input = max(args.background_input_tokens, args.interferer_input_tokens)
    if len(body_ids) < max_input:
        raise RuntimeError(
            f"synthetic corpus has {len(body_ids)} tokens; need {max_input}"
        )

    timeout = httpx.Timeout(connect=30.0, read=None, write=180.0, pool=30.0)
    limits = httpx.Limits(
        max_connections=args.background_concurrency + args.interferer_count + 8,
        max_keepalive_connections=args.background_concurrency + 4,
    )
    traces: List[Dict[str, Any]] = []
    content_events: List[Dict[str, Any]] = []
    metric_samples: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "stage": "S8-interference",
        "status": "running",
        "run_stamp": run_stamp,
        "config_label": args.config_label,
        "replicate": args.replicate,
        "trial_kind": args.trial_kind,
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "background_concurrency": args.background_concurrency,
        "background_input_tokens": args.background_input_tokens,
        "background_output_tokens": args.background_output_tokens,
        "interferer_input_tokens": args.interferer_input_tokens,
        "interferer_output_tokens": args.interferer_output_tokens,
        "interferer_count": args.interferer_count,
        "warmup_seconds": args.warmup_seconds,
        "control_impact_seconds": args.control_impact_seconds,
        "recovery_seconds": args.recovery_seconds,
        "metrics_interval_seconds": args.metrics_interval,
        "injection_t_s": None,
        "impact_end_t_s": None,
        "trial_end_t_s": None,
        "error": None,
    }

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(f"{args.base_url}/health")
        health.raise_for_status()
        before = await wait_idle(client, args.base_url, args.idle_timeout)
        metadata["metrics_before"] = before
        trial_t0 = time.perf_counter()
        metric_stop = asyncio.Event()
        background_stop = asyncio.Event()
        metrics_task = asyncio.create_task(
            poll_metrics(
                client,
                args.base_url,
                trial_t0,
                args.metrics_interval,
                metric_stop,
                metric_samples,
            )
        )
        background_tasks = [
            asyncio.create_task(
                background_worker(
                    worker_index,
                    client,
                    args,
                    body_ids,
                    tokenizer,
                    trial_t0,
                    background_stop,
                    traces,
                    content_events,
                )
            )
            for worker_index in range(args.background_concurrency)
        ]

        try:
            await asyncio.sleep(args.warmup_seconds)
            injection_t_s = time.perf_counter() - trial_t0
            metadata["injection_t_s"] = injection_t_s

            if args.trial_kind == "inject":
                injection_tasks = []
                for index in range(args.interferer_count):
                    prompt_ids = exact_prompt(
                        body_ids,
                        tokenizer,
                        args.interferer_input_tokens,
                        "interferer",
                        index,
                        args.seed,
                    )
                    injection_tasks.append(
                        asyncio.create_task(
                            streaming_request(
                                client=client,
                                base_url=args.base_url,
                                model=args.model,
                                prompt_ids=prompt_ids,
                                output_tokens=args.interferer_output_tokens,
                                request_id=f"long-{index:03d}",
                                role="interferer",
                                sampling_seed=args.seed + 900000 + index,
                                trial_t0=trial_t0,
                                traces=traces,
                                content_events=content_events,
                            )
                        )
                    )
                injection_results = await asyncio.wait_for(
                    asyncio.gather(*injection_tasks),
                    timeout=args.interferer_timeout,
                )
                failures = [
                    row for row in injection_results if row["status"] != "ok"
                ]
                if failures:
                    raise RuntimeError(f"interferer request failures: {failures}")
                first_content_times = [
                    float(row["first_content_t_s"])
                    for row in injection_results
                    if row["first_content_t_s"] is not None
                ]
                if len(first_content_times) != args.interferer_count:
                    raise RuntimeError(
                        "an interferer returned no visible content event"
                    )
                metadata["impact_end_t_s"] = max(first_content_times)
            else:
                await asyncio.sleep(args.control_impact_seconds)
                metadata["impact_end_t_s"] = time.perf_counter() - trial_t0

            await asyncio.sleep(args.recovery_seconds)
            metadata["status"] = "complete"
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = repr(exc)
            raise
        finally:
            metadata["trial_end_t_s"] = time.perf_counter() - trial_t0
            background_stop.set()
            for task in background_tasks:
                task.cancel()
            await asyncio.gather(*background_tasks, return_exceptions=True)
            metric_stop.set()
            await metrics_task
            write_outputs(out_dir, metadata, traces, content_events, metric_samples)

        after = await wait_idle(client, args.base_url, args.idle_timeout)
        metadata["metrics_after"] = after
        metadata["preemptions_delta"] = (
            after["preemptions"] - before["preemptions"]
        )
        write_outputs(out_dir, metadata, traces, content_events, metric_samples)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trial-kind", choices=["control", "inject"], required=True)
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--background-concurrency", type=int, default=32)
    parser.add_argument("--background-input-tokens", type=int, default=256)
    parser.add_argument("--background-output-tokens", type=int, default=8192)
    parser.add_argument("--interferer-input-tokens", type=int, default=16384)
    parser.add_argument("--interferer-output-tokens", type=int, default=16)
    parser.add_argument("--interferer-count", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=15.0)
    parser.add_argument("--control-impact-seconds", type=float, default=2.0)
    parser.add_argument("--recovery-seconds", type=float, default=15.0)
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    parser.add_argument("--interferer-timeout", type=float, default=180.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    positive_ints = {
        "background_concurrency": args.background_concurrency,
        "background_input_tokens": args.background_input_tokens,
        "background_output_tokens": args.background_output_tokens,
        "interferer_input_tokens": args.interferer_input_tokens,
        "interferer_output_tokens": args.interferer_output_tokens,
        "interferer_count": args.interferer_count,
        "replicate": args.replicate,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in [
        "warmup_seconds", "control_impact_seconds", "metrics_interval",
        "idle_timeout", "interferer_timeout",
    ]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.recovery_seconds < 0:
        parser.error("--recovery-seconds must be non-negative")
    if (
        args.background_input_tokens + args.background_output_tokens
        > args.max_model_len
    ):
        parser.error("background input + output exceeds max model length")
    if (
        args.interferer_input_tokens + args.interferer_output_tokens
        > args.max_model_len
    ):
        parser.error("interferer input + output exceeds max model length")
    return args


def main() -> None:
    args = parse_args()
    out_dir = asyncio.run(execute(args))
    print(f"S8 trial complete: {out_dir}")


if __name__ == "__main__":
    main()
