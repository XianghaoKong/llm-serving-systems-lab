#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8002"


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def metric_values(text, metric_name):
    pattern = (
        rf"^{re.escape(metric_name)}"
        rf"(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$"
    )
    return [
        float(x)
        for x in re.findall(pattern, text, flags=re.MULTILINE)
    ]


def metric_value(text, metric_name, default=0.0):
    vals = metric_values(text, metric_name)
    return vals[0] if vals else default


def build_synthetic_body(seed=2026, paragraphs=800):
    """
    Deterministic synthetic document with varied, unique records.
    Avoids simply repeating one sentence thousands of times.
    """
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
    locations = [
        "north sector", "south sector", "central facility", "coastal station",
        "mountain site", "test chamber", "operations room", "remote node",
        "simulation zone", "control cluster",
    ]

    lines = []
    for i in range(paragraphs):
        subject = rng.choice(subjects)
        action = rng.choice(actions)
        prop1 = rng.choice(properties)
        prop2 = rng.choice(properties)
        location = rng.choice(locations)

        a = rng.randint(10, 9999)
        b = rng.randint(10, 9999)
        c = rng.randint(10, 9999)

        lines.append(
            f"Record {i:05d}: The {subject} at the {location} {action} "
            f"{prop1} value {a} and {prop2} value {b}. "
            f"Observation code {c} was retained for later comparison. "
            f"This record is uniquely indexed as experiment item {i:05d}."
        )

    return "\n".join(lines)


async def fetch_metrics(client, base_url):
    r = await client.get(f"{base_url}/metrics")
    r.raise_for_status()
    text = r.text

    return {
        "running": metric_value(text, "vllm:num_requests_running"),
        "waiting": metric_value(text, "vllm:num_requests_waiting"),
        "kv_usage": metric_value(text, "vllm:kv_cache_usage_perc"),
        "preemptions": metric_value(text, "vllm:num_preemptions_total"),
        "prompt_tokens_total": metric_value(
            text, "vllm:prompt_tokens_total"
        ),
        "generation_tokens_total": metric_value(
            text, "vllm:generation_tokens_total"
        ),
    }


async def wait_idle(client, base_url, timeout_s=60):
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        m = await fetch_metrics(client, base_url)
        if (
            m["running"] == 0
            and m["waiting"] == 0
            and m["kv_usage"] <= 1e-9
        ):
            return m
        await asyncio.sleep(0.2)

    raise RuntimeError("Server did not return to idle state within timeout.")


async def poll_metrics(client, base_url, stop_event, samples, interval):
    while not stop_event.is_set():
        try:
            m = await fetch_metrics(client, base_url)
            m["t_monotonic"] = time.perf_counter()
            samples.append(m)
        except Exception as exc:
            samples.append(
                {
                    "metrics_error": repr(exc),
                    "t_monotonic": time.perf_counter(),
                }
            )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_request(
    client,
    base_url,
    model,
    prompt_ids,
    target_input_tokens,
    output_tokens,
    poll_interval,
    run_label,
):
    await wait_idle(client, base_url)

    before = await fetch_metrics(client, base_url)

    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "seed": 2026,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    metric_samples = []
    stop_event = asyncio.Event()

    poll_task = asyncio.create_task(
        poll_metrics(
            client,
            base_url,
            stop_event,
            metric_samples,
            poll_interval,
        )
    )

    start = time.perf_counter()
    first_event_s = None
    first_content_s = None
    content_event_times = []
    usage = None
    finish_reason = None

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

                if first_event_s is None:
                    first_event_s = now - start

                obj = json.loads(data)

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get("choices") or []
                if choices:
                    choice = choices[0]
                    text_delta = choice.get("text") or ""

                    if text_delta:
                        t = now - start
                        content_event_times.append(t)
                        if first_content_s is None:
                            first_content_s = t

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

    finally:
        stop_event.set()
        await poll_task

    end = time.perf_counter()

    after = await fetch_metrics(client, base_url)

    if usage is None:
        raise RuntimeError(
            "Streaming response did not contain usage information."
        )

    actual_prompt = usage.get("prompt_tokens")
    actual_output = usage.get("completion_tokens")

    if actual_prompt != target_input_tokens:
        raise RuntimeError(
            f"Prompt token mismatch: target={target_input_tokens}, "
            f"server={actual_prompt}"
        )

    if actual_output != output_tokens:
        raise RuntimeError(
            f"Output token mismatch: target={output_tokens}, "
            f"server={actual_output}"
        )

    itls = [
        b - a
        for a, b in zip(
            content_event_times[:-1],
            content_event_times[1:],
        )
    ]

    good_samples = [
        x for x in metric_samples
        if "metrics_error" not in x
    ]

    max_kv = max(
        (x["kv_usage"] for x in good_samples),
        default=0.0,
    )
    max_running = max(
        (x["running"] for x in good_samples),
        default=0.0,
    )
    max_waiting = max(
        (x["waiting"] for x in good_samples),
        default=0.0,
    )

    result = {
        "run_label": run_label,
        "target_input_tokens": target_input_tokens,
        "actual_input_tokens": actual_prompt,
        "target_output_tokens": output_tokens,
        "actual_output_tokens": actual_output,
        "first_event_ttft_ms": (
            first_event_s * 1000
            if first_event_s is not None else None
        ),
        "visible_ttft_ms": (
            first_content_s * 1000
            if first_content_s is not None else None
        ),
        "e2e_ms": (end - start) * 1000,
        "content_event_count": len(content_event_times),
        "content_event_itl_mean_ms": (
            statistics.mean(itls) * 1000 if itls else None
        ),
        "content_event_itl_p50_ms": (
            percentile(itls, 0.50) * 1000 if itls else None
        ),
        "content_event_itl_p95_ms": (
            percentile(itls, 0.95) * 1000 if itls else None
        ),
        "max_kv_cache_usage": max_kv,
        "max_running": max_running,
        "max_waiting": max_waiting,
        "preemptions_delta": (
            after["preemptions"] - before["preemptions"]
        ),
        "finish_reason": finish_reason,
        "metrics_samples": len(good_samples),
    }

    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32640],
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()

    max_total = max(args.targets) + args.output_tokens
    if max_total > 32768:
        raise ValueError(
            f"input + output exceeds 32768: {max_total}"
        )

    run_id = time.strftime(
        "%Y%m%dT%H%M%SZ",
        time.gmtime(),
    )

    out_dir = Path(
        f"results/s5/raw/context_scaling/{run_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=os.environ.get(
            "HF_HOME",
            "/workspace/hf-cache",
        ),
    )

    print("Building deterministic synthetic document...")
    body_text = build_synthetic_body()
    body_ids = tokenizer.encode(
        body_text,
        add_special_tokens=False,
    )

    query_text = (
        "\n\nFinal task: Review the preceding records and provide "
        "a concise technical summary of the observed information."
    )
    query_ids = tokenizer.encode(
        query_text,
        add_special_tokens=False,
    )

    required_body_tokens = max(args.targets) - len(query_ids)

    if len(body_ids) < required_body_tokens:
        raise RuntimeError(
            f"Synthetic body too short: have {len(body_ids)} tokens, "
            f"need {required_body_tokens}"
        )

    prompts = {}

    for target in args.targets:
        n_body = target - len(query_ids)
        if n_body <= 0:
            raise RuntimeError(
                f"Target {target} is too short for query suffix."
            )

        prompt_ids = body_ids[:n_body] + query_ids

        assert len(prompt_ids) == target
        prompts[target] = prompt_ids

    config = {
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "targets": args.targets,
        "output_tokens": args.output_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "poll_interval_s": args.poll_interval,
        "prompt_mode": "exact token-id list",
        "synthetic_seed": 2026,
        "prefix_cache_expected": False,
    }

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    timeout = httpx.Timeout(
        connect=30.0,
        read=None,
        write=60.0,
        pool=30.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
    ) as client:

        health = await client.get(
            f"{args.base_url}/health"
        )
        health.raise_for_status()

        baseline_metrics = await client.get(
            f"{args.base_url}/metrics"
        )
        baseline_metrics.raise_for_status()

        (out_dir / "metrics_baseline.txt").write_text(
            baseline_metrics.text,
            encoding="utf-8",
        )

        all_results = []

        for target in args.targets:
            print(
                f"\n=== target input {target} tokens ==="
            )

            for w in range(args.warmups):
                label = f"{target}_warmup_{w + 1}"
                print(f"Warmup: {label}")

                result = await run_request(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    prompt_ids=prompts[target],
                    target_input_tokens=target,
                    output_tokens=args.output_tokens,
                    poll_interval=args.poll_interval,
                    run_label=label,
                )

                print(
                    f"  TTFT={result['visible_ttft_ms']:.2f} ms "
                    f"E2E={result['e2e_ms']:.2f} ms "
                    f"KVmax={result['max_kv_cache_usage']:.4f}"
                )

            for r in range(args.repeats):
                label = f"{target}_measured_{r + 1}"
                print(f"Measured: {label}")

                result = await run_request(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    prompt_ids=prompts[target],
                    target_input_tokens=target,
                    output_tokens=args.output_tokens,
                    poll_interval=args.poll_interval,
                    run_label=label,
                )

                all_results.append(result)

                print(
                    f"  input={result['actual_input_tokens']} "
                    f"output={result['actual_output_tokens']} "
                    f"TTFT={result['visible_ttft_ms']:.2f} ms "
                    f"E2E={result['e2e_ms']:.2f} ms "
                    f"ITLmean={result['content_event_itl_mean_ms']:.3f} ms "
                    f"KVmax={result['max_kv_cache_usage']:.4f} "
                    f"running={result['max_running']:.0f} "
                    f"waiting={result['max_waiting']:.0f}"
                )

                with (
                    out_dir / "requests.jsonl"
                ).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(result) + "\n"
                    )

        summary_rows = []

        for target in args.targets:
            rows = [
                r for r in all_results
                if r["target_input_tokens"] == target
            ]

            ttft = [
                r["visible_ttft_ms"] for r in rows
                if r["visible_ttft_ms"] is not None
            ]
            e2e = [r["e2e_ms"] for r in rows]
            itl = [
                r["content_event_itl_mean_ms"]
                for r in rows
                if r["content_event_itl_mean_ms"] is not None
            ]

            summary_rows.append(
                {
                    "input_tokens": target,
                    "output_tokens": args.output_tokens,
                    "n": len(rows),
                    "ttft_p50_ms": percentile(ttft, 0.50),
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "e2e_p50_ms": percentile(e2e, 0.50),
                    "e2e_p95_ms": percentile(e2e, 0.95),
                    "mean_content_event_itl_p50_ms": (
                        percentile(itl, 0.50)
                    ),
                    "max_observed_kv_usage": max(
                        r["max_kv_cache_usage"]
                        for r in rows
                    ),
                    "max_running": max(
                        r["max_running"] for r in rows
                    ),
                    "max_waiting": max(
                        r["max_waiting"] for r in rows
                    ),
                    "total_preemptions": sum(
                        r["preemptions_delta"]
                        for r in rows
                    ),
                }
            )

        summary_path = out_dir / "summary.csv"

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=summary_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n=== S5-A summary ===")
        for row in summary_rows:
            print(
                f"{row['input_tokens']:>5} tokens | "
                f"TTFT p50 {row['ttft_p50_ms']:.2f} ms | "
                f"E2E p50 {row['e2e_p50_ms']:.2f} ms | "
                f"ITL p50 {row['mean_content_event_itl_p50_ms']:.3f} ms | "
                f"KV max {row['max_observed_kv_usage']:.4f}"
            )

        print(f"\nRaw results: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
