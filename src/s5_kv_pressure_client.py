#!/usr/bin/env python3

import argparse
import asyncio
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


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://127.0.0.1:8002"


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


def metric_values(text, name):
    pattern = (
        rf"^{re.escape(name)}"
        rf"(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$"
    )
    return [
        float(x)
        for x in re.findall(pattern, text, flags=re.MULTILINE)
    ]


def metric_value(text, name, default=0.0):
    vals = metric_values(text, name)
    return vals[0] if vals else default


def metric_value_with_label(
    text, name, label_key, label_value, default=0.0
):
    pattern = (
        rf"^{re.escape(name)}"
        rf"\{{[^}}]*{re.escape(label_key)}="
        rf'\"{re.escape(label_value)}\"[^}}]*\}}'
        rf"\s+([-+0-9.eE]+)$"
    )
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else default


async def fetch_metrics(client, base_url):
    r = await client.get(f"{base_url}/metrics")
    r.raise_for_status()
    text = r.text

    return {
        "running": metric_value(
            text, "vllm:num_requests_running"
        ),
        "waiting": metric_value(
            text, "vllm:num_requests_waiting"
        ),
        "waiting_capacity": metric_value_with_label(
            text,
            "vllm:num_requests_waiting_by_reason",
            "reason",
            "capacity",
        ),
        "kv_usage": metric_value(
            text, "vllm:kv_cache_usage_perc"
        ),
        "preemptions": metric_value(
            text, "vllm:num_preemptions_total"
        ),
        "prompt_tokens_total": metric_value(
            text, "vllm:prompt_tokens_total"
        ),
        "generation_tokens_total": metric_value(
            text, "vllm:generation_tokens_total"
        ),
    }


async def wait_idle(client, base_url, timeout_s=120):
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

    raise RuntimeError(
        "Server failed to return to idle state."
    )


async def poll_metrics(
    client,
    base_url,
    stop_event,
    samples,
    interval_s,
    t0,
):
    while not stop_event.is_set():
        try:
            m = await fetch_metrics(client, base_url)
            m["elapsed_s"] = time.perf_counter() - t0
            samples.append(m)
        except Exception as exc:
            samples.append({
                "elapsed_s": time.perf_counter() - t0,
                "metrics_error": repr(exc),
            })

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=interval_s,
            )
        except asyncio.TimeoutError:
            pass


async def run_one_request(
    client,
    base_url,
    model,
    prompt_ids,
    output_tokens,
    request_index,
    start_event,
):
    await start_event.wait()

    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "seed": 2026 + request_index,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    start = time.perf_counter()

    first_event = None
    first_content = None
    content_times = []

    usage = None
    finish_reason = None

    async with client.stream(
        "POST",
        f"{base_url}/v1/completions",
        json=payload,
    ) as response:

        if response.status_code != 200:
            body = await response.aread()

            raise RuntimeError(
                f"request {request_index}: "
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

            if first_event is None:
                first_event = now - start

            obj = json.loads(data)

            if obj.get("usage"):
                usage = obj["usage"]

            choices = obj.get("choices") or []

            if choices:
                choice = choices[0]
                text_delta = choice.get("text") or ""

                if text_delta:
                    t = now - start
                    content_times.append(t)

                    if first_content is None:
                        first_content = t

                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

    end = time.perf_counter()

    if usage is None:
        raise RuntimeError(
            f"request {request_index}: no usage returned"
        )

    itls = [
        b - a
        for a, b in zip(
            content_times[:-1],
            content_times[1:],
        )
    ]

    return {
        "request_index": request_index,
        "actual_input_tokens": usage["prompt_tokens"],
        "actual_output_tokens": usage["completion_tokens"],
        "first_event_ttft_ms": (
            first_event * 1000
            if first_event is not None
            else None
        ),
        "visible_ttft_ms": (
            first_content * 1000
            if first_content is not None
            else None
        ),
        "e2e_ms": (end - start) * 1000,
        "content_event_count": len(content_times),
        "content_event_itl_mean_ms": (
            statistics.mean(itls) * 1000
            if itls else None
        ),
        "content_event_itl_p95_ms": (
            percentile(itls, 0.95) * 1000
            if itls else None
        ),
        "finish_reason": finish_reason,
    }


def build_synthetic_body(seed=2026, paragraphs=1000):
    rng = random.Random(seed)

    subjects = [
        "satellite", "battery", "compiler",
        "database", "sensor", "railway",
        "hospital", "warehouse", "network",
        "turbine", "robot", "datacenter",
    ]

    actions = [
        "recorded", "validated", "transmitted",
        "estimated", "compared", "inspected",
        "aggregated", "classified", "measured",
    ]

    props = [
        "latency", "temperature", "throughput",
        "pressure", "utilization", "capacity",
        "bandwidth", "load", "variance",
        "memory usage",
    ]

    lines = []

    for i in range(paragraphs):
        lines.append(
            f"Record {i:05d}: The "
            f"{rng.choice(subjects)} "
            f"{rng.choice(actions)} "
            f"{rng.choice(props)} value "
            f"{rng.randint(10, 9999)} and "
            f"{rng.choice(props)} value "
            f"{rng.randint(10, 9999)}. "
            f"Observation code "
            f"{rng.randint(10, 9999)} "
            f"was retained. This entry is "
            f"uniquely indexed as item {i:05d}."
        )

    return "\n".join(lines)


async def run_concurrency_point(
    client,
    args,
    prompt_ids,
    concurrency,
    out_dir,
):
    print(
        f"\n=== concurrency {concurrency} ==="
    )

    await wait_idle(client, args.base_url)

    before = await fetch_metrics(
        client, args.base_url
    )

    start_event = asyncio.Event()
    metric_samples = []
    stop_event = asyncio.Event()

    batch_t0 = time.perf_counter()

    poll_task = asyncio.create_task(
        poll_metrics(
            client,
            args.base_url,
            stop_event,
            metric_samples,
            args.poll_interval,
            batch_t0,
        )
    )

    tasks = [
        asyncio.create_task(
            run_one_request(
                client=client,
                base_url=args.base_url,
                model=args.model,
                prompt_ids=prompt_ids,
                output_tokens=args.output_tokens,
                request_index=i,
                start_event=start_event,
            )
        )
        for i in range(concurrency)
    ]

    # Let every coroutine reach the barrier.
    await asyncio.sleep(0.1)

    batch_t0 = time.perf_counter()
    start_event.set()

    results = await asyncio.gather(*tasks)

    batch_end = time.perf_counter()

    stop_event.set()
    await poll_task

    after = await fetch_metrics(
        client, args.base_url
    )

    for r in results:
        if r["actual_input_tokens"] != args.input_tokens:
            raise RuntimeError(
                f"Input mismatch: "
                f"{r['actual_input_tokens']} "
                f"!= {args.input_tokens}"
            )

        if r["actual_output_tokens"] != args.output_tokens:
            raise RuntimeError(
                f"Output mismatch: "
                f"{r['actual_output_tokens']} "
                f"!= {args.output_tokens}"
            )

    good_samples = [
        x for x in metric_samples
        if "metrics_error" not in x
    ]

    ttfts = [
        r["visible_ttft_ms"]
        for r in results
        if r["visible_ttft_ms"] is not None
    ]

    e2es = [
        r["e2e_ms"]
        for r in results
    ]

    itls = [
        r["content_event_itl_mean_ms"]
        for r in results
        if r["content_event_itl_mean_ms"] is not None
    ]

    summary = {
        "concurrency": concurrency,
        "input_tokens_per_request": args.input_tokens,
        "output_tokens_per_request": args.output_tokens,
        "total_input_tokens": (
            concurrency * args.input_tokens
        ),
        "total_output_tokens": (
            concurrency * args.output_tokens
        ),
        "batch_makespan_s": (
            batch_end - batch_t0
        ),
        "request_throughput_rps": (
            concurrency /
            (batch_end - batch_t0)
        ),
        "output_throughput_tps": (
            concurrency * args.output_tokens /
            (batch_end - batch_t0)
        ),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "e2e_p50_ms": percentile(e2es, 0.50),
        "e2e_p95_ms": percentile(e2es, 0.95),
        "mean_content_event_itl_p50_ms": (
            percentile(itls, 0.50)
        ),
        "mean_content_event_itl_p95_ms": (
            percentile(itls, 0.95)
        ),
        "max_kv_usage": max(
            (
                x["kv_usage"]
                for x in good_samples
            ),
            default=0.0,
        ),
        "max_running": max(
            (
                x["running"]
                for x in good_samples
            ),
            default=0.0,
        ),
        "max_waiting": max(
            (
                x["waiting"]
                for x in good_samples
            ),
            default=0.0,
        ),
        "max_waiting_capacity": max(
            (
                x["waiting_capacity"]
                for x in good_samples
            ),
            default=0.0,
        ),
        "samples_with_capacity_waiting": sum(
            x["waiting_capacity"] > 0
            for x in good_samples
        ),
        "capacity_waiting_sample_fraction": (
            sum(
                x["waiting_capacity"] > 0
                for x in good_samples
            ) / len(good_samples)
            if good_samples else 0.0
        ),
        "preemptions_delta": (
            after["preemptions"] -
            before["preemptions"]
        ),
        "metrics_samples": len(good_samples),
    }

    point_dir = (
        out_dir /
        f"concurrency_{concurrency}"
    )
    point_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (point_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (
        point_dir / "requests.jsonl"
    ).open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with (
        point_dir / "metrics.jsonl"
    ).open("w", encoding="utf-8") as f:
        for s in metric_samples:
            f.write(json.dumps(s) + "\n")

    print(
        f"KV max       : "
        f"{summary['max_kv_usage']:.4f} "
        f"({summary['max_kv_usage'] * 100:.2f}%)"
    )
    print(
        f"max running  : "
        f"{summary['max_running']:.0f}"
    )
    print(
        f"max waiting  : "
        f"{summary['max_waiting']:.0f}"
    )
    print(
        f"preemptions  : "
        f"{summary['preemptions_delta']:.0f}"
    )
    print(
        f"TTFT p50/p95 : "
        f"{summary['ttft_p50_ms']:.2f} / "
        f"{summary['ttft_p95_ms']:.2f} ms"
    )
    print(
        f"ITL p50/p95  : "
        f"{summary['mean_content_event_itl_p50_ms']:.3f} / "
        f"{summary['mean_content_event_itl_p95_ms']:.3f} ms"
    )
    print(
        f"output tput  : "
        f"{summary['output_throughput_tps']:.2f} tok/s"
    )

    await wait_idle(client, args.base_url)

    return summary


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default=BASE_URL,
    )
    parser.add_argument(
        "--model",
        default=MODEL,
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=32640,
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[8, 16, 24],
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output-root",
        default="results/s5/raw/kv_pressure_pilot",
    )
    parser.add_argument(
        "--run-label",
        default=None,
    )
    parser.add_argument(
        "--purpose",
        default="KV pressure experiment",
    )

    args = parser.parse_args()

    if (
        args.input_tokens +
        args.output_tokens >
        32768
    ):
        raise ValueError(
            "input + output exceeds max_model_len"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=os.environ.get(
            "HF_HOME",
            "/workspace/hf-cache",
        ),
    )

    print("Building exact 32K synthetic prompt...")

    body = build_synthetic_body()

    body_ids = tokenizer.encode(
        body,
        add_special_tokens=False,
    )

    suffix = tokenizer.encode(
        "\n\nFinal task: summarize the preceding "
        "technical records concisely.",
        add_special_tokens=False,
    )

    required = args.input_tokens - len(suffix)

    if len(body_ids) < required:
        raise RuntimeError(
            f"Synthetic body too short: "
            f"{len(body_ids)} < {required}"
        )

    prompt_ids = (
        body_ids[:required] +
        suffix
    )

    assert len(prompt_ids) == args.input_tokens

    run_id = time.strftime(
        "%Y%m%dT%H%M%SZ",
        time.gmtime(),
    )

    out_dir = Path(args.output_root) / run_id

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "run_id": run_id,
        "model": args.model,
        "base_url": args.base_url,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "concurrency_points": args.concurrency,
        "poll_interval_s": args.poll_interval,
        "prefix_caching": False,
        "run_label": args.run_label,
        "purpose": args.purpose,
    }

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    timeout = httpx.Timeout(
        connect=30,
        read=None,
        write=120,
        pool=30,
    )

    max_conn = (
        max(args.concurrency) + 10
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_conn,
            max_keepalive_connections=max_conn,
        ),
    ) as client:

        health = await client.get(
            f"{args.base_url}/health"
        )
        health.raise_for_status()

        summaries = []

        for c in args.concurrency:
            s = await run_concurrency_point(
                client,
                args,
                prompt_ids,
                c,
                out_dir,
            )
            summaries.append(s)

    (out_dir / "pilot_summary.json").write_text(
        json.dumps(
            summaries,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== S5-B pilot summary ===")

    for s in summaries:
        print(
            f"C={s['concurrency']:>2} | "
            f"KV={s['max_kv_usage'] * 100:>6.2f}% | "
            f"running={s['max_running']:>4.0f} | "
            f"waiting={s['max_waiting']:>4.0f} | "
            f"preempt={s['preemptions_delta']:>3.0f} | "
            f"TTFTp95={s['ttft_p95_ms']:>8.2f} ms | "
            f"ITLp95={s['mean_content_event_itl_p95_ms']:>7.3f} ms | "
            f"out={s['output_throughput_tps']:>7.2f} tok/s"
        )

    print(f"\nRaw results: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
