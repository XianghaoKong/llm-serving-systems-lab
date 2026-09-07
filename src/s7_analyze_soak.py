#!/usr/bin/env python3

import csv
import json
import math
import re
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

RUN_ID = "20260907T125350Z"

ROOT = Path("/workspace/llm-serving-systems-lab")
RUN = ROOT / "results/s7/raw/soak" / RUN_ID
OUT = ROOT / "results/s7/analysis/soak" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

PROM = "http://127.0.0.1:9090"


def prom_range(query, start, end, step=1):
    params = urllib.parse.urlencode({
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    })

    with urllib.request.urlopen(
        f"{PROM}/api/v1/query_range?{params}",
        timeout=30,
    ) as r:
        obj = json.load(r)

    result = obj.get("data", {}).get("result", [])

    values = []
    for series in result:
        for ts, value in series.get("values", []):
            try:
                values.append((float(ts), float(value)))
            except Exception:
                pass

    values.sort(key=lambda x: x[0])
    return values


def vals(series, start=None, end=None):
    result = []
    for ts, v in series:
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        if math.isfinite(v):
            result.append(v)
    return result


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def percentile(xs, p):
    if not xs:
        return float("nan")

    xs = sorted(xs)

    if len(xs) == 1:
        return xs[0]

    k = (len(xs) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return xs[lo]

    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def maximum(xs):
    return max(xs) if xs else float("nan")


def minimum(xs):
    return min(xs) if xs else float("nan")


# --------------------------------------------------
# Timeline
# --------------------------------------------------

timeline = {}

with (RUN / "timeline.csv").open() as f:
    reader = csv.DictReader(f)

    for row in reader:
        timeline[(row["stage"], row["event"])] = int(row["unix_ts"])


segments = []

for i in range(1, 6):
    name = f"segment_{i}"

    segments.append({
        "name": name,
        "start": timeline[(name, "start")],
        "end": timeline[(name, "end")],
    })


# --------------------------------------------------
# Client metrics
# --------------------------------------------------

patterns = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "request_throughput_rps": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_throughput_tps": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "p95_ttft_ms": r"P95 TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "p95_tpot_ms": r"P95 TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "p95_itl_ms": r"P95 ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
    "max_itl_ms": r"Max ITL \(ms\):\s+([0-9.]+)",
}

client_rows = []

for seg in segments:
    text = (RUN / f"{seg['name']}.stdout.txt").read_text()

    row = {"segment": seg["name"]}

    for key, pattern in patterns.items():
        m = re.search(pattern, text)

        row[key] = float(m.group(1)) if m else float("nan")

    client_rows.append(row)


with (OUT / "segment_client_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(client_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(client_rows)


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

queries = {
    "running": 'vllm:num_requests_running',
    "waiting": 'vllm:num_requests_waiting',
    "waiting_capacity":
        'vllm:num_requests_waiting_by_reason{reason="capacity"}',
    "waiting_deferred":
        'vllm:num_requests_waiting_by_reason{reason="deferred"}',
    "kv":
        'vllm:kv_cache_usage_perc',
    "preemptions":
        'vllm:num_preemptions_total',

    "gpu":
        'DCGM_FI_DEV_GPU_UTIL',
    "fb_used":
        'DCGM_FI_DEV_FB_USED',
    "power":
        'DCGM_FI_DEV_POWER_USAGE',
    "temp":
        'DCGM_FI_DEV_GPU_TEMP',

    "engine_up":
        'up{job="llm_engine"}',
    "dcgm_up":
        'up{job="dcgm"}',
}


full_start = timeline[("idle_before_soak", "start")]
full_end = timeline[("idle_after_soak", "end")]

series = {}

for name, query in queries.items():
    try:
        series[name] = prom_range(
            query,
            full_start,
            full_end,
            step=1,
        )
    except Exception as e:
        print(f"WARNING querying {name}: {e}")
        series[name] = []


# --------------------------------------------------
# Segment telemetry
# --------------------------------------------------

telemetry_rows = []

for seg in segments:
    name = seg["name"]
    start = seg["start"]
    end = seg["end"]

    # Determine actual serving-active window from vLLM running > 0.
    active_samples = [
        (ts, v)
        for ts, v in series["running"]
        if start <= ts <= end and v > 0
    ]

    if active_samples:
        active_start = active_samples[0][0]
        active_end = active_samples[-1][0]
    else:
        active_start = start
        active_end = end

    def get(metric):
        return vals(
            series[metric],
            active_start,
            active_end,
        )

    running = get("running")
    waiting = get("waiting")
    wait_cap = get("waiting_capacity")
    wait_def = get("waiting_deferred")
    kv = get("kv")
    gpu = get("gpu")
    fb = get("fb_used")
    power = get("power")
    temp = get("temp")
    engine_up = get("engine_up")
    dcgm_up = get("dcgm_up")

    pre = [
        (ts, v)
        for ts, v in series["preemptions"]
        if active_start <= ts <= active_end
    ]

    if len(pre) >= 2:
        preemption_delta = pre[-1][1] - pre[0][1]
    else:
        preemption_delta = float("nan")

    telemetry_rows.append({
        "segment": name,
        "active_start": active_start,
        "active_end": active_end,
        "active_duration_s": active_end - active_start,

        "running_mean": mean(running),
        "running_max": maximum(running),

        "waiting_mean": mean(waiting),
        "waiting_p95": percentile(waiting, 0.95),
        "waiting_max": maximum(waiting),

        "waiting_capacity_mean": mean(wait_cap),
        "waiting_deferred_mean": mean(wait_def),

        "kv_mean_pct": mean(kv) * 100 if kv else float("nan"),
        "kv_max_pct": maximum(kv) * 100 if kv else float("nan"),

        "gpu_mean_pct": mean(gpu),
        "gpu_p05_pct": percentile(gpu, 0.05),
        "gpu_min_pct": minimum(gpu),

        "fb_used_mean": mean(fb),
        "fb_used_max": maximum(fb),

        "power_mean_w": mean(power),
        "power_max_w": maximum(power),

        "temp_mean_c": mean(temp),
        "temp_max_c": maximum(temp),

        "preemption_delta": preemption_delta,

        "engine_up_min": minimum(engine_up),
        "dcgm_up_min": minimum(dcgm_up),
    })


with (OUT / "segment_telemetry.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(telemetry_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(telemetry_rows)


# --------------------------------------------------
# Idle before/after HBM comparison
# --------------------------------------------------

idle_before_start = timeline[("idle_before_soak", "start")]
idle_before_end = timeline[("idle_before_soak", "end")]

idle_after_start = timeline[("idle_after_soak", "start")]
idle_after_end = timeline[("idle_after_soak", "end")]

fb_before = vals(
    series["fb_used"],
    idle_before_start,
    idle_before_end,
)

fb_after = vals(
    series["fb_used"],
    idle_after_start,
    idle_after_end,
)

kv_before = vals(
    series["kv"],
    idle_before_start,
    idle_before_end,
)

kv_after = vals(
    series["kv"],
    idle_after_start,
    idle_after_end,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

def drift(first, last):
    if not math.isfinite(first) or first == 0:
        return float("nan")
    return (last / first - 1.0) * 100.0


with (OUT / "soak_summary.txt").open("w") as f:

    f.write("S7 FORMAL SOAK SUMMARY\n")
    f.write("======================\n\n")

    f.write(f"Run ID: {RUN_ID}\n")
    f.write("Concurrency: 64\n")
    f.write("Segments: 5\n")
    f.write("Requests/segment: 20,000\n")
    f.write("Total successful requests: 100,000\n")

    wall = (
        timeline[("soak", "end")]
        - timeline[("soak", "start")]
    )

    active_client = sum(
        r["benchmark_duration_s"]
        for r in client_rows
    )

    f.write(f"Soak wall-clock: {wall} s\n")
    f.write(
        f"Total client serving duration: "
        f"{active_client:.2f} s\n\n"
    )

    first = client_rows[0]
    last = client_rows[-1]

    for metric in [
        "output_throughput_tps",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "p95_itl_ms",
        "p99_itl_ms",
    ]:
        d = drift(first[metric], last[metric])

        f.write(
            f"{metric} drift segment1->5: "
            f"{d:+.3f}%\n"
        )

    f.write("\nTelemetry by segment:\n")

    for row in telemetry_rows:
        f.write(
            f"{row['segment']}: "
            f"active={row['active_duration_s']:.0f}s "
            f"GPU={row['gpu_mean_pct']:.2f}% "
            f"wait={row['waiting_mean']:.2f} "
            f"KV={row['kv_mean_pct']:.3f}% "
            f"power={row['power_mean_w']:.1f}W "
            f"temp_max={row['temp_max_c']:.1f}C "
            f"preemptions={row['preemption_delta']} "
            f"engine_up_min={row['engine_up_min']} "
            f"dcgm_up_min={row['dcgm_up_min']}\n"
        )

    f.write("\nIdle HBM/KV comparison:\n")
    f.write(
        f"FB used before median: "
        f"{median(fb_before):.3f}\n"
    )
    f.write(
        f"FB used after median: "
        f"{median(fb_after):.3f}\n"
    )

    if fb_before and fb_after:
        f.write(
            f"FB used delta: "
            f"{median(fb_after)-median(fb_before):+.3f}\n"
        )

    f.write(
        f"KV before median: "
        f"{median(kv_before)*100 if kv_before else float('nan'):.4f}%\n"
    )
    f.write(
        f"KV after median: "
        f"{median(kv_after)*100 if kv_after else float('nan'):.4f}%\n"
    )


print()
print("Analysis complete:")
print(OUT)
print()

print((OUT / "soak_summary.txt").read_text())
