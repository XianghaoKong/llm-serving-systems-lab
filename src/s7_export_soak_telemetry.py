#!/usr/bin/env python3

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/workspace/llm-serving-systems-lab")
RUN_ID = "20260907T125350Z"

RAW = ROOT / "results/s7/raw/soak" / RUN_ID
ANALYSIS = ROOT / "results/s7/analysis/soak" / RUN_ID
FIGURES = ANALYSIS / "figures"

ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

PROM = "http://127.0.0.1:9090"


def query_range(query, start, end, step=1):
    params = urllib.parse.urlencode({
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    })

    url = f"{PROM}/api/v1/query_range?{params}"

    with urllib.request.urlopen(url, timeout=60) as response:
        obj = json.load(response)

    if obj.get("status") != "success":
        raise RuntimeError(obj)

    result = obj["data"]["result"]

    if not result:
        print(f"WARNING: no data for {query}")
        return pd.Series(dtype=float)

    # This experiment has one engine and one GPU.
    values = result[0]["values"]

    s = pd.Series(
        {int(float(ts)): float(v) for ts, v in values},
        dtype=float,
    )

    return s.sort_index()


# -------------------------------------------------------
# Timeline
# -------------------------------------------------------

timeline = {}

with (RAW / "timeline.csv").open() as f:
    for row in csv.DictReader(f):
        timeline[(row["stage"], row["event"])] = int(row["unix_ts"])


start = timeline[("idle_before_soak", "start")]
end = timeline[("idle_after_soak", "end")]


# -------------------------------------------------------
# Export Prometheus telemetry
# -------------------------------------------------------

queries = {
    "running_requests":
        'vllm:num_requests_running',

    "waiting_requests":
        'vllm:num_requests_waiting',

    "waiting_capacity":
        'vllm:num_requests_waiting_by_reason{reason="capacity"}',

    "waiting_deferred":
        'vllm:num_requests_waiting_by_reason{reason="deferred"}',

    "kv_cache_usage":
        'vllm:kv_cache_usage_perc',

    "preemptions_total":
        'vllm:num_preemptions_total',

    "gpu_util_pct":
        'DCGM_FI_DEV_GPU_UTIL',

    "fb_used_mb":
        'DCGM_FI_DEV_FB_USED',

    "power_w":
        'DCGM_FI_DEV_POWER_USAGE',

    "gpu_temp_c":
        'DCGM_FI_DEV_GPU_TEMP',

    "engine_up":
        'up{job="llm_engine"}',

    "dcgm_up":
        'up{job="dcgm"}',
}


data = {}

for name, query in queries.items():
    print(f"Querying {name} ...")
    data[name] = query_range(query, start, end, step=1)


df = pd.DataFrame(data)

df.index.name = "unix_ts"
df = df.sort_index()

df.insert(
    0,
    "elapsed_s",
    df.index - timeline[("soak", "start")]
)

df.insert(
    1,
    "utc_time",
    pd.to_datetime(df.index, unit="s", utc=True)
)

out_csv = ANALYSIS / "prometheus_timeseries_1hz.csv"
df.to_csv(out_csv)

print()
print("Saved:")
print(out_csv)


# -------------------------------------------------------
# Segment boundaries helper
# -------------------------------------------------------

segment_bounds = []

for i in range(1, 6):
    name = f"segment_{i}"

    segment_bounds.append((
        name,
        timeline[(name, "start")] - timeline[("soak", "start")],
        timeline[(name, "end")] - timeline[("soak", "start")],
    ))


def boundaries(ax):
    for name, s, e in segment_bounds:
        ax.axvline(s, linewidth=0.6, alpha=0.25)
        ax.axvline(e, linewidth=0.6, alpha=0.25)

        mid = (s + e) / 2
        ymax = ax.get_ylim()[1]

        ax.text(
            mid,
            ymax,
            name.replace("_", " "),
            ha="center",
            va="bottom",
            fontsize=8,
        )


# -------------------------------------------------------
# Figure 1: GPU utilization
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    df["elapsed_s"],
    df["gpu_util_pct"],
    linewidth=1,
)

ax.set_xlabel("Elapsed time from soak start (s)")
ax.set_ylabel("GPU utilization (%)")
ax.set_title("S7 Formal Soak — GPU Utilization")
ax.set_ylim(0, 105)

boundaries(ax)

fig.tight_layout()
fig.savefig(
    FIGURES / "soak_gpu_utilization.png",
    dpi=180,
)
plt.close(fig)


# -------------------------------------------------------
# Figure 2: scheduler
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    df["elapsed_s"],
    df["running_requests"],
    label="Running",
    linewidth=1,
)

ax.plot(
    df["elapsed_s"],
    df["waiting_requests"],
    label="Waiting",
    linewidth=1,
)

ax.set_xlabel("Elapsed time from soak start (s)")
ax.set_ylabel("Requests")
ax.set_title("S7 Formal Soak — vLLM Scheduler State")
ax.legend()

boundaries(ax)

fig.tight_layout()
fig.savefig(
    FIGURES / "soak_scheduler_state.png",
    dpi=180,
)
plt.close(fig)


# -------------------------------------------------------
# Figure 3: KV occupancy
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    df["elapsed_s"],
    df["kv_cache_usage"] * 100,
    linewidth=1,
)

ax.set_xlabel("Elapsed time from soak start (s)")
ax.set_ylabel("KV cache occupancy (%)")
ax.set_title("S7 Formal Soak — KV Cache Occupancy")

boundaries(ax)

fig.tight_layout()
fig.savefig(
    FIGURES / "soak_kv_cache.png",
    dpi=180,
)
plt.close(fig)


# -------------------------------------------------------
# Figure 4: power
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    df["elapsed_s"],
    df["power_w"],
    linewidth=1,
)

ax.set_xlabel("Elapsed time from soak start (s)")
ax.set_ylabel("GPU power (W)")
ax.set_title("S7 Formal Soak — GPU Power")

boundaries(ax)

fig.tight_layout()
fig.savefig(
    FIGURES / "soak_gpu_power.png",
    dpi=180,
)
plt.close(fig)


# -------------------------------------------------------
# Figure 5: client metric drift
# -------------------------------------------------------

client = pd.read_csv(
    ANALYSIS / "segment_client_metrics.csv"
)

metrics = {
    "Output throughput":
        "output_throughput_tps",

    "P95 TTFT":
        "p95_ttft_ms",

    "P99 TTFT":
        "p99_ttft_ms",

    "P95 ITL":
        "p95_itl_ms",
}

fig, ax = plt.subplots(figsize=(9, 5))

x = range(1, 6)

for label, column in metrics.items():
    base = client[column].iloc[0]

    normalized = (
        client[column] / base - 1
    ) * 100

    ax.plot(
        x,
        normalized,
        marker="o",
        label=label,
    )

ax.axhline(0, linewidth=0.8)

ax.set_xticks(list(x))
ax.set_xlabel("Soak segment")
ax.set_ylabel("Change relative to segment 1 (%)")
ax.set_title(
    "S7 Formal Soak — Client-Side Performance Drift"
)
ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES / "soak_client_metric_drift.png",
    dpi=180,
)
plt.close(fig)


print()
print("Figures:")
for p in sorted(FIGURES.glob("*.png")):
    print(" ", p.name)

print()
print("S7 telemetry export complete.")
