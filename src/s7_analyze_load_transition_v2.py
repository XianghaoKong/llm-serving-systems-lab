#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt


TELEMETRY_QUERIES = {
    "gpu_util_pct": "DCGM_FI_DEV_GPU_UTIL",
    "hbm_used_mib": "DCGM_FI_DEV_FB_USED",
    "power_w": "DCGM_FI_DEV_POWER_USAGE",
    "gpu_temp_c": "DCGM_FI_DEV_GPU_TEMP",

    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "waiting_capacity": 'vllm:num_requests_waiting_by_reason{reason="capacity"}',
    "waiting_deferred": 'vllm:num_requests_waiting_by_reason{reason="deferred"}',
    "kv_usage_pct": "100 * vllm:kv_cache_usage_perc",

    "preemptions_total": "vllm:num_preemptions_total",

    "llm_up": 'up{job="llm_engine"}',
    "dcgm_up": 'up{job="dcgm"}',
}


CLIENT_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9]+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9]+)",
    "total_generated_tokens": r"Total generated tokens:\s+([0-9]+)",
    "request_throughput_rps": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_throughput_tps": r"Output token throughput \(tok/s\):\s+([0-9.]+)",

    "median_e2e_ms": r"Median E2E Latency \(ms\):\s+([0-9.]+)",
    "p95_e2e_ms": r"P95 E2E Latency \(ms\):\s+([0-9.]+)",
    "p99_e2e_ms": r"P99 E2E Latency \(ms\):\s+([0-9.]+)",

    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p95_ttft_ms": r"P95 TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",

    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p95_tpot_ms": r"P95 TPOT \(ms\):\s+([0-9.]+)",

    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p95_itl_ms": r"P95 ITL \(ms\):\s+([0-9.]+)",
    "max_itl_ms": r"Max ITL \(ms\):\s+([0-9.]+)",
}


LOAD_STAGES = ["c8", "c32", "c64_a", "c128", "c64_b"]


def prom_query_range(base_url, query, start, end, step=1):
    params = urllib.parse.urlencode({
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    })

    url = f"{base_url}/api/v1/query_range?{params}"

    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.load(r)

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {query}")

    result = payload["data"]["result"]

    if not result:
        return []

    if len(result) > 1:
        print(
            f"WARNING: query returned {len(result)} series; "
            f"using first series: {query}"
        )

    values = []

    for ts, value in result[0]["values"]:
        try:
            v = float(value)
        except ValueError:
            continue

        if math.isfinite(v):
            values.append((float(ts), v))

    return values


def parse_timeline(path):
    rows = list(csv.DictReader(path.open()))

    grouped = {}

    for row in rows:
        stage = row["stage"]
        grouped.setdefault(stage, {
            "stage": stage,
            "concurrency": int(row["concurrency"]),
            "num_prompts": int(row["num_prompts"]),
        })

        grouped[stage][row["event"]] = int(row["unix_ts"])

    stages = []

    for item in grouped.values():
        if "start" in item and "end" in item:
            stages.append(item)

    return sorted(stages, key=lambda x: x["start"])


def parse_client_stdout(path):
    text = path.read_text(errors="replace")
    out = {}

    for key, pattern in CLIENT_PATTERNS.items():
        match = re.search(pattern, text)

        if not match:
            raise RuntimeError(f"Missing {key} in {path}")

        value = match.group(1)

        if key in {
            "successful_requests",
            "total_input_tokens",
            "total_generated_tokens",
        }:
            out[key] = int(value)
        else:
            out[key] = float(value)

    return out


def vals_between(series, start, end):
    return [
        value
        for ts, value in series
        if start <= ts <= end
    ]


def safe_mean(xs):
    return statistics.mean(xs) if xs else math.nan


def safe_median(xs):
    return statistics.median(xs) if xs else math.nan


def safe_max(xs):
    return max(xs) if xs else math.nan


def safe_min(xs):
    return min(xs) if xs else math.nan


def counter_delta(xs):
    if len(xs) < 2:
        return math.nan

    delta = xs[-1] - xs[0]

    # Counter reset fallback
    if delta < 0:
        return math.nan

    return delta


def fmt(v, digits=2):
    if v is None or not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def active_window(running_series, shell_start, shell_end):
    active = [
        ts
        for ts, value in running_series
        if shell_start <= ts <= shell_end and value > 0
    ]

    if not active:
        return None, None

    return min(active), max(active)


def plot_series(
    series,
    stages,
    experiment_start,
    ylabel,
    title,
    out_path,
):
    if not series:
        return

    xs = [ts - experiment_start for ts, _ in series]
    ys = [v for _, v in series]

    plt.figure(figsize=(10, 4.8))
    plt.plot(xs, ys)

    ymax = max(ys) if ys else 1

    for stage in stages:
        if stage["concurrency"] <= 0:
            continue

        x = stage["start"] - experiment_start

        plt.axvline(
            x=x,
            linestyle="--",
            alpha=0.3,
        )

        plt.text(
            x,
            ymax,
            f"C{stage['concurrency']}",
            rotation=90,
            verticalalignment="top",
            fontsize=8,
        )

    plt.xlabel("Seconds since experiment start")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dir",
        type=Path,
    )

    parser.add_argument(
        "--prometheus",
        default="http://127.0.0.1:9090",
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_id = run_dir.name

    timeline_path = run_dir / "timeline.csv"

    if not timeline_path.exists():
        raise SystemExit(
            f"Missing timeline: {timeline_path}"
        )

    stages = parse_timeline(timeline_path)

    experiment_start = min(s["start"] for s in stages)
    experiment_end = max(s["end"] for s in stages)

    query_start = experiment_start - 5
    query_end = experiment_end + 5

    out_dir = (
        Path("results/s7/analysis/load_transition_v2")
        / run_id
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(
        f"Experiment window: "
        f"{experiment_start} -> {experiment_end}"
    )
    print()

    telemetry = {}

    for name, query in TELEMETRY_QUERIES.items():
        try:
            series = prom_query_range(
                args.prometheus,
                query,
                query_start,
                query_end,
                step=1,
            )

            telemetry[name] = series

            print(
                f"{name:<20} samples={len(series)}"
            )

        except Exception as e:
            print(
                f"WARNING {name}: {e}"
            )
            telemetry[name] = []

    print()

    summary_rows = []

    for stage_name in LOAD_STAGES:
        stage = next(
            (s for s in stages if s["stage"] == stage_name),
            None,
        )

        if stage is None:
            raise RuntimeError(
                f"Missing stage {stage_name}"
            )

        stdout_path = (
            run_dir / f"{stage_name}.stdout.txt"
        )

        if not stdout_path.exists():
            raise RuntimeError(
                f"Missing stdout: {stdout_path}"
            )

        client = parse_client_stdout(stdout_path)

        expected_input = (
            stage["num_prompts"] * 512
        )

        expected_output = (
            stage["num_prompts"] * 128
        )

        validation_pass = (
            client["successful_requests"]
            == stage["num_prompts"]
            and client["total_input_tokens"]
            == expected_input
            and client["total_generated_tokens"]
            == expected_output
        )

        active_start, active_end = active_window(
            telemetry["running"],
            stage["start"],
            stage["end"],
        )

        row = {
            "stage": stage_name,
            "concurrency": stage["concurrency"],
            "num_prompts": stage["num_prompts"],
            "shell_duration_s":
                stage["end"] - stage["start"],
            "active_start": active_start or "",
            "active_end": active_end or "",
            "active_duration_s":
                (
                    active_end - active_start
                    if active_start is not None
                    and active_end is not None
                    else math.nan
                ),
            "client_validation_pass":
                validation_pass,
            **client,
        }

        if active_start is not None:
            for metric in [
                "gpu_util_pct",
                "hbm_used_mib",
                "power_w",
                "gpu_temp_c",
                "running",
                "waiting",
                "waiting_capacity",
                "waiting_deferred",
                "kv_usage_pct",
                "llm_up",
                "dcgm_up",
            ]:
                values = vals_between(
                    telemetry[metric],
                    active_start,
                    active_end,
                )

                row[f"{metric}_mean"] = safe_mean(values)
                row[f"{metric}_median"] = safe_median(values)
                row[f"{metric}_max"] = safe_max(values)
                row[f"{metric}_min"] = safe_min(values)

            preemptions = vals_between(
                telemetry["preemptions_total"],
                active_start,
                active_end,
            )

            row["preemptions_delta"] = (
                counter_delta(preemptions)
            )

            gpu_values = vals_between(
                telemetry["gpu_util_pct"],
                active_start,
                active_end,
            )

            row["gpu_active_sample_count"] = len(
                gpu_values
            )

            if gpu_values:
                row["gpu_nonzero_fraction"] = (
                    sum(v > 0 for v in gpu_values)
                    / len(gpu_values)
                )
            else:
                row["gpu_nonzero_fraction"] = math.nan

        summary_rows.append(row)

    # Write summary
    all_fields = []

    for row in summary_rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    with (
        out_dir / "stage_summary_v2.csv"
    ).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save raw telemetry
    with (
        out_dir / "telemetry_raw.csv"
    ).open("w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(
            ["metric", "unix_ts", "value"]
        )

        for metric, series in telemetry.items():
            for ts, value in series:
                writer.writerow(
                    [metric, ts, value]
                )

    # Plots
    plot_specs = [
        (
            "gpu_util_pct",
            "GPU utilization (%)",
            "GPU utilization",
            "gpu_util_over_time.png",
        ),
        (
            "hbm_used_mib",
            "HBM used (MiB)",
            "GPU HBM usage",
            "hbm_used_over_time.png",
        ),
        (
            "power_w",
            "Power (W)",
            "GPU power",
            "power_over_time.png",
        ),
        (
            "running",
            "Running requests",
            "Running requests",
            "running_over_time.png",
        ),
        (
            "waiting",
            "Waiting requests",
            "Waiting requests",
            "waiting_over_time.png",
        ),
        (
            "waiting_capacity",
            "Capacity-waiting requests",
            "Capacity waiting",
            "waiting_capacity_over_time.png",
        ),
        (
            "kv_usage_pct",
            "KV cache usage (%)",
            "KV-cache occupancy",
            "kv_usage_over_time.png",
        ),
    ]

    for metric, ylabel, title, filename in plot_specs:
        plot_series(
            telemetry[metric],
            stages,
            experiment_start,
            ylabel,
            title,
            out_dir / filename,
        )

    print("Active-window stage summary")
    print()

    header = (
        f"{'Stage':<9}"
        f"{'C':>5}"
        f"{'Dur':>7}"
        f"{'GPU%':>9}"
        f"{'Run':>8}"
        f"{'Wait':>8}"
        f"{'KV%':>8}"
        f"{'Out tok/s':>12}"
        f"{'P95TTFT':>11}"
        f"{'P95E2E':>10}"
        f"{'Preempt':>9}"
    )

    print(header)
    print("-" * len(header))

    for r in summary_rows:
        print(
            f"{r['stage']:<9}"
            f"{r['concurrency']:>5}"
            f"{fmt(r['active_duration_s'], 0):>7}"
            f"{fmt(r.get('gpu_util_pct_mean')):>9}"
            f"{fmt(r.get('running_mean')):>8}"
            f"{fmt(r.get('waiting_mean')):>8}"
            f"{fmt(r.get('kv_usage_pct_mean')):>8}"
            f"{fmt(r['output_throughput_tps']):>12}"
            f"{fmt(r['p95_ttft_ms']):>11}"
            f"{fmt(r['p95_e2e_ms']):>10}"
            f"{fmt(r.get('preemptions_delta'), 0):>9}"
        )

    print()
    print("Validation")
    print()

    for r in summary_rows:
        print(
            f"{r['stage']}: "
            f"client={'PASS' if r['client_validation_pass'] else 'FAIL'}, "
            f"GPU samples={r.get('gpu_active_sample_count', 0)}, "
            f"GPU nonzero="
            f"{fmt(100 * r.get('gpu_nonzero_fraction', math.nan))}%"
        )

    lookup = {
        r["stage"]: r
        for r in summary_rows
    }

    if "c64_a" in lookup and "c64_b" in lookup:
        a = lookup["c64_a"]
        b = lookup["c64_b"]

        print()
        print("C64 recovery after C128")
        print()

        checks = [
            ("Output tok/s", "output_throughput_tps"),
            ("P95 TTFT", "p95_ttft_ms"),
            ("P95 E2E", "p95_e2e_ms"),
            ("GPU util", "gpu_util_pct_mean"),
            ("Waiting", "waiting_mean"),
            ("KV usage", "kv_usage_pct_mean"),
        ]

        for label, key in checks:
            av = a.get(key, math.nan)
            bv = b.get(key, math.nan)

            if (
                math.isfinite(av)
                and math.isfinite(bv)
                and av != 0
            ):
                delta = (
                    100 * (bv - av) / av
                )

                print(
                    f"{label:<13} "
                    f"{fmt(av)} -> {fmt(bv)} "
                    f"({delta:+.2f}%)"
                )
            else:
                print(
                    f"{label:<13} "
                    f"{fmt(av)} -> {fmt(bv)}"
                )

    print()
    print("Analysis complete.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
