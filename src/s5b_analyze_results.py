#!/usr/bin/env python3
"""Analyze S5-B KV-pressure formal results.

Usage:
  python src/s5b_analyze_results.py \
    --root results/s5/raw/kv_pressure_formal/<FORMAL_ID>

Outputs:
  results/s5/summary/s5b_kv_pressure_summary.csv
  results/s5/summary/s5b_kv_pressure_repeats.csv
  results/s5/summary/s5b_kv_pressure_pooled_requests.csv
  results/s5/figures/s5b_ttft_p95.png
  results/s5/figures/s5b_output_throughput.png
  results/s5/figures/s5b_max_running.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "baseline_66p54GiB",
    "kv_14GiB",
    "kv_7p78GiB",
    "kv_7GiB",
    "kv_6p75GiB",
]
DISPLAY = {
    "baseline_66p54GiB": "66.54 GiB",
    "kv_14GiB": "14.00 GiB",
    "kv_7p78GiB": "7.78 GiB",
    "kv_7GiB": "7.00 GiB",
    "kv_6p75GiB": "6.75 GiB",
}


def percentile(values, q):
    return float(
        np.percentile(
            np.asarray(list(values), dtype=float),
            q,
            method="linear",
        )
    )


def parse_cache_info(path: Path):
    text = path.read_text()

    def get(key):
        match = re.search(
            rf'{re.escape(key)}="([^"]+)"',
            text,
        )
        if not match:
            raise RuntimeError(
                f"{key} not found in {path}"
            )
        return match.group(1)

    return {
        "kv_cache_memory_bytes": int(
            get("kv_cache_memory_bytes")
        ),
        "kv_cache_size_tokens": int(
            get("kv_cache_size_tokens")
        ),
        "kv_cache_max_concurrency": float(
            get("kv_cache_max_concurrency")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/s5/summary"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/s5/figures"),
    )
    args = parser.parse_args()

    args.summary_dir.mkdir(
        parents=True, exist_ok=True
    )
    args.figures_dir.mkdir(
        parents=True, exist_ok=True
    )

    repeat_rows = []
    request_rows = []
    config_rows = []

    for label in ORDER:
        config_dir = args.root / label
        if not config_dir.is_dir():
            raise RuntimeError(
                f"Missing config: {config_dir}"
            )

        cache = parse_cache_info(
            config_dir / "cache_config_info.txt"
        )
        cache["label"] = label
        cache["kv_pool_gib"] = (
            cache["kv_cache_memory_bytes"] / 2**30
        )
        config_rows.append(cache)

        for rep in range(1, 6):
            rep_root = (
                config_dir /
                f"measured_repeat_{rep}"
            )
            run_dirs = [
                p for p in rep_root.iterdir()
                if p.is_dir()
            ]
            if len(run_dirs) != 1:
                raise RuntimeError(
                    f"{rep_root}: expected exactly "
                    f"one timestamp directory"
                )

            run_dir = run_dirs[0]
            point = (
                run_dir /
                "concurrency_8"
            )

            summary = json.loads(
                (point / "summary.json").read_text()
            )
            summary["label"] = label
            summary["repeat"] = rep
            repeat_rows.append(summary)

            request_lines = [
                line
                for line in (
                    point / "requests.jsonl"
                ).read_text().splitlines()
                if line.strip()
            ]
            if len(request_lines) != 8:
                raise RuntimeError(
                    f"{point}: expected 8 requests, "
                    f"found {len(request_lines)}"
                )

            for line in request_lines:
                request = json.loads(line)
                request["label"] = label
                request["repeat"] = rep
                request_rows.append(request)

    repeats = pd.DataFrame(repeat_rows)
    requests = pd.DataFrame(request_rows)
    configs = pd.DataFrame(config_rows)

    if len(repeats) != 25:
        raise RuntimeError(
            f"Expected 25 measured batches, "
            f"found {len(repeats)}"
        )
    if len(requests) != 200:
        raise RuntimeError(
            f"Expected 200 measured requests, "
            f"found {len(requests)}"
        )
    if set(requests["actual_input_tokens"]) != {32640}:
        raise RuntimeError(
            "Prompt token count mismatch"
        )
    if set(requests["actual_output_tokens"]) != {128}:
        raise RuntimeError(
            "Completion token count mismatch"
        )
    if set(requests["finish_reason"]) != {"length"}:
        raise RuntimeError(
            "Unexpected finish reason"
        )

    repeats = repeats.merge(
        configs,
        on="label",
        how="left",
    )
    repeats["max_kv_usage_pct"] = (
        repeats["max_kv_usage"] * 100
    )
    repeats[
        "capacity_waiting_sample_fraction_pct"
    ] = (
        repeats[
            "capacity_waiting_sample_fraction"
        ] * 100
    )

    metrics = [
        "max_kv_usage_pct",
        "max_running",
        "max_waiting",
        "max_waiting_capacity",
        "capacity_waiting_sample_fraction_pct",
        "preemptions_delta",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "mean_content_event_itl_p50_ms",
        "mean_content_event_itl_p95_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "request_throughput_rps",
        "output_throughput_tps",
        "batch_makespan_s",
    ]

    summary = (
        repeats.groupby("label")[metrics]
        .median()
        .reset_index()
        .merge(configs, on="label", how="left")
    )

    for metric in [
        "ttft_p95_ms",
        "output_throughput_tps",
        "mean_content_event_itl_p95_ms",
        "e2e_p95_ms",
    ]:
        cv = (
            repeats.groupby("label")[metric]
            .agg(
                lambda x:
                float(
                    x.std(ddof=1)
                    / x.mean()
                    * 100
                )
            )
        )
        summary[f"{metric}_cv_pct"] = (
            summary["label"].map(cv)
        )

    pooled = []
    for label in ORDER:
        group = requests[
            requests["label"] == label
        ]
        pooled.append({
            "label": label,
            "n_requests": len(group),
            "pooled_visible_ttft_p50_ms":
                percentile(
                    group["visible_ttft_ms"], 50
                ),
            "pooled_visible_ttft_p95_ms":
                percentile(
                    group["visible_ttft_ms"], 95
                ),
            "pooled_visible_ttft_p99_ms":
                percentile(
                    group["visible_ttft_ms"], 99
                ),
            "pooled_e2e_p50_ms":
                percentile(group["e2e_ms"], 50),
            "pooled_e2e_p95_ms":
                percentile(group["e2e_ms"], 95),
            "pooled_per_request_mean_content_itl_p50_ms":
                percentile(
                    group[
                        "content_event_itl_mean_ms"
                    ],
                    50,
                ),
            "pooled_per_request_mean_content_itl_p95_ms":
                percentile(
                    group[
                        "content_event_itl_mean_ms"
                    ],
                    95,
                ),
        })
    pooled = pd.DataFrame(pooled)

    rank = {label: i for i, label in enumerate(ORDER)}
    summary["_order"] = summary["label"].map(rank)
    repeats["_order"] = repeats["label"].map(rank)
    pooled["_order"] = pooled["label"].map(rank)

    summary = (
        summary.sort_values("_order")
        .drop(columns="_order")
    )
    repeats = (
        repeats.sort_values(["_order", "repeat"])
        .drop(columns="_order")
    )
    pooled = (
        pooled.sort_values("_order")
        .drop(columns="_order")
    )

    summary.to_csv(
        args.summary_dir /
        "s5b_kv_pressure_summary.csv",
        index=False,
    )
    repeats.to_csv(
        args.summary_dir /
        "s5b_kv_pressure_repeats.csv",
        index=False,
    )
    pooled.to_csv(
        args.summary_dir /
        "s5b_kv_pressure_pooled_requests.csv",
        index=False,
    )

    plot = (
        summary.set_index("label")
        .loc[ORDER]
    )
    labels = [DISPLAY[x] for x in ORDER]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(
        labels,
        plot["ttft_p95_ms"],
        marker="o",
    )
    plt.xlabel("Configured KV-cache pool")
    plt.ylabel(
        "Median-of-repeats P95 visible TTFT (ms)"
    )
    plt.title("S5-B: KV Capacity vs P95 TTFT")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(
        args.figures_dir /
        "s5b_ttft_p95.png",
        dpi=180,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(
        labels,
        plot["output_throughput_tps"],
        marker="o",
    )
    plt.xlabel("Configured KV-cache pool")
    plt.ylabel(
        "Median output throughput (tokens/s)"
    )
    plt.title(
        "S5-B: KV Capacity vs Output Throughput"
    )
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(
        args.figures_dir /
        "s5b_output_throughput.png",
        dpi=180,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(
        labels,
        plot["max_running"],
        marker="o",
    )
    plt.xlabel("Configured KV-cache pool")
    plt.ylabel(
        "Median maximum running requests"
    )
    plt.title(
        "S5-B: Admission Changes at "
        "the KV-Capacity Boundary"
    )
    plt.xticks(rotation=25)
    plt.ylim(0, 9)
    plt.tight_layout()
    plt.savefig(
        args.figures_dir /
        "s5b_max_running.png",
        dpi=180,
    )
    plt.close(fig)

    print(
        summary[
            [
                "label",
                "kv_pool_gib",
                "kv_cache_max_concurrency",
                "max_kv_usage_pct",
                "max_running",
                "ttft_p95_ms",
                "mean_content_event_itl_p95_ms",
                "output_throughput_tps",
                "preemptions_delta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
