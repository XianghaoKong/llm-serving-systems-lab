#!/usr/bin/env python3
"""Analyze S8 causal prefill/decode interference trials."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


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


def finite(values: Iterable[Any]) -> List[float]:
    output = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_events(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["t_s"] = float(row["t_s"])
        row["content_event_index"] = int(row["content_event_index"])
        row["content_chars"] = int(row["content_chars"])
    return rows


def gap_intervals(
    events: Sequence[Dict[str, Any]],
    role: str,
) -> List[Tuple[float, float, float]]:
    by_request: Dict[str, List[float]] = defaultdict(list)
    for event in events:
        if event["role"] == role:
            by_request[str(event["request_id"])].append(float(event["t_s"]))
    intervals = []
    for times in by_request.values():
        times.sort()
        intervals.extend(
            (left, right, (right - left) * 1000.0)
            for left, right in zip(times[:-1], times[1:], strict=True)
        )
    return intervals


def overlapping_gaps(
    intervals: Sequence[Tuple[float, float, float]],
    start: float,
    end: float,
) -> List[float]:
    return [gap for left, right, gap in intervals if right >= start and left <= end]


def event_rate(
    events: Sequence[Dict[str, Any]],
    role: str,
    start: float,
    end: float,
) -> Optional[float]:
    duration = end - start
    if duration <= 0:
        return None
    count = sum(
        event["role"] == role and start <= float(event["t_s"]) <= end
        for event in events
    )
    return count / duration


def window_metrics(
    rows: Sequence[Dict[str, Any]],
    start: float,
    end: float,
) -> Dict[str, Optional[float]]:
    good = [
        row for row in rows
        if "metrics_error" not in row and start <= float(row["t_s"]) <= end
    ]
    return {
        "max_running": max(finite(row.get("running") for row in good), default=None),
        "max_waiting": max(finite(row.get("waiting") for row in good), default=None),
        "max_kv_usage": max(finite(row.get("kv_usage") for row in good), default=None),
    }


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def summarize_trial(trial_path: Path, baseline_seconds: float) -> Dict[str, Any]:
    trial_dir = trial_path.parent
    metadata = json.loads(trial_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError(f"incomplete trial: {trial_dir}")
    events = read_events(trial_dir / "content_events.csv.gz")
    requests = read_jsonl(trial_dir / "requests.jsonl")
    metrics = read_jsonl(trial_dir / "metrics.jsonl")
    marker = float(metadata["injection_t_s"])
    impact_end = float(metadata["impact_end_t_s"])
    baseline_start = marker - baseline_seconds
    baseline_end = marker - 1.0
    if baseline_start < 0 or baseline_end <= baseline_start:
        raise RuntimeError(
            f"insufficient warmup for baseline window in {trial_dir}"
        )

    intervals = gap_intervals(events, "background")
    baseline_gaps = overlapping_gaps(
        intervals, baseline_start, baseline_end
    )
    impact_gaps = overlapping_gaps(intervals, marker, impact_end)
    if len(baseline_gaps) < 20:
        raise RuntimeError(
            f"only {len(baseline_gaps)} baseline background gaps in {trial_dir}"
        )
    if len(impact_gaps) < 1:
        raise RuntimeError(f"no impact-window background gaps in {trial_dir}")

    baseline_p95 = percentile(baseline_gaps, 0.95)
    baseline_p99 = percentile(baseline_gaps, 0.99)
    impact_p95 = percentile(impact_gaps, 0.95)
    impact_p99 = percentile(impact_gaps, 0.99)
    impact_max = max(impact_gaps)
    baseline_rate = event_rate(
        events, "background", baseline_start, baseline_end
    )
    impact_rate = event_rate(events, "background", marker, impact_end)

    interferers = [row for row in requests if row.get("role") == "interferer"]
    long_ttfts = finite(
        (
            float(row["first_content_t_s"]) - float(row["start_t_s"])
        ) * 1000.0
        if row.get("first_content_t_s") is not None else None
        for row in interferers
    )
    long_e2e = finite(
        (float(row["end_t_s"]) - float(row["start_t_s"])) * 1000.0
        for row in interferers
        if row.get("status") == "ok"
    )
    completed = [
        row for row in requests
        if row.get("status") == "ok" and row.get("role") == "background"
    ]
    event_token_deltas = finite(
        row.get("content_event_minus_output_tokens") for row in completed
    )
    impact_system = window_metrics(metrics, marker, impact_end)

    label = str(metadata["config_label"])
    budget_match = re.search(r"(\d+)$", label)
    budget = int(budget_match.group(1)) if budget_match else None
    return {
        "trial_dir": str(trial_dir),
        "config_label": label,
        "chunked_prefill": int(label.startswith("on_")),
        "max_num_batched_tokens": budget,
        "replicate": int(metadata["replicate"]),
        "trial_kind": metadata["trial_kind"],
        "background_concurrency": int(metadata["background_concurrency"]),
        "interferer_input_tokens": int(metadata["interferer_input_tokens"]),
        "interferer_count": int(metadata["interferer_count"]),
        "baseline_window_s": baseline_end - baseline_start,
        "impact_window_s": impact_end - marker,
        "baseline_gap_count": len(baseline_gaps),
        "impact_gap_count": len(impact_gaps),
        "baseline_gap_p95_ms": baseline_p95,
        "baseline_gap_p99_ms": baseline_p99,
        "impact_gap_p95_ms": impact_p95,
        "impact_gap_p99_ms": impact_p99,
        "impact_gap_max_ms": impact_max,
        "p95_stall_ratio": ratio(impact_p95, baseline_p95),
        "p99_stall_ratio": ratio(impact_p99, baseline_p99),
        "background_event_rate_baseline_s": baseline_rate,
        "background_event_rate_impact_s": impact_rate,
        "background_event_rate_ratio": ratio(impact_rate, baseline_rate),
        "long_ttft_p50_ms": percentile(long_ttfts, 0.50),
        "long_ttft_p95_ms": percentile(long_ttfts, 0.95),
        "long_e2e_p50_ms": percentile(long_e2e, 0.50),
        "impact_max_running": impact_system["max_running"],
        "impact_max_waiting": impact_system["max_waiting"],
        "impact_max_kv_usage": impact_system["max_kv_usage"],
        "preemptions_delta": metadata.get("preemptions_delta"),
        "completed_background_requests": len(completed),
        "event_token_delta_min": min(event_token_deltas, default=None),
        "event_token_delta_max": max(event_token_deltas, default=None),
    }


def bootstrap_median_ci(
    values: Sequence[float],
    seed: int,
    samples: int = 5000,
) -> Tuple[Optional[float], Optional[float]]:
    xs = finite(values)
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], xs[0]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(xs, k=len(xs)))
        for _ in range(samples)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def aggregate(rows: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    keys = [
        "config_label", "chunked_prefill", "max_num_batched_tokens",
        "trial_kind", "background_concurrency", "interferer_input_tokens",
        "interferer_count",
    ]
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    metrics = [
        "p95_stall_ratio", "p99_stall_ratio", "impact_gap_max_ms",
        "background_event_rate_ratio", "long_ttft_p50_ms",
        "long_e2e_p50_ms", "impact_max_waiting", "impact_max_kv_usage",
    ]
    output = []
    for group_index, (group_key, group_rows) in enumerate(grouped.items()):
        item = dict(zip(keys, group_key, strict=True))
        item["n_runs"] = len(group_rows)
        for metric in metrics:
            values = finite(row.get(metric) for row in group_rows)
            item[f"{metric}_median"] = (
                statistics.median(values) if values else None
            )
            low, high = bootstrap_median_ci(
                values, seed=seed + group_index * 100 + len(metric)
            )
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            row["trial_kind"], row["interferer_input_tokens"],
            row["background_concurrency"], row["chunked_prefill"],
            -(row["max_num_batched_tokens"] or 0),
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_by_length(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    inject = [row for row in rows if row["trial_kind"] == "inject"]
    lengths = sorted({int(row["interferer_input_tokens"]) for row in inject})
    if not inject:
        return

    config_order = [
        "off_32768", "on_32768", "on_16384", "on_8192",
        "on_4096", "on_2048", "on_1024", "on_512",
    ]
    order_index = {label: index for index, label in enumerate(config_order)}
    display_labels = ["off\n32K", "32K", "16K", "8K", "4K", "2K", "1K", "512"]

    plt.figure(figsize=(9, 5.5))
    for length in lengths:
        selected = [
            row for row in inject
            if int(row["interferer_input_tokens"]) == length
        ]
        selected.sort(key=lambda row: order_index[str(row["config_label"])])
        medians = [row["p99_stall_ratio_median"] for row in selected]
        lows = [row["p99_stall_ratio_ci_low"] for row in selected]
        highs = [row["p99_stall_ratio_ci_high"] for row in selected]
        plt.errorbar(
            range(len(selected)), medians,
            yerr=[
                [median - low for median, low in zip(medians, lows, strict=True)],
                [high - median for median, high in zip(medians, highs, strict=True)],
            ],
            marker="o", capsize=3, linewidth=2,
            label=f"{length // 1024}K prompt",
        )
    plt.axhline(1.0, color="black", linewidth=1, alpha=0.5)
    plt.axvspan(5.65, 6.35, color="#2ca02c", alpha=0.08)
    plt.yscale("log", base=2)
    plt.yticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    plt.xticks(range(len(display_labels)), display_labels)
    plt.ylabel("Background P99 content-event gap ratio")
    plt.xlabel("Chunked-prefill token budget (off = unchunked baseline)")
    plt.title("Smaller Prefill Chunks Bound Decode Stalls")
    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stall_ratio_by_config.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, len(lengths), figsize=(14, 4.8), sharey=True)
    short_labels = ["off", "32K", "16K", "8K", "4K", "2K", "1K", "512"]
    colors = plt.get_cmap("tab10")
    for axis, length in zip(axes, lengths, strict=True):
        selected = [
            row for row in inject
            if int(row["interferer_input_tokens"]) == length
        ]
        selected.sort(key=lambda row: order_index[str(row["config_label"])])
        xs = [row["long_ttft_p50_ms_median"] for row in selected]
        ys = [row["p99_stall_ratio_median"] for row in selected]
        axis.plot(xs, ys, color="0.72", linewidth=1.5, zorder=1)
        for index, (row, x, y) in enumerate(zip(selected, xs, ys, strict=True)):
            color = "#d62728" if row["config_label"] == "on_1024" else colors(index)
            size = 72 if row["config_label"] == "on_1024" else 48
            axis.scatter(x, y, s=size, color=color, zorder=2)
            budget = row["max_num_batched_tokens"]
            if row["config_label"] == "off_32768" or (
                budget is not None and budget < length
            ):
                axis.annotate(
                    short_labels[index], (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8,
                    fontweight="bold" if row["config_label"] == "on_1024" else "normal",
                )
        axis.set_title(f"{length // 1024}K-token prompt")
        axis.set_xlabel("Long-request median TTFT (ms)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Background P99 gap ratio")
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks([1, 2, 4, 8, 16, 32, 64])
    axes[0].set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
    fig.suptitle("Chunk Size Trades Long-Request TTFT for Decode Fairness", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "ttft_stall_pareto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else input_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = sorted(
        path for path in input_root.rglob("trial.json")
        if not any(part.startswith("excluded_") for part in path.parts)
    )
    if not trial_paths:
        raise FileNotFoundError(f"no trial.json files below {input_root}")
    rows = [summarize_trial(path, args.baseline_seconds) for path in trial_paths]
    aggregates = aggregate(rows, args.bootstrap_seed)
    write_csv(output_dir / "trial_summary.csv", rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregates)
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_root": str(input_root),
                "baseline_seconds": args.baseline_seconds,
                "run_is_statistical_unit": True,
                "bootstrap_samples": 5000,
                "trials": rows,
                "aggregates": aggregates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_by_length(aggregates, output_dir)
    print(f"Analyzed {len(rows)} trials")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
