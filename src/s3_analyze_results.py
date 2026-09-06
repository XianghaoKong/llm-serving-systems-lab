#!/usr/bin/env python3
"""
S3 — vLLM continuous-batching results analyzer.

Public-output analyzer for:
  S3-C: natural-generation capacity sweep
  S3-E: max_num_seqs concurrency-control experiment
  S3-D: length-matched controlled replay

The script reads local raw results and writes only compact public summaries/figures.

Expected raw layouts (the discovery code is intentionally tolerant):

  results/s3/raw/poisson/coarse/lambda_*/
      requests.csv
      run_metadata.json
      scheduler_metrics.csv.gz

  results/s3/raw/poisson/concurrency_optimization/**/
      requests.csv
      run_metadata.json
      scheduler_metrics.csv.gz

  results/s3/raw/length_matched/**/
      requests.csv
      run_metadata.json
      scheduler_metrics.csv.gz

It also recognizes older concurrency directories such as
`results/s3/raw/poisson/coarse/lambda_4_12_seqs`.

Outputs:
  results/s3/summary/s3_capacity_summary.csv
  results/s3/summary/s3_concurrency_summary.csv
  results/s3/summary/s3_length_matched_summary.csv
  results/s3/summary/s3_results_summary.json
  results/s3/summary/s3_key_findings.md

  results/s3/figures/01_throughput_vs_arrival_rate.png
  results/s3/figures/02_ttft_vs_arrival_rate.png
  results/s3/figures/03_itl_vs_arrival_rate.png
  results/s3/figures/04_concurrency_control_ttft.png
  results/s3/figures/05_concurrency_control_itl.png

Run:
    python src/s3_analyze_results.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

S3_RAW = PROJECT_ROOT / "results" / "s3" / "raw"
CAPACITY_ROOT = S3_RAW / "poisson" / "coarse"
CONCURRENCY_ROOT = S3_RAW / "poisson" / "concurrency_optimization"
LENGTH_MATCHED_ROOT = S3_RAW / "length_matched"

SUMMARY_DIR = PROJECT_ROOT / "results" / "s3" / "summary"
FIG_DIR = PROJECT_ROOT / "results" / "s3" / "figures"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def finite_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def finite_int(value: Any) -> Optional[int]:
    x = finite_float(value)
    return int(x) if x is not None else None


def pct(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_maybe_gzip_csv(path: Path) -> List[Dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return read_csv(path)


def first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def nested(meta: Dict[str, Any], section: str, *keys: str) -> Any:
    obj = meta.get(section)
    if isinstance(obj, dict):
        value = first_present(obj, *keys)
        if value not in (None, ""):
            return value
    return first_present(meta, *keys)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def successful_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    ok = []
    for row in rows:
        status = str(row.get("status", "ok")).strip().lower()
        if status in ("", "ok", "success", "successful"):
            ok.append(row)
    return ok


def numeric_column(
    rows: Sequence[Dict[str, str]],
    candidates: Sequence[str],
) -> List[float]:
    for key in candidates:
        vals = []
        present = False
        for row in rows:
            if key in row:
                present = True
                x = finite_float(row.get(key))
                if x is not None:
                    vals.append(x)
        if present and vals:
            return vals
    return []


def sum_column(
    rows: Sequence[Dict[str, str]],
    candidates: Sequence[str],
) -> Optional[int]:
    vals = numeric_column(rows, candidates)
    if not vals:
        return None
    return int(round(sum(vals)))


def metric_file(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "scheduler_metrics.csv.gz",
        run_dir / "metrics.csv.gz",
        run_dir / "scheduler_metrics.csv",
        run_dir / "metrics.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def request_file(run_dir: Path) -> Path:
    p = run_dir / "requests.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing requests.csv in {run_dir}")
    return p


def metadata_file(run_dir: Path) -> Path:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing run_metadata.json in {run_dir}")
    return p


def metrics_max(
    meta: Dict[str, Any],
    run_dir: Path,
    meta_keys: Sequence[str],
    csv_keys: Sequence[str],
) -> Optional[float]:
    m = meta.get("metrics_polling")
    if isinstance(m, dict):
        value = first_present(m, *meta_keys)
        x = finite_float(value)
        if x is not None:
            return x

    mf = metric_file(run_dir)
    if not mf:
        return None

    rows = read_maybe_gzip_csv(mf)
    vals = numeric_column(rows, csv_keys)
    return max(vals) if vals else None


def configured_lambda(meta: Dict[str, Any]) -> Optional[float]:
    return finite_float(first_present(
        meta,
        "configured_arrival_rate_req_s",
        "configured_lambda_req_s",
        "arrival_rate_req_s",
        "arrival_rate",
    ))


def realized_lambda(meta: Dict[str, Any]) -> Optional[float]:
    return finite_float(first_present(
        meta,
        "realized_schedule_arrival_rate_req_s",
        "realized_arrival_rate_req_s",
        "realized_lambda_req_s",
    ))


def measured_makespan(meta: Dict[str, Any]) -> Optional[float]:
    return finite_float(first_present(
        meta,
        "measured_makespan_s",
        "elapsed_s",
        "measured_elapsed_s",
    ))


def drain_seconds(meta: Dict[str, Any]) -> Optional[float]:
    return finite_float(first_present(
        meta,
        "drain_after_last_arrival_s",
        "drain_s",
    ))


def summary_value(meta: Dict[str, Any], *keys: str) -> Optional[float]:
    return finite_float(nested(meta, "summary", *keys))


def audit_value(meta: Dict[str, Any], *keys: str) -> Any:
    return nested(meta, "audit", *keys)


def server_expected_max_num_seqs(meta: Dict[str, Any]) -> Optional[int]:
    config = meta.get("server_expected_config")
    if isinstance(config, dict):
        x = finite_int(first_present(
            config,
            "max_num_seqs",
            "max_sequences",
        ))
        if x is not None:
            return x

    x = finite_int(first_present(
        meta,
        "server_max_num_seqs",
        "expected_server_max_num_seqs",
        "max_num_seqs",
    ))
    return x


def infer_maxseq_from_name(path: Path) -> Optional[int]:
    text = "/".join(path.parts[-4:]).lower()
    patterns = [
        r"maxseq[_-]?(\d+)",
        r"max[_-]?seqs?[_-]?(\d+)",
        r"(\d+)[_-]?seqs?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    return None


def infer_label(path: Path, meta: Dict[str, Any]) -> str:
    label = first_present(meta, "label", "run_label", "config_label")
    if label:
        return str(label)
    n = server_expected_max_num_seqs(meta)
    if n is None:
        n = infer_maxseq_from_name(path)
    return "default" if n is None else f"maxseq{n}"


def compute_request_metrics(
    rows: Sequence[Dict[str, str]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    ok = successful_rows(rows)

    ttft = numeric_column(ok, [
        "client_first_token_event_ttft_ms",
        "client_ttft_ms",
        "ttft_ms",
    ])
    visible_ttft = numeric_column(ok, [
        "client_first_visible_text_ttft_ms",
        "visible_ttft_ms",
    ])
    e2e = numeric_column(ok, [
        "client_e2e_ms",
        "e2e_ms",
    ])
    itl = numeric_column(ok, [
        "mean_content_event_itl_ms",
        "request_mean_content_event_itl_ms",
        "mean_sse_content_event_itl_ms",
    ])
    output_tokens = sum_column(ok, [
        "output_tokens",
        "completion_tokens",
        "server_completion_tokens",
    ])
    input_tokens = sum_column(ok, [
        "input_tokens",
        "server_prompt_tokens",
        "prompt_tokens",
    ])

    makespan = measured_makespan(meta)
    if makespan is None or makespan <= 0:
        # Keep the analyzer strict rather than silently inventing a denominator.
        req_throughput = summary_value(
            meta, "achieved_request_throughput_req_s"
        )
        out_throughput = summary_value(
            meta, "aggregate_output_tokens_per_s"
        )
    else:
        req_throughput = len(ok) / makespan
        out_throughput = (
            output_tokens / makespan
            if output_tokens is not None else None
        )

    return {
        "requests_total": len(rows),
        "requests_successful": len(ok),
        "p50_ttft_ms": pct(ttft, 0.50),
        "p95_ttft_ms": pct(ttft, 0.95),
        "p99_ttft_ms": pct(ttft, 0.99),
        "p50_visible_ttft_ms": pct(visible_ttft, 0.50),
        "p95_visible_ttft_ms": pct(visible_ttft, 0.95),
        "p50_e2e_ms": pct(e2e, 0.50),
        "p95_e2e_ms": pct(e2e, 0.95),
        "p99_e2e_ms": pct(e2e, 0.99),
        "p50_mean_content_event_itl_ms": pct(itl, 0.50),
        "p95_mean_content_event_itl_ms": pct(itl, 0.95),
        "total_output_tokens": output_tokens,
        "total_input_tokens": input_tokens,
        "achieved_request_throughput_req_s": req_throughput,
        "aggregate_output_tokens_per_s": out_throughput,
    }


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    meta = read_json(metadata_file(run_dir))
    rows = read_csv(request_file(run_dir))
    req = compute_request_metrics(rows, meta)

    max_running = metrics_max(
        meta,
        run_dir,
        ["max_num_requests_running", "max_running"],
        ["num_requests_running", "running", "vllm_running"],
    )
    max_waiting = metrics_max(
        meta,
        run_dir,
        ["max_num_requests_waiting", "max_waiting"],
        ["num_requests_waiting", "waiting", "vllm_waiting"],
    )
    max_kv = metrics_max(
        meta,
        run_dir,
        [
            "max_kv_cache_usage_perc",
            "max_gpu_cache_usage_perc",
            "max_kv_cache_usage",
        ],
        [
            "kv_cache_usage_perc",
            "gpu_cache_usage_perc",
            "kv_cache_usage",
        ],
    )

    configured = configured_lambda(meta)
    realized = realized_lambda(meta)

    # Metadata is authoritative when available.
    for key, aliases in {
        "p50_ttft_ms": ("p50_client_ttft_ms", "p50_ttft_ms"),
        "p95_ttft_ms": ("p95_client_ttft_ms", "p95_ttft_ms"),
        "p99_ttft_ms": ("p99_client_ttft_ms", "p99_ttft_ms"),
        "p50_visible_ttft_ms": ("p50_visible_ttft_ms",),
        "p95_visible_ttft_ms": ("p95_visible_ttft_ms",),
        "p50_e2e_ms": ("p50_e2e_ms",),
        "p95_e2e_ms": ("p95_e2e_ms",),
        "p99_e2e_ms": ("p99_e2e_ms",),
        "p50_mean_content_event_itl_ms": (
            "p50_request_mean_content_event_itl_ms",
            "p50_mean_content_event_itl_ms",
        ),
        "p95_mean_content_event_itl_ms": (
            "p95_request_mean_content_event_itl_ms",
            "p95_mean_content_event_itl_ms",
        ),
        "achieved_request_throughput_req_s": (
            "achieved_request_throughput_req_s",
        ),
        "aggregate_output_tokens_per_s": (
            "aggregate_output_tokens_per_s",
        ),
    }.items():
        v = summary_value(meta, *aliases)
        if v is not None:
            req[key] = v

    metadata_total_output = finite_int(audit_value(
        meta, "total_output_tokens", "observed_output_tokens"
    ))
    if metadata_total_output is not None:
        req["total_output_tokens"] = metadata_total_output

    metadata_total_input = finite_int(audit_value(
        meta, "total_input_tokens"
    ))
    if metadata_total_input is not None:
        req["total_input_tokens"] = metadata_total_input

    return {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "configured_lambda_req_s": configured,
        "realized_lambda_req_s": realized,
        "measured_makespan_s": measured_makespan(meta),
        "drain_s": drain_seconds(meta),
        "max_num_seqs": (
            server_expected_max_num_seqs(meta)
            if server_expected_max_num_seqs(meta) is not None
            else infer_maxseq_from_name(run_dir)
        ),
        "max_running": max_running,
        "max_waiting": max_waiting,
        "max_kv_cache_usage": max_kv,
        **req,
    }


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def complete_run_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    result = []
    for meta in root.rglob("run_metadata.json"):
        d = meta.parent
        if (d / "requests.csv").exists():
            result.append(d)
    return sorted(set(result))


def discover_capacity_runs() -> List[Path]:
    if not CAPACITY_ROOT.exists():
        raise FileNotFoundError(
            f"S3-C capacity directory does not exist:\n  {CAPACITY_ROOT}"
        )

    runs = []
    for d in sorted(CAPACITY_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()

        if "cache_suspect" in name:
            continue

        # Config-variant directories belong to S3-E, not the S3-C curve.
        if (
            "seq" in name
            or "maxseq" in name
            or "default" in name
        ):
            continue

        if (d / "run_metadata.json").exists() and (d / "requests.csv").exists():
            runs.append(d)

    if not runs:
        raise RuntimeError(
            "No formal S3-C load points found under "
            f"{CAPACITY_ROOT}"
        )

    # Fail on duplicate configured lambdas instead of silently choosing one.
    seen: Dict[float, Path] = {}
    for d in runs:
        meta = read_json(d / "run_metadata.json")
        lam = configured_lambda(meta)
        if lam is None:
            raise RuntimeError(f"Missing configured lambda: {d}")
        rounded = round(lam, 9)
        if rounded in seen:
            raise RuntimeError(
                "Duplicate formal S3-C load points for "
                f"lambda={lam}: {seen[rounded]} and {d}"
            )
        seen[rounded] = d

    return runs


def discover_concurrency_runs() -> List[Path]:
    candidates = complete_run_dirs(CONCURRENCY_ROOT)

    # Backward-compatible discovery if optimization runs still live in coarse/.
    if CAPACITY_ROOT.exists():
        for d in complete_run_dirs(CAPACITY_ROOT):
            n = infer_maxseq_from_name(d)
            if n is not None:
                candidates.append(d)

    unique = sorted(set(candidates))
    return unique


def discover_length_matched_runs() -> List[Path]:
    return complete_run_dirs(LENGTH_MATCHED_ROOT)


# -----------------------------------------------------------------------------
# Build summaries
# -----------------------------------------------------------------------------

def build_capacity_summary() -> List[Dict[str, Any]]:
    rows = [summarize_run(d) for d in discover_capacity_runs()]
    rows.sort(key=lambda r: float(r["configured_lambda_req_s"]))

    for row in rows:
        row["experiment"] = "S3-C natural generation"
        row["config"] = "default"

    return rows


def build_concurrency_summary(
    capacity_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Include the formal natural λ=4 default as the comparison baseline.
    defaults = [
        dict(r)
        for r in capacity_rows
        if r.get("configured_lambda_req_s") is not None
        and abs(float(r["configured_lambda_req_s"]) - 4.0) < 1e-9
    ]
    for row in defaults:
        row["experiment"] = "S3-E natural generation"
        row["config"] = "default"
        row["max_num_seqs"] = None
        rows.append(row)

    for d in discover_concurrency_runs():
        row = summarize_run(d)
        meta = read_json(d / "run_metadata.json")
        n = row.get("max_num_seqs")
        label = infer_label(d, meta)
        if n is not None:
            label = f"maxseq{int(n)}"
        row["experiment"] = "S3-E natural generation"
        row["config"] = label
        rows.append(row)

    def order_key(r: Dict[str, Any]) -> tuple:
        if r["config"] == "default":
            return (0, 0)
        n = r.get("max_num_seqs")
        return (1, -(int(n) if n is not None else -1))

    rows.sort(key=order_key)
    return rows


def build_length_matched_summary() -> List[Dict[str, Any]]:
    rows = []
    for d in discover_length_matched_runs():
        row = summarize_run(d)
        meta = read_json(d / "run_metadata.json")
        row["experiment"] = "S3-D length-matched replay"
        row["config"] = infer_label(d, meta)

        expected = finite_int(audit_value(
            meta,
            "expected_total_output_tokens",
            "expected_output_tokens",
            "target_total_output_tokens",
            "target_total",
        ))
        observed = finite_int(audit_value(
            meta,
            "observed_total_output_tokens",
            "observed_output_tokens",
            "total_output_tokens",
            "output_total",
        ))
        length_matches = finite_int(audit_value(
            meta,
            "length_matches",
            "matched_output_lengths",
        ))

        row["expected_output_tokens"] = expected
        row["observed_output_tokens"] = (
            observed
            if observed is not None
            else row.get("total_output_tokens")
        )
        row["length_matches"] = length_matches
        rows.append(row)

    def order_key(r: Dict[str, Any]) -> tuple:
        return (0, 0) if r["config"] == "default" else (
            1,
            r.get("max_num_seqs") or 999999,
        )

    rows.sort(key=order_key)
    return rows


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def finite_xy(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[List[float], List[float]]:
    points = []
    for row in rows:
        x = finite_float(row.get(x_key))
        y = finite_float(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def save_line(
    rows: Sequence[Dict[str, Any]],
    y_key: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    second_y_key: Optional[str] = None,
    second_label: Optional[str] = None,
) -> None:
    x, y = finite_xy(rows, "realized_lambda_req_s", y_key)
    if not x:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label=ylabel)

    if second_y_key:
        x2, y2 = finite_xy(rows, "realized_lambda_req_s", second_y_key)
        if x2:
            ax.plot(x2, y2, marker="o", label=second_label or second_y_key)

    ax.set_xlabel("Realized arrival rate (requests/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if second_y_key:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_capacity(capacity: Sequence[Dict[str, Any]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    save_line(
        capacity,
        "aggregate_output_tokens_per_s",
        "Output throughput (tokens/s)",
        "S3-C: Continuous-batching throughput scaling",
        "01_throughput_vs_arrival_rate.png",
        second_y_key=None,
    )

    x95, y95 = finite_xy(
        capacity, "realized_lambda_req_s", "p95_ttft_ms"
    )
    x99, y99 = finite_xy(
        capacity, "realized_lambda_req_s", "p99_ttft_ms"
    )
    if x95:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x95, y95, marker="o", label="P95 TTFT")
        if x99:
            ax.plot(x99, y99, marker="o", label="P99 TTFT")
        ax.set_xlabel("Realized arrival rate (requests/s)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("S3-C: Tail TTFT and the high-load latency knee")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "02_ttft_vs_arrival_rate.png", dpi=180)
        plt.close(fig)

    x50, y50 = finite_xy(
        capacity, "realized_lambda_req_s", "p50_mean_content_event_itl_ms"
    )
    x95, y95 = finite_xy(
        capacity, "realized_lambda_req_s", "p95_mean_content_event_itl_ms"
    )
    if x50:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x50, y50, marker="o", label="P50 request-mean content-event ITL")
        if x95:
            ax.plot(x95, y95, marker="o", label="P95 request-mean content-event ITL")
        ax.set_xlabel("Realized arrival rate (requests/s)")
        ax.set_ylabel("SSE content-event ITL (ms)")
        ax.set_title("S3-C: Streaming cadence under load")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "03_itl_vs_arrival_rate.png", dpi=180)
        plt.close(fig)


def plot_concurrency(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return

    labels = [str(r["config"]) for r in rows]
    p95 = [finite_float(r.get("p95_ttft_ms")) for r in rows]
    p99 = [finite_float(r.get("p99_ttft_ms")) for r in rows]

    if any(v is not None for v in p95):
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, [math.nan if v is None else v for v in p95],
                marker="o", label="P95 TTFT")
        ax.plot(x, [math.nan if v is None else v for v in p99],
                marker="o", label="P99 TTFT")
        ax.set_xticks(x, labels)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("S3-E: Concurrency-control tail TTFT at λ=4.0")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "04_concurrency_control_ttft.png", dpi=180)
        plt.close(fig)

    itl = [
        finite_float(r.get("p95_mean_content_event_itl_ms"))
        for r in rows
    ]
    if any(v is not None for v in itl):
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, [math.nan if v is None else v for v in itl],
                marker="o")
        ax.set_xticks(x, labels)
        ax.set_ylabel("P95 request-mean content-event ITL (ms)")
        ax.set_title("S3-E: Streaming cadence vs concurrency cap")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_concurrency_control_itl.png", dpi=180)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Key findings
# -----------------------------------------------------------------------------

def fmt(x: Any, digits: int = 2) -> str:
    v = finite_float(x)
    return "n/a" if v is None else f"{v:.{digits}f}"


def find_capacity_row(
    rows: Sequence[Dict[str, Any]],
    configured: float,
) -> Optional[Dict[str, Any]]:
    for row in rows:
        x = finite_float(row.get("configured_lambda_req_s"))
        if x is not None and abs(x - configured) < 1e-9:
            return row
    return None


def find_config(
    rows: Sequence[Dict[str, Any]],
    config: str,
) -> Optional[Dict[str, Any]]:
    return next((r for r in rows if r.get("config") == config), None)


def write_key_findings(
    capacity: Sequence[Dict[str, Any]],
    concurrency: Sequence[Dict[str, Any]],
    matched: Sequence[Dict[str, Any]],
) -> None:
    r375 = find_capacity_row(capacity, 3.75)
    r4 = find_capacity_row(capacity, 4.0)

    default = find_config(concurrency, "default")
    max12 = find_config(concurrency, "maxseq12")
    max10 = find_config(concurrency, "maxseq10")
    max8 = find_config(concurrency, "maxseq8")

    m_default = find_config(matched, "default")
    m_12 = find_config(matched, "maxseq12")

    lines = [
        "# S3 — vLLM Continuous-Batching Key Findings",
        "",
        "## S3-C — Natural-generation capacity sweep",
        "",
        "The capacity sweep uses fresh vLLM processes for each formal load point. "
        "Prefix caching remains enabled, but cross-load-point cache contamination is avoided.",
        "",
    ]

    if r375 and r4:
        lines += [
            "A sharp tail-latency knee was observed between configured "
            "`λ=3.75` and `λ=4.0` req/s:",
            "",
            f"- Realized arrival rate: **{fmt(r375['realized_lambda_req_s'], 3)} → "
            f"{fmt(r4['realized_lambda_req_s'], 3)} req/s**.",
            f"- Output throughput: **{fmt(r375['aggregate_output_tokens_per_s'], 1)} → "
            f"{fmt(r4['aggregate_output_tokens_per_s'], 1)} tok/s**.",
            f"- P95 TTFT: **{fmt(r375['p95_ttft_ms'], 1)} → "
            f"{fmt(r4['p95_ttft_ms'], 1)} ms**.",
            f"- P99 TTFT: **{fmt(r375['p99_ttft_ms'], 1)} → "
            f"{fmt(r4['p99_ttft_ms'], 1)} ms**.",
            f"- P95 request-mean SSE content-event ITL: "
            f"**{fmt(r375['p95_mean_content_event_itl_ms'], 2)} → "
            f"{fmt(r4['p95_mean_content_event_itl_ms'], 2)} ms**.",
            "",
            "Raw throughput was still increasing at the highest tested point, so this "
            "is best described as a **latency knee**, not a demonstrated hard throughput ceiling.",
            "",
        ]

    if r4:
        lines += [
            "At the high-load point, scheduler waiting alone did not identify the onset "
            "of degradation:",
            "",
            f"- Max running requests: **{fmt(r4['max_running'], 0)}**.",
            f"- Max waiting requests: **{fmt(r4['max_waiting'], 0)}**.",
            f"- Peak reported KV-cache usage: **{fmt((finite_float(r4['max_kv_cache_usage']) or 0) * 100, 1)}%** "
            "if the vLLM metric is represented as a 0–1 fraction.",
            "",
            "This supports the interpretation that tail degradation can appear through "
            "active-batch execution/interference before an explicit waiting queue or KV-cache "
            "capacity limit becomes dominant.",
            "",
        ]

    lines += [
        "## S3-E — Concurrency control",
        "",
    ]

    if default and max12:
        lines += [
            "In natural-generation runs at configured `λ=4.0`, limiting "
            "`max_num_seqs` changed the active-concurrency regime:",
            "",
            f"- Default: P95/P99 TTFT **{fmt(default['p95_ttft_ms'],1)} / "
            f"{fmt(default['p99_ttft_ms'],1)} ms**, max running "
            f"**{fmt(default['max_running'],0)}**.",
            f"- `max_num_seqs=12`: P95/P99 TTFT **{fmt(max12['p95_ttft_ms'],1)} / "
            f"{fmt(max12['p99_ttft_ms'],1)} ms**, max running "
            f"**{fmt(max12['max_running'],0)}**, max waiting "
            f"**{fmt(max12['max_waiting'],0)}**.",
        ]
        if max10:
            lines.append(
                f"- `max_num_seqs=10`: P95 TTFT **{fmt(max10['p95_ttft_ms'],1)} ms**, "
                f"max waiting **{fmt(max10['max_waiting'],0)}**."
            )
        if max8:
            lines.append(
                f"- `max_num_seqs=8`: P95 TTFT **{fmt(max8['p95_ttft_ms'],1)} ms**, "
                f"max waiting **{fmt(max8['max_waiting'],0)}**."
            )
        lines += [
            "",
            "The natural-generation results are consistent with a trade-off: excessive "
            "active concurrency can worsen execution interference, while tighter caps can "
            "shift delay into scheduler waiting. `max_num_seqs=12` was the strongest tested "
            "natural-run TTFT operating point, but this is not treated as a universal optimum.",
            "",
        ]

    lines += [
        "## S3-D — Length-matched controlled replay",
        "",
    ]

    if m_default and m_12:
        lines += [
            "The controlled replay forced the same output-token work on the default and "
            "`max_num_seqs=12` configurations.",
            "",
            f"- Default max running: **{fmt(m_default['max_running'],0)}**.",
            f"- `max_num_seqs=12` max running: **{fmt(m_12['max_running'],0)}**.",
            f"- Default P95 TTFT: **{fmt(m_default['p95_ttft_ms'],1)} ms**.",
            f"- `max_num_seqs=12` P95 TTFT: **{fmt(m_12['p95_ttft_ms'],1)} ms**.",
            f"- Default / capped output throughput: "
            f"**{fmt(m_default['aggregate_output_tokens_per_s'],1)} / "
            f"{fmt(m_12['aggregate_output_tokens_per_s'],1)} tok/s**.",
            "",
            "Because the default replay did not exceed the 12-sequence cap, the cap was "
            "not activated and the two configurations were nearly identical. Therefore, "
            "the natural-run improvement should not be described as a universal direct "
            "causal speedup from setting `max_num_seqs=12`; it is conditional on the "
            "request-lifetime/output trajectory producing enough active concurrency for "
            "the cap to matter.",
            "",
        ]
    else:
        lines += [
            "No complete default + maxseq12 length-matched pair was discovered. "
            "The analyzer therefore does not make a controlled-replay comparison.",
            "",
        ]

    lines += [
        "## Measurement boundaries",
        "",
        "- SSE content-event ITL is an application/streaming metric, not GPU TPOT.",
        "- Natural-generation output trajectories can differ across runs even with fixed seeds, "
        "so throughput differences across scheduler configurations are not treated as pure "
        "scheduler-causal effects.",
        "- A zero waiting gauge does not by itself imply that the serving system is healthy.",
        "- The observed knee is specific to this model, GPU, software stack, workload, and finite trace.",
        "",
    ]

    (SUMMARY_DIR / "s3_key_findings.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    capacity = build_capacity_summary()
    concurrency = build_concurrency_summary(capacity)
    matched = build_length_matched_summary()

    write_csv(SUMMARY_DIR / "s3_capacity_summary.csv", capacity)
    write_csv(SUMMARY_DIR / "s3_concurrency_summary.csv", concurrency)
    write_csv(SUMMARY_DIR / "s3_length_matched_summary.csv", matched)

    result = {
        "analysis": "S3 vLLM continuous batching",
        "capacity": capacity,
        "concurrency_control": concurrency,
        "length_matched_replay": matched,
        "interpretation_boundaries": {
            "natural_generation": (
                "Output trajectories/request lifetimes may differ across runs; "
                "do not interpret throughput differences as pure scheduler causality."
            ),
            "length_matched": (
                "Controlled replay changes EOS semantics via fixed token work and "
                "is a secondary validation rather than production-like UX."
            ),
            "streaming_itl": (
                "Content-event ITL is an SSE/application metric, not GPU TPOT."
            ),
        },
    }

    (SUMMARY_DIR / "s3_results_summary.json").write_text(
        json.dumps(clean_for_json(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_capacity(capacity)
    plot_concurrency(concurrency)
    write_key_findings(capacity, concurrency, matched)

    print("=" * 96)
    print("S3 — PUBLIC SUMMARY ANALYSIS COMPLETE")
    print("=" * 96)
    print(f"Capacity points:       {len(capacity)}")
    print(f"Concurrency variants: {len(concurrency)}")
    print(f"Length-matched runs:  {len(matched)}")
    print()
    print(f"Summary directory:    {SUMMARY_DIR}")
    print(f"Figure directory:     {FIG_DIR}")

    r375 = find_capacity_row(capacity, 3.75)
    r4 = find_capacity_row(capacity, 4.0)
    if r375 and r4:
        print()
        print("High-load knee check:")
        print(
            f"  lambda 3.75: P95 TTFT={fmt(r375['p95_ttft_ms'],1)} ms, "
            f"P95 ITL={fmt(r375['p95_mean_content_event_itl_ms'],2)} ms, "
            f"output={fmt(r375['aggregate_output_tokens_per_s'],1)} tok/s"
        )
        print(
            f"  lambda 4.00: P95 TTFT={fmt(r4['p95_ttft_ms'],1)} ms, "
            f"P95 ITL={fmt(r4['p95_mean_content_event_itl_ms'],2)} ms, "
            f"output={fmt(r4['aggregate_output_tokens_per_s'],1)} tok/s"
        )


if __name__ == "__main__":
    main()
