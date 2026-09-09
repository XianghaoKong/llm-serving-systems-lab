#!/usr/bin/env python3
"""Extract portable GPU-kernel and busy-interval data from Nsight SQLite.

Nsight Systems changes its SQLite schema across releases. This extractor
discovers the kernel and string columns instead of depending on one toolkit
version, then emits a stable CSV/JSON schema for the S8-B analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


KERNEL_TABLE_CANDIDATES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
)
NAME_COLUMN_CANDIDATES = (
    "demangledName",
    "shortName",
    "mangledName",
    "name",
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_names(connection: sqlite3.Connection) -> List[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]


def table_columns(connection: sqlite3.Connection, table: str) -> List[str]:
    return [
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({quote_identifier(table)})"
        )
    ]


def choose_kernel_table(tables: Sequence[str]) -> str:
    by_lower = {name.lower(): name for name in tables}
    for candidate in KERNEL_TABLE_CANDIDATES:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    matches = [name for name in tables if "kernel" in name.lower()]
    if not matches:
        raise RuntimeError("Nsight SQLite contains no GPU kernel table")
    return sorted(matches)[0]


def load_string_ids(
    connection: sqlite3.Connection,
    tables: Sequence[str],
) -> Mapping[int, str]:
    table = next((name for name in tables if name.lower() == "stringids"), None)
    if table is None:
        return {}
    columns = table_columns(connection, table)
    id_column = next((name for name in columns if name.lower() == "id"), None)
    value_column = next(
        (name for name in columns if name.lower() in {"value", "string"}),
        None,
    )
    if id_column is None or value_column is None:
        return {}
    query = (
        f"SELECT {quote_identifier(id_column)}, "
        f"{quote_identifier(value_column)} FROM {quote_identifier(table)}"
    )
    return {
        int(identifier): str(value)
        for identifier, value in connection.execute(query)
        if identifier is not None and value is not None
    }


def resolve_name(raw_name: Any, strings: Mapping[int, str]) -> str:
    if raw_name is None:
        return "<unknown>"
    if isinstance(raw_name, int) and raw_name in strings:
        return strings[raw_name]
    return str(raw_name)


def extract_kernel_rows(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    tables = table_names(connection)
    table = choose_kernel_table(tables)
    columns = table_columns(connection, table)
    by_lower = {name.lower(): name for name in columns}
    start_column = by_lower.get("start")
    end_column = by_lower.get("end")
    if start_column is None or end_column is None:
        raise RuntimeError(f"kernel table {table} has no start/end columns")
    name_column = next(
        (by_lower.get(candidate.lower()) for candidate in NAME_COLUMN_CANDIDATES
         if by_lower.get(candidate.lower()) is not None),
        None,
    )
    stream_column = by_lower.get("streamid") or by_lower.get("stream")
    selected = [start_column, end_column]
    if name_column is not None:
        selected.append(name_column)
    if stream_column is not None:
        selected.append(stream_column)
    query = (
        "SELECT " + ", ".join(quote_identifier(name) for name in selected)
        + f" FROM {quote_identifier(table)} ORDER BY {quote_identifier(start_column)}"
    )
    strings = load_string_ids(connection, tables)
    rows: List[Dict[str, Any]] = []
    for values in connection.execute(query):
        start_ns = int(values[0])
        end_ns = int(values[1])
        if end_ns < start_ns:
            continue
        offset = 2
        raw_name: Any = None
        if name_column is not None:
            raw_name = values[offset]
            offset += 1
        stream_id: Optional[Any] = None
        if stream_column is not None:
            stream_id = values[offset]
        rows.append(
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "name": resolve_name(raw_name, strings),
                "stream_id": stream_id,
            }
        )
    if not rows:
        raise RuntimeError(f"kernel table {table} contains no usable rows")
    origin_ns = min(int(row["start_ns"]) for row in rows)
    for row in rows:
        row["relative_start_ms"] = (int(row["start_ns"]) - origin_ns) / 1e6
        row["relative_end_ms"] = (int(row["end_ns"]) - origin_ns) / 1e6
        row["duration_ms"] = int(row["duration_ns"]) / 1e6
    return rows


def merge_busy_intervals(
    kernels: Iterable[Mapping[str, Any]],
    gap_threshold_ns: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(
        (
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in kernels
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    if not ordered:
        return []
    origin_ns = ordered[0][0]
    intervals: List[Dict[str, Any]] = []
    start_ns, end_ns = ordered[0]
    kernel_count = 1
    for next_start, next_end in ordered[1:]:
        if next_start <= end_ns + gap_threshold_ns:
            end_ns = max(end_ns, next_end)
            kernel_count += 1
            continue
        intervals.append(
            busy_interval_row(start_ns, end_ns, kernel_count, origin_ns)
        )
        start_ns, end_ns, kernel_count = next_start, next_end, 1
    intervals.append(busy_interval_row(start_ns, end_ns, kernel_count, origin_ns))
    return intervals


def busy_interval_row(
    start_ns: int,
    end_ns: int,
    kernel_count: int,
    origin_ns: int,
) -> Dict[str, Any]:
    return {
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "relative_start_ms": (start_ns - origin_ns) / 1e6,
        "relative_end_ms": (end_ns - origin_ns) / 1e6,
        "duration_ms": (end_ns - start_ns) / 1e6,
        "kernel_count": kernel_count,
    }


def write_csv_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--busy-gap-us",
        type=float,
        default=50.0,
        help="Merge adjacent kernels separated by at most this many microseconds.",
    )
    args = parser.parse_args()
    if args.busy_gap_us < 0:
        parser.error("--busy-gap-us must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.sqlite) as connection:
        kernels = extract_kernel_rows(connection)
    busy = merge_busy_intervals(kernels, int(args.busy_gap_us * 1000.0))
    write_csv_gz(args.output_dir / "kernel_events.csv.gz", kernels)
    write_csv_gz(args.output_dir / "busy_intervals.csv.gz", busy)
    span_ns = max(int(row["end_ns"]) for row in kernels) - min(
        int(row["start_ns"]) for row in kernels
    )
    active_ns = sum(int(row["duration_ns"]) for row in busy)
    summary = {
        "schema_version": 1,
        "sqlite": str(args.sqlite),
        "busy_gap_us": args.busy_gap_us,
        "kernel_count": len(kernels),
        "busy_interval_count": len(busy),
        "gpu_span_ms": span_ns / 1e6,
        "gpu_active_ms": active_ns / 1e6,
        "gpu_active_fraction": active_ns / span_ns if span_ns else None,
        "max_kernel_duration_ms": max(float(row["duration_ms"]) for row in kernels),
        "max_busy_interval_ms": max(float(row["duration_ms"]) for row in busy),
    }
    (args.output_dir / "nsys_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
