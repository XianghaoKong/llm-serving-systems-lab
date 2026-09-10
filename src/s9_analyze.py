"""Aggregate independent timing blocks, never treat graph samples as repeats."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, q):
    a = sorted(values)
    p = (len(a) - 1) * q
    i = int(p)
    return a[i] + (a[min(i+1, len(a)-1)] - a[i]) * (p-i)


def median_ci(values, seed=2026):
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(4000)]
    return quantile(estimates, .025), quantile(estimates, .975)


def aggregate(rows):
    grouped = defaultdict(list)
    keys = ("op", "rows", "width", "dtype", "backend", "phase")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, records in sorted(grouped.items()):
        blocks = [r["block"] for r in records]
        if len(blocks) != len(set(blocks)):
            raise ValueError(f"duplicate timing block: {key}")
        if not all(r["correctness"] for r in records):
            raise ValueError(f"failed correctness: {key}")
        p50s = [r["p50_us"] for r in records]
        lo, hi = median_ci(p50s)
        out.append(dict(zip(keys, key), blocks=len(records), median_p50_us=statistics.median(p50s),
            median_p95_us=statistics.median(r["p95_us"] for r in records),
            median_p50_ci_low_us=lo, median_p50_ci_high_us=hi,
            max_abs_error=max(r["output_max_abs_error"] for r in records),
            peak_allocated_bytes=max(r["incremental_peak_allocated_bytes"] for r in records)))
    by_key = {tuple(r[k] for k in keys):r for r in out}
    for row in out:
        base = "unfused" if row["op"] == "w4a16" else "eager"
        ref = by_key.get(tuple(row[k] if k != "backend" else base for k in keys))
        row["speedup_vs_eager_or_unfused"] = ref["median_p50_us"] / row["median_p50_us"] if ref else None
        strong = "cublas_dense" if row["op"] == "w4a16" else "compile"
        ref = by_key.get(tuple(row[k] if k != "backend" else strong for k in keys))
        row["speedup_vs_compile_or_dense"] = ref["median_p50_us"] / row["median_p50_us"] if ref else None
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    rows=[]
    for name in args.inputs:
        p=Path(name)
        if not (p.parent/"complete.json").exists():
            raise ValueError(f"not a completed run: {p}")
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    result=aggregate(rows)
    if any(r["blocks"] != 5 for r in result):
        raise ValueError("formal microbenchmark requires exactly five blocks per cell")
    output=Path(args.output);output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result, indent=2))
    with (output/"summary.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=result[0]);writer.writeheader();writer.writerows(result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(13,4.5))
    colors={"eager":"#9299a6","compile":"#333c4a","liger":"#d38b25","triton":"#247e91","tilelang":"#9955a2"}
    for ax,op,width in zip(axes,("rms","swiglu","w4a16"),(3584,3584,3584)):
        subset=[r for r in result if r["op"]==op and r["width"]==width and r["dtype"]=="torch.bfloat16" and r["phase"]=="forward"]
        for backend in sorted({r["backend"] for r in subset}):
            points=sorted((r for r in subset if r["backend"]==backend),key=lambda r:r["rows"])
            ax.plot([r["rows"] for r in points],[r["median_p50_us"] for r in points],"o-",label=backend,color=colors.get(backend))
        ax.set(xscale="log",yscale="log",xlabel="Rows",ylabel="Median block P50 (µs)",title=f"{op} · BF16 · width {width}")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("A100 80GB PCIe · CUDA graph replay · lower is better")
    fig.tight_layout();fig.savefig(output/"kernel_latency.png",dpi=180);plt.close(fig)
    print(json.dumps({"measurement_blocks":len(rows),"cells":len(result)}))


if __name__=="__main__":
    main()
