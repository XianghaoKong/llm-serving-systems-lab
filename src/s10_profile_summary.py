"""Compact Kineto traces; interval overlap is measured on this rank's GPU.

NCCL identification uses kernel names. These timings exclude CPU-only network
work and are not hardware performance counters or a causal stall attribution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def total(intervals):
    return sum(end-start for start, end in merge(intervals))


def summarize(path):
    raw = path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur",0)>0]
    if not kernels:
        raise ValueError(f"no GPU kernels captured in {path}")
    communication = [e for e in kernels if "nccl" in e.get("name","").lower()]
    compute = [e for e in kernels if "nccl" not in e.get("name","").lower()]
    def intervals(items):
        return [(e["ts"],e["ts"]+e["dur"]) for e in items]
    comm_us = total(intervals(communication))
    compute_us = total(intervals(compute))
    active_us = total(intervals(kernels))
    counts = Counter(e["name"] for e in kernels)
    durations = Counter()
    for e in kernels:
        durations[e["name"]] += e["dur"]
    return {"trace":str(path), "sha256":hashlib.sha256(raw).hexdigest(),
            "gpu_kernel_count":len(kernels), "nccl_kernel_count":len(communication),
            "gpu_active_union_ms":active_us/1000, "nccl_union_ms":comm_us/1000,
            "non_nccl_union_ms":compute_us/1000,
            "nccl_compute_overlap_ms":max(0,comm_us+compute_us-active_us)/1000,
            "kernels":[{"name":name,"count":counts[name],"summed_ms":value/1000}
                       for name,value in durations.most_common()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted(args.root.rglob("trace-rank*.json"))]
    args.output.write_text(json.dumps(rows,indent=2))
    print(f"Validated {len(rows)} GPU traces")


if __name__ == "__main__":
    main()
