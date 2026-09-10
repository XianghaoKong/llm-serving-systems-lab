"""Validate all-rank formal S10 artifacts and summarize independent runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered)-1)*quantile
    left = int(position)
    right = min(left+1, len(ordered)-1)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)


def summarize_run(directory):
    launch = json.loads((directory/"launch.json").read_text())
    case = launch["case"]
    result = dict(case)
    result["run"] = directory.name
    result["accepted"] = False
    if launch.get("returncode") != 0:
        result["reason"] = f"process failed: {launch.get('returncode')}"
        return result
    ranks = []
    for rank in range(case["world"]):
        path = directory/f"rank-{rank}.json"
        if not path.exists():
            result["reason"] = f"missing rank {rank}"
            return result
        ranks.append(json.loads(path.read_text()))
    if not all(r.get("complete") and r.get("formal") for r in ranks):
        result["reason"] = "incomplete, smoke or profiled run"
        return result
    sequences = []
    for rank in ranks:
        if case["backend"] == "zero":
            times = rank["steps_seconds"]
            losses = rank["loss"]
            norms = rank["gradient_norm"]
        else:
            measured = [r for r in rank["records"] if r["measured"]]
            times = [r["seconds"] for r in measured]
            losses = [v for r in measured for v in r["loss"].values()]
            norms = [r["gradient_norm"] for r in measured]
        if len(times) != case.get("steps", 100) or any(not math.isfinite(t) or t<=0 for t in times):
            raise ValueError(f"invalid step count/timing: {directory}")
        if any(not math.isfinite(v) for v in losses+norms if v is not None):
            raise ValueError(f"nonfinite values: {directory}")
        sequences.append(times)
    if any(ts != sequences[0] for ts in sequences[1:]):
        raise ValueError(f"slowest-rank times disagree: {directory}")
    times = sequences[0]
    result.update(accepted=True, steps=len(times), median_step_seconds=statistics.median(times),
                  p95_step_seconds=percentile(times, .95),
                  tokens_per_second=case.get("global_tokens",8192)*len(times)/sum(times),
                  max_rank_peak_allocated_gib=max(r["peak_allocated_bytes"] for r in ranks)/2**30,
                  max_rank_peak_reserved_gib=max(r["peak_reserved_bytes"] for r in ranks)/2**30,
                  per_rank_peak_allocated_gib=[r["peak_allocated_bytes"]/2**30 for r in ranks])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize_run(p.parent) for p in sorted(args.root.rglob("launch.json"))]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"runs.json").write_text(json.dumps(rows, indent=2))
    accepted = [r for r in rows if r["accepted"]]
    fields = ["run","backend","model","world","stage","tp","pp","steps",
              "median_step_seconds","p95_step_seconds","tokens_per_second",
              "max_rank_peak_allocated_gib","max_rank_peak_reserved_gib"]
    with (args.output/"runs.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted)
    print(json.dumps({"runs":len(rows),"accepted_formal":len(accepted),
                      "optimizer_updates":sum(r["steps"] for r in accepted)}))


if __name__ == "__main__":
    main()
