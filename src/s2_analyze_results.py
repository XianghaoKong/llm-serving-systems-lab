#!/usr/bin/env python3
"""
Analyze S2 coarse Poisson-arrival runs from the project directory.

Expected layout:
results/s2/raw/coarse/lambda_*/requests.csv
results/s2/raw/coarse/lambda_*/run_metadata.json

Outputs:
results/s2/summary/s2_coarse_summary.csv
results/s2/summary/s2_category_summary.csv
results/s2/figures/*.png
results/s2/summary/S2_COARSE_SUMMARY.md
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "s2" / "raw" / "coarse"
SUMMARY_DIR = ROOT / "results" / "s2" / "summary"
FIG_DIR = ROOT / "results" / "s2" / "figures"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def pct(s, q):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else np.nan

runs = []
for d in sorted(RAW.glob("lambda_*")):
    req = d / "requests.csv"
    meta = d / "run_metadata.json"
    if req.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("phase") == "coarse" and m.get("status") == "complete":
            runs.append((req, meta, m))

if not runs:
    raise SystemExit(f"No complete S2 coarse runs found under {RAW}")

summary_rows = []
category_rows = []

for req_path, meta_path, meta in runs:
    df = pd.read_csv(req_path)
    ok = df[df["status"] == "ok"].copy()
    lam = float(meta["configured_arrival_rate_req_s"])
    realized = float(meta["realized_schedule_arrival_rate_req_s"])
    rho = float(meta["realized_rho_from_s1_mean_service"])

    summary_rows.append({
        "configured_lambda_req_s": lam,
        "realized_lambda_req_s": realized,
        "realized_rho": rho,
        "requests": len(df),
        "successful": len(ok),
        "failed": int((df["status"] != "ok").sum()),
        "s1_hash_matches": int(pd.to_numeric(ok["s1_hash_match"], errors="coerce").fillna(0).sum()),
        "max_client_inflight": int(meta["max_client_inflight"]),
        "dispatch_lag_p50_ms": pct(df["dispatch_lag_ms"], .50),
        "dispatch_lag_p95_ms": pct(df["dispatch_lag_ms"], .95),
        "queue_mean_ms": float(pd.to_numeric(ok["server_queue_ms"], errors="coerce").mean()),
        "queue_p50_ms": pct(ok["server_queue_ms"], .50),
        "queue_p95_ms": pct(ok["server_queue_ms"], .95),
        "queue_p99_ms": pct(ok["server_queue_ms"], .99),
        "queue_max_ms": float(pd.to_numeric(ok["server_queue_ms"], errors="coerce").max()),
        "client_ttft_p50_ms": pct(ok["client_first_token_event_ttft_ms"], .50),
        "client_ttft_p95_ms": pct(ok["client_first_token_event_ttft_ms"], .95),
        "client_ttft_p99_ms": pct(ok["client_first_token_event_ttft_ms"], .99),
        "visible_ttft_p50_ms": pct(ok["client_first_visible_text_ttft_ms"], .50),
        "visible_ttft_p95_ms": pct(ok["client_first_visible_text_ttft_ms"], .95),
        "client_e2e_p50_ms": pct(ok["client_e2e_ms"], .50),
        "client_e2e_p95_ms": pct(ok["client_e2e_ms"], .95),
        "client_e2e_p99_ms": pct(ok["client_e2e_ms"], .99),
        "model_ttft_p50_ms": pct(ok["server_model_ttft_ms"], .50),
        "model_ttft_p95_ms": pct(ok["server_model_ttft_ms"], .95),
        "tpot_p50_ms": pct(ok["server_mean_tpot_ms"], .50),
        "tpot_p95_ms": pct(ok["server_mean_tpot_ms"], .95),
        "queue_ttft_inflation_corr": float(
            pd.to_numeric(ok["server_queue_ms"], errors="coerce")
              .corr(pd.to_numeric(ok["client_ttft_inflation_vs_s1_ms"], errors="coerce"))
        ),
    })

    for cat, g in ok.groupby("workload_category"):
        category_rows.append({
            "configured_lambda_req_s": lam,
            "realized_lambda_req_s": realized,
            "realized_rho": rho,
            "workload_category": cat,
            "n": len(g),
            "queue_p50_ms": pct(g["server_queue_ms"], .50),
            "queue_p95_ms": pct(g["server_queue_ms"], .95),
            "client_ttft_p50_ms": pct(g["client_first_token_event_ttft_ms"], .50),
            "client_ttft_p95_ms": pct(g["client_first_token_event_ttft_ms"], .95),
            "client_e2e_p50_ms": pct(g["client_e2e_ms"], .50),
            "tpot_p50_ms": pct(g["server_mean_tpot_ms"], .50),
        })

summary = pd.DataFrame(summary_rows).sort_values("configured_lambda_req_s").reset_index(drop=True)
cats = pd.DataFrame(category_rows).sort_values(["configured_lambda_req_s", "workload_category"])
summary.to_csv(SUMMARY_DIR / "s2_coarse_summary.csv", index=False)
cats.to_csv(SUMMARY_DIR / "s2_category_summary.csv", index=False)

x = summary["realized_lambda_req_s"].to_numpy()

def line(cols, labels, ylabel, title, name):
    plt.figure(figsize=(8,5))
    for c, label in zip(cols, labels):
        plt.plot(x, summary[c].to_numpy(), marker="o", label=label)
    plt.xlabel("Realized arrival rate (requests/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(cols) > 1:
        plt.legend()
    plt.grid(True, alpha=.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180)
    plt.close()

line(["queue_p50_ms","queue_p95_ms","queue_p99_ms"], ["P50","P95","P99"],
     "Queue latency (ms)", "S2 Queueing Delay vs Realized Arrival Rate",
     "01_queue_vs_arrival_rate.png")
line(["client_ttft_p50_ms","client_ttft_p95_ms","client_ttft_p99_ms"], ["P50","P95","P99"],
     "Client first-token-event TTFT (ms)", "S2 TTFT vs Realized Arrival Rate",
     "02_ttft_vs_arrival_rate.png")

plt.figure(figsize=(8,5))
plt.plot(summary["realized_rho"], summary["queue_p50_ms"], marker="o", label="Queue P50")
plt.plot(summary["realized_rho"], summary["queue_p95_ms"], marker="o", label="Queue P95")
plt.axvline(1.0, linestyle="--", linewidth=1.2, label="ρ = 1")
plt.xlabel("Realized offered load ρ")
plt.ylabel("Queue latency (ms)")
plt.title("S2 Queueing Delay vs Offered Load")
plt.legend()
plt.grid(True, alpha=.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_queue_vs_rho.png", dpi=180)
plt.close()

line(["tpot_p50_ms","tpot_p95_ms"], ["TPOT P50","TPOT P95"],
     "Server mean TPOT (ms/token)", "S2 GPU Decode Cost Remains Stable Under Load",
     "04_tpot_vs_arrival_rate.png")
line(["client_e2e_p50_ms","client_e2e_p95_ms","client_e2e_p99_ms"], ["P50","P95","P99"],
     "Client E2E latency (ms)", "S2 End-to-End Latency vs Realized Arrival Rate",
     "06_e2e_vs_arrival_rate.png")

plt.figure(figsize=(8,5))
plt.plot(x, summary["max_client_inflight"], marker="o")
plt.xlabel("Realized arrival rate (requests/s)")
plt.ylabel("Max concurrent in-flight requests")
plt.title("S2 Backlog Growth vs Realized Arrival Rate")
plt.grid(True, alpha=.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_max_inflight_vs_arrival_rate.png", dpi=180)
plt.close()

print(summary.to_string(index=False))
print(f"\nWrote summary to {SUMMARY_DIR}")
print(f"Wrote figures to {FIG_DIR}")
