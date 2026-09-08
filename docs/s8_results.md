# S8 formal results: prefill–decode interference

S8 measures a scheduler-level latency trade-off on one NVIDIA A100 80GB PCIe.
A closed-loop population of 64 long-running decode requests is established
first; one deterministic long prompt is then injected at a known timestamp.
The experiment asks whether limiting the chunked-prefill token budget protects
decode cadence, and what that protection costs in long-request TTFT.

The formal run used vLLM 0.28.0, BF16, Qwen2.5-1.5B-Instruct, FCFS scheduling,
prefix caching disabled, and five rotated blocks. Each server configuration had
one excluded stabilization trial, one control, and one injection at 8K, 16K,
and 24K input tokens. A run is the statistical unit; intervals crossing the
injection boundary are included.

## Main result

Chunked prefill reduced decode stalls only when its token budget was small
enough to force the injected prompt to yield. For a 24K-token prompt, an
unchunked prefill increased the background P99 content-event gap by **51.89×**.
A 1K-token chunk budget reduced that ratio to **4.23×**, a **91.8% reduction**,
and reduced the median maximum gap from **701.2 ms to 70.2 ms**. The trade-off
was an increase in long-request median TTFT from **752.2 ms to 1,415.6 ms**.

![Background decode stall ratio across chunk budgets](../results/s8/analysis/formal/20260908T144356Z/stall_ratio_by_config.png)

The 1K budget is the practical knee in this sweep. Moving from 1K to 512 tokens
reduced the median stall ratio by only 10–13% across prompt lengths, while
increasing long-request TTFT by approximately 49–53%.

![TTFT versus decode-stall trade-off](../results/s8/analysis/formal/20260908T144356Z/ttft_stall_pareto.png)

## Registered 24K-prompt sweep

The P99 ratio is the within-run impact-window P99 background content-event gap
divided by the pre-injection P99 gap. Confidence intervals are bootstrap 95%
intervals over five runs.

| Configuration | P99 stall ratio, median [95% CI] | Median max gap | Event-rate ratio | Long-request TTFT | Stall change vs off |
|---|---:|---:|---:|---:|---:|
| chunking off, 32K budget | 51.89× [39.19, 62.81] | 701.2 ms | 0.06 | 752.2 ms | baseline |
| chunking on, 32K budget | 60.72× [43.32, 63.92] | 696.1 ms | 0.06 | 749.5 ms | +17.0% |
| chunking on, 16K budget | 27.98× [21.95, 34.40] | 394.7 ms | 0.07 | 785.1 ms | −46.1% |
| chunking on, 8K budget | 21.09× [19.47, 27.27] | 316.8 ms | 0.08 | 841.0 ms | −59.4% |
| chunking on, 4K budget | 16.50× [11.00, 17.24] | 181.5 ms | 0.09 | 911.0 ms | −68.2% |
| chunking on, 2K budget | 8.96× [4.10, 9.21] | 105.9 ms | 0.12 | 1,067.4 ms | −82.7% |
| **chunking on, 1K budget** | **4.23× [3.92, 5.47]** | **70.2 ms** | **0.15** | **1,415.6 ms** | **−91.8%** |
| chunking on, 512 budget | 3.78× [3.60, 4.84] | 59.0 ms | 0.20 | 2,158.8 ms | −92.7% |

The equal-budget off/on pair does not improve isolation. Likewise, an 8K
budget does not help the 8K prompt, while 4K and smaller budgets do. For the
24K prompt, the effect begins at 16K. This threshold relationship supports a
mechanism in which scheduling opportunities between prefill chunks, rather
than the feature toggle alone, allow decode work to resume.

## The 1K knee across prompt lengths

| Prompt | Unchunked stall / TTFT | 1K stall / TTFT | 512 stall / TTFT |
|---:|---:|---:|---:|
| 8K | 13.87× / 220.4 ms | 3.02× / 403.8 ms | 2.64× / 600.4 ms |
| 16K | 31.43× / 470.2 ms | 3.78× / 867.9 ms | 3.40× / 1,296.5 ms |
| 24K | 51.89× / 752.2 ms | 4.23× / 1,415.6 ms | 3.78× / 2,158.8 ms |

The unchunked stall grows with prompt length. A 1K budget compresses all three
cases into a much narrower 3.02–4.23× range, at an approximately 83–88% TTFT
increase relative to the corresponding unchunked run.

## Controls and validity checks

- All **200/200 trial directories** completed: 40 excluded stabilization, 40
  control, and 120 injection trials.
- The analyzer consumed **160 registered statistical trials** and produced 32
  aggregate rows: eight configurations × four trial conditions.
- All **120/120 interferers** completed successfully with the exact requested
  input length and 16 output tokens; there were no request errors.
- Two interferers delivered 16 generated tokens in 15 SSE content events. This
  is why the primary metric is named content-event gap. It does not affect the
  background-gap metric or the long-request first-event TTFT.
- Metric scraping produced zero errors. Scheduler waiting and preemptions were
  zero in every formal trial. Median peak KV occupancy stayed below 8.1%, so the
  effect is not explained by request admission or KV exhaustion.
- No-injection controls had median P99 ratios between 0.99× and 1.06× across
  configurations, supporting the within-run baseline design.

These results are specific to the tested A100, model, vLLM version, background
shape, and FCFS policy. They establish a causal scheduler trade-off under a
controlled decode population; they do not claim a universal production
default for every workload.

## Artifacts and reproduction

The public analysis directory contains the complete aggregate and per-trial
summaries, environment record, protocol snapshot, plots, and structured JSON:

- [`aggregate_summary.csv`](../results/s8/analysis/formal/20260908T144356Z/aggregate_summary.csv)
- [`trial_summary.csv`](../results/s8/analysis/formal/20260908T144356Z/trial_summary.csv)
- [`analysis.json`](../results/s8/analysis/formal/20260908T144356Z/analysis.json)
- [`audit.json`](../results/s8/analysis/formal/20260908T144356Z/audit.json)
- [`environment.txt`](../results/s8/analysis/formal/20260908T144356Z/environment.txt)
- [`protocol.txt`](../results/s8/analysis/formal/20260908T144356Z/protocol.txt)

The raw run identifier is `20260908T144356Z`. Raw prompts, content-event traces,
metrics, and server logs remain on the persistent Runpod volume and are excluded
from Git. Re-run the registered experiment with:

```bash
S8_PHASE=formal \
S8_PROTOCOL_FROZEN=1 \
S8_BACKGROUND_CONCURRENCY=64 \
S8_INTERFERER_COUNT=1 \
bash src/run_s8_interference.sh
```
