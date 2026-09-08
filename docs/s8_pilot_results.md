# S8 calibration and pilot results

These results freeze the workload parameters for the formal prefill/decode
interference experiment. They were collected on one NVIDIA A100 80GB PCIe
with vLLM 0.28.0, PyTorch 2.13.0+cu130, BF16, Qwen2.5-1.5B-Instruct, prefix
caching disabled, FCFS scheduling, and a 32,768-token model limit.

## Background concurrency calibration

Calibration used chunked prefill with an 8,192-token scheduler budget and one
16,384-token injected prefill. The baseline columns come from the pre-impact
window in each injection trial.

| Concurrency | Baseline P95 gap (ms) | Waiting | Preemptions | Peak KV use | Impact P99 gap (ms) |
|---:|---:|---:|---:|---:|---:|
| 8 | 8.71 | 0 | 0 | 1.52% | 227.73 |
| 16 | 9.04 | 0 | 0 | 2.30% | 218.64 |
| 32 | 9.29 | 0 | 0 | 3.64% | 231.18 |
| 64 | 8.97 | 0 | 0 | 5.82% | 245.80 |

Concurrency 64 is the highest registered point. It has no scheduler waiting,
no preemption, and no baseline P95 gap knee, so the formal experiment freezes
background concurrency at 64.

Raw calibration run identifiers:

- C=8: `20260908T141345Z`
- C=16: `20260908T141724Z`
- C=32: `20260908T141943Z`
- C=64: `20260908T142203Z`

## Pilot scheduler sweep

The pilot used concurrency 64, one 16,384-token interferer, and run identifier
`20260908T142545Z`.

| Configuration | Impact P99 gap (ms) | P99 stall ratio | Event-rate ratio | Long-request TTFT (ms) |
|---|---:|---:|---:|---:|
| chunking off, budget 32,768 | 385.36 | 37.47x | 0.102 | 449.47 |
| chunking on, budget 32,768 | 390.11 | 36.16x | 0.097 | 448.31 |
| chunking on, budget 8,192 | 240.60 | 14.91x | 0.158 | 511.00 |
| chunking on, budget 2,048 | 79.84 | 7.47x | 0.162 | 615.52 |
| chunking on, budget 512 | 40.19 | 4.06x | 0.226 | 1,134.70 |

The equal-budget off/on pair behaves similarly because a 32,768-token budget
does not force this 16,384-token prompt to yield. Smaller budgets sharply
reduce decode stalls while increasing long-request TTFT, exposing the intended
latency tradeoff.

The unchunked P99 stall ratio is above the preregistered 1.20 threshold, so the
formal interferer count remains one. All five injected requests reported exact
16,384-token input and 16-token output counts. Each produced 16 SSE content
events, all 64 background requests remained active through the impact window,
metric scraping had no errors, and no trial recorded scheduler waiting or
preemption.
