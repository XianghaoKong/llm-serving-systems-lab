# S8-D: Qwen2.5-7B directional validation

## Objective

S8-D checks whether the direction of the 1.5B scheduler result persists when
model compute density increases. It is a bounded external-validity check, not
a second full chunk-budget sweep.

## Calibration

Qwen2.5-7B-Instruct uses the same A100, BF16 dtype, FCFS policy, disabled
prefix cache, 256/8192 background shape, and 24K/16 injected request shape as
S8. Background concurrency is selected before the formal run from C16 and C32.
The rule chooses the smallest candidate with median GPU utilization at or above
90% and scheduler waiting at or below one; if neither reaches 90%, it chooses
the highest queue-safe candidate.

```bash
S8D_PHASE=calibration bash src/run_s8_model_validation.sh
```

The calibration retains one-second `nvidia-smi` telemetry, client-side SSE
events, vLLM metrics, server logs, the selection rule, and its result.

## Frozen formal matrix

After calibration, S8-D runs five rotated blocks of three configurations:

| Configuration | Injection | Chunk budget |
|---|---:|---:|
| `off_32768` | 24576/16 | chunking off, 32K |
| `on_4096` | 24576/16 | 4K |
| `on_1024` | 24576/16 | 1K |

Each of the 15 trials starts a fresh vLLM process, warms the background for 15
seconds, injects one request, and records a 15-second recovery window. A run is
the statistical unit; the analyzer reports medians and bootstrap 95%
intervals.

```bash
S8D_PHASE=formal \
S8D_BACKGROUND_CONCURRENCY=<calibrated value> \
bash src/run_s8_model_validation.sh
```

The extension supports the original direction when 4K and 1K reduce the
background P99 stall ratio relative to unchunked execution. TTFT is reported
as the corresponding cost. The claim remains specific to the tested A100,
model, workload, and engine version.
