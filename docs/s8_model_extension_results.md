# S8-D: Qwen2.5-7B directional validation results

## Calibration

The extension used Qwen2.5-7B-Instruct in BF16 on the same NVIDIA A100 80GB
PCIe. C16 and C32 were calibrated before the formal matrix. Both reached 100%
median utilization among nonzero GPU samples with zero scheduler waiting and
zero preemptions. The frozen selection rule chose the smallest saturated,
queue-safe candidate, C16.

| Candidate | Active GPU samples | Busy-sample fraction | Active utilization median | Maximum waiting |
|---:|---:|---:|---:|---:|
| C16 | 20 / 53 | 37.7% | 100% | 0 |
| C32 | 21 / 51 | 41.2% | 100% | 0 |

The busy-sample fraction keeps tokenizer and request-preparation idle time
visible instead of treating zero-utilization samples as model execution.

## Formal result

All 15 preregistered trials completed: five rotated runs per configuration at
C16 with one 24K-input, 16-output-token injection. There were zero request
errors and zero preemptions.

| Chunked-prefill budget | P99 stall ratio, median [95% CI] | Maximum SSE gap, median | Long-request TTFT, median | Stall reduction vs off |
|---:|---:|---:|---:|---:|
| off / 32K | 112.72x [105.57, 122.45] | 2,212.4 ms | 2,257.3 ms | baseline |
| on / 4K | 24.94x [20.10, 26.95] | 487.4 ms | 2,401.3 ms | 77.9% |
| on / 1K | 7.53x [6.26, 8.05] | 366.8 ms | 2,735.5 ms | 93.3% |

![Qwen2.5-7B directional validation](../results/s8/analysis/model_validation_7b/20260909T215626Z/model_extension_7b.png)

The direction from the 1.5B study persists at 7B. A 4K budget reduced the
background stall ratio by 77.9% for a 6.4% TTFT increase. A 1K budget reduced
the stall ratio by 93.3% for a 21.2% TTFT increase. The absolute 7B stalls are
larger and its best operating point is not inferred from this bounded matrix;
the result establishes directional external validity for the scheduler trade-
off on this GPU and software stack.

## Artifacts

- `results/s8/analysis/model_validation_7b/calibration/20260909T214447Z/selection.json`
- `results/s8/analysis/model_validation_7b/calibration/20260909T214447Z/calibration_summary.csv`
- `results/s8/analysis/model_validation_7b/20260909T215626Z/aggregate_summary.csv`
- `results/s8/analysis/model_validation_7b/20260909T215626Z/trial_summary.csv`
- `results/s8/analysis/model_validation_7b/20260909T215626Z/audit.json`
- `results/s8/analysis/model_validation_7b/20260909T215626Z/environment.txt`
- `results/s8/analysis/model_validation_7b/20260909T215626Z/protocol.txt`

Raw client traces, GPU samples, metrics, and server logs remain on the
experiment volume and are excluded from Git.
