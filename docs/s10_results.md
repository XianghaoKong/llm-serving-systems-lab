# S10: capacity, throughput and communication in distributed training

Status: core experiment complete on 2026-09-10; full archive verified and compute stopped.

## What this experiment measures

This is a training-system extension to the serving and kernel studies. It tests
how optimizer-state sharding and tensor/pipeline parallelism trade memory for
throughput on one four-GPU host. DeepSpeed and Megatron supply the distributed
algorithms; this repository supplies the controlled workload, correctness gate,
measurement wrappers, capacity probes and analysis.

The DeepSpeed model contains **7,615,616,512 parameters**: random Qwen-shaped
weights with 28 layers, hidden size 3584, intermediate size 18944, 28 query heads,
four KV heads and vocabulary 152064. Megatron uses matching headline dimensions,
RMSNorm, SwiGLU, rotary positions and untied embeddings, but its implementation,
initialization and mock-data path differ. Compare layouts **within each backend**.
No pretrained checkpoint is downloaded or evaluated for model quality.

## Hardware and measurement contract

| Property | Setting |
|---|---|
| Host | 4 × NVIDIA A100-SXM4-80GB; every GPU pair reports NV12 |
| Runtime | PyTorch 2.8.0+cu128; DeepSpeed 0.17.5 |
| Megatron | core_v0.14.0, commit `23e00ed0963c35382dfe8a5a94fb3cda4d21e133` |
| Precision | BF16 model; AdamW with FP32 master/optimizer state |
| Optimizer | LR 1e-4, betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, clip norm 1 |
| Batch | Sequence 512, microbatch 1, 8192 global input tokens per update |
| Repetition | Three independently launched runs per layout, rotated order |
| Timing | 20 warmup + 100 measured optimizer updates per run |
| Memory | Maximum over ranks of peak allocated and reserved CUDA memory, reset after warmup |
| Profiling | Separate processes; 20 warmup + two captured updates, all four ranks |

Global batch size is 16 sequences. DeepSpeed accumulation is eight microbatches
on two ranks and four on four ranks. The tested Megatron layouts have DP=1 and
16 microbatches per update. CUDA synchronization brackets each complete optimizer
update; timing uses the slowest rank. Inter-update barriers, timing reductions
and logging are outside the measured interval. DeepSpeed token tensors are
preloaded on the GPU; Megatron's upstream data fetch is inside `train_step`.

DeepSpeed uses Torch AdamW with `foreach=False`, 50-million-element communication
buckets and `overlap_comm=False`. Megatron uses its local transformer backend
and Torch fallback normalization/optimizer, without Transformer Engine, Apex or
the optional fused paths disabled in the recorded launch command. Neither matrix
uses CPU offload or activation recomputation. These choices define the baseline;
the study does not claim to exhaust production tuning options.

Run throughput is `8192 × measured updates / sum(slowest-rank update seconds)`.
Aggregate values are medians over the three runs; ranges retain process-to-process
variation. P95 is computed within each run, then summarized across runs. The 100
correlated updates are not treated as 100 independent experiments, and three runs
do not support a strong confidence-interval claim.

## Numerical gate

A finite-loss smoke test was insufficient: DeepSpeed 0.17.6 ZeRO-2 failed a
controlled one-update reference comparison with **114.43% relative update L2
error**. Version 0.18.3 passed the initial update-only check but was excluded by
a strengthened gradient-norm check. Version 0.17.5 passed both gates on two and
four GPUs for ZeRO-1/2/3: 18 rank-level records, maximum update error 3.74% and
maximum gradient-norm discrepancy 0.00244%.

See the [numerical regression case study](s10_numerical_regression.md) for the
reference arithmetic, rejected-version evidence and limits. This validates one
representative mixed-precision update, not all graphs or long-run convergence.
Repeated synthetic batches can be memorized quickly; decreasing or near-zero
loss is not presented as a model-quality result.

## Capacity probes

| Backend | Layout | Outcome under the recorded configuration |
|---|---|---|
| DeepSpeed 0.17.5 | One GPU, ZeRO-0 | OOM |
| DeepSpeed 0.17.5 | Two GPUs, ZeRO-1 or ZeRO-2 | OOM during optimizer-state allocation |
| DeepSpeed 0.17.5 | Two GPUs, ZeRO-3 | Passed; admitted to formal matrix |
| DeepSpeed 0.17.5 | Four GPUs, ZeRO-1/2/3 | Passed; admitted to formal matrix |
| Megatron | Two GPUs, TP2 or PP2 | OOM |
| Megatron | Four GPUs, TP4, PP4 or TP2+PP2 | Passed; admitted to formal matrix |

These are capacity observations for these allocator, buffer and optimizer
settings. They do not imply that a 7B-class model can never train on two GPUs
with other settings. State-size arithmetic is a lower bound: actual accumulation
dtypes, activation buffers, allocator reservations and optimizer temporaries
change the peak. Failed cases have no throughput score.

## Formal results

All **21 formal runs / 2100 measured updates / 78 rank records** passed the
acceptance checks. Each cell below summarizes three independent runs. Throughput
is shown as median [minimum, maximum]; latency and memory are run medians.

| Backend / layout | Input tok/s, median [range] | Step P50 (s) | Step P95 (s) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| ZeRO-3 / 2 GPUs | 2,577.8 [2,510.6, 2,598.4] | 3.1572 | 3.2982 | 68.26 | 75.87 |
| ZeRO-1 / 4 GPUs | 8,174.6 [8,174.2, 8,175.8] | 1.0019 | 1.0046 | 56.76 | 73.08 |
| ZeRO-2 / 4 GPUs | 5,851.4 [5,849.0, 5,852.8] | 1.4003 | 1.4026 | 56.76 | 68.00 |
| ZeRO-3 / 4 GPUs | 4,875.1 [4,802.2, 4,904.6] | 1.6426 | 1.8754 | 39.77 | 57.82 |
| TP1 + PP4 / 4 GPUs | 6,637.7 [6,635.6, 6,637.7] | 1.2347 | 1.2372 | 44.70 | 48.60 |
| TP2 + PP2 / 4 GPUs | 6,196.9 [6,169.8, 6,279.0] | 1.3217 | 1.3320 | 39.15 | 41.69 |
| TP4 + PP1 / 4 GPUs | 4,463.6 [4,450.6, 4,749.7] | 1.8241 | 1.8798 | 39.12 | 40.22 |

![S10 throughput and memory](../results/s10/analysis/distributed_training.png)

At four ranks, ZeRO-3 reduces measured peak allocated memory by **29.9%** relative to ZeRO-1, at a **40.4%** throughput cost. ZeRO-2's peak is unchanged in this setup, while throughput is **28.4%** lower. The state-sharding label alone does not predict the complete-update memory peak.

ZeRO-3 scales from two to four ranks by **1.89×**, or **94.6%** strong-scaling efficiency at fixed global tokens. The two-rank runs still occupy the same rented four-GPU Pod; the unused GPUs do not reduce the rental bill.

Within Megatron's local backend, PP4 has **1.49×** TP4 throughput, using **5.58 GiB** more peak allocated memory. TP2+PP2 has **1.39×** TP4 throughput with a **0.04 GiB** peak difference. TP4's process-to-process spread is visible in the range and plot; these are workload-specific observations, not a universal parallelism ranking.

## Separate communication profiles

Six separate profiling runs produced **24 GPU traces**, one per rank. The
table shows ranges across four ranks in a two-update capture; ranks are not
independent repeat trials.

| Four-GPU layout | NCCL kernel count per rank | NCCL / GPU-active union | NCCL/non-NCCL overlap (ms) |
|---|---:|---:|---:|
| TP1 + PP4 / 4 GPUs | 49–89 | 12.7–35.4% | 0.0–0.0 |
| TP2 + PP2 / 4 GPUs | 1,969–2,065 | 16.1–42.5% | 70.9–162.4 |
| TP4 + PP1 / 4 GPUs | 3,919–3,919 | 42.3–50.5% | 186.7–241.4 |
| ZeRO-1 / 4 GPUs | 240–240 | 19.5–19.9% | 0.0–0.0 |
| ZeRO-2 / 4 GPUs | 930–930 | 42.1–42.9% | 0.0–0.0 |
| ZeRO-3 / 4 GPUs | 2,724–2,724 | 54.1–71.8% | 559.4–950.2 |

The two-update capture contains 240 NCCL kernels per rank for ZeRO-1, 930 for
ZeRO-2 and 2724 for ZeRO-3. This is direct evidence of increased communication
launch activity in the tested sharding configurations. PP4 records 49–89 NCCL
kernels per rank versus 3919 for TP4, alongside different active-time fractions
and rank balance. These observations help characterize the throughput trade-off;
they do not isolate communication as its only cause.

Kineto kernel names identify NCCL work. The analysis unions intervals on each
rank's GPU and measures overlap with non-NCCL kernels. An active-time fraction
is not the fraction of wall time lost to communication: NCCL kernels can wait,
compute can overlap communication, and profiles include synchronization used by
the measurement harness. Kernel timings are not HBM/NVLink hardware counters,
and these traces alone cannot prove bandwidth saturation or a causal bottleneck.

## Artifacts and reproduction

The complete archive was downloaded and SHA-256 verified before the experiment
Pod received its stop request at **2026-09-10T10:27:08.3980756Z**. A subsequent API check
confirmed the Pod stopped. Existing stopped Pods and retained data were preserved.

- Archive: `s10-artifacts-20260910.tgz` (local backup; raw traces are not committed).
- SHA-256: `3322eb94fc40a37db123c7214f8609a0d953d65a154a62fcf8cf6feaabe556f5`.
- Integrity: 21 formal runs, 2100 updates, 78 rank records, 18 accepted numerical records and 24 raw-trace hashes; formal launch source hashes match archived source.
- Compute rate: $6.36/hour for the four-GPU host.
- Creation-to-stop estimate: **$12.83** over **2.017 hours**, including setup, rejected pilots and archive transfer. This is a timestamp-based estimate, not a finalized invoice; retained storage is billed separately.

See [archive verification](../results/s10/environment/archive-verification.json),
[cost basis](../results/s10/environment/cost.json),
[accepted per-run statistics](../results/s10/analysis/runs.csv),
[all-run aggregates](../results/s10/analysis/summary.json) and
[profile summaries](../results/s10/profile/summary.json).

Published data include all-rank formal JSON, capacity logs, numerical checks,
compact profile summaries with raw-trace hashes, and hardware/software records.
Full raw traces remain in the verified local archive and retained Pod volume.

CPU-only analysis from the repository root:

```bash
python src/s10_analyze.py results/s10/runs --output results/s10/analysis
python src/s10_plot.py results/s10/analysis/runs.json results/s10/analysis/distributed_training.png
python src/s10_verify_results.py
```

To reproduce on a suitable CUDA host, inspect and run `src/run_s10_setup.sh`,
then run the numerical checks before the capacity, formal and profile matrices.
`src/s10_run.py` accepts the JSON matrices in `configs/s10/`. Their output
directories must be new; the runner refuses to overwrite an existing run.
The [protocol](s10_distributed_protocol.md) defines the acceptance boundaries.

This core study excludes 14B, multi-node networks, convergence evaluation and
S11/RL. Its training throughput is not an inference or serving speedup.
