# S10: training-state sharding and model parallelism

Status: implementation and smoke validation; no accepted results yet.
S10 resumed by the user on 2026-09-10. The S9+S10 incremental budget remains
USD 50; this session targets at most USD 30 additional spend. Stop compute and
retain data after the session. Old Pods stay stopped.

## Scope

Use one multi-GPU host and record GPU topology before comparing layouts.
DeepSpeed uses random Qwen2.5-7B-shaped weights (3584 hidden, 28 layers, 28 query
heads, 4 KV heads, 18944 intermediate, 152064 vocabulary), synthetic tokens,
BF16 model parameters and AdamW. Small-model smoke runs validate all stages
before capacity tests. ZeRO 0/1/2/3 are tested on available 1/2/4-rank layouts;
an out-of-memory case is a capacity result, never a zero-throughput success.

Megatron-LM is pinned to core_v0.14.0, commit
23e00ed0963c35382dfe8a5a94fb3cda4d21e133. Use its official GPT training entrypoint,
local transformer implementation and synthetic mock dataset. Compare selected
DP, TP, PP and hybrid configurations within this backend. Do not interpret
cross-framework differences as causal parallelism improvements: implementation,
initialization and dataset construction differ.

## Measurement contract

- Fix sequence length, microbatch and global tokens per optimizer update within
  each matrix; derive gradient accumulation from data-parallel degree.
- Twenty warmup and 100 measured optimizer updates per formal run. Run profiles
  separately. Short smoke runs are never pooled with formal data.
- Time complete updates with CUDA synchronization and MAX over ranks. Exclude
  logging, the measurement reduction and inter-update barrier from step timing.
- Save every rank's peak allocated/reserved memory, loss and gradient norm where
  available; reject nonfinite values or skipped updates. This does not establish
  training convergence or numerical equivalence across implementations.
- Repeat selected configurations in separate processes; report per-run medians
  and P95, and do not pretend individual correlated steps are independent trials.
- Measure topology and software versions; archive source, launch flags, logs,
  failures, GPU telemetry and raw profiles before stopping the Pod.

## References

- https://www.deepspeed.ai/tutorials/zero/
- https://www.deepspeed.ai/docs/config-json/
- https://github.com/NVIDIA/Megatron-LM/tree/core_v0.14.0

DeepSpeed and Megatron supply the distributed algorithms. The local code supplies
experiment configuration, measurement, validation and artifact analysis.
