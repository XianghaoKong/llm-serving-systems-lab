# S9 — fused kernels, numerical compatibility, and model-path validation

**A100 80GB PCIe; FP16/BF16; 1,024 operator cells, five blocks each.** Custom
RMSNorm is faster than the tested dynamic `torch.compile` configuration, while
SwiGLU is often already well fused by the compiler. Naive W4 fusion reduces
temporary allocation and launches but does not beat predequantized cuBLAS.
Model-level numerical validation rejects the fully fused RMSNorm candidate;
a partial-fusion path preserves the original reduction and checked outputs,
but makes the tested model path approximately **5.9% slower**.

See the [protocol](s9_kernel_protocol.md), [full operator table](../results/s9/summary/summary.csv),
[timing records](../results/s9/runs/), [correctness reports](../results/s9/correctness/),
and [profiling summary](../results/s9/profile/summary.json).

![Operator latency on A100](../results/s9/summary/kernel_latency.png)

The chart shows medians of block P50s, with bootstrap bands, for BF16 width
3584. W4 uses N = K. Lower is better. CUDA Graph replay is a hot-input operator
measurement, not a request-latency distribution. Liger SwiGLU includes input
preservation copies; predequantized cuBLAS retains dense weights.

## 1. Compare with compiled and fused baselines

These are unweighted summaries of **per-shape speedups against the tested
dynamic compiler configuration**, not a workload-weighted model speedup.
Values above 1 are faster; below 1 are slower. The compiler was not exhaustively
autotuned and static-shape compilation is not part of this matrix.

| Operator / phase | Shapes per backend | Triton median speedup (min–max) | TileLang median speedup (min–max) |
|---|---:|---:|---:|
| RMSNorm forward | 36 | 1.906× (1.397–3.493) | 1.840× (1.329–3.200) |
| RMSNorm forward + backward | 36 | 1.467× (0.846–2.196) | 1.532× (0.855–2.166) |
| SwiGLU forward | 60 | 1.021× (0.995–1.811) | 1.008× (0.934–1.748) |
| SwiGLU forward + backward | 60 | 1.032× (0.958–1.355) | 1.018× (0.940–1.350) |

For BF16 RMSNorm at **512 × 3584**, forward latency is **11.054 µs compiled,
4.723 µs Triton, 4.755 µs TileLang**. Triton is **2.340×** faster than this
compiler configuration. At **1 × 3584**, SwiGLU is effectively tied:
**1.870 µs compiled versus 1.866 µs Triton**. A few hundredths of a microsecond
are not a useful universal optimization claim.

Custom RMSNorm also approaches the existing fused Liger implementation; this
is not a claim of a new state-of-the-art RMSNorm algorithm. Some backward
configurations regress against compilation. The eager FP32-reference baseline
includes casts and temporaries, so eager-only speedups are secondary evidence.

## 2. W4 fusion saves allocation, but the tile strategy matters

The packed format is two unsigned INT4 values per uint8, group size 128,
per-output-channel scale and zero point, BF16/FP16 dequantized operands and FP32
accumulation. Accepted runs disable cuBLAS reduced-precision reductions.

**BF16, N = K = 3584; median block P50, µs:**

| Rows | Unfused dequant + GEMM | Triton fused | TileLang fused | Predequantized cuBLAS |
|---:|---:|---:|---:|---:|
| 1 | 335.203 | 102.048 | 64.994 | 16.600 |
| 512 | 406.936 | 906.170 | 469.142 | 91.568 |

At one row, TileLang is **5.157×** faster than the unfused reference, yet
predequantized cuBLAS is still much faster. At 512 rows, both custom fused
implementations are slower even than the unfused reference. Across all 16 W4
shapes/dtypes, neither custom implementation beats the dense baseline.

For 512 rows, extra one-call allocator memory falls from **150.38 MiB** in the
unfused path to **3.50 MiB** in either fused path. Predequantized cuBLAS also
needs only the output allocation during the call, because its dense weight
matrix is already resident. These are incremental temporary allocations, not
total process/model VRAM or an end-to-end quantization compression ratio.

The custom GEMM fixes a 16 × 64 × 32 tile and repeatedly unpacks weight tiles.
The result demonstrates why eliminating a temporary tensor does not guarantee
higher throughput. This is a standalone packed-GEMM study, not an AWQ/GPTQ or
Marlin implementation and not a claim about quantized model quality.

## 3. What the traces establish

Representative BF16 cases show these **kernel** counts; D2D copies are separate:

| Operation | Eager / unfused | Compile / dense | Triton | TileLang |
|---|---:|---:|---:|---:|
| RMSNorm forward, 512 × 3584 | 10 | 1 | 1 | 1 |
| RMSNorm forward + backward, 512 × 3584 | 27 | 4 | 4 | 3 |
| SwiGLU forward, 512 × 3584 | 5 | 1 | 1 | 1 |
| W4 forward, 32 × 3584 × 3584 | 10 | 1 | 1 | 1 |

Liger SwiGLU forward has **one kernel plus two D2D copies** in the non-mutating
adapter. The copies preserve inputs overwritten by upstream backward; their
cost must not be described as an intrinsic cost of the bare Liger kernel.

RMSNorm compiled and custom forward both use one kernel. Their launch/resource
metadata differs: the compiled kernel uses **512 threads/block, 32 registers
per thread, 4,096 bytes shared memory**; Triton uses **128 threads/block,
62 registers per thread, 16 bytes shared memory**. Thus launch-count reduction
alone does not explain the 2.340× comparison against compilation.

For W4 at 32 rows, both custom kernels use a 2 × 56 grid, 128 threads/block and
6,144 bytes shared memory. Triton reports **168 registers/thread**, TileLang
**64**. These observations motivate investigation of tile layout, register use
and weight reuse. They do not prove a particular stall or HBM bottleneck.

**Counter limitation:** Nsight Compute 2025.1.1 returned `ERR_NVGPUCTRPERM` on
the cloud host. No hardware-counter occupancy, memory transactions or stall
metrics were obtained. Kineto's estimated occupancy field is not substituted
for a hardware measurement. Profile durations are excluded from formal timing.

## 4. Pretrained Qwen validation

The model is Qwen2.5-1.5B-Instruct, BF16, Transformers SDPA, with revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The fully fused model adapter failed the **2% logit-NRMSE gate** on the first
pilot request: **3.453%** with default FMA contraction, **3.006%** with it
disabled. Passing an isolated operator tolerance was insufficient to accept
the change in the pretrained model. Those timings are not accepted speedups.

The compatibility path retains ATen's FP32 variance reduction and fuses the
remaining pointwise operations, preserving Qwen's intermediate BF16 cast. Its
20-request pilot produced identical checked logits and greedy predictions.
Request length is a runtime kernel argument, avoiding per-length compilation
inside measured prefill. The fully fused and compatibility paths remain
distinct in code and result labels.

The formal run uses **200 length-matched synthetic requests, three blocks,
two backends, 32 fixed continuation positions**. Original prompt text was not
available in the checkout or recovered S8 volume, so this is not an original
corpus replay. Category proportions and input lengths come from the frozen
public manifest. Both backends consume identical baseline continuation tokens.
Tokenization, HTTP/SSE, detokenization, EOS stopping and serving queues are
excluded. No answer-quality claim is made.

**Completed: 1,200 calls, 600 paired comparisons, 200 distinct request IDs.**
All checked first/last-position logits match exactly (maximum absolute error
and NRMSE both zero); all **19,200 compatible-path** greedy-token predictions
match the eager reference (38,400 predicted tokens across both paths). This is
numerical compatibility under the replay, not semantic
answer-quality evidence.

| Metric | Eager median, ms | Compatible median, ms | Median request speedup [95% cluster CI] |
|---|---:|---:|---:|
| Prefill / first token | 21.273 | 22.574 | 0.954× [0.951, 0.960] |
| Decode P50 | 18.414 | 19.623 | 0.942× [0.941, 0.944] |
| Decode P95 | 20.828 | 21.845 | 0.949× [0.945, 0.954] |
| Model-loop wall time | 619.503 | 654.893 | 0.944× [0.942, 0.945] |

The last column takes per-request paired ratios across repeated blocks; it is
not the ratio of the two marginal medians. The wall-time ratio implies about
**5.9% slowdown**, so the compatibility path is **not recommended as a speed
optimization** on this tested Hugging Face execution path.

Separate 512-input / four-position model traces show **5,013 → 4,101 kernels**
and **33.695 → 31.344 ms** summed kernel duration. Fewer kernels and less summed
GPU execution time did not yield lower unprofiled model-loop latency. This is
consistent with costs outside active kernel execution offsetting the savings;
the profiling run does not establish an exact causal breakdown. It also does
not predict vLLM/SGLang gains, since those engines already use fused operators.

See [model records and summary](../results/s9/model/) and the
[grouped model profile](../results/s9/model/model-profile-summary.json).
Full timelines and rejected pilots are retained in the verified artifact archive.

## 5. Failed pilots and corrective actions

- **W4 default reductions:** one FP16 result out of 1,835,008 exceeded the
  original absolute tolerance (0.010191 versus 0.01). Disabling reduced-precision
  cuBLAS reductions aligned the baseline with the accumulation contract; all
  320 records were rerun and passed. The tolerance was not loosened.
- **Unprimed Kineto:** first-kernel events were missing. A sentinel inside the
  active profiler session, outside the target annotation, fixed collection.
  The accepted 44-case profile is the primed rerun.
- **Full Qwen fusion:** failed the model-logit gate as described above.
- **Static-length compatibility pilot:** numerical checks passed, but new
  lengths triggered compilation. Runtime length dispatch removed that source
  of first-call latency before the formal run.
- **Nsight counters:** access denied by the host; no counter data fabricated.

See [environment records](../results/s9/environment/) and the
[execution protocol](s9_kernel_protocol.md) for pins, tolerance, timing and
reproduction details. CPU CI tests analysis logic; GPU acceptance is documented
by completed run artifacts, not by the CPU CI badge. The completed
[S10 distributed-training study](s10_results.md) extends these numerical checks
to state sharding and model parallelism on four A100s.
The complete archive was verified locally before compute stopped at
**2026-09-10 05:15:10 UTC**; its SHA-256 is recorded in
[archive metadata](../results/s9/environment/artifact-archive.json).
