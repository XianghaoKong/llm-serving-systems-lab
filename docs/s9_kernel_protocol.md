# S9: fused operator and Qwen model-path protocol

## Questions

1. How much do fusion and FP32 intermediate elimination help compared with
   eager PyTorch, and how much remains against `torch.compile` and Liger?
2. Where do custom RMSNorm/SwiGLU forward and backward kernels lose their
   advantage as row counts grow?
3. Does fusing packed INT4 dequantization into GEMM save allocation/launches,
   and when does a dense cuBLAS baseline still win?
4. Does one numerically compatible RMSNorm replacement affect a pretrained
   Qwen execution path after attention, linear layers and Python overhead?

## Hardware and software

One A100 80GB PCIe, isolated serial GPU jobs; BF16 and FP16, FP32 reductions and
accumulation. `requirements-s9.txt` records direct package pins. The artifact
bundle includes the driver, device limits, package freeze, model revision and
source checksums. Clocks are not locked; a timestamped telemetry sample covers
part of the run. No multi-GPU scaling claim is made.

## Operator matrix and arithmetic

| Operator | Rows | Width | Backends | Phases |
|---|---|---|---|---|
| RMSNorm | 1, 8, 32, 128, 512, 2048 | 1536, 3584, 5120 | eager, compile, Liger, Triton, TileLang | forward; forward + backward |
| SwiGLU | same | 1536, 3584, 5120, 8960, 18944 | same | same |
| W4A16 GEMM | 1, 32, 128, 512 | N = K = 1536, 3584 | unfused, predequantized cuBLAS, Triton, TileLang | forward |

Both dtypes are tested. The planned total is **1,024 cells × 5 blocks = 5,120
measurement records**. Completion markers, not planned counts, determine which
runs enter the report. The small failed/diagnostic pilots are excluded.

RMSNorm normalizes and multiplies the weight in FP32 before one output cast.
SwiGLU computes SiLU and multiplication in FP32 before one output cast. This
contract deliberately makes intermediate materialization visible in eager
PyTorch. It is not identical to every framework's low-precision arithmetic.
The eager reference includes repeated casts and is a secondary baseline;
comparisons against the specified compiler configuration and fused libraries
are more informative than eager-only speedups.
`torch.compile` uses fullgraph and dynamic shapes. Liger RMSNorm uses its Gemma
casting mode with zero weight offset to match this contract.

**Liger SwiGLU backward overwrites its saved input buffers.** Its adapter clones
both inputs, including the clone cost in the non-mutating API comparison. This
is not a bare Liger kernel speed comparison; the profile should expose those
extra copies. TileLang wrappers copy noncontiguous inputs. Formal timing inputs
are contiguous, while the correctness suite also tests strided views.

W4 weights are uint8 `[N, K/2]`, two unsigned nibbles per byte, with group size
128 and per-output-channel scale/zero point. Dequantized values are cast to
activation precision before GEMM; accumulation is FP32. The predequantized
cuBLAS baseline retains a dense weight matrix and excludes dequantization time:
it has a different storage contract. It must not be described as a packed INT4
serving engine. No AWQ/GPTQ/Marlin compatibility or model quantization quality
claim is made.
Both cuBLAS reduced-precision-reduction flags are disabled for accepted W4
runs. A pilot with default FP16 reduction settings failed the original tolerance
and was excluded; the tolerance was not relaxed to accept it.

## Correctness gates

Compare output and both gradients to explicit FP32-intermediate references for
RMSNorm/SwiGLU. Include empty rows, non-power-of-two/odd widths, noncontiguous
activations, invalid ranks/zero width and repeated execution. W4 also includes
partial row/column tiles and compares against FP32 matmul of the same rounded
dequantized weights. Every formal shape/backend is checked before timing.

Norm/SwiGLU tolerance: FP16 absolute 0.004, relative 0.003; BF16 absolute 0.03,
relative 0.02. W4 formal tolerance: FP16 absolute 0.01, BF16 absolute 0.06,
relative 0.025. Absolute maximum error is reported alongside the tolerance;
passing a relative tolerance does not imply zero error or bitwise identity.

## Timing and analysis

- Compile and correctness are outside timing. First forward/backward call time
  is recorded separately; it can include compilation, dispatch and cache hits.
- Three warm calls, ten operator invocations per CUDA Graph replay, thirty
  event samples per block. Rotate backend order across five blocks.
- Inputs and capture use one non-default CUDA stream to avoid backward graph
  dependencies on the legacy stream. Gradient outputs are freshly produced.
- Each record contains replay-amortized P50/P95. These are hot-input operator
  execution metrics, **not request latency tails** or uncaptured Python time.
- Report medians across five block P50/P95 values. Bootstrap block medians
  (4,000 draws, fixed seed); do not count 30 samples as 30 independent repeats.
  Five blocks provide limited precision, particularly for tail estimates.
- Peak allocated memory is a separate one-call incremental allocator reading.
  It is not total model VRAM and does not include driver allocations.
- Effective bandwidth uses logical tensor bytes, not measured HBM traffic.
  W4 TFLOP/s counts GEMM FLOPs, excluding dequantization arithmetic.

## Profiling

Use separate Kineto traces for 44 representative forward/backward calls. Record
kernel names, launch counts, summed kernel duration and available launch/resource
metadata. Profile times are excluded from the formal table. Any occupancy field
is a Kineto estimate; hardware-counter occupancy, HBM transactions and stall
reasons are not collected: Nsight Compute 2025.1.1 was attempted and returned
`ERR_NVGPUCTRPERM`. Inspecting the generated trace supports fusion/launch
claims but cannot alone establish a specific memory or compute bottleneck.
Two additional pretrained-model traces (512 input tokens, four continuation
positions) check whether the RMSNorm launch reduction remains visible in Qwen.
Prime Kineto with a sentinel kernel inside the active profiler session, then
filter events to the synchronized target annotation. The unprimed pilot missed
the first CUDA kernel and is excluded. Device-to-device copies are separate
from kernel launches; Liger SwiGLU forward has two preservation copies.

## Pretrained Qwen integration

Load pinned Qwen2.5-1.5B-Instruct in BF16 with Transformers SDPA. Replace only
RMSNorm and preserve Qwen's intermediate cast **before** weight multiplication;
use an inference-only adapter, not the microbenchmark's different arithmetic.
The fully fused model adapter failed the 2% logit-NRMSE gate both with default
FMA contraction and with contraction disabled. The accepted candidate retains
ATen's FP32 variance computation and fuses the remaining pointwise operations,
with FMA contraction disabled. This is a partial-fusion compatibility path,
not the one-kernel microbenchmark path. Token count is a runtime parameter so
new input lengths do not compile a new kernel inside measured prefill calls.

Original frozen prompt text was unavailable in the local checkout and recovered
S8 volume. Select 200 public-manifest records in original category proportions
(60/40/30/30/20/20), using seed 2026. Generate deterministic synthetic token IDs
with exactly those input lengths (no padding or truncation). Publish selected
IDs, lengths, categories and manifest checksum. This is **length-matched replay,
not an original-prompt, semantic-quality or source-corpus evaluation**.

For each request, use 32 fixed continuation positions, ignoring EOS. The eager
reference supplies continuation tokens; later calls consume the same tokens in
both paths. Run three blocks, alternating paired order after reference creation.
Warm both implementations on six representative lengths. Measure CUDA-event
prefill/first-token time and decode P50/P95, plus synchronized model-loop wall
time. Tokenization, HTTP/SSE, detokenization and serving queues are excluded.

Check finite logits and normalized RMS error at the first and last position
(fail above 2%), and report greedy-token agreement at all 32 positions under
the common prefix. Agreement is a numerical diagnostic, not an answer-quality
score. Aggregate paired speedups; bootstrap request-level medians across blocks
to avoid treating repeated versions of the same request as independent data.
The baseline is unfused Hugging Face RMSNorm. vLLM/SGLang already use fused
operators, so a gain here must not be transferred to those engines as a claim.

## Reproduce

Use the pinned CUDA environment and run from the repository root:

```bash
python -m pip install -r requirements-s9.txt
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir models/Qwen2.5-1.5B-Instruct
MODEL=/path/to/Qwen2.5-1.5B-Instruct bash src/run_s9.sh
```

Output paths must be new: measurement files are opened exclusively so reruns
cannot silently append or overwrite accepted data. GPU correctness/benchmarks
are explicit manual jobs; CPU CI checks analysis logic and source syntax.
`python src/s9_verify_results.py` checks the published record counts, block
uniqueness, sample medians, model agreement and manifest identity without a GPU.

On a host granting counter access, a representative optional NCU invocation is:

```bash
ncu --section LaunchStats --section Occupancy --section SpeedOfLight \
  --nvtx --nvtx-include 's9_target/' --kernel-name 'regex:_rms_fwd' \
  --launch-count 1 --csv --log-file rms-forward.csv -o rms-forward \
  python src/s9_ncu_case.py --op rms --rows 512 --width 3584
```

Run this separately from timing jobs. The study host denied counter access;
the command is provided for reproducibility, not as a successful measurement.

Method references: [Triton](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html),
[Liger](https://github.com/linkedin/Liger-Kernel),
[TileLang](https://github.com/tile-ai/tilelang),
[Qwen2 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py).
The repository's Triton and TileLang kernels are independently written.
