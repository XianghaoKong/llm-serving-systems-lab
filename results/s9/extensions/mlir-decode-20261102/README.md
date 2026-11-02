# S9 extension: exact RMSNorm reduction and decode CUDA Graphs

This extends the [original S9 study](../../../../docs/s9_results.md) from
commit `45a5bb5eddd333a336752aec7c4380ff6eff123d`. It resolves the observed
RMSNorm numerical discrepancy, parallelizes that compatible reduction, and
reduces model-loop host submission overhead. The original A100 measurements
remain separate and unchanged.

**Environment:** one desktop RTX 4070 12GB, WSL2 Ubuntu 24.04, PyTorch
2.8.0+cu128, Triton 3.4.0, Transformers 4.57.1, LLVM/MLIR 19.1.1;
Qwen2.5-1.5B-Instruct, BF16, SDPA, batch 1, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Clocks were not locked and the GPU
was shared with the desktop. Results apply to these measured cases.

## Changes and evidence

1. **Locate the first difference.** The square stage matches, but Triton's
   reduction tree differs from ATen's. A small FP32 mean difference can cross
   a BF16 rounding boundary. Substituting the ATen mean removes the observed
   downstream discrepancy. The 346-token case uses synthetic tokens with a
   fixed seed; it reproduces the mechanism, not the original A100 request text.
   See [per-layer diagnosis](numerics/diagnosis.json).
2. **Encode the arithmetic contract in MLIR.** The `s9.aten_mean` scalar oracle
   and `s9.aten_mean_warp` lowering preserve PyTorch 2.8's four accumulators,
   combination tree, shuffle order and FP32 scaling for width 1536. One
   physical warp cooperates on each row; threads no longer each recompute the
   entire mean. The bitcode links into the Triton RMSNorm tail, producing one
   fused GPU kernel. This lowers an explicitly extracted mean region; it is
   not an automatic rewrite of arbitrary Triton IR. See
   [compiler source](../../../../compiler/s9-numerics/) and
   [17-case validation](numerics/validation.json).
3. **Explain the apparent 346-token regression.** The earlier cyclic ordering
   was unbalanced, and CUDA event intervals included host queue starvation.
   Ten ABBA/BAAB blocks across five lengths produced 200 requests: at 346,
   old/new mean wall latency was 419.22/419.42 ms; the block-bootstrap 95%
   interval for the mean difference was −1.41 to +1.73 ms. Adjacent lengths
   behaved similarly. There was no 346-specific fallback. See
   [paired records](diagnosis-346/paired-records.jsonl),
   [summary](diagnosis-346/paired-summary.json) and
   [detailed investigation](../../../../docs/s9_mlir_numerics.md).
4. **Reduce host overhead.** Validate once, cache launch specifications, and
   remove the mean diagnostic allocation/store from normal calls. Output
   tensors still have independent storage; `.run()` retains diagnostic means.
   The [frozen pre-change adapter](baseline/s9_mlir_adapter_before.py) is the
   contemporaneously remeasured baseline.
5. **Replay exact-length decode graphs.** Each prompt-length bucket uses 31
   graphs, one per actual decode position. Double-buffered contiguous KV
   storage preserves eager attention shapes and strides. A new request runs
   fresh prefill, copies its KV, then advances tokens and KV on graph replay.
   The graph does not replace the model with a single GPU kernel: a separately
   profiled step still contains 769 kernels, including 57 RMSNorm kernels,
   submitted with one `cudaGraphLaunch` plus one token `cudaMemcpyAsync`.

## Measured performance

The earlier parallel-reduction experiment measured the following operator GPU
medians, in microseconds. These are historical operator measurements, separate
from the final model run; see [complete operator summary](parallel-reduction/operators.json).

| Rows | Scalar MLIR prototype | ATen-mean compatibility | Parallel MLIR |
|---:|---:|---:|---:|
| 1 | 36.966 | 5.427 | 1.843 |
| 346 | 113.181 | 12.210 | 2.770 |
| 4096 | 1256.136 | 194.438 | 23.774 |

The accepted final model run uses five paths, three prompt lengths and ten
Williams-balanced blocks: **150 timed requests**, each with one prefill and
31 decode forwards. Inputs and eager continuation prefixes are shared across
paths; EOS does not shorten requests. Compilation, graph preparation,
profiling and correctness checks are outside timing. Initial graph KV copies
and per-step token copies are inside timing. HTTP, tokenization, sampling and
argmax are excluded. Wall latency is the primary metric; CUDA-event intervals
are not pure active-kernel time.

| Prompt tokens | Eager ms | ATen-mean compatibility ms | Parallel MLIR before host changes ms | Host changes ms | Host changes + graph ms |
|---:|---:|---:|---:|---:|---:|
| 33 | 402.427 | 416.311 | 409.286 | 366.737 | 249.551 |
| 346 | 411.072 | 419.588 | 411.918 | 366.421 | 263.126 |
| 4096 | 646.767 | 644.199 | 633.053 | 587.923 | 507.869 |

| Prompt tokens | Host vs before | Graph vs host | Graph vs before | Graph vs ATen-mean compatibility |
|---:|---:|---:|---:|---:|
| 33 | −11.19% | −31.89% | −39.52% | −40.06% |
| 346 | −11.63% | −27.91% | −36.19% | −37.11% |
| 4096 | −7.07% | −13.38% | −19.79% | −21.43% |

Times are medians by path. Percentages are medians of within-block ratios,
not ratios of the table's medians. Each reported comparison improves in all
ten blocks at each length. See [raw records](host-graph/model-records.jsonl),
[aggregates](host-graph/models.json), [host API measurements](host-graph/host-micro.json),
and [profiler summary](host-graph/profile-summary.json).

Graph preparation costs **3.39–3.48 seconds per bucket** after model/operator
warmup, excluded from warm latency. Relative to the host-optimized path, the
measured savings amortize preparation after approximately 30, 34 and 45
requests at lengths 33, 346 and 4096. Persistent KV double buffers occupy
3.5, 20.6 and 225.7 MiB respectively, in addition to graph pools, logits and
driver allocations. See [preparation and memory records](host-graph/preparation.json).
One cold request does not recover this cost.

## Correctness and scope

- 17 operator cases cover 9,358 rows / 14,373,888 output elements, with exact
  CPU-oracle mean, GPU mean and final-output comparisons.
- The final model run checks **480 logit positions**, all bitwise equal to
  eager, with zero fallback. It includes a second prompt reusing the same
  graphs with actual greedy-token feedback, matching all 32 predictions.
- 55 targeted GPU/ordering/analysis tests passed in the recorded environment;
  the [test output](validation/gpu-tests.txt) is retained. CPU CI verifies
  published data and CPU tests; it does not rerun GPU acceptance.
- The fused specialization covers CUDA BF16, width 1536 and generated row
  shapes. FP16 or unavailable shapes use the ATen compatibility fallback.
  Input alignment may require a clone. The implementation is inference-only.
- `ExactDecodeGraph` requires the prepared Qwen2 SDPA model, eval mode,
  `torch.inference_mode()`, batch 1, the same CUDA stream, a fixed prompt
  length/horizon, and one active request. Reprepare after model storage,
  backend or shape changes. Returned graph logits borrow storage and must be
  copied if retained across reuse of the corresponding graph.

This is a local model-loop study, not an HTTP benchmark, production serving
integration, A100 rerun or repeat of the original 200-request workload.

## Reproduce

From the repository root, verify published artifacts without a GPU:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q tests
python src/s9_verify_results.py
python src/s9_verify_extension.py
```

For GPU reproduction, install LLVM/MLIR 19.1.1, Clang 19, CMake and Ninja on
Linux/WSL2. Use a fresh environment and output directories:

```bash
python3 -m venv .venv-s9-numerics
source .venv-s9-numerics/bin/activate
pip install -r compiler/s9-numerics/requirements.txt
cmake -S compiler/s9-numerics -B work/s9-mlir-build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-19
cmake --build work/s9-mlir-build -j2
ctest --test-dir work/s9-mlir-build --output-on-failure
python compiler/s9-numerics/build_libraries.py \
  --tool work/s9-mlir-build/s9-opt --output work/s9-libraries \
  --parallel --rows 1 2 4 8 16 33 346 4096

export TRITON_CACHE_DIR="$PWD/work/s9-triton-cache"
export S9_MLIR_LIBRARIES="$PWD/work/s9-libraries"
# Download Qwen/Qwen2.5-1.5B-Instruct at the revision specified above.
# Point MODEL to that complete local snapshot; scripts load local files only.
MODEL=/path/to/pinned/Qwen2.5-1.5B-Instruct
python src/s9_numeric_probe.py --model "$MODEL" \
  --output work/s9-probe --length 346 --steps 32
python src/s9_validate_mlir.py --libraries "$S9_MLIR_LIBRARIES" \
  --fixture work/s9-probe/captured_input.pt --output work/s9-validation \
  --model "$MODEL" --replay-dir work/s9-probe
python -m pytest -q tests/test_s9_mlir_adapter.py tests/test_s9_decode_graph.py \
  tests/test_s9_benchmark_order.py tests/test_s9_analyze.py
python src/s9_benchmark_host.py --libraries "$S9_MLIR_LIBRARIES" \
  --before-adapter results/s9/extensions/mlir-decode-20261102/baseline/s9_mlir_adapter_before.py \
  --model "$MODEL" --output work/s9-host-graph-new --lengths 33 346 4096 --blocks 10
```

The model benchmark requires complete ten-block cycles for five paths. The
frozen adapter and optimized adapter use the same parallel bitcode. The
scalar prototype can be built separately by omitting `--parallel` when
investigating the earlier operator comparison.

## Artifact provenance

[provenance.json](provenance.json) identifies the accepted runs and original
file hashes. Published command arguments omit local machine paths; measured
values are unchanged. [manifest.json](manifest.json) covers this directory's
published text, normalizing CRLF to LF for cross-platform checkout. The final
run's source hashes are checked against the corresponding repository files.
Weights, tensor fixtures, compiled native libraries and full profiler
timelines are generated or retained locally, not checked into this directory.
The [Chinese stage-by-stage notes](../../../../docs/s9_mlir_numerics.md)
preserve the investigation and earlier experiments.
