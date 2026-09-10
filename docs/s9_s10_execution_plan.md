# S9–S10 execution plan

S9 studies custom fused kernels; S10 studies training-state sharding and
distributed model execution. Existing R0–S8 data and workloads stay frozen.
The incremental GPU budget is USD 50, authorized on 2026-09-10. Stop paid
compute after each experiment session and retain artifacts on volume storage.

Update (2026-09-10): finish S9 only, then stop the experiment Pod. S10 is paused;
its local scaffold is not a completed experiment and no multi-GPU Pod is needed.

Later update (2026-09-10): S9 completed and its Pod stopped. The user resumed S10.
See `s10_distributed_protocol.md` for the active scope. One four-A100-SXM Pod is
used at USD 6.36/hour with a four-hour stop watchdog; the old Pods stay stopped.

## S9: fused kernels

1. RMSNorm and SwiGLU forward/backward: PyTorch eager and compile, Liger,
   custom Triton, custom TileLang. Use FP32 reductions and compare output,
   input gradients, and RMSNorm weight gradients against an explicit reference.
2. Rows 1/8/32/128/512/2048; RMSNorm widths 1536/3584/5120, FP16/BF16.
   Include odd row counts, non-power-of-two widths, noncontiguous views,
   invalid/empty inputs, and repeated execution checks. SwiGLU also uses
   Qwen intermediate widths 8960/18944.
3. W4A16: unsigned nibbles packed along K, group-size 128, per-output-channel
   scale and integer zero point. Compare dequantize-plus-cuBLAS, predequantized
   cuBLAS (a different storage contract), and fused Triton/TileLang.
4. Warm compilation separately. Rotate implementations across five blocks;
   report CUDA-event P50/P95, compile/first-call time, incremental allocator
   memory, error, modeled bytes/FLOPs. Profile representative shapes separately.
   Modeled effective bandwidth is not measured HBM traffic.
5. Integrate a correct kernel into a pretrained Qwen path. The original frozen
   request text was not found in the checkout or the recovered S8 volume, so
   use a stratified 200-request length-matched replay from the public manifest,
   with deterministic synthetic token IDs. Mark it as model-path timing, not
   an original-corpus or HTTP/SSE evaluation. Report changes even if the kernel
   speedup vanishes. Preserve Qwen's intermediate normalization cast explicitly.

## S10: distributed training

1. DeepSpeed ZeRO 0/1/2/3, world sizes 1/2/4 where affordable. Use synthetic
   token batches, BF16, AdamW, identical model and global tokens per update.
   Fix microbatch size and derive accumulation from data-parallel degree;
   accumulation cannot remain constant when both global batch and DP change.
2. Megatron-Core: DP, TP, PP and hybrid layouts on the same node. Record
   `nvidia-smi topo -m`, GPU model, NCCL/software versions and model config.
   Do not compare different networks as if the parallelism configuration were
   the only changed variable.
3. Twenty warm-up and 100 measured optimizer steps for accepted formal runs.
   Separate reduced-step smoke tests from formal data. Report per-rank peak
   allocated/reserved memory, tokens/s, slowest-rank step latency and relative
   scaling efficiency. Record loss/gradient finiteness; this is not convergence.
4. Start with a small smoke model, then memory-plan a 7B-class random model.
   Capacity failures are recorded, not silently replaced by smaller models.
   A 14B and full 4-GPU sweep are contingent on runtime and available balance.

## References and attribution

Method references (implementations in this repository are independently written):

- https://github.com/linkedin/Liger-Kernel (correctness and strong fused baselines)
- https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- https://github.com/tile-ai/tilelang/tree/main/examples/norm
- https://github.com/tile-ai/tilelang/tree/main/examples/dequantize_gemm
- https://www.deepspeed.ai/tutorials/zero/
- https://github.com/NVIDIA/Megatron-LM

Pin installed versions and capture source revisions in every run. Label missing
backends, profiling counters and incomplete matrix cells explicitly.
