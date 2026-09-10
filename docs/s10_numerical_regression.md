# S10: a finite loss was not a sufficient correctness check

The initial DeepSpeed 0.17.6 smoke runs completed with finite losses for all
ZeRO stages. However, a controlled optimizer-update check rejected ZeRO-2.
S10 therefore pins 0.17.5 for accepted formal comparisons and preserves the
rejected pilots. This is version selection backed by a local reproduction,
not a claim that this project implemented or fixed DeepSpeed's algorithms.

## Controlled experiment

`src/s10_validate_update.py` constructs a two-layer Qwen-shaped model with
hidden size 128, intermediate size 256, four query heads, two KV heads and
vocabulary 256. All implementations receive identical initial BF16 weights and
the same eight synthetic sequences of 32 tokens. Each GPU processes microbatches
of one sequence; accumulation is four microbatches on two GPUs and two on four.

The reference computes BF16 forward/backward, accumulates gradients in FP32,
averages gradients across data-parallel ranks, clips the global norm to 1.0,
and applies FP32 AdamW to master weights. Learning rate is 1e-4, betas 0.9/0.95,
epsilon 1e-8 and weight decay 0.01. ZeRO's full FP32 parameters are reconstructed
after one update and compared with the reference.

The update error is `||theta_DS - theta_ref|| / ||theta_ref - theta_initial||`.
It is relative to the small optimizer update, **not relative model-weight error**.
The initial gate required at most 5% relative update error. Inspection of the
reported gradient norms led to a stricter additional gate: at most 1% relative
gradient-norm error. Both gates are now executable checks.

## Observations

| DeepSpeed | ZeRO | Ranks tested | Maximum update L2 error | Maximum relative gradient-norm discrepancy | Decision |
|---|---:|---|---:|---:|---|
| 0.17.6 | 1 | 2 | 3.43% | 0.00138% | Diagnostic only |
| 0.17.6 | 2 | 2 | 114.43% | 48.61% | Rejected |
| 0.17.6 | 3 | 2 | 3.50% | 0.00102% | Diagnostic only |
| 0.18.3 | 1 | 2, 4 | 3.56% | 75.00% | Rejected by strengthened norm gate |
| 0.18.3 | 2 | 2, 4 | 3.74% | 75.00% | Rejected by strengthened norm gate |
| 0.18.3 | 3 | 2, 4 | 3.61% | 0.00102% | Diagnostic only |
| **0.17.5** | **1** | **2, 4** | **3.56%** | **0.00244%** | **Accepted** |
| **0.17.5** | **2** | **2, 4** | **3.74%** | **0.00233%** | **Accepted** |
| **0.17.5** | **3** | **2, 4** | **3.61%** | **0.00102%** | **Accepted** |

On 0.17.6 ZeRO-2, the initial forward losses matched exactly, yet the update
failed the reference check. This is consistent with the kind of regression
described in [DeepSpeed issue 7718](https://github.com/deepspeedai/DeepSpeed/issues/7718).
We have not proved that our graph hits exactly the same internal defect.

On 0.18.3, a first-step AdamW parameter comparison alone passed. The reported
ZeRO-1/2 gradient norm was approximately the reference divided by the accumulation
factor. Adam's first update can be insensitive to a uniform gradient scale,
which is why update agreement alone did not settle the discrepancy. We did not
establish whether this was exclusively reporting or also optimizer semantics;
the version was excluded conservatively rather than declaring it correct.

The accepted version passed 18 rank-level records: three stages on two and four
GPUs. This is a representative one-update check in mixed precision. It does not
prove every model graph, exact equality, long-run convergence, or Megatron's
numerical equivalence to DeepSpeed. The 0.18.3 raw pilot files retain their
original update-only `passed` flag; the additional norm gate explains why they
are not accepted evidence.

## Reproduction

With the pinned runtime installed, run separately from any performance test:

```bash
torchrun --standalone --nproc_per_node=2 src/s10_validate_update.py \
  --stage 2 --output results/s10-validation-local
```

Repeat stages 1/2/3 with two and four ranks. Change the DeepSpeed pin only in an
isolated environment when reproducing rejected versions. Keep raw results;
never combine timing measurements from different versions in one comparison.
