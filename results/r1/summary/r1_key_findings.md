# R1 Sequential Serving Baseline — Key Findings

## Protocol

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dtype: `torch.bfloat16`
- Concurrency: `1`
- Batch size: `1`
- Frozen corpus SHA-256: `7593429c095064a7f375e12d40db43ebf174d0f3a57f49bc51db030a0619d014`
- Requests per backend: 1,000
- Eager and Flash used the same frozen corpus and deterministic execution order.
- Flash was verified through the forced Flash SDPA backend context.

## Main findings

1. **Long-context prefill:** For 3,073–4,096-token requests, median paired
   request-TTFT speedup was **2.25x**. Across the dedicated
   Long-context QA category, Eager median request TTFT was
   **574.9 ms** versus
   **254.0 ms** for Flash.

2. **Decode:** Across **375,683** measured decode-token
   events, median GPU token latency fell from
   **17.90 ms** to
   **14.77 ms**. Request-level median decode
   throughput increased from **55.62**
   to **67.37 tok/s**.

3. **Tail token latency:** P99 GPU decode latency was
   **19.97 ms** for Eager and
   **16.36 ms** for Flash.

4. **Memory:** Long-context median peak allocated GPU memory fell from
   **4.476 GiB** to
   **3.996 GiB**
   (**10.7% reduction**).

5. **Same-output sanity check:** Only
   **91/1000** requests produced exactly the
   same output token sequence across stochastic Eager and Flash runs. On those
   exactly matched generations, the median paired token-level GPU speedup was
   still **1.214x**.

6. **Observed service time:** The sum of request service times fell from
   **57.07 min** to
   **49.24 min**
   (**13.7% lower**) even though Flash
   generated **+5.6%** more output
   tokens in this stochastic run.

## Interpretation boundary

E2E latency is a production-observed metric, not a pure kernel-causal metric,
because stochastic generation trajectories diverge across backends. TTFT,
TPOT, streaming ITL, decode throughput, and memory are the primary
backend-performance metrics.

CUDA `reserved` memory is allocator-reserved capacity and must not be described
as active/resident tensor memory.

The R1 production-like BF16 workload does not by itself establish the earlier
controlled 512-token boundary effect. See `boundary_512_summary.csv` for the
508–516-token measurements.
