# S3 — vLLM Continuous-Batching Key Findings

## S3-C — Natural-generation capacity sweep

The capacity sweep uses fresh vLLM processes for each formal load point. Prefix caching remains enabled, but cross-load-point cache contamination is avoided.

A sharp tail-latency knee was observed between configured `λ=3.75` and `λ=4.0` req/s:

- Realized arrival rate: **3.239 → 3.455 req/s**.
- Output throughput: **547.1 → 564.4 tok/s**.
- P95 TTFT: **232.7 → 559.3 ms**.
- P99 TTFT: **251.3 → 1622.2 ms**.
- P95 request-mean SSE content-event ITL: **11.83 → 17.96 ms**.

Raw throughput was still increasing at the highest tested point, so this is best described as a **latency knee**, not a demonstrated hard throughput ceiling.

At the high-load point, scheduler waiting alone did not identify the onset of degradation:

- Max running requests: **14**.
- Max waiting requests: **0**.
- Peak reported KV-cache usage: **5.3%** if the vLLM metric is represented as a 0–1 fraction.

This supports the interpretation that tail degradation can appear through active-batch execution/interference before an explicit waiting queue or KV-cache capacity limit becomes dominant.

## S3-E — Concurrency control

In natural-generation runs at configured `λ=4.0`, limiting `max_num_seqs` changed the active-concurrency regime:

- Default: P95/P99 TTFT **559.3 / 1622.2 ms**, max running **14**.
- `max_num_seqs=12`: P95/P99 TTFT **234.9 / 253.9 ms**, max running **12**, max waiting **1**.
- `max_num_seqs=10`: P95 TTFT **442.4 ms**, max waiting **3**.
- `max_num_seqs=8`: P95 TTFT **791.5 ms**, max waiting **6**.

The natural-generation results are consistent with a trade-off: excessive active concurrency can worsen execution interference, while tighter caps can shift delay into scheduler waiting. `max_num_seqs=12` was the strongest tested natural-run TTFT operating point, but this is not treated as a universal optimum.

## S3-D — Length-matched controlled replay

The controlled replay forced the same output-token work on the default and `max_num_seqs=12` configurations.

- Default max running: **11**.
- `max_num_seqs=12` max running: **11**.
- Default P95 TTFT: **232.0 ms**.
- `max_num_seqs=12` P95 TTFT: **234.0 ms**.
- Default / capped output throughput: **573.0 / 572.6 tok/s**.

Because the default replay did not exceed the 12-sequence cap, the cap was not activated and the two configurations were nearly identical. Therefore, the natural-run improvement should not be described as a universal direct causal speedup from setting `max_num_seqs=12`; it is conditional on the request-lifetime/output trajectory producing enough active concurrency for the cap to matter.

## Measurement boundaries

- SSE content-event ITL is an application/streaming metric, not GPU TPOT.
- Natural-generation output trajectories can differ across runs even with fixed seeds, so throughput differences across scheduler configurations are not treated as pure scheduler-causal effects.
- A zero waiting gauge does not by itself imply that the serving system is healthy.
- The observed knee is specific to this model, GPU, software stack, workload, and finite trace.
