# LLM Inference & Serving Systems Lab

A systems-oriented investigation of LLM inference performance, from GPU attention kernels to production-like serving behavior.

This project studies how inference performance changes across multiple layers of the stack:

- attention kernel implementation
- long-context prefill
- autoregressive decoding
- KV-cache growth
- realistic heterogeneous workloads
- HTTP/SSE streaming
- bursty request arrivals
- queueing
- continuous batching with vLLM
- concurrency control
- SLO-qualified goodput

The main goal is to answer a practical systems question:

> **Does a faster model kernel necessarily produce a more responsive LLM serving system?**

The experiments show that it does not. Flash Attention substantially improves prefill, decode efficiency, and GPU memory use, but under increasing traffic the dominant bottleneck shifts toward queueing, batching, active concurrency, and scheduler behavior.

---

## Key Results

### Kernel-level optimization

On Qwen2.5-1.5B-Instruct running on a single RTX 4070:

- Flash Attention improved 3K–4K-token request TTFT by approximately **2.25×**
- Median decode throughput increased from approximately **55.6 tok/s to 67.4 tok/s**
- Long-context peak GPU allocation decreased
- In a controlled 512-token decode workload, Flash reduced E2E latency from approximately **9.01 s to 7.62 s**

### Queueing behavior

Under bursty Poisson arrivals with a single FCFS worker:

- request TTFT increased from sub-second latency to tens of seconds
- GPU token latency remained approximately **15–16 ms**
- user-visible latency became dominated by **queueing rather than slower token execution**

### Continuous batching

Using vLLM continuous batching:

- aggregate output throughput scaled beyond **500 tok/s**
- median streaming cadence remained relatively stable over a wide load range
- a sharp **tail-latency knee** appeared between configured arrival rates of approximately **3.75 and 4.0 req/s**

From λ=3.75 to λ=4.0:

- output throughput: **547.1 → 564.4 tok/s**
- P95 TTFT: **232.7 → 559.3 ms**
- P99 TTFT: **~251 → ~1622 ms**
- P95 streaming ITL: **11.83 → 17.96 ms**

Raw throughput was still increasing, so this is a **latency knee rather than a demonstrated hard throughput ceiling**.

### Concurrency control

At λ=4.0, natural-generation experiments compared different `max_num_seqs` limits:

| Configuration | Max Running | Max Waiting | P95 TTFT | P99 TTFT | P95 ITL |
|---|---:|---:|---:|---:|---:|
| Default | 14 | 0 | 559.3 ms | 1622.2 ms | 17.96 ms |
| `max_num_seqs=12` | 12 | 1 | 234.9 ms | 253.9 ms | 12.27 ms |
| `max_num_seqs=10` | 10 | 3 | 442.4 ms | 563.4 ms | 12.26 ms |
| `max_num_seqs=8` | 8 | 6 | 791.5 ms | 1007.9 ms | 11.61 ms |

The results show a trade-off:

- excessive active concurrency can increase execution interference
- overly restrictive concurrency shifts delay into scheduler waiting
- `max_num_seqs=12` provided the best tested natural-run TTFT trade-off

However, this is **not treated as a universal optimum**.

A later length-matched controlled replay forced identical output-token work for default and `max_num_seqs=12`. Both configurations reached only 11 concurrent running requests, so the 12-sequence limit was never activated. Their performance was consequently nearly identical.

This indicates that concurrency control is useful specifically when it actually prevents over-admission.

### SLO-qualified goodput

A request was considered SLO-compliant when both conditions were satisfied:

- TTFT ≤ **300 ms**
- mean SSE content-event ITL ≤ **15 ms**

Request goodput was defined as:

```text
SLO-passing successful requests / measured makespan

The highest measured request goodput occurred at configured λ=3.75:

Configured λ	SLO Pass Rate	Raw Req/s	Request Goodput
3.50	98.33%	2.795	2.748
3.75	98.33%	2.913	2.864
4.00	86.67%	2.953	2.559

Moving from λ=3.75 to λ=4.0:

raw request throughput increased slightly
request goodput decreased by approximately 10.7%
output goodput decreased from approximately 546.4 tok/s to 477.5 tok/s

This demonstrates that:

Maximum raw throughput is not necessarily the optimal production operating point.

Experiment Progression

The project follows a bottom-up performance investigation:

GPU kernel behavior
        ↓
long-context prefill
        ↓
autoregressive decode
        ↓
realistic workload
        ↓
HTTP/SSE serving
        ↓
Poisson arrivals and queueing
        ↓
continuous batching
        ↓
concurrency control
        ↓
SLO-qualified goodput
Experimental Environment
Base inference experiments
Model: Qwen/Qwen2.5-1.5B-Instruct
GPU: NVIDIA GeForce RTX 4070, 12 GB VRAM
OS: WSL2 / Ubuntu 22.04
Framework: PyTorch + Hugging Face Transformers
GPU count: 1
vLLM serving experiments
vLLM: 0.28.0
PyTorch: 2.13
CUDA runtime: 13.2
Precision: BF16
Attention backend: FlashAttention 2
torch.compile: enabled
CUDA graphs: enabled
Prefix caching: enabled
Chunked prefill: enabled
GPU memory utilization target: 0.85

The vLLM experiments use a different software stack from the custom PyTorch serving baseline. Therefore, differences between the two are treated as serving-stack comparisons rather than pure scheduler-only speedups.

R0 — Realistic Workload Construction

A frozen heterogeneous workload of 1,000 requests was constructed from public datasets.

Category	Requests
Short Interactive	300
Knowledge QA	200
Coding Request	150
Document QA	150
Long-context QA	100
Long Output	100
Total	1000

Input ranges include:

short interactive / knowledge requests
document QA around 1K–2K input tokens
long-context QA around 3K–4K input tokens
long-output generation workloads

The public repository contains only the metadata-oriented workload manifest. Full prompt content is intentionally kept out of the public repository.

Frozen workload checksum:

7593429c095064a7f375e12d40db43ebf174d0f3a57f49bc51db030a0619d014

Public manifest:

workloads/final/public_workload_manifest.csv
R1 — Sequential Eager vs Flash Baseline

R1 evaluates the complete 1,000-request workload using a persistent single-request inference path.

The benchmark compares:

Eager attention
PyTorch SDPA Flash attention
TTFT

Representative request-level results:

Metric	Eager	Flash
P50 request TTFT	21.55 ms	17.72 ms
P95 request TTFT	574.3 ms	253.7 ms
P99 request TTFT	693.8 ms	287.5 ms

For 3073–4096-token inputs:

~574.9 ms → ~254.0 ms
≈ 2.25× speedup
Decode
Metric	Eager	Flash
Median TPOT	17.98 ms	14.84 ms
Median decode throughput	55.62 tok/s	67.37 tok/s

Flash improved median decode throughput by approximately 21%.

Memory

Maximum peak allocated GPU memory:

4.963 GiB → 4.163 GiB

Long-context peak allocation was also lower under Flash.

Service time

Total measured service time across the full workload:

~57.07 min → ~49.24 min

The Flash run generated slightly more tokens because stochastic generation trajectories differed across backends.

Therefore:

E2E latency is treated as a production-observed metric rather than a pure kernel-causal measurement.

Primary backend-performance metrics are TTFT, TPOT, streaming ITL, decode throughput, and GPU memory.

Figures

S1 — HTTP / SSE Streaming Serving

The inference path was exposed through:

FastAPI
persistent GPU model
HTTP/1.1
SSE streaming

The S1 implementation preserved the R1 Flash generation trajectory for all 1,000 requests.

Observed localhost client/server TTFT overhead was approximately 1 ms at the median.

An important measurement issue was also identified:

An SSE content event is not necessarily equivalent to a visible token.

Some streaming events contain empty text payloads, so later experiments separately track:

first content-event TTFT
first visible-text TTFT

SSE content-event ITL is therefore treated as an application-level streaming metric, not GPU TPOT.

S2 — Poisson Arrivals and Queueing

S2 introduces bursty traffic while keeping:

one persistent worker
FCFS execution
no batching

Requests are generated from an exponential inter-arrival process.

The same master random arrival trace is reused across load points and scaled by configured λ. Both configured and realized arrival rates are reported.

Result

As arrival rate increases:

queueing delay grows dramatically
TTFT grows with queueing delay
GPU token latency remains approximately stable

The central observation is:

A faster GPU kernel does not guarantee low user-visible latency once queueing dominates the serving path.

S3 — vLLM Continuous Batching

S3 replaces the single-request serving architecture with vLLM continuous batching.

S3-A — Sequential validation

A 60-request balanced pilot was first used to validate the vLLM stack.

Representative results:

P50 TTFT: ~17.6 ms
P95 TTFT: ~231 ms

Long-context request TTFT remained primarily dominated by prefill, while steady SSE content-event ITL stayed around 8 ms.

S3-C — Capacity Sweep

Formal natural-generation capacity runs were executed with a fresh vLLM process for each load point.

This avoids cross-load-point prefix-cache contamination while keeping prefix caching enabled within each experiment.

The main finding is a tail-latency knee between configured λ=3.75 and λ=4.0.

The system was still increasing aggregate throughput while tail latency degraded sharply.

This indicates that latency capacity can be reached before hard throughput saturation.

S3-E — Concurrency Control

The λ=4.0 stress point was used to study max_num_seqs.

The natural-generation experiment suggests a concurrency trade-off:

too much active concurrency
        ↓
execution interference
        ↓
tail latency increases

too little active concurrency
        ↓
scheduler waiting
        ↓
TTFT increases

The tested max_num_seqs=12 configuration provided the strongest TTFT result in this natural workload.

S3-D — Length-Matched Replay

To test whether the apparent max_num_seqs=12 improvement was caused directly by the concurrency cap, a controlled replay was performed.

Each request was forced to generate exactly the same number of output tokens as its corresponding S1 Flash request.

Both configurations produced:

120 / 120 successful requests
120 / 120 output-length matches
22,225 expected output tokens
22,225 observed output tokens

Results:

Metric	Default	maxseq12
Max running	11	11
Max waiting	0	0
P95 TTFT	231.98 ms	234.00 ms
P95 ITL	11.79 ms	11.87 ms
Output throughput	573.01 tok/s	572.58 tok/s

The default scheduler never exceeded 11 active requests, so the 12-sequence cap was never activated.

Therefore:

The natural-run improvement from max_num_seqs=12 is conditional on the workload producing enough active concurrency for the cap to matter.

S4 — SLO and Goodput

S4 converts the latency results into a production-oriented capacity metric.

A request passes the SLO when:

TTFT <= 300 ms
AND
mean SSE content-event ITL <= 15 ms

Request goodput:

SLO-passing requests / measured makespan

Output goodput:

output tokens from SLO-passing requests / measured makespan

At configured λ=3.75:

SLO pass rate      98.33%
raw throughput     2.913 req/s
request goodput    2.864 req/s
output goodput     546.4 tok/s

At configured λ=4.0:

SLO pass rate      86.67%
raw throughput     2.953 req/s
request goodput    2.559 req/s
output goodput     477.5 tok/s

Raw throughput increased slightly, but SLO-qualified goodput decreased.

This establishes the project's central serving result:

The throughput-maximizing operating point lies beyond the SLO-optimal operating point.

Bottleneck Evolution

The experiments show a clear change in the dominant performance bottleneck as the system becomes more realistic.

Stage	Dominant concern
E1	Long-context prefill
E2	Attention kernel efficiency
E3	Decode latency and KV-cache growth
R1	Realistic kernel-level inference
S1	API / streaming measurement
S2	Queueing
S3	Continuous batching and active concurrency
S3-E	Concurrency-control trade-off
S4	SLO-qualified capacity

This progression is one of the main conclusions of the project:

The bottleneck moves upward through the serving stack as lower-level computation becomes more efficient.

Repository Structure
.
├── README.md
├── requirements.txt
├── src/
│   ├── r0a_source_acquisition.py
│   ├── r0b_build_candidate_pools.py
│   ├── r0c_classify_requests.py
│   ├── r0d_construct_token_bands.py
│   ├── r0e_finalize_workload.py
│   ├── r1_sequential_benchmark.py
│   ├── r1_analyze_results.py
│   ├── s1_streaming_server.py
│   ├── s1_streaming_client.py
│   ├── s2_poisson_client.py
│   ├── s2_analyze_results.py
│   ├── s3_vllm_sequential_client_v2.py
│   ├── s3_vllm_poisson_client.py
│   ├── s3_vllm_length_matched_client.py
│   ├── s3_analyze_results.py
│   └── s4_analyze_goodput.py
│
├── workloads/
│   └── final/
│       ├── public_workload_manifest.csv
│       └── SHA256SUMS.txt
│
└── results/
    ├── r1/
    │   ├── figures/
    │   └── summary/
    ├── s2/
    │   ├── figures/
    │   ├── schedules/
    │   └── summary/
    ├── s3/
    │   ├── figures/
    │   └── summary/
    └── s4/
        ├── figures/
        └── summary/

Raw benchmark traces and full workload prompts are intentionally excluded from the public repository.

Reproducing the Analysis

After collecting experiment results:

python src/r1_analyze_results.py
python src/s2_analyze_results.py
python src/s3_analyze_results.py
python src/s4_analyze_goodput.py

Generated public summaries and figures are written under:

results/<stage>/summary/
results/<stage>/figures/
Measurement Notes

Several interpretation boundaries are intentionally preserved.

Natural generation

Even with fixed seeds, generation trajectories may differ across execution stacks or concurrent scheduling regimes due to numerical differences.

Therefore, output-throughput differences between natural-generation configurations should not automatically be interpreted as pure scheduler-causal speedups.

SSE ITL

The streaming ITL reported in S1–S4 is based on SSE content-event timing.

It is an application-level serving metric and should not be confused with GPU TPOT.

CUDA reserved memory

CUDA reserved memory represents caching-allocator reservation, not active tensor allocation.

Sequence-length boundary

A reproducible latency effect was observed near a 512-token cached-sequence boundary in an earlier controlled FP16 decode configuration.

The effect was not consistently reproduced in the later BF16 production-like serving path and is therefore treated as configuration-dependent rather than universal.

Capacity

The observed λ≈3.75–4.0 latency knee belongs to this specific:

model
GPU
software stack
workload distribution
finite Poisson trace

It should not be interpreted as a universal vLLM capacity limit.

Main Takeaway

This project started with a GPU-kernel optimization question and ended with a serving-systems conclusion:

LLM performance is a stack-level property.

Flash Attention can significantly reduce compute and memory cost, but production responsiveness also depends on:

arrival patterns
service-time variance
queueing
batching
request lifetimes
concurrency admission
scheduler behavior
latency SLOs

The highest-throughput configuration is therefore not necessarily the best production configuration.

For this workload and hardware, SLO-qualified goodput peaked before maximum observed raw throughput.