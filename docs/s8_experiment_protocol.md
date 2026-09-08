# S8: Prefill–Decode Interference and Chunked-Prefill Scheduling

## Status

This document is the pre-pilot protocol. Formal parameters must be frozen after
the calibration and pilot gates below. Pilot observations may select the
background concurrency and interferer count, but formal results must not be used
to revise the protocol.

## Objective

S8 tests how a long prefill affects decode requests already running on one GPU,
then measures how vLLM's chunked-prefill token budget changes the trade-off
between:

- background decode content-event gaps;
- long-request time to first token (TTFT); and
- aggregate output throughput.

The primary causal event is a deterministic long-prompt injection into a stable
closed-loop decode population. A later open-arrival validation will use the
configurations selected from this causal experiment.

## Hypotheses

1. A long prefill increases the P99 content-event gap of background decode
   requests relative to the same run's pre-injection window.
2. Reducing `max_num_batched_tokens` reduces the decode stall but increases the
   long request's TTFT.
3. An intermediate token budget lies on the TTFT–decode-stall–throughput Pareto
   frontier; an extreme small budget may add scheduling and launch overhead.

Failure to support a hypothesis is a valid result. In particular, the A100 may
process the 1.5B model's single long prefill too quickly to produce a material
stall.

## Fixed system configuration

| Property | Value |
|---|---|
| GPU | 1× NVIDIA A100 80GB PCIe |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Precision | BF16 |
| vLLM | 0.28.0 for the initial experiment |
| Maximum context | 32,768 tokens |
| GPU memory utilization | 0.90 |
| Maximum sequences | 128 |
| Scheduling policy | FCFS |
| Concurrent partial prefills | 1 |
| Long-prefill threshold | 0 |
| Prefix caching | Disabled |
| Speculative decoding | Disabled |

Prompts are deterministic exact token-ID lists. Generation uses temperature 0,
`ignore_eos=true`, and exact output-length validation. Background and long
requests receive unique suffixes even though prefix caching is disabled.

## Server configurations

| Label | Chunked prefill | `max_num_batched_tokens` |
|---|---:|---:|
| `off_32768` | no | 32768 |
| `on_32768` | yes | 32768 |
| `on_16384` | yes | 16384 |
| `on_8192` | yes | 8192 |
| `on_4096` | yes | 4096 |
| `on_2048` | yes | 2048 |
| `on_1024` | yes | 1024 |
| `on_512` | yes | 512 |

The `off_32768` versus `on_32768` pair holds the token budget constant. It is
the control for the scheduling-policy effect. The smaller enabled budgets then
measure the additional effect of forcing partial prefills.

## Calibration gate

The background request shape is 256 input tokens and 8,192 output tokens. Test
background concurrency 8, 16, 32, and 64 with `on_8192`. Select the highest
point meeting all predeclared criteria during the steady observation window:

- no persistent scheduler waiting;
- no preemptions;
- P95 background content-event gap has not entered a sharp knee; and
- GPU utilization is high enough to represent a meaningful decode load.

Use the existing S7 telemetry to interpret GPU utilization. Do not select a
concurrency solely because it makes the later interference effect larger.

Calibration commands:

```bash
for c in 8 16 32 64; do
  S8_PHASE=pilot \
  S8_CONFIGS="on_8192" \
  S8_INPUT_LENGTHS="16384" \
  S8_BACKGROUND_CONCURRENCY="$c" \
  bash src/run_s8_interference.sh
done
```

## Pilot gate

The pilot tests a 16,384-token prefill with one injected request against five
representative configurations. If the median P99 stall ratio for
`off_32768` is below 1.20, freeze the formal interferer count at four and repeat
the pilot. Otherwise freeze it at one.

```bash
S8_PHASE=pilot \
S8_BACKGROUND_CONCURRENCY=<calibrated value> \
bash src/run_s8_interference.sh
```

The pilot must also confirm:

- every injected request reports exactly the requested input and output tokens;
- the background population remains present throughout the impact window;
- metric scraping produces no material errors; and
- completed responses disclose whether SSE content-event count equals generated
  token count.

If SSE event count differs from output token count, the primary metric remains
named `content-event gap`; it must not be renamed token-level ITL.

## Formal causal experiment

Each formal block rotates configuration and prompt-length order. Every server
configuration starts in a fresh vLLM process. An excluded stabilization trial is
followed by one control and one injection at each long input length:

- 8,192 input / 16 output tokens;
- 16,384 input / 16 output tokens; and
- 24,576 input / 16 output tokens.

Each trial contains:

1. 15 seconds of background decode warm-up;
2. one marked control or injection window; and
3. 15 seconds of post-impact observation.

Five independent blocks are required. A run, not an individual token or SSE
event, is the statistical unit.

```bash
S8_PHASE=formal \
S8_PROTOCOL_FROZEN=1 \
S8_BACKGROUND_CONCURRENCY=<calibrated value> \
S8_INTERFERER_COUNT=<frozen value> \
bash src/run_s8_interference.sh
```

## Primary measurements

For each background request, construct consecutive client-side content-event
intervals using one monotonic clock.

- Baseline window: five to one seconds before injection.
- Impact window: injection dispatch through the last injected request's first
  visible content event.
- Primary effect: impact-window P99 gap divided by baseline-window P99 gap.
- Secondary effects: P95 gap ratio, maximum gap, background content-event rate,
  long-request TTFT, scheduler waiting, KV occupancy, and preemptions.

Intervals crossing a window boundary are included, so the interval containing a
complete decode stall is not accidentally omitted. Aggregate estimates report
the median across runs and a bootstrap 95% confidence interval over runs.

## Profiling follow-up

Profile only three representative configurations using the exact formal trace:

1. `off_32768`;
2. `on_32768`; and
3. the selected Pareto-knee configuration.

Nsight Systems captures are diagnostic and are not used as formal latency
measurements. Start the server with dynamic CUDA-profiler capture:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --profiler-config.profiler cuda \
  <the matching formal server arguments>
```

Use `/start_profile` immediately before the marked injection interval and
`/stop_profile` after the short recovery window. Run
`--enable-logging-iteration-details` separately because detailed logging can
perturb timing.

## Output layout

Raw trial artifacts are excluded from version control:

```text
results/s8/raw/interference_<phase>/<run-id>/
├── environment.txt
├── protocol.txt
├── block_<n>/<config>/
│   ├── server_command.txt
│   ├── control/
│   └── inject_<input-tokens>/
└── analysis/
    ├── trial_summary.csv
    ├── aggregate_summary.csv
    ├── analysis.json
    ├── stall_ratio_by_config.png
    └── ttft_stall_pareto.png
```

After the causal experiment, only the non-chunked baseline, equal-budget
chunked control, best decode-stall configuration, and Pareto-knee configuration
advance to the open-arrival mixed-workload validation.
