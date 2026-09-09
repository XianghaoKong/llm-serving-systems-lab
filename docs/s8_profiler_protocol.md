# S8-B: GPU execution mechanism validation

## Objective

The completed S8 experiment established that a 24K-token prefill can create a
large background decode stall and that a 1K chunk budget reduces the stall at
the cost of longer prefill TTFT. S8-B tests the mechanism directly:

> Does chunking replace one long GPU-busy interval with shorter intervals that
> create scheduling opportunities for active decode streams?

S8-B is diagnostic. Nsight traces are not used to replace the client-side
latency measurements from the preregistered S8 experiment because profiling
changes execution overhead.

## Open-source method references

The capture procedure follows the vLLM developer profiling workflow:

- run the OpenAI-compatible server under `nsys profile`;
- set `VLLM_WORKER_MULTIPROC_METHOD=spawn`;
- use `--trace-fork-before-exec=true` and `--cuda-graph-trace=node`;
- select `cudaProfilerApi` as the dynamic capture range; and
- call `/start_profile` and `/stop_profile` around a small request window.

Reference: [vLLM profiling documentation](https://docs.vllm.ai/en/stable/contributing/profiling/).
The repository records the installed vLLM and Nsight versions in every run and
does not treat results from the upstream project as local measurements.

## Fixed experiment matrix

| Case | Background | Injection | Chunk budget |
|---|---:|---:|---:|
| `control_off_32768` | C64, 256/8192 | none | chunking off, 32K |
| `inject_off_32768` | C64, 256/8192 | 24576/16 | chunking off, 32K |
| `inject_on_4096` | C64, 256/8192 | 24576/16 | 4K |
| `inject_on_1024` | C64, 256/8192 | 24576/16 | 1K |

Five rotated blocks produce 20 diagnostic trials. A fresh vLLM process is used
for each trial. The model, dtype, prompt construction, FCFS policy, disabled
prefix cache, and request validation match the completed S8 experiment.
Medians and bootstrap 95% intervals are computed over the five runs, with a
run treated as the statistical unit.

## Capture and alignment

Each trial performs 15 seconds of unprofiled background warm-up. The client
then calls `/start_profile`, records the endpoint's request and return times on
the same monotonic clock as all SSE events, and immediately records the control
or injection marker. Background work is cancelled after a two-second recovery
window, after which `/stop_profile` finalizes the capture.

The extractor converts the Nsight SQLite export into two portable tables:

- `kernel_events.csv.gz`: every GPU kernel and its duration;
- `busy_intervals.csv.gz`: the union of overlapping kernels, merging adjacent
  kernels separated by at most 50 microseconds.

GPU timestamps are expressed relative to the first captured kernel. Injection
alignment uses the client-clock return from `/start_profile` as an approximate
zero. Endpoint latency and this limitation are retained in the output so the
plots do not imply sub-millisecond cross-process clock accuracy.

## Primary evidence

The mechanism is supported when the injected unchunked case has both a longer
maximum GPU-busy interval and a larger background SSE gap than the no-injection
control, while the 4K and 1K cases shorten both quantities. The report must also
show long-request TTFT, scheduler waiting, KV occupancy, preemptions, kernel
count, GPU active fraction, and the longest individual kernel.

The analysis does not claim that a single long CUDA kernel alone causes the
stall. A GPU-busy interval may contain many back-to-back kernels from one or
more streams.

## Run

```bash
S8_PROJECT_ROOT=/workspace/llm-serving-systems-lab \
S8_PROFILE_REPEATS=5 \
bash src/run_s8_profile.sh
```

For a one-block validation before the formal diagnostic run:

```bash
S8_PROFILE_REPEATS=1 bash src/run_s8_profile.sh
```

Raw `.nsys-rep`, SQLite, logs, request traces, and event tables remain under
`results/s8/raw/profile/` and are ignored by Git. After review, only aggregate
CSV files, an audit, a representative compressed trace or screenshot when
small enough, and the final figure should be published.
