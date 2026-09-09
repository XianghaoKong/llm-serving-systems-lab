import csv
import gzip
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from src.s8_analyze_interference import aggregate, summarize_trial
from src.s8_interference_client import call_profile_endpoint, streaming_request
from src.s8_analyze_profile import aggregate as aggregate_profile
from src.s8_nsys_extract import extract_kernel_rows, merge_busy_intervals


class S8InterferenceAnalysisTest(unittest.TestCase):
    def test_boundary_crossing_stall_is_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial_dir = Path(temporary) / "block_1" / "on_2048" / "inject_16384"
            trial_dir.mkdir(parents=True)
            metadata = {
                "status": "complete",
                "config_label": "on_2048",
                "replicate": 1,
                "trial_kind": "inject",
                "background_concurrency": 1,
                "interferer_input_tokens": 16384,
                "interferer_count": 1,
                "injection_t_s": 5.0,
                "impact_end_t_s": 5.1,
                "preemptions_delta": 0,
            }
            (trial_dir / "trial.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            requests = [
                {
                    "request_id": "bg-0",
                    "role": "background",
                    "status": "cancelled",
                    "start_t_s": 0.0,
                    "end_t_s": 6.0,
                },
                {
                    "request_id": "long-0",
                    "role": "interferer",
                    "status": "ok",
                    "start_t_s": 5.0,
                    "first_content_t_s": 5.1,
                    "end_t_s": 5.2,
                },
            ]
            with (trial_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
                for row in requests:
                    handle.write(json.dumps(row) + "\n")

            event_times = [index / 100 for index in range(0, 500)]
            event_times += [5.1 + index / 100 for index in range(0, 91)]
            with gzip.open(
                trial_dir / "content_events.csv.gz",
                "wt",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "t_s", "role", "request_id",
                        "content_event_index", "content_chars",
                    ],
                )
                writer.writeheader()
                for index, event_time in enumerate(event_times, start=1):
                    writer.writerow(
                        {
                            "t_s": event_time,
                            "role": "background",
                            "request_id": "bg-0",
                            "content_event_index": index,
                            "content_chars": 1,
                        }
                    )
            with (trial_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for event_time in [5.0, 5.05, 5.1]:
                    handle.write(
                        json.dumps(
                            {
                                "t_s": event_time,
                                "running": 2,
                                "waiting": 0,
                                "kv_usage": 0.01,
                            }
                        ) + "\n"
                    )

            row = summarize_trial(trial_dir / "trial.json", baseline_seconds=4.0)
            self.assertAlmostEqual(row["long_ttft_p50_ms"], 100.0)
            self.assertGreater(row["impact_gap_max_ms"], 100.0)
            self.assertGreater(row["p99_stall_ratio"], 5.0)
            grouped = aggregate([row, row], seed=2026)
            self.assertEqual(grouped[0]["n_runs"], 2)
            self.assertEqual(grouped[0]["background_concurrency"], 1)


class S8StreamingParserTest(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_usage_and_event_count_are_validated(self):
        async def handler(_request):
            body = "\n".join(
                [
                    'data: {"choices":[{"text":"a","finish_reason":null}]}',
                    "",
                    'data: {"choices":[{"text":"b","finish_reason":"length"}]}',
                    "",
                    "data: {\"choices\":[],\"usage\":"
                    "{\"prompt_tokens\":2,\"completion_tokens\":2}}",
                    "",
                    "data: [DONE]",
                    "",
                ]
            )
            return httpx.Response(200, content=body.encode("utf-8"))

        traces = []
        events = []
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            trace = await streaming_request(
                client=client,
                base_url="http://test",
                model="test-model",
                prompt_ids=[10, 11],
                output_tokens=2,
                request_id="test-0",
                role="interferer",
                sampling_seed=2026,
                trial_t0=time.perf_counter(),
                traces=traces,
                content_events=events,
            )
        self.assertEqual(trace["status"], "ok")
        self.assertEqual(trace["actual_input_tokens"], 2)
        self.assertEqual(trace["actual_output_tokens"], 2)
        self.assertEqual(trace["content_event_minus_output_tokens"], 0)
        self.assertEqual(len(events), 2)

    async def test_profile_endpoint_retains_alignment_metadata(self):
        async def handler(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/start_profile")
            return httpx.Response(200, json={"status": "started"})

        trial_t0 = time.perf_counter()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await call_profile_endpoint(
                client,
                "http://test",
                "/start_profile",
                trial_t0,
            )
        self.assertEqual(result["status_code"], 200)
        self.assertGreaterEqual(result["return_t_s"], result["request_t_s"])
        self.assertGreaterEqual(result["latency_ms"], 0.0)


class S8NsightExtractionTest(unittest.TestCase):
    def test_nsight_schema_discovery_and_busy_interval_union(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)"
            )
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                "(start INTEGER, end INTEGER, demangledName INTEGER, streamId INTEGER)"
            )
            connection.executemany(
                "INSERT INTO StringIds(id, value) VALUES (?, ?)",
                [(1, "prefill_gemm"), (2, "decode_attention")],
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                [
                    (1_000_000, 2_000_000, 1, 7),
                    (2_025_000, 2_500_000, 2, 8),
                    (2_700_000, 3_000_000, 2, 8),
                ],
            )
            rows = extract_kernel_rows(connection)
        self.assertEqual([row["name"] for row in rows], [
            "prefill_gemm", "decode_attention", "decode_attention"
        ])
        intervals = merge_busy_intervals(rows, gap_threshold_ns=50_000)
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0]["duration_ms"], 1.5)
        self.assertEqual(intervals[0]["kernel_count"], 2)


class S8ProfileAggregationTest(unittest.TestCase):
    def test_profile_aggregate_reports_reproducible_bootstrap_intervals(self):
        rows = [
            {
                "case": "inject_on_1024",
                "trial_kind": "inject",
                "config_label": "on_1024",
                "impact_max_busy_interval_ms": value,
                "max_kernel_duration_ms": value / 10,
                "gpu_active_fraction": 0.99,
                "background_p99_stall_ratio": value / 100,
                "background_impact_max_gap_ms": value / 2,
                "long_request_ttft_p50_ms": value * 2,
                "impact_max_waiting": 0,
                "impact_max_kv_usage": 0.08,
                "preemptions_delta": 0,
            }
            for value in [100.0, 200.0, 300.0, 400.0, 500.0]
        ]
        first = aggregate_profile(rows, seed=2026)[0]
        second = aggregate_profile(rows, seed=2026)[0]
        self.assertEqual(first, second)
        self.assertEqual(first["n_runs"], 5)
        self.assertEqual(first["impact_max_busy_interval_ms_median"], 300.0)
        self.assertLessEqual(
            first["impact_max_busy_interval_ms_ci_low"], 300.0
        )
        self.assertGreaterEqual(
            first["impact_max_busy_interval_ms_ci_high"], 300.0
        )


if __name__ == "__main__":
    unittest.main()
