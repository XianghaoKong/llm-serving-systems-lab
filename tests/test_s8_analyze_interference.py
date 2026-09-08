import csv
import gzip
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from src.s8_analyze_interference import aggregate, summarize_trial
from src.s8_interference_client import streaming_request


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


if __name__ == "__main__":
    unittest.main()
