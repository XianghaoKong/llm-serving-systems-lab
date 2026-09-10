import json
from pathlib import Path
import tempfile
import unittest

from src.s10_analyze import summarize_run


class ArtifactContracts(unittest.TestCase):
    def make_run(self, root):
        case = {"backend":"zero", "world":2, "steps":100, "global_tokens":8192}
        (root/"launch.json").write_text(json.dumps({"case":case,"returncode":0}))
        rank = {"complete":True,"formal":True,"steps_seconds":[2.0]*100,
                "loss":[1.0]*100,"gradient_norm":[0.5]*100,
                "peak_allocated_bytes":2**30,"peak_reserved_bytes":2**31}
        for index in range(2):
            (root/f"rank-{index}.json").write_text(json.dumps(rank))
        return rank

    def test_throughput_uses_global_tokens_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root)
            result = summarize_run(root)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["tokens_per_second"],4096)

    def test_failed_rank_rejects_whole_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rank = self.make_run(root)
            rank["complete"] = False
            (root/"rank-1.json").write_text(json.dumps(rank))
            self.assertFalse(summarize_run(root)["accepted"])

    def test_inconsistent_max_rank_times_are_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rank = self.make_run(root)
            rank["steps_seconds"][0] = 3.0
            (root/"rank-1.json").write_text(json.dumps(rank))
            with self.assertRaises(ValueError):
                summarize_run(root)
