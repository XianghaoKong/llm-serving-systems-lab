import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from s9_analyze import aggregate


class AggregationTests(unittest.TestCase):
    def row(self, backend, block, us):
        return dict(op="rms", rows=1, width=1536, dtype="torch.float16", backend=backend,
                    phase="forward", block=block, correctness=True,p50_us=us,p95_us=us*2,
                    output_max_abs_error=0,incremental_peak_allocated_bytes=128)

    def test_block_medians_and_baseline_direction(self):
        rows=[self.row("eager",i+1,x) for i,x in enumerate([10,10,10,10,1000])]
        rows += [self.row("triton",i+1,2) for i in range(5)]
        out=aggregate(rows)
        triton=next(r for r in out if r["backend"]=="triton")
        self.assertEqual(triton["speedup_vs_eager_or_unfused"],5)
        self.assertEqual(triton["blocks"],5)

    def test_duplicate_block_rejected(self):
        r=self.row("eager",1,10)
        with self.assertRaises(ValueError): aggregate([r,r])

    def test_failed_correctness_rejected(self):
        r=self.row("eager",1,10);r["correctness"]=False
        with self.assertRaises(ValueError): aggregate([r])
