import unittest
from src.s10_config import accumulation,parallel_degrees,zero_state_bytes


class TrainingContracts(unittest.TestCase):
    def test_global_tokens_are_preserved_across_dp(self):
        for world in (1,2,4):
            acc=accumulation(8192,512,1,world)
            self.assertEqual(acc*world*512,8192)
        with self.assertRaises(ValueError):accumulation(8193,512,1,2)

    def test_parallel_layout_and_layer_divisibility(self):
        self.assertEqual(parallel_degrees(4,2,1,28,28),2)
        with self.assertRaises(ValueError):parallel_degrees(4,2,4,28,28)
        with self.assertRaises(ValueError):parallel_degrees(4,1,4,30,28)

    def test_state_sharding_is_identity_at_world_one(self):
        for stage in range(4):
            self.assertEqual(sum(zero_state_bytes(1000,1,stage).values()),16000)
        self.assertEqual(sum(zero_state_bytes(1000,4,3).values()),4000)


if __name__=="__main__":unittest.main()
