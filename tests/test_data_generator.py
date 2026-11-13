# tests/test_data_generator.py
import unittest
import os
import shutil
import torch
from research.baseline.data_generator import generate_benchmark_dataset

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "tests/tmp_data_gen"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_generate_rotation_dataset(self):
        path, data_hash, meta = generate_benchmark_dataset(
            sys_name="rotation_family",
            num_trajectories=5,
            T=30,
            dt=0.02,
            seed=42,
            save_dir=self.tmp_dir,
            sys_kwargs={"a": 0.5}
        )
        self.assertTrue(os.path.exists(path))
        self.assertEqual(len(data_hash), 64)
        data = torch.load(path)
        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]["x_seq"].shape, (30, 2))
        self.assertEqual(data[0]["u_seq"].shape, (30, 1))
        self.assertIn("action_mean", meta)
        self.assertIn("state_bounds", meta)

    def test_reproducible_hash(self):
        _, hash1, _ = generate_benchmark_dataset("unicycle", 3, 20, 0.02, 123, self.tmp_dir)
        _, hash2, _ = generate_benchmark_dataset("unicycle", 3, 20, 0.02, 123, self.tmp_dir)
        self.assertEqual(hash1, hash2)

    def test_planar_quadrotor_dataset(self):
        path, data_hash, meta = generate_benchmark_dataset("planar_quadrotor", 2, 15, 0.01, 77, self.tmp_dir)
        self.assertTrue(os.path.exists(path))
        data = torch.load(path)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["x_seq"].shape, (15, 6))
        self.assertEqual(data[0]["u_seq"].shape, (15, 2))

if __name__ == "__main__":
    unittest.main()

