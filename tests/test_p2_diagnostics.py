# tests/test_p2_diagnostics.py
import unittest
import os
import shutil
import yaml
from research.diagnostics.run_p2_diagnostics import run_p2_stress_diagnosis

class TestP2DiagnosticsIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "tests/tmp_p2_diag"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_smoke_stress_diagnosis(self):
        report_path = run_p2_stress_diagnosis(output_dir=self.tmp_dir, quick_mode=True)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("rotation_sweep", data)
        self.assertIn("benchmark_evaluations", data)
        self.assertIn("failure_mode_separation", data)
        self.assertIn("summary", data)

if __name__ == "__main__":
    unittest.main()

