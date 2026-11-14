"""CartPole operator and end-to-end runner validation, without training."""
import subprocess
import sys
import unittest

import torch

from research.baseline.common import ROOT, build_module, load_config

CONFIG = 'configs/research/cartpole_controllable_win15_frac0.05_seed42.yml'


class TestCartPoleRunnerContract(unittest.TestCase):
    def test_operator_matrix_dimensions(self):
        module = build_module(load_config(ROOT / CONFIG))
        op = module.model.operator.core
        self.assertEqual(op.A_free.shape, (22, 25))
        self.assertEqual(op.B_free.shape, (21, 1))
        self.assertEqual(op.H_head.shape, (4, 4))
        self.assertEqual(op.H_tail.shape, (21, 21))
        matrix = op.build_P()
        self.assertEqual(matrix.shape, (25, 25))
        self.assertTrue(torch.allclose(matrix, matrix.T))
        self.assertTrue((torch.linalg.eigvalsh(matrix.double()) > 0).all())

    def test_train_koopman_dry_run_execution(self):
        result = subprocess.run(
            [sys.executable, '-m', 'research.baseline.train_koopman',
             '--config', CONFIG, '--dry-run'],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('dry run: config, dataset and model contract are valid', result.stdout)
