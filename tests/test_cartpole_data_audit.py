"""Integrity checks on real data and deliberately corrupted small fixtures."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

import torch

from dynamics.cartpole import CartPoleDynamics
from research.baseline.common import ROOT
from research.baseline.data_generator import rk4_step
from research.baseline.cartpole_data_audit import (
    audit_tensor_structure, audit_actuator_bounds, audit_initial_state_bounds,
    audit_dynamics_mutation, audit_transition_consistency,
    audit_train_test_separation, audit_cartpole_dataset, AuditFailure, main,
)


def trajectory(position=0.0, steps=3):
    x = torch.tensor([[position, 0., 0.01, 0.]], dtype=torch.float64)
    x0 = x.clone()
    controls = torch.zeros(steps, 1, dtype=torch.float64)
    states = []
    for i, u in enumerate(controls):
        x = rk4_step(CartPoleDynamics().dynamics, i * .02, x, u, .02)
        states.append(x)
    return dict(x0=x0, u_seq=controls, x_seq=torch.stack(states))


class TestCartPoleDataAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = torch.load(ROOT / 'data/cartpole_test.pt', weights_only=True)

    def test_tensor_structure_and_shapes(self):
        stats = audit_tensor_structure(self.data, 1000, 250)
        for key, value in dict(count=1000, nan_count=0, inf_count=0,
                               x0_shape=[1, 4], u_seq_shape=[250, 1],
                               x_seq_shape=[250, 1, 4]).items():
            self.assertEqual(stats[key], value)

    def test_structure_rejects_corruption(self):
        for key in ('x0', 'u_seq', 'x_seq'):
            for bad in (float('nan'), float('inf')):
                data = [trajectory()]
                data[0][key].flatten()[0] = bad
                with self.assertRaises(ValueError):
                    audit_tensor_structure(data, 1, 3)
        for key in ('x0', 'u_seq', 'x_seq'):
            data = [trajectory()]
            data[0][key] = data[0][key].long()
            with self.assertRaises(ValueError):
                audit_tensor_structure(data, 1, 3)
        with self.assertRaises(ValueError):
            audit_tensor_structure([], 1, 3)
        data = [trajectory()]
        data[0]['x_seq'] = data[0]['x_seq'].squeeze(1)
        with self.assertRaises(ValueError):
            audit_tensor_structure(data, 1, 3)

    def test_actuator_bounds(self):
        self.assertTrue(audit_actuator_bounds(self.data)['bounds_satisfied'])
        for value in (-10.01, 10.01, float('nan')):
            data = [trajectory()]
            data[0]['u_seq'][0] = value
            with self.assertRaises(ValueError):
                audit_actuator_bounds(data)

    def test_initial_state_bounds(self):
        stats = audit_initial_state_bounds(self.data)
        self.assertTrue(stats['bounds_satisfied'])
        self.assertEqual(stats['x_dot_max'], 0.)
        self.assertEqual(stats['theta_dot_max'], 0.)
        for index, value in ((0, 4.01), (1, .001), (2, .101), (3, .001)):
            data = [trajectory()]
            data[0]['x0'][0, index] = value
            with self.assertRaises(ValueError):
                audit_initial_state_bounds(data)

    def test_dynamics_mutation_free(self):
        safe, message = audit_dynamics_mutation()
        self.assertTrue(safe, message)

    def test_transition_consistency_sample(self):
        stats = audit_transition_consistency(self.data[:10])
        self.assertTrue(stats['consistent'])
        self.assertLess(stats['max_error'], 1e-4)

    def test_transition_rejects_first_and_last_corruption(self):
        for index in (0, -1):
            data = [trajectory()]
            data[0]['x_seq'][index, 0, 0] += .01
            with self.assertRaises(ValueError):
                audit_transition_consistency(data)
        for tol in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                audit_transition_consistency([trajectory()], tol)

    def test_train_test_separation(self):
        train = torch.load(ROOT / 'data/cartpole_train.pt', weights_only=True)
        stats = audit_train_test_separation(train, self.data)
        self.assertTrue(stats['disjoint'])
        self.assertGreaterEqual(stats['min_x0_distance'], 1e-6)

    def test_separation_rejects_leakage(self):
        first = trajectory()
        for other in (copy.deepcopy(first), trajectory(5e-7)):
            with self.assertRaises(ValueError):
                audit_train_test_separation([first], [other])
        other = trajectory(.1)
        other['x_seq'] = first['x_seq'].clone()
        with self.assertRaises(ValueError):
            audit_train_test_separation([first], [other])
        stats = audit_train_test_separation([first], [trajectory(.1)])
        self.assertTrue(stats['disjoint'])

    def test_full_audit_preserves_real_transition_failure(self):
        with self.assertRaises(AuditFailure) as caught:
            audit_cartpole_dataset(ROOT / 'data/cartpole_train.pt', ROOT / 'data/cartpole_test.pt')
        report = caught.exception.report
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['splits']['train']['transitions']['transitions_checked'], 2500000)
        self.assertFalse(report['splits']['train']['transitions']['consistent'])
        self.assertTrue(report['splits']['test']['transitions']['consistent'])
        self.assertTrue(report['separation']['disjoint'])

    def test_cli_writes_failure_report_and_exits_nonzero(self):
        report = dict(status='failed', errors=['transition mismatch'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit.yaml'
            with patch('sys.argv', ['audit', '--output', str(output)]), patch(
                'research.baseline.cartpole_data_audit.audit_cartpole_dataset',
                side_effect=AuditFailure('transition mismatch', report)
            ), self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(yaml.safe_load(output.read_text()), report)
