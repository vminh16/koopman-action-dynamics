import unittest

import torch

from research.diagnostics import p1_upstream_faithful as f
from research.diagnostics.p1_audit import experiment as e


class UpstreamFaithfulTest(unittest.TestCase):
    def test_fraction_config_and_s5_sampler(self):
        cfg = f.config('pendulum', 42, 0.3)
        self.assertEqual(cfg['num_train_trajectories'], 1500)
        raw = [dict(x0=torch.zeros(2), x_seq=torch.zeros(250, 1, 2), u_seq=torch.zeros(250, 1)) for _ in range(6)]
        (x, u, y), info = f.windows(raw, [0, 2, 4], 42, 'traj_s5', 0.05)
        self.assertEqual(info, dict(windows=15, trajectories=3, max_start=4))
        self.assertEqual(tuple(u.shape), (15, 15, 1))

    def test_dense_p_matches_official_forward_and_is_sign_free(self):
        model = f.make_model('cartpole', 42, 'dense')
        op = model.model.operator.core
        P = op.build_P()
        self.assertTrue(torch.allclose(P, P.T))
        self.assertGreater(float(P[:4, 4:].abs().max()), 0)  # dense, not block-diagonal
        x, u = torch.randn(3, 4), torch.randn(3, 15, 1)
        A, B = e.matrices(model)
        self.assertTrue(torch.allclose(model(x, u), e.rollout(model, x, u, A, B), atol=1e-6))
        with torch.no_grad():  # physical input column can be negative, unlike block SPD
            op.H_dense.zero_(); op.H_dense[3, 3] = 1.0; op.H_dense[4, 3] = -2.0; op.B_free.zero_(); op.B_free[0] = 1.0
        self.assertLess(float(op.get_AB()[1][3, 0]), 0)


if __name__ == '__main__':
    unittest.main()
