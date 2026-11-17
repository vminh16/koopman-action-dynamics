import unittest

import torch

from research.diagnostics import h2_diagnosis as h
from research.diagnostics import p2_config_check as c
from research.diagnostics import p2_followup as old


class H2DiagnosisTest(unittest.TestCase):
    def test_fixed_b_response_is_cb_and_rollout_matches_baseline(self):
        model = h.make_model(92)
        X = torch.randn(5, 2)
        B = old.matrices(model)[1]
        torch.testing.assert_close(h.response(model, X), B[:2, 0].expand(5, 2))
        u = torch.randn(5, 15, 1)
        torch.testing.assert_close(h.rollout(model, X, u), c.rollout(model, X, u))

    def test_base_arm_is_exactly_the_frozen_baseline(self):
        ours, frozen = h.make_model(92).state_dict(), old.make_model(92, True).state_dict()
        self.assertEqual(ours.keys(), frozen.keys())
        for k in ours:
            torch.testing.assert_close(ours[k], frozen[k], rtol=0, atol=0)

    def test_g5_response_is_state_dependent_scalar_gain_along_cb(self):
        model = h.make_model(92, modulated=True)
        X = torch.randn(7, 2)
        R = h.response(model, X)
        cb = old.matrices(model)[1][:2, 0]
        cross = R[:, 0]*cb[1]-R[:, 1]*cb[0]  # parallel to CB: G.5 cannot rotate the input direction
        self.assertLess(float(cross.abs().max()), 1e-5)
        self.assertGreater(float(R[:, 0].std()), 0)

    def test_every_arm_gets_equal_update_budget(self):
        for arm, (_, ntrain, _, _) in h.ARMS.items():
            windows = 20*ntrain
            self.assertEqual((h.UPDATES*64//windows)*-(-windows//64), h.UPDATES, arm)


if __name__ == '__main__':
    unittest.main()
