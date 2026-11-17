import unittest

import torch

from research.diagnostics import h2_diagnosis as h
from research.diagnostics import h2_g5_tests as t
from research.diagnostics import p2_config_check as c
from research.diagnostics import p2_followup as old


class G5TestsHarness(unittest.TestCase):
    def test_generic_loss_and_matrices_equal_rotation_baseline(self):
        model = h.make_model(92)
        batch = (torch.randn(4, 2), torch.randn(4, 15, 1), torch.randn(4, 15, 2))
        for x, y in zip(t.matrices(model), old.matrices(model)):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(t.loss(model, batch), c.loss(model, batch))

    def test_multi_input_fixed_b_response_is_constant_cb(self):
        model = t.make_model('planar_quadrotor', 42, False)
        R = t.response(model, torch.randn(5, 6))
        B = t.matrices(model)[1][:6]
        torch.testing.assert_close(R, B.expand(5, 6, 2))
        g5 = t.make_model('unicycle', 42, True)
        self.assertEqual(tuple(t.response(g5, torch.randn(3, 3)).shape), (3, 3, 2))


if __name__ == '__main__':
    unittest.main()
