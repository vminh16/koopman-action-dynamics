import unittest

import torch

from research.diagnostics import h2_gate as g


class H2GateTest(unittest.TestCase):
    def test_cos_feature_gain_sees_cos_theta_and_round_trips(self):
        model = g.cart_model('cartpole', 42, 'dense')
        x, u = torch.randn(4, 4), torch.ones(4, 1)
        shifted = x.clone(); shifted[:, 2] += 2*torch.pi  # unwrapped angle: same physical state
        enc = model.model.u_encoder
        torch.testing.assert_close(enc(x, u), enc(shifted, u))
        again = g.cart_model('cartpole', 42, 'dense')
        again.load_state_dict(model.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
