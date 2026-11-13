# tests/test_koopman_operator.py
import unittest

import torch

from models.koopman_operator import (
    KoopmanOperator,
    MultiInputControllableKoopmanOperator,
    SingleInputControllableKoopmanOperator,
)


def output_controllability_rank(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> int:
    N = A.shape[0]
    blocks = [B]
    for _ in range(N - 1):
        blocks.append(A @ blocks[-1])
    return int(torch.linalg.matrix_rank(C @ torch.cat(blocks, dim=1)).item())


class TestMultiInputOperator(unittest.TestCase):
    def test_build_and_similarity_identities(self):
        # Previously raised NameError: name 'device' is not defined.
        op = MultiInputControllableKoopmanOperator(latent_dim=12, state_dim=6, control_dim=2)
        A, B = op.get_AB()
        self.assertEqual(tuple(A.shape), (12, 12))
        self.assertEqual(tuple(B.shape), (12, 2))

        Ac, Bc, P = op.build_Ac(), op.build_Bc(), op.build_P()
        self.assertTrue(
            torch.allclose(A, P @ Ac @ torch.linalg.inv(P), atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(torch.allclose(B, P @ Bc, atol=1e-5, rtol=1e-5))

        eig = torch.linalg.eigvalsh(P)
        self.assertGreater(float(eig.detach().min()), 0.0)
        self.assertTrue(torch.allclose(P, P.T, atol=1e-6))

    def test_dynamic_dtype_and_device(self):
        op = MultiInputControllableKoopmanOperator(12, 6, 2).double()
        A, B = op.get_AB()
        self.assertEqual(A.dtype, torch.float64)
        self.assertEqual(B.dtype, torch.float64)
        self.assertEqual(op.build_P().dtype, torch.float64)

    def test_gradients_flow_to_all_free_parameters(self):
        op = MultiInputControllableKoopmanOperator(12, 6, 2)
        A, B = op.get_AB()
        loss = A.pow(2).sum() + B.pow(2).sum()
        loss.backward()
        for name in ("A_corelast_free", "A_tail_free", "B_tail", "H_head", "H_tail"):
            grad = getattr(op, name).grad
            self.assertIsNotNone(grad, f"no gradient for {name}")
            self.assertGreater(float(grad.abs().sum()), 0.0, f"zero gradient for {name}")

    def test_output_controllability_is_preserved(self):
        op = MultiInputControllableKoopmanOperator(12, 6, 2)
        A, B = op.get_AB()
        C = torch.eye(6, 12)
        rank = output_controllability_rank(A, B, C)
        self.assertEqual(rank, 6)

    def test_legacy_H_still_loadable(self):
        # Old checkpoints contain a leftover dense `H`; keep it registered.
        op = MultiInputControllableKoopmanOperator(8, 4, 2)
        self.assertIn("H", op.state_dict())
        state = op.state_dict()
        op2 = MultiInputControllableKoopmanOperator(8, 4, 2)
        op2.load_state_dict(state, strict=True)

    def test_factory_multi_input(self):
        op = KoopmanOperator(latent_dim=8, control_dim=2, state_dim=4, mode="controllable")
        self.assertIsInstance(op.core, MultiInputControllableKoopmanOperator)
        A, B = op.get_AB()
        self.assertEqual(tuple(B.shape), (8, 2))


class TestSingleInputOperator(unittest.TestCase):
    def test_similarity_and_margin(self):
        op = SingleInputControllableKoopmanOperator(latent_dim=6, state_dim=2)
        A, B = op.get_AB()
        Ac, Bc, P = op.build_Ac(), op.build_Bc(), op.build_P()
        self.assertTrue(torch.allclose(A, P @ Ac @ torch.linalg.inv(P), atol=1e-5))
        self.assertTrue(torch.allclose(B, P @ Bc, atol=1e-5))
        self.assertGreater(float(torch.linalg.eigvalsh(P).detach().min()), 0.0)
        self.assertEqual(output_controllability_rank(A, B, torch.eye(2, 6)), 2)


if __name__ == "__main__":
    unittest.main()
