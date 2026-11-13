# tests/test_p2_diagnostics.py
import os
import shutil
import unittest

import torch
import yaml

from research.baseline.data_generator import generate_benchmark_dataset, rk4_step
from research.benchmarks import load_benchmark
from research.diagnostics.run_p2_diagnostics import (
    build_official_module,
    build_training_dataset,
    predict_one_step,
    run_p2_stress_diagnosis,
    train_official_baseline,
)


class TestTransitionAlignment(unittest.TestCase):
    def test_windows_align_current_action_with_next_state(self):
        # Stored format from the generator: x_seq[k] = x_{k+1}, u_seq[k] = u_k.
        traj = {
            "x0": torch.tensor([[10.0, 20.0]]),
            "u_seq": torch.tensor([[1.0], [2.0], [3.0]]),
            "x_seq": torch.tensor([[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]),
        }
        ds = build_training_dataset(
            [traj], window_len=2, max_windows_per_traj=10, seed=0
        )
        starts = {start for _, start in ds.samples}
        self.assertEqual(starts, {0, 1})
        for idx, (traj_idx, start) in enumerate(ds.samples):
            self.assertEqual(traj_idx, 0)
            x0, u_win, x_tgt = ds[idx]
            if start == 0:
                self.assertTrue(torch.equal(x0, traj["x0"].squeeze(0)))
            else:
                self.assertTrue(torch.equal(x0, traj["x_seq"][start - 1].squeeze(0)))
            # u_t at the same index as the returned state, never u_{t-1}.
            self.assertTrue(torch.equal(u_win, traj["u_seq"][start : start + 2]))
            self.assertTrue(torch.equal(x_tgt, traj["x_seq"][start : start + 2]))

    def test_generated_transitions_match_rk4(self):
        path, _, meta = generate_benchmark_dataset(
            "rotation_family", num_trajectories=2, T=5, dt=0.02, seed=7,
            save_dir="tests/tmp_p2_data", sys_kwargs={"a": 0.5},
        )
        sys = load_benchmark("rotation_family", a=0.5)["instance"]
        self.assertAlmostEqual(meta["dt"], sys.dt)
        data = torch.load(path)
        x = data[0]["x0"].squeeze(0)
        for t in range(5):
            u_t = data[0]["u_seq"][t]
            x = rk4_step(sys.dynamics, 0.0, x.unsqueeze(0), u_t.unsqueeze(0), sys.dt).squeeze(0)
            self.assertTrue(torch.allclose(x, data[0]["x_seq"][t], atol=1e-6))
        shutil.rmtree("tests/tmp_p2_data", ignore_errors=True)


class TestOfficialBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = "tests/tmp_p2_official"
        os.makedirs(cls.tmp, exist_ok=True)
        cls.path, cls.hash, cls.meta = generate_benchmark_dataset(
            "planar_quadrotor", num_trajectories=3, T=20, dt=0.01, seed=3,
            save_dir=cls.tmp,
        )
        cls.trajectories = torch.load(cls.path)
        cls.sys = load_benchmark("planar_quadrotor")["instance"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_module_is_upstream_and_uses_benchmark_dt(self):
        module = build_official_module(self.sys, latent_dim=12)
        self.assertEqual(type(module).__name__, "KoopmanLitModule")
        self.assertEqual(module.model.dt, 0.01)
        self.assertEqual(module.dt, 0.01)

    def test_one_step_prediction_uses_reported_dt(self):
        module = build_official_module(self.sys, latent_dim=12).double()
        module.eval()
        x = torch.tensor([[0.1, 0.2, 0.05, -0.1, 0.3, 0.02]], dtype=torch.float64)
        u = torch.tensor([[8.0, 9.0]], dtype=torch.float64)
        A, B = module.model.operator.get_AB()
        z = module.model.encode(x)
        z_next = z + module.dt * module.model.dynamics(0.0, z, u)
        expected = z_next[:, : self.sys.Nstates]
        got = predict_one_step(module, x, u, dt=self.sys.dt)
        self.assertTrue(torch.allclose(got, expected, atol=1e-10))

    def test_training_checkpoint_provenance(self):
        ckpt = os.path.join(self.tmp, "official.pt")
        module, prov = train_official_baseline(
            self.trajectories, self.sys, latent_dim=12, epochs=1, seed=0,
            window_len=5, max_windows_per_traj=2, batch_size=4,
            checkpoint_path=ckpt,
            dataset_sha256=self.hash,
        )
        self.assertEqual(prov["model_class"], "KoopmanLitModule")
        self.assertEqual(prov["config"]["koopman_mode"], "controllable")
        self.assertEqual(prov["dataset"]["sha256"], self.hash)
        self.assertEqual(prov["seed"], 0)
        self.assertEqual(prov["dt"], 0.01)
        self.assertTrue(os.path.exists(ckpt))
        self.assertEqual(prov["checkpoint"]["sha256"], _sha256(ckpt))

        reloaded = build_official_module(self.sys, latent_dim=12)
        reloaded.load_state_dict(torch.load(ckpt, weights_only=True), strict=True)


def _sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestP2DiagnosticsIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "tests/tmp_p2_diag"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_quick_report_integrity(self):
        report_path = run_p2_stress_diagnosis(output_dir=self.tmp_dir, quick_mode=True)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r") as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["baseline"]["model_class"], "KoopmanLitModule")
        self.assertEqual(data["baseline"]["model_config"]["koopman_mode"], "controllable")
        self.assertIn("provenance", data)
        self.assertTrue(data["provenance"]["git_head"])
        self.assertIn("acceptance_criteria", data)

        for a_key, entry in data["rotation_sweep"].items():
            self.assertIn("datasets", entry)
            self.assertIn("seeds", entry)
            for seed_entry in entry["seeds"].values():
                self.assertIn("one_step_mse", seed_entry)
                self.assertIn("open_loop_mse", seed_entry)
                self.assertEqual(set(seed_entry["open_loop_mse"]), {"15", "30", "50"})
                self.assertIn("derivative_error", seed_entry)
                self.assertIn("E_drift", seed_entry)
                self.assertIn("manifold_error", seed_entry)
                self.assertIn("rho_T", seed_entry["gramian"])
                self.assertIn("rank_deficient", seed_entry["gramian"])
                self.assertIn("E_response", seed_entry["response"])
                self.assertIn("candidate_regret", seed_entry)
                self.assertIn("h3", seed_entry)
            # Same data/training budget across the rotation sweep.
            cfg = entry["budget"]
            self.assertEqual(cfg["num_trajectories"], data["rotation_sweep"]["a_0.0"]["budget"]["num_trajectories"])
            self.assertEqual(cfg["epochs"], data["rotation_sweep"]["a_0.0"]["budget"]["epochs"])

        self.assertIn("h3_feasibility", data)
        self.assertIn("gates", data)
        for gate in ("p2", "h2_fixed_b_limitation", "h2_readiness", "h3"):
            self.assertIn(gate, data["gates"])
            self.assertIn(data["gates"][gate]["verdict"], {"PASS", "FAIL", "GO", "NO-GO", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"})
        self.assertEqual(
            data["gates"]["p2"]["criteria"]["official_baseline_used"],
            "KoopmanLitModule",
        )
        self.assertIn("failure_mode_separation", data)
        self.assertIn("mpc_evaluations", data)

        # No artificial positive margin when the Gramian is rank deficient.
        for entry in data["rotation_sweep"].values():
            for seed_entry in entry["seeds"].values():
                g = seed_entry["gramian"]
                if g["rank_deficient"]:
                    self.assertIsNone(g["rho_T"])


if __name__ == "__main__":
    unittest.main()
