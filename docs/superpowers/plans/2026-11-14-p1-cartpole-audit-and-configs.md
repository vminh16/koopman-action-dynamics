# CartPole Data Integrity Audit & Locked Configurations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a mathematically rigorous data integrity audit for the CartPole dataset and construct locked multi-seed configuration files (`.yml`) to guarantee faithful, drift-free reproduction of the ICLR 2026 Controllable Koopman baseline.

**Architecture:** Implement a comprehensive mathematical verification suite in `research/baseline/cartpole_data_audit.py` to audit tensor dimensions, finite regularity, actuator constraints, state-space geometry, ODE transition fidelity, and train/test disjointness. Generate 12 locked experiment configs in `configs/research/*.yml` encoding paper Table 5 specifications ($N=25$, $dt=0.02$, $H=15$, Adam $10^{-3}$, 100 epochs) across 4 data fractions (1%, 5%, 10%, 30%) and 3 seeds (42, 43, 44), verified by contract tests.

**Tech Stack:** PyTorch (`torch`), PyTorch Lightning, NumPy, PyYAML (`yaml`), unittest.

**Spec:** [ROADMAP.md](file:///home/minh/Desktop/Controllable-Koopman/ROADMAP.md#L163-L174), [TRACKING.md](file:///home/minh/Desktop/Controllable-Koopman/TRACKING.md#L1-L85), [RRD_v0.5.md](file:///home/minh/Desktop/Controllable-Koopman/research/common/RRD_v0.5.md), ICLR 2026 Paper Section G.1, G.2, Table 1, Table 5.

## Global Constraints

- **Python environment:** Python 3.10+, PyTorch with CPU/CUDA support via `.venv/bin/python`.
- **Dataset location & layout:** `data/cartpole_train.pt` (10,000 trajectories) and `data/cartpole_test.pt` (1,000 trajectories); each trajectory is a dict with keys `{'x0', 'u_seq', 'x_seq'}`.
- **Config format:** Must use `.yml` extension in `configs/research/` as requested by user.
- **Paper parameters for CartPole:** $N = 25$ observables, $n = 4$ state dims, $m = 1$ control dims, window length $H = 15$, learning rate $1 \times 10^{-3}$, batch size 64, epochs 100.
- **Float precision for verification:** High-precision float64 for mathematical checks (Gramian, eigenvalues, and integration error metrics).
- **Test execution:** Use standard library runner `PYTHONPATH=. .venv/bin/python -m unittest discover tests` (zero external test runner dependencies).

## Review Focus

1. **Actuator bounds violation:** Trajectory controls exceeding $[-10.0, 10.0]$ N must fail audit immediately.
2. **In-place state mutation:** Evaluating `CartPoleDynamics.dynamics(t, x, u)` must never mutate tensor `x` in-place, preserving autograd backward compatibility.
3. **Train/Test data leakage:** Any identical initial condition $\|x_0^{\text{train}} - x_0^{\text{test}}\|_2 < 10^{-6}$ or trajectory overlap must trigger a fatal audit error.
4. **Observable dimension mismatch:** CartPole requires $N=25$ (not Pendulum's $N=16$); any config with $N \neq 25$ must fail contract verification.
5. **Effective hyperparameter drift:** Any divergence between YAML config entries and `KoopmanLitModule` constructed attributes must abort before training.

---

## Tasks

### Task 1: CartPole Mathematical & Data Integrity Audit Engine

**Files:**
- Create: `research/baseline/cartpole_data_audit.py`
- Test: `tests/test_cartpole_data_audit.py`

**Interfaces:**
- Consumes: `data/cartpole_train.pt`, `data/cartpole_test.pt`, `dynamics.cartpole.CartPoleDynamics`.
- Produces: `audit_cartpole_dataset(train_path: str | Path, test_path: str | Path, tol: float = 1e-4) -> dict` and CLI tool outputting `outputs/reproduction/cartpole_data_audit.yaml`.

- [ ] **Step 1: Write the failing unit tests for CartPole data audit**

```python
# tests/test_cartpole_data_audit.py
import unittest
import torch
from pathlib import Path
from research.baseline.cartpole_data_audit import (
    audit_tensor_structure,
    audit_actuator_bounds,
    audit_initial_state_bounds,
    audit_dynamics_mutation,
    audit_transition_consistency,
    audit_train_test_separation,
    audit_cartpole_dataset,
)

class TestCartPoleDataAudit(unittest.TestCase):
    def setUp(self):
        self.train_path = Path("data/cartpole_train.pt")
        self.test_path = Path("data/cartpole_test.pt")

    def test_tensor_structure_and_shapes(self):
        data = torch.load(self.test_path)
        stats = audit_tensor_structure(data, expected_count=1000, expected_T=250)
        self.assertEqual(stats["count"], 1000)
        self.assertEqual(stats["nan_count"], 0)
        self.assertEqual(stats["inf_count"], 0)
        self.assertEqual(stats["x0_shape"], [1, 4])
        self.assertEqual(stats["u_seq_shape"], [250, 1])
        self.assertEqual(stats["x_seq_shape"], [250, 1, 4])

    def test_actuator_bounds(self):
        data = torch.load(self.test_path)
        u_stats = audit_actuator_bounds(data, u_min=-10.0, u_max=10.0)
        self.assertTrue(u_stats["bounds_satisfied"])
        self.assertGreaterEqual(u_stats["min_u"], -10.0)
        self.assertLessEqual(u_stats["max_u"], 10.0)

    def test_initial_state_bounds(self):
        data = torch.load(self.test_path)
        init_stats = audit_initial_state_bounds(data)
        self.assertTrue(init_stats["bounds_satisfied"])
        self.assertAlmostEqual(init_stats["x_dot_max"], 0.0, places=5)
        self.assertAlmostEqual(init_stats["theta_dot_max"], 0.0, places=5)

    def test_dynamics_mutation_free(self):
        is_safe, error_msg = audit_dynamics_mutation()
        self.assertTrue(is_safe, f"CartPole dynamics mutated input: {error_msg}")

    def test_transition_consistency_sample(self):
        data = torch.load(self.test_path)
        trans_stats = audit_transition_consistency(data[:10], tol=1e-4)
        self.assertTrue(trans_stats["consistent"])
        self.assertLess(trans_stats["max_error"], 1e-4)

    def test_train_test_separation(self):
        train_data = torch.load(self.train_path)
        test_data = torch.load(self.test_path)
        sep_stats = audit_train_test_separation(train_data, test_data)
        self.assertTrue(sep_stats["disjoint"])
        self.assertGreater(sep_stats["min_x0_distance"], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_cartpole_data_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.baseline.cartpole_data_audit'`

- [ ] **Step 3: Implement `research/baseline/cartpole_data_audit.py`**

Implement comprehensive mathematical check functions:
1. `audit_tensor_structure(data, expected_count, expected_T)`: Validates shapes `(1, 4)`, `(250, 1)`, `(250, 1, 4)`, dtypes, NaN/Inf check across all trajectories.
2. `audit_actuator_bounds(data, u_min=-10.0, u_max=10.0)`: Checks $\min(u) \ge u_{\min}$, $\max(u) \le u_{\max}$, mean $\approx 0$, empirical variance $\approx (20)^2 / 12$.
3. `audit_initial_state_bounds(data)`: Checks $x_0 \in [-4.0, 4.0]$, $\theta_0 \in [-0.1, 0.1]$, $\dot{x}_0 = 0$, $\dot{\theta}_0 = 0$.
4. `audit_dynamics_mutation()`: Instantiates `CartPoleDynamics`, creates a tensor with `requires_grad=True`, computes `dynamics(0, x, u)`, verifies input $x$ is unmodified and backward passes cleanly.
5. `audit_transition_consistency(trajectories, tol=1e-4)`: Re-integrates $(x_t, u_t) \to x_{t+1}$ using ground-truth RK4 with $dt=0.02$, measures maximum error $\|x_{t+1} - x_{t+1}^{\text{recorded}}\|_{\infty}$.
6. `audit_train_test_separation(train_data, test_data)`: Verifies zero trajectory hash collision and positive minimum distance between train and test initial conditions.
7. `audit_cartpole_dataset(train_path, test_path, tol=1e-4)`: Aggregates all checks, computes SHA-256 for both `.pt` files, outputs a structured YAML report and prints human-readable summary.
8. CLI entrypoint: `python -m research.baseline.cartpole_data_audit --output outputs/reproduction/cartpole_data_audit.yaml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_cartpole_data_audit.py -v`
Expected: PASS with 6 passing tests.

- [ ] **Step 5: Run full audit on disk and save report**

Run: `PYTHONPATH=. .venv/bin/python -m research.baseline.cartpole_data_audit --output outputs/reproduction/cartpole_data_audit.yaml`
Expected: Audit output showing `ALL MATHEMATICAL INTEGRITY CHECKS PASSED`, train SHA-256 `f016889ea0...`, test SHA-256 `9b8f99b244...`.

---

### Task 2: Locked Experiment Configurations in `.yml` for CartPole

**Files:**
- Create: `configs/research/cartpole_controllable_win15_frac0.01_seed42.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.01_seed43.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.01_seed44.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.05_seed42.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.05_seed43.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.05_seed44.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.1_seed42.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.1_seed43.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.1_seed44.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.3_seed42.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.3_seed43.yml`
- Create: `configs/research/cartpole_controllable_win15_frac0.3_seed44.yml`
- Test: `tests/test_cartpole_config_contract.py`

**Interfaces:**
- Consumes: SHA-256 hashes from Task 1, Paper Table 5 hyperparameters.
- Produces: 12 `.yml` locked config files verified by contract test.

- [ ] **Step 1: Write the failing unit tests for CartPole configs**

```python
# tests/test_cartpole_config_contract.py
import unittest
from pathlib import Path
from research.baseline.common import (
    ROOT,
    build_module,
    check_effective_hyperparameters,
    dataset_fingerprint,
    load_config,
)

CONFIGS = [
    f"configs/research/cartpole_controllable_win15_frac{frac}_seed{seed}.yml"
    for frac in ["0.01", "0.05", "0.1", "0.3"]
    for seed in [42, 43, 44]
]

class TestCartPoleConfigContract(unittest.TestCase):
    def test_all_configs_exist(self):
        for rel in CONFIGS:
            path = ROOT / rel
            self.assertTrue(path.exists(), f"Missing config: {rel}")

    def test_config_contract_and_effective_hyperparameters(self):
        for rel in CONFIGS:
            cfg = load_config(ROOT / rel)
            self.assertEqual(cfg["system"], "cartpole")
            self.assertEqual(cfg["latent_dim"], 25)
            self.assertEqual(cfg["num_layers"], 3)
            self.assertEqual(cfg["hidden_size"], 64)
            self.assertEqual(cfg["window_len"], 15)
            self.assertEqual(cfg["batch_size"], 64)
            self.assertEqual(cfg["lr"], 0.001)
            self.assertEqual(cfg["epochs"], 100)

            # Verify dataset fingerprint
            fingerprint = dataset_fingerprint(cfg)
            self.assertIn("train", fingerprint)
            self.assertIn("test", fingerprint)

            # Build module and check no drift
            module = build_module(cfg)
            effective = check_effective_hyperparameters(module, cfg)
            self.assertEqual(effective["latent_dim"], 25)
            self.assertEqual(effective["state_dim"], 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_cartpole_config_contract.py -v`
Expected: FAIL with `AssertionError: Missing config: configs/research/cartpole_controllable_win15_frac0.01_seed42.yml`

- [ ] **Step 3: Generate the 12 locked `.yml` configuration files**

Each configuration file follows the exact YAML contract of the baseline runner:
- `run_id`: `C01-s{seed}` for 1%, `C02-s{seed}` for 5%, `C03-s{seed}` for 10%, `C04-s{seed}` for 30%.
- `system: cartpole`
- `seed: {seed}`
- `model_class: KoopmanLitModule`
- `koopman_mode: controllable`
- `control_loss_flag: true`
- `encoder_loss_flag: true`
- `use_modulated_u: false`
- `latent_dim: 25`
- `num_layers: 3`
- `hidden_size: 64`
- `u_encoder_layers: 2`
- `u_encoder_hidden: 64`
- `window_len: 15`
- `max_windows_per_traj: 20`
- `window_start_max: 235`
- `train_fraction: {frac}`
- `num_train_trajectories: {100 | 500 | 1000 | 3000}`
- `batch_size: 64`
- `epochs: 100`
- `deterministic: false`
- `optimizer: AdamW`
- `lr: 0.001`
- `weight_decay: 1.0e-05`
- `scheduler: ReduceLROnPlateau`
- `scheduler_factor: 0.1`
- `scheduler_patience: 10`
- `prediction_gamma: 0.8`
- `lambda_encoder_pred: 0.5`
- `lambda_control: 0.005`
- `gramian_horizon_steps: 15`
- `data_train: data/cartpole_train.pt`
- `data_test: data/cartpole_test.pt`
- `data_train_sha256: f016889ea0090a91c31c545ec37ca25fcaee8d6658d2364bebabb8e0ecea449b`
- `data_test_sha256: 9b8f99b2445f8c6e737f683d544e3efe357a9f21b75e006948541cfdbd5ef35e`
- `checkpoint: checkpoints/{run_id}.pt`

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_cartpole_config_contract.py -v`
Expected: PASS for all 12 configs without hyperparameter drift.

---

### Task 3: Dry-Run Runner Contract & Operator Dimension Verification

**Files:**
- Test: `tests/test_cartpole_runner_contract.py`

**Interfaces:**
- Consumes: `research.baseline.train_koopman`, locked `.yml` configs from Task 2.
- Produces: Verified end-to-end dry-run execution and operator parameter checks.

- [ ] **Step 1: Write unit test for CartPole runner dry-run and operator structure**

```python
# tests/test_cartpole_runner_contract.py
import unittest
import subprocess
from pathlib import Path
from research.baseline.common import ROOT, build_module, load_config

class TestCartPoleRunnerContract(unittest.TestCase):
    def test_operator_matrix_dimensions(self):
        cfg = load_config(ROOT / "configs/research/cartpole_controllable_win15_frac0.05_seed42.yml")
        module = build_module(cfg)
        op = module.model.operator
        # SingleInputControllableKoopmanOperator structure for n=4, N=25
        self.assertEqual(op.A_free.shape, (22, 25))
        self.assertEqual(op.B_free.shape, (21, 1))
        self.assertEqual(op.H_head.shape, (4, 4))
        self.assertEqual(op.H_tail.shape, (21, 21))

        # Check positive definiteness of similarity transform P
        P = op.build_P()
        self.assertEqual(P.shape, (25, 25))
        eigs = torch.linalg.eigvalsh(P)
        self.assertTrue((eigs > 0).all(), "P must be strictly positive definite")

    def test_train_koopman_dry_run_execution(self):
        cmd = [
            ".venv/bin/python", "-m", "research.baseline.train_koopman",
            "--config", "configs/research/cartpole_controllable_win15_frac0.05_seed42.yml",
            "--dry-run"
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Dry-run failed:\n{res.stderr}\n{res.stdout}")
        self.assertIn("dry run: config, dataset and model contract are valid", res.stdout)
```

- [ ] **Step 2: Run test to verify operator dimensions and dry-run execution**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_cartpole_runner_contract.py -v`
Expected: PASS

- [ ] **Step 3: Run comprehensive test discovery**

Run: `PYTHONPATH=. .venv/bin/python -m unittest discover tests -v`
Expected: All existing tests + new CartPole audit & config tests PASS.

---

## Self-Review

1. **Spec coverage:**
   - Covers [ROADMAP.md](file:///home/minh/Desktop/Controllable-Koopman/ROADMAP.md) checklist items for P1: freeze settings, validate CartPole dataset (shape, finiteness, actuator limits, mutation, train/test separation, hashes), and generate locked configs for multi-seed reproduction.
2. **Step scan:**
   - Every step specifies exact file paths, explicit test code, exact commands, and expected pass/fail outputs.
3. **Type consistency:**
   - Naming conventions (`cartpole_controllable_win15_frac{frac}_seed{seed}.yml`, `run_id: C01-s42`) and parameter signatures are consistent across all tasks.
4. **Review Focus addressed:**
   - Actuator bounds: Task 1 Step 1 & 3 test $u \in [-10, 10]$.
   - In-place mutation: Task 1 Step 1 & 3 test backward pass through dynamics.
   - Train/test leakage: Task 1 Step 1 & 3 test zero collision and disjoint $x_0$.
   - Observable dimension $N=25$: Task 2 Step 1 & Task 3 Step 1 verify dimensions $(22, 25)$, $(21, 1)$, $(4, 4)$, $(21, 21)$.
   - Effective hyperparameter drift: Task 2 Step 1 tests `check_effective_hyperparameters`.
5. **Proportion:**
   - Plan is concise, focused strictly on Steps 1 and 2, providing bite-sized actionable steps without redundant bloat.
