# P2 Baseline Diagnosis, QP MPC, and Stress-Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining 4 components of Phase P2: generate stress-test datasets and train the official Koopman baseline, implement the mathematical diagnostics engine ($E_{\text{response}}, E_{\text{lift}}, E_{\text{drift}}, d_{\text{orient}}, \rho_T$), implement and test a verified convex QP MPC controller in `research/control/`, and execute the controlled stress-strength sweep with separation of failure modes into `outputs/diagnosis/`.

**Architecture:** 
1. `research/baseline/data_generator.py`: Generates deterministic simulation datasets for the new benchmarks (`rotation_family` sweep, `unicycle`, `planar_quadrotor`) with SHA-256 fingerprints, action statistics, and state coverage logs.
2. `research/diagnostics/metrics.py`: Evaluates physical and lifted errors using autograd Jacobians $J_\phi(x)$, same-state action-response oracle comparisons, candidate-action regret, and float64 continuous Gramian quadrature for $\rho_T$.
3. `research/control/qp_mpc.py`: Formulates discrete-time LTI convex QP MPC via `cvxpy` + `osqp`, discretizing continuous $(A, B)$ using matrix exponential (`scipy.linalg.expm`), enforcing actuator bounds $u_{\min} \le u_t \le u_{\max}$, verified on an LTI toy before evaluating learned models.
4. `research/diagnostics/run_p2_diagnostics.py`: Orchestrates baseline training on the stress benchmarks, executes rotation-strength sweeps, runs closed-loop MPC tracking, and plots diagnostic curves while categorizing errors into model-class limitation, data coverage, and numerical issues.

**Tech Stack:** Python 3.10, PyTorch, PyTorch Lightning, NumPy, SciPy (`solve_continuous_lyapunov`, `expm`, `quad`), CVXPY, OSQP, Matplotlib, PyYAML.

**Spec:** [ROADMAP.md](file:///home/minh/Desktop/Controllable-Koopman/ROADMAP.md) (Phase P2, lines 182-188) and [EXPERIMENT.md](file:///home/minh/Desktop/Controllable-Koopman/EXPERIMENT.md) (§4 Metrics 4.1–4.6, §5 E05–E09).

## Global Constraints

- **Only official Koopman:** In Phase P2, only train the author's official `KoopmanLitModule` (`A_free/B_free` or controllable canonical form). Do NOT train any factorized (H2) or certified state-dependent (H3) models.
- **Oracle boundary:** Ground truth simulator vector fields and Jacobians are strictly used for evaluation and diagnosis, never passed to the loss during training.
- **Float64 precision for Gramians:** Compute continuous-time reachability Gramian $W_y(T)$ and controllability index $\rho_T$ in float64 using SciPy quadrature, eliminating the upstream Euler `15*dt` horizon bug.
- **Exact Convex QP:** The baseline MPC must be an exact linear convex QP on the lifted space $z$, strictly respecting actuator saturation bounds without ad-hoc clipping outside the solver.
- **Failure separation:** Reports must distinguish between: (a) structural model-class failure (e.g. state-dependent $G_x(x)$ cannot be represented by fixed $B$); (b) dataset distribution coverage; and (c) numerical conditioning of the solver.

## Review Focus

1. **Monotonic error growth with rotation strength $a$:** As $a$ increases from $0.0$ to $2.0$ in `rotation_family`, prediction MSE may remain deceptively moderate, but action-response error $E_{\text{response}}$ and lifted error $\|B - J_\phi(x) G_x(x)\|$ must grow significantly.
2. **Actuator constraint satisfaction:** In QP MPC, optimal commands $u_t$ returned by the solver must strictly satisfy $u_{\min} \le u_t \le u_{\max}$ across all steps.
3. **Autograd Jacobian memory efficiency:** When computing $J_\phi(x) = \frac{\partial \phi(x)}{\partial x}$ across a batch of sample states, avoid full batch outer-product explosions by using vectorized Jacobian routines (`torch.vmap` or list comprehension over state dimension).
4. **Coordinate frame alignment in MPC:** Closed-loop plant simulation must correctly feed physical state $x_t$ into encoder $\phi(x_t) = z_t$ at every control step (receding horizon).
5. **Deterministic reproducibility:** All datasets, training seeds, and diagnostic evaluations must record hashes, git commit, and explicit random seeds.

---

### Task 1: Dataset Generation & Baseline Training Pipeline (`research/baseline/`)

**Files:**
- Create: `research/baseline/data_generator.py`
- Modify: `research/baseline/common.py` (add benchmark support)
- Test: `tests/test_data_generator.py`

**Interfaces:**
- Consumes: `research.benchmarks.load_benchmark`, `my_utils.simulate.simulate_trajectory`
- Produces:
  - `generate_benchmark_dataset(sys_name, num_trajectories, T, dt, seed, save_dir, sys_kwargs=None) -> (path, sha256_hash, metadata)`
  - Metadata includes: action distribution (mean, std, min, max), state bounds, coverage volume estimate.
  - Generates reproducible `.pt` files matching upstream format `[{"x0": ..., "u_seq": ..., "x_seq": ...}, ...]`.

- [x] **Step 1: Write failing test for dataset generator**

```python
# tests/test_data_generator.py
import unittest
import os
import shutil
import torch
from research.baseline.data_generator import generate_benchmark_dataset

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "tests/tmp_data_gen"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_generate_rotation_dataset(self):
        path, data_hash, meta = generate_benchmark_dataset(
            sys_name="rotation_family",
            num_trajectories=5,
            T=30,
            dt=0.02,
            seed=42,
            save_dir=self.tmp_dir,
            sys_kwargs={"a": 0.5}
        )
        self.assertTrue(os.path.exists(path))
        self.assertEqual(len(data_hash), 64)
        data = torch.load(path)
        self.assertEqual(len(data), 5)
        self.assertEqual(data[0]["x_seq"].shape, (30, 2))
        self.assertEqual(data[0]["u_seq"].shape, (30, 1))
        self.assertIn("action_mean", meta)
        self.assertIn("state_bounds", meta)

    def test_reproducible_hash(self):
        _, hash1, _ = generate_benchmark_dataset("unicycle", 3, 20, 0.02, 123, self.tmp_dir)
        _, hash2, _ = generate_benchmark_dataset("unicycle", 3, 20, 0.02, 123, self.tmp_dir)
        self.assertEqual(hash1, hash2)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_data_generator.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.baseline.data_generator'

- [x] **Step 3: Implement `generate_benchmark_dataset` in `research/baseline/data_generator.py`**

- Use `load_benchmark(sys_name, **sys_kwargs)` to load dynamic system.
- Sample initial states $x_0 \sim \text{sample\_initial\_state}$.
- Sample random control actions uniformly from $[u_{\min}, u_{\max}]$.
- Integrate trajectories using RK4 or ODE integrator with step $dt$.
- Compute dataset statistics: state min/max, action mean/variance.
- Save tensor list to `.pt`, compute SHA-256 hash, and return metadata dict.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_data_generator.py`
Expected: PASS (ran 2 tests)

---

### Task 2: Mathematical Diagnostics Engine (`research/diagnostics/metrics.py`)

**Files:**
- Create: `research/diagnostics/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: PyTorch model (`KoopmanLitModule`), Benchmark Dynamics (`research.benchmarks`)
- Produces:
  - `compute_action_response_error(model, sys, x_samples, u_samples, u_ref) -> float` ($E_{\text{response}}$)
  - `compute_vector_field_mismatches(model, sys, x_samples) -> dict` ($E_{\text{phys}}, E_{\text{lift}}, E_{\text{drift}}, d_{\text{orient}}$)
  - `compute_continuous_gramian_and_margin(A, B, C, T, num_points=500) -> dict` ($\rho_T, \lambda_{\min}(W_y), \lambda_{\max}(Q_y)$ in float64)
  - `compute_candidate_regret(model, sys, x_samples, candidate_actions, horizon=5) -> float` ($R_{\text{cand}}$)

- [x] **Step 1: Write failing test for diagnostics metrics**

```python
# tests/test_metrics.py
import unittest
import torch
import numpy as np
from research.benchmarks.rotation_family import RotationFamilyDynamics
from research.diagnostics.metrics import (
    compute_action_response_error,
    compute_vector_field_mismatches,
    compute_continuous_gramian_and_margin,
    compute_candidate_regret
)

class MockModel(torch.nn.Module):
    def __init__(self, state_dim=2, latent_dim=4, control_dim=1):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        # A and B matrices
        self.A = torch.nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.B = torch.nn.Parameter(torch.zeros(latent_dim, control_dim))
        self.B.data[0, 0] = 1.0 # acts along first physical state

    def encode(self, x):
        # Lift: [x1, x2, 0, 0]
        pad = torch.zeros(x.shape[0], self.latent_dim - self.state_dim)
        return torch.cat([x, pad], dim=-1)

    def decode(self, z):
        return z[:, :self.state_dim]

class TestMetrics(unittest.TestCase):
    def test_gramian_quadrature(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]], dtype=np.float64)
        B = np.array([[0.0], [1.0]], dtype=np.float64)
        C = np.array([[1.0, 0.0]], dtype=np.float64)
        res = compute_continuous_gramian_and_margin(A, B, C, T=1.0)
        self.assertIn("rho_T", res)
        self.assertGreater(res["rho_T"], 0.0)
        self.assertGreater(res["lambda_min_Wy"], 0.0)

    def test_response_error_calculation(self):
        model = MockModel()
        sys = RotationFamilyDynamics(a=0.0)
        x = torch.randn(10, 2)
        u = torch.ones(10, 1)
        u_ref = torch.zeros(10, 1)
        err = compute_action_response_error(model, sys, x, u, u_ref)
        self.assertIsInstance(err, float)
        self.assertGreaterEqual(err, 0.0)

    def test_vector_field_mismatches(self):
        model = MockModel()
        sys = RotationFamilyDynamics(a=1.0)
        x = torch.randn(5, 2)
        res = compute_vector_field_mismatches(model, sys, x)
        self.assertIn("E_phys", res)
        self.assertIn("E_lift", res)
        self.assertIn("E_drift", res)
        self.assertIn("d_orient", res)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_metrics.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.diagnostics.metrics'

- [x] **Step 3: Implement metrics in `research/diagnostics/metrics.py`**

Formulas from EXPERIMENT.md §4:
- $E_{\text{response}} = \frac{1}{K} \sum_i \| C(f_z(\phi(x_i), u_i) - f_z(\phi(x_i), u_{0i})) - (f(x_i, u_i) - f(x_i, u_{0i})) \|_2^2$
- Vectorized autograd Jacobian: compute $J_\phi(x) = \left[ \nabla \phi_1(x), \dots, \nabla \phi_N(x) \right]^T \in \mathbb{R}^{d_z \times d_x}$.
- Lifted error $E_{\text{lift}}(x) = \|B - J_\phi(x) G_x(x)\|_F$.
- Physical error $E_{\text{phys}}(x) = \|CB - G_x(x)\|_F$.
- Drift error $E_{\text{drift}}(x) = \|A\phi(x) - J_\phi(x) f_0(x)\|_2$.
- Orientation metric $d_{\text{orient}}(x) = \| (I - \Pi_{J_\phi(x) G_x(x)}) B \|_2$.
- Continuous-time Gramians computed in float64 using matrix exponential integration:
  $$W_y(T) = \int_0^T C e^{As} B B^T e^{A^T s} C^T ds, \quad Q_y(T) = \int_0^T C e^{As} e^{A^T s} C^T ds$$
  $$\rho_T = \frac{\sqrt{\lambda_{\min}(W_y(T))}}{\sqrt{\lambda_{\max}(Q_y(T))}}$$
- Candidate regret $R_{\text{cand}}$: evaluate candidate input sequences on model vs ground-truth plant.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_metrics.py`
Expected: PASS (ran 3 tests)

---

### Task 3: Baseline Convex QP MPC Engine (`research/control/qp_mpc.py`)

**Files:**
- Create: `research/control/qp_mpc.py`
- Test: `tests/test_qp_mpc.py`

**Interfaces:**
- Consumes: $A \in \mathbb{R}^{d_z \times d_z}, B \in \mathbb{R}^{d_z \times d_u}, C \in \mathbb{R}^{d_x \times d_z}$, encoder $\phi(x)$, `cvxpy`, `osqp`
- Produces: Class `KoopmanConvexMPC`:
  - `__init__(A, B, C, dt, horizon=15, Q=None, R=None, umin=None, umax=None, Qf=None, encoder=None)`
  - `solve(x_current, x_target) -> (u_opt, status, info)`
  - `step(x_current, x_target) -> u0` (returns first optimal control vector $u_0$)
  - `simulate_closed_loop(sys, x0, x_target, num_steps) -> dict` (history of states, controls, costs)

- [x] **Step 1: Write failing test for convex QP MPC**

```python
# tests/test_qp_mpc.py
import unittest
import numpy as np
from research.control.qp_mpc import KoopmanConvexMPC

class TestQPMPC(unittest.TestCase):
    def test_double_integrator_stabilization(self):
        # LTI toy system: x_dot = [x2, u]
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        dt = 0.05
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=dt, horizon=20,
            Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
            umin=np.array([-1.5]), umax=np.array([1.5])
        )
        x0 = np.array([2.0, 0.0])
        u_opt, status, _ = mpc.solve(x0, x_target=np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        # Control must brake/accelerate towards origin (u < 0)
        self.assertLess(u_opt[0], 0.0)
        # Actuator bounds strictly respected
        self.assertTrue(-1.5 <= u_opt[0] <= 1.5)

    def test_actuator_saturation_enforcement(self):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        # Narrow bounds [-0.2, 0.2]
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=0.05, horizon=10,
            umin=np.array([-0.2]), umax=np.array([0.2])
        )
        x_far = np.array([100.0, 50.0]) # huge error
        u_opt, status, _ = mpc.solve(x_far, x_target=np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        self.assertAlmostEqual(u_opt[0], -0.2, places=4)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_qp_mpc.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.control.qp_mpc'

- [x] **Step 3: Implement `KoopmanConvexMPC` in `research/control/qp_mpc.py`**

- Continuous to discrete conversion:
  $A_d = e^{A \Delta t}$, $B_d = \int_0^{\Delta t} e^{A s} ds \cdot B$ via `scipy.linalg.expm`.
- Formulate CVXPY variables: $Z \in \mathbb{R}^{d_z \times (H+1)}$, $U \in \mathbb{R}^{d_u \times H}$.
- Parameters for warm starting: `z_init = cp.Parameter(d_z)`, `x_ref = cp.Parameter(d_x)`.
- Cost:
  $$\min \sum_{k=0}^{H-1} \left( \|C z_k - x_{\text{ref}}\|_Q^2 + \|u_k\|_R^2 \right) + \|C z_H - x_{\text{ref}}\|_{Q_f}^2$$
- Constraints:
  $$z_0 = z_{\text{init}}, \quad z_{k+1} = A_d z_k + B_d u_k, \quad u_{\min} \le u_k \le u_{\max}$$
- Solver: `problem.solve(solver=cp.OSQP, warm_start=True)`.
- Method `simulate_closed_loop(sys, x0, x_target, num_steps)`: runs receding horizon control with true plant feedback.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_qp_mpc.py`
Expected: PASS (ran 2 tests)

---

### Task 4: Stress-Sweep Diagnosis Suite & Failure Separation (`research/diagnostics/run_p2_diagnostics.py`)

**Files:**
- Create: `research/diagnostics/run_p2_diagnostics.py`
- Test: `tests/test_p2_diagnostics.py`
- Outputs: `outputs/diagnosis/p2_failure_report.yaml`, `outputs/diagnosis/rotation_sweep.png`, `outputs/diagnosis/mpc_trajectories.png`

**Interfaces:**
- Consumes: Tasks 1, 2, 3 and `research.benchmarks`
- Produces:
  - Command-line entrypoint and programmatic function `run_p2_stress_diagnosis(output_dir, quick_mode=False)`
  - Generates comprehensive tables and plots separating:
    1. **Structural model-class limitation** ($E_{\text{response}}$ vs $a$).
    2. **State distribution coverage**.
    3. **Solver / numerical conditioning** ($\rho_T$, condition number of $W_y$).
  - Full report saved to `outputs/diagnosis/p2_failure_report.yaml`.

- [x] **Step 1: Write integration test for diagnostic suite**

```python
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
        self.assertIn("failure_mode_separation", data)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_p2_diagnostics.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.diagnostics.run_p2_diagnostics'

- [x] **Step 3: Implement `run_p2_stress_diagnosis` in `research/diagnostics/run_p2_diagnostics.py`**

Orchestration logic:
1. **Rotation Sweep Experiment (E06)**:
   - For $a \in [0.0, 0.5, 1.0, 2.0]$: generate datasets, train lightweight official `KoopmanLitModule` (15 epochs in quick mode, 50 epochs standard).
   - Evaluate one-step MSE, multi-step open loop MSE, and $E_{\text{response}}$.
   - Verify hypothesis: prediction MSE remains low while $E_{\text{response}}$ degrades sharply as $a$ grows.
2. **Benchmark Systems Evaluation (E07)**:
   - Evaluate official Koopman on `unicycle` and `planar_quadrotor`.
   - Measure $E_{\text{phys}}$, $E_{\text{lift}} = \|B - J_\phi G_x\|$, $d_{\text{orient}}$.
3. **Closed-Loop QP MPC Evaluation (E08)**:
   - Run closed-loop tracking of target waypoints using `KoopmanConvexMPC`.
   - Compare tracking success rate, closed-loop cost, and candidate regret.
4. **Failure-Mode Separation**:
   - Classify findings into:
     - *Model-class error*: $\|B - J_\phi G_x\|_F > 0$ and $E_{\text{response}} > \epsilon$.
     - *Coverage error*: distance from training trajectory manifold.
     - *Numerical error*: conditioning of Gramian $W_y(T)$ and solver convergence.
5. **Plotting & Logging**:
   - Save figures (`rotation_sweep.png`, `mpc_trajectories.png`) and YAML report.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests/test_p2_diagnostics.py`
Expected: PASS (ran 1 test)

- [x] **Step 5: Run full project test suite**

Run: `PYTHONPATH=. .venv/bin/python -m unittest discover tests`
Expected: PASS (all tests pass, 0 failures)

