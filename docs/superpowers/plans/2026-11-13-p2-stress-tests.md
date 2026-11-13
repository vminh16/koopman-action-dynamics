# P2 Controlled Stress Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase P2 diagnostic stress-testing pipeline (Rotation Family, Unicycle, Planar Quadrotor, mathematical diagnostics engine, and convex QP MPC) to isolate failure modes of the official Koopman baseline without training any H2/H3 methods.

**Architecture:** A modular overlay under `research/` leaving upstream core untouched: `research/benchmarks/` houses synthetic and physical continuous-time control-affine benchmark dynamics with analytic drift $f_0(x)$ and input matrix $G_x(x)$; `research/diagnostics/` implements physical and lifted error metrics ($E_{\text{response}}, E_{\text{lift}}, E_{\text{drift}}, d_{\text{orient}}, \rho_T$); `research/control/` implements a verified convex QP MPC controller using `cvxpy` + `osqp`; and `research/baseline/` provides a clean runner for training the official `KoopmanLitModule` on stress benchmarks and generating diagnosis reports in `outputs/diagnosis/`.

**Tech Stack:** Python 3.10, PyTorch, PyTorch Lightning, NumPy, SciPy (`solve_continuous_lyapunov`, `expm`, `solve_ivp`), CVXPY + OSQP, Matplotlib, PyYAML.

**Spec:** [ROADMAP.md](file:///home/minh/Desktop/Controllable-Koopman/ROADMAP.md) (Phase P2, lines 175-188) and [EXPERIMENT.md](file:///home/minh/Desktop/Controllable-Koopman/EXPERIMENT.md) (§4 Metrics & Diagnostics, §5 Registry E05-E09).

## Global Constraints

- **No H2/H3 model training:** Through Phase P2, only the author's official `KoopmanLitModule` is trained and evaluated; do not train any factorized (H2) or certified input-map (H3) models.
- **Upstream immutability:** Do not edit upstream modules (`models/`, `training/`, `dynamics/`) except for documented standalone bugfixes. All new code belongs under `research/`, `scripts/`, or `tests/`.
- **True Dynamics Oracle boundary:** Simulator oracles for ground truth vector fields $f(x, u)$ and analytic $G_x(x)$ are used **strictly for evaluation/diagnosis**, never as training supervision.
- **Float64 precision for control-theoretic metrics:** Gramian computations, condition numbers, and $\rho_T$ margins must be computed in float64 using SciPy or quadrature, bypassing the upstream Euler `15*dt` overwrite.
- **Code reuse over reinventing:** Benchmark dynamics and QP MPC formulations are adapted from standard verified templates (SciPy, CVXPY standard QP MPC) rather than built from scratch.

## Review Focus

1. **State-dependent input rotation:** When rotation strength $a=0$, $G_x(x)$ is constant and Koopman $E_{\text{response}}$ should be near zero; when $a > 0$, error must increase monotonically with $a$.
2. **Jacobian computation on non-differentiable points:** The encoder $\phi(x)$ uses ReLU activations; autograd Jacobians $J_\phi(x)$ must handle or safely evaluate batch Jacobian without memory blowup.
3. **Actuator constraint satisfaction in QP MPC:** The CVXPY formulation must strictly enforce $u_{\min} \le u_t \le u_{\max}$ under discrete-time LTI dynamics $z_{t+1} = A_d z_t + B_d u_t$.
4. **Coordinate conventions for Unicycle and Quadrotor:** Unicycle controls must be $(v, \omega)$ with state $(x, y, \theta)$; Quadrotor must use world-frame vertical and horizontal accelerations driven by tilted body thrust.
5. **Deterministic reproducibility:** All benchmark data generation and trajectory evaluations must use explicit random seeds and hashes.

---

### Task 1: Benchmark Dynamics & Oracles (`research/benchmarks/`)

**Files:**
- Create: `research/benchmarks/__init__.py`
- Create: `research/benchmarks/rotation_family.py`
- Create: `research/benchmarks/unicycle.py`
- Create: `research/benchmarks/planar_quadrotor.py`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `torch`, `numpy`, `scipy`
- Produces: Classes `RotationFamilyDynamics`, `UnicycleDynamics`, `PlanarQuadrotorDynamics` conforming to:
  - `dynamics(x, u) -> dx_dt` (PyTorch tensor and NumPy callable)
  - `analytic_f0(x) -> f0` (Drift vector field)
  - `analytic_Gx(x) -> Gx` (Control-affine matrix, shape `[batch, state_dim, action_dim]`)
  - `sample_initial_state(num_samples) -> x0`
  - Attributes: `Nstates`, `Ncontrols`, `umin`, `umax`, `dt`

- [x] **Step 1: Write failing tests for benchmark dynamics**

```python
# tests/test_benchmarks.py
import unittest
import torch
import numpy as np
from research.benchmarks.rotation_family import RotationFamilyDynamics
from research.benchmarks.unicycle import UnicycleDynamics
from research.benchmarks.planar_quadrotor import PlanarQuadrotorDynamics

class TestBenchmarks(unittest.TestCase):
    def test_rotation_family_properties(self):
        sys_a0 = RotationFamilyDynamics(a=0.0)
        x = torch.tensor([[1.0, 0.5]])
        u = torch.tensor([[0.5]])
        Gx_a0 = sys_a0.analytic_Gx(x)
        self.assertEqual(Gx_a0.shape, (1, 2, 1))
        self.assertTrue(torch.allclose(Gx_a0[0, :, 0], torch.tensor([1.0, 0.0])))
        
        # Test a != 0 rotates input
        sys_a1 = RotationFamilyDynamics(a=1.0)
        Gx_a1 = sys_a1.analytic_Gx(x)
        self.assertFalse(torch.allclose(Gx_a1, Gx_a0))
        
        # Verify control-affine identity: f(x, u) == f0(x) + Gx(x) @ u
        f_total = sys_a1.dynamics(x, u)
        f_affine = sys_a1.analytic_f0(x) + torch.bmm(Gx_a1, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_total, f_affine, atol=1e-5))

    def test_unicycle_dynamics(self):
        sys = UnicycleDynamics()
        self.assertEqual(sys.Nstates, 3)
        self.assertEqual(sys.Ncontrols, 2)
        x = torch.tensor([[0.0, 0.0, 0.0]]) # heading 0
        u = torch.tensor([[1.0, 0.5]])       # v=1, omega=0.5
        dx = sys.dynamics(x, u)
        self.assertTrue(torch.allclose(dx, torch.tensor([[1.0, 0.0, 0.5]]), atol=1e-5))

    def test_planar_quadrotor_dynamics(self):
        sys = PlanarQuadrotorDynamics()
        self.assertEqual(sys.Nstates, 6)
        self.assertEqual(sys.Ncontrols, 2)
        # hover equilibrium test: phi=0, thrust = m*g / 2 each
        x0 = torch.zeros((1, 6))
        u_hover = torch.tensor([[sys.m * sys.g / 2.0, sys.m * sys.g / 2.0]])
        dx = sys.dynamics(x0, u_hover)
        self.assertTrue(torch.allclose(dx, torch.zeros((1, 6)), atol=1e-4))

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_benchmarks.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.benchmarks'

- [x] **Step 3: Implement `RotationFamilyDynamics`, `UnicycleDynamics`, and `PlanarQuadrotorDynamics`**

Copy & customize standard kinematics and equations of motion:
- `research/benchmarks/rotation_family.py`:
  $$\dot{x} = A_0 x + R(a \cdot x_1) B_0 u, \quad B_0 = [1, 0]^T, \quad R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
- `research/benchmarks/unicycle.py`:
  $$\dot{x} = v \cos\theta, \quad \dot{y} = v \sin\theta, \quad \dot{\theta} = \omega$$
- `research/benchmarks/planar_quadrotor.py`:
  State $(y, z, \phi, \dot{y}, \dot{z}, \dot{\phi})$, controls $(u_1, u_2)$, adapted from PVTOL dynamics:
  $$\ddot{y} = -\frac{u_1+u_2}{m} \sin\phi, \quad \ddot{z} = \frac{u_1+u_2}{m}\cos\phi - g, \quad \ddot{\phi} = \frac{L(u_2-u_1)}{I_{yy}}$$

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_benchmarks.py`
Expected: PASS (ran 3 tests)

- [x] **Step 5: Commit**

```bash
git add research/benchmarks/ tests/test_benchmarks.py
git commit -m "feat(p2): implement rotation family, unicycle, and planar quadrotor benchmark dynamics"
```

---

### Task 2: Mathematical Diagnostics Engine (`research/diagnostics/`)

**Files:**
- Create: `research/diagnostics/__init__.py`
- Create: `research/diagnostics/metrics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: PyTorch model, benchmark dynamics oracle, `scipy.linalg`
- Produces:
  - `compute_action_response_error(model, sys, x_samples, u_samples, u_ref) -> float`
  - `compute_lifted_and_physical_error(model, sys, x_samples) -> dict` (`E_phys`, `E_lift`, `E_drift`)
  - `compute_orientation_mismatch(model, sys, x_samples) -> float` ($d_{\text{orient}}$)
  - `compute_koopman_margin_rho(A, B, C, T, dt=0.02) -> dict` ($\rho_T$, $\lambda_{\min}(W_y)$, $\lambda_{\max}(Q_y)$ in float64)

- [x] **Step 1: Write failing tests for diagnostic metrics**

```python
# tests/test_diagnostics.py
import unittest
import torch
import numpy as np
from research.diagnostics.metrics import (
    compute_action_response_error,
    compute_lifted_and_physical_error,
    compute_koopman_margin_rho,
    compute_orientation_mismatch
)

class TestDiagnostics(unittest.TestCase):
    def test_koopman_margin_analytic(self):
        A = np.array([[0.0, 1.0], [-1.0, -0.5]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        res = compute_koopman_margin_rho(A, B, C, T=2.0)
        self.assertIn("rho_T", res)
        self.assertIn("lambda_min_W", res)
        self.assertGreater(res["rho_T"], 0.0)
        self.assertGreater(res["lambda_min_W"], 0.0)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_diagnostics.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.diagnostics'

- [x] **Step 3: Implement diagnostic metrics in `research/diagnostics/metrics.py`**

Formulas from EXPERIMENT.md §4:
- $E_{\text{response}} = \frac{1}{K} \sum_i \| C (f_z(\phi(x_i), u_i) - f_z(\phi(x_i), u_{0i})) - (f(x_i, u_i) - f(x_i, u_{0i})) \|_2^2$
- $J_\phi(x)$ via `torch.autograd.functional.jacobian` or batched vmap.
- $E_{\text{phys}} = \| C B - G_x(x) \|_F$, $E_{\text{lift}} = \| B - J_\phi(x) G_x(x) \|_F$, $E_{\text{drift}} = \| A \phi(x) - J_\phi(x) f_0(x) \|_2$.
- $d_{\text{orient}}(x) = \| (I - \Pi_{J_\phi(x) G_x(x)}) B \|_2$.
- Accurate continuous-time Gramian quadrature in float64:
  $$W_y(T) = \int_0^T C e^{As} B B^T e^{A^T s} C^T ds, \quad Q_y(T) = \int_0^T C e^{As} e^{A^T s} C^T ds$$

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_diagnostics.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add research/diagnostics/ tests/test_diagnostics.py
git commit -m "feat(p2): implement mathematical diagnostics engine (response, lifted, drift, rho_T)"
```

---

### Task 3: Baseline Convex QP MPC Engine (`research/control/`)

**Files:**
- Create: `research/control/__init__.py`
- Create: `research/control/qp_mpc.py`
- Test: `tests/test_qp_mpc.py`

**Interfaces:**
- Consumes: $A, B, C$ continuous or discrete matrices, `cvxpy`, `osqp`
- Produces:
  - Class `KoopmanConvexMPC`:
    - `__init__(A, B, C, dt, horizon, Q, R, umin, umax, Qf=None)`
    - `solve(x0, x_ref_seq) -> u_opt, status, info`
    - `simulate_closed_loop(sys, x0, x_target, num_steps) -> traj_dict`

- [x] **Step 1: Write failing tests for QP MPC**

```python
# tests/test_qp_mpc.py
import unittest
import numpy as np
from research.control.qp_mpc import KoopmanConvexMPC

class TestQPMPC(unittest.TestCase):
    def test_qp_mpc_double_integrator_regulation(self):
        # Double integrator: x_dot = v, v_dot = u
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        C = np.eye(2)
        dt = 0.05
        mpc = KoopmanConvexMPC(
            A=A, B=B, C=C, dt=dt, horizon=20,
            Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
            umin=np.array([-1.0]), umax=np.array([1.0])
        )
        x0 = np.array([2.0, 0.0]) # displaced position
        u0, status, _ = mpc.solve(x0, x_ref=np.zeros(2))
        self.assertIn(status, ["optimal", "optimal_inaccurate"])
        self.assertLess(u0[0], 0.0) # negative acceleration
        self.assertTrue(-1.0 <= u0[0] <= 1.0)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_qp_mpc.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.control'

- [x] **Step 3: Implement `KoopmanConvexMPC` in `research/control/qp_mpc.py`**

Adapt standard CVXPY MPC template:
- Discretize continuous $(A, B)$ using matrix exponential: $A_d = e^{A \cdot dt}, B_d = \int_0^{dt} e^{As} ds \cdot B$ via `scipy.linalg.expm`.
- Setup QP parameters (`z_init`, `x_ref`) and variables ($Z \in \mathbb{R}^{d_z \times (H+1)}, U \in \mathbb{R}^{d_u \times H}$).
- Objective: $\sum_{k=0}^{H-1} (C z_k - x_{\text{ref}, k})^T Q (C z_k - x_{\text{ref}, k}) + u_k^T R u_k + (C z_H - x_{\text{ref}, H})^T Q_f (C z_H - x_{\text{ref}, H})$.
- Constraints: $z_0 = \phi(x_0)$, $z_{k+1} = A_d z_k + B_d u_k$, $u_{\min} \le u_k \le u_{\max}$.
- Solver: `solver=cp.OSQP, warm_start=True`.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_qp_mpc.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add research/control/ tests/test_qp_mpc.py
git commit -m "feat(p2): implement baseline convex QP MPC engine with cvxpy and osqp"
```

---

### Task 4: Research Baseline Runner & Dataset Generator (`research/baseline/`)

**Files:**
- Create: `research/baseline/data_generator.py`
- Modify: `research/baseline/common.py` (integrate benchmark dynamics)
- Reused: `research/baseline/train_koopman.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Benchmark dynamics (`research.benchmarks`), `training.train.KoopmanLitModule`, `my_utils.dataset`
- Produces:
  - `generate_benchmark_dataset(sys_name, num_trajectories, T, dt, seed, save_dir, kwargs) -> filepath, hash`
  - Integration with `train_koopman.py` supporting benchmark systems `rotation_family`, `unicycle`, and `planar_quadrotor`.

- [x] **Step 1: Write failing test for dataset generator and runner**

```python
# tests/test_runner.py
import unittest
import os
import torch
from research.baseline.data_generator import generate_benchmark_dataset

class TestRunner(unittest.TestCase):
    def test_generate_smoke_dataset(self):
        tmp_dir = "tests/tmp_test_data"
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            out_file, data_hash = generate_benchmark_dataset(
                sys_name="rotation_family",
                num_trajectories=5,
                T=20,
                dt=0.02,
                seed=123,
                save_dir=tmp_dir,
                kwargs={"a": 0.5}
            )
            self.assertTrue(os.path.exists(out_file))
            data = torch.load(out_file)
            self.assertEqual(len(data), 5)
            self.assertEqual(len(data_hash), 64) # sha256
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_runner.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.baseline'

- [x] **Step 3: Implement data generator and thin Koopman runner**

- `research/baseline/data_generator.py`: Generates trajectories using RK4 numerical integration of `dynamics(x, u)`, calculates SHA-256 hash, saves to `.pt`.
- `research/baseline/runner.py`: Wraps `KoopmanLitModule` training cleanly, logs hyperparameters, saves checkpoint without `_MLPmodel` typo, records manifest YAML.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_runner.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add research/baseline/ tests/test_runner.py
git commit -m "feat(p2): implement research data generator and thin baseline Koopman runner"
```

---

### Task 5: Controlled Stress-Test Suite & Diagnosis Report (`research/diagnostics/run_p2_diagnostics.py`)

**Files:**
- Create: `research/diagnostics/run_p2_diagnostics.py`
- Test: `tests/test_p2_integration.py`
- Outputs: `outputs/diagnosis/p2_report.yaml`, `outputs/diagnosis/rotation_sweep.png`

**Interfaces:**
- Consumes: Tasks 1-4
- Produces: End-to-end diagnosis script running:
  - E05: Baseline diagnostics on existing Pendulum checkpoint.
  - E06: Sweep of rotation parameter $a \in [0.0, 0.5, 1.0, 2.0]$ in Rotation Family.
  - E07: Diagnostics on Unicycle and Planar Quadrotor.
  - E08: Verification of convex QP MPC tracking on learned models.
  - Generates comprehensive tables and charts in `outputs/diagnosis/`.

- [x] **Step 1: Write integration test for diagnosis suite**

```python
# tests/test_p2_integration.py
import unittest
import os
import yaml
from research.diagnostics.run_p2_diagnostics import run_smoke_diagnosis

class TestP2Integration(unittest.TestCase):
    def test_smoke_diagnosis(self):
        tmp_dir = "tests/tmp_diagnosis"
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            report_path = run_smoke_diagnosis(output_dir=tmp_dir)
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r") as f:
                report = yaml.safe_load(f)
            self.assertIn("pendulum_diagnostics", report)
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_p2_integration.py`
Expected: FAIL with ModuleNotFoundError

- [x] **Step 3: Implement `run_p2_diagnostics.py`**

Orchestrates:
1. Load existing Pendulum checkpoint, compute $E_{\text{response}}$, $E_{\text{lift}}$, $\rho_T$, $d_{\text{orient}}$.
2. Train/evaluate official Koopman on rotation sweep $a \in \{0.0, 0.5, 1.0, 2.0\}$.
3. Plot metric curves ($a$ vs $E_{\text{response}}$ vs Prediction MSE).
4. Save YAML summary and diagnostic plots to `outputs/diagnosis/`.

- [x] **Step 4: Run integration test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_p2_integration.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add research/diagnostics/run_p2_diagnostics.py tests/test_p2_integration.py
git commit -m "feat(p2): implement end-to-end P2 stress test suite and diagnosis report generator"
```
