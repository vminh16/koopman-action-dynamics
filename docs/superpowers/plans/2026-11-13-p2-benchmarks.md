# P2 Benchmark Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three core benchmark dynamical systems required for Phase P2 controlled stress tests: `rotation_family.py`, `unicycle.py`, and `planar_quadrotor.py`, each featuring analytic drift $f_0(x)$, analytic state-dependent input matrix $G_x(x)$, actuator bounds, initial state sampling, and strict compatibility with the repository's simulation pipeline.

**Architecture:** A modular benchmark package at `research/benchmarks/` containing standalone classes `RotationFamilyDynamics`, `UnicycleDynamics`, and `PlanarQuadrotorDynamics`. All classes provide:
1. `dynamics(self, t, x=None, u=None)` accepting both `(t, x, u)` and `(x, u)` call conventions.
2. `analytic_f0(self, x)` returning drift tensor of shape `[B, state_dim]`.
3. `analytic_Gx(self, x)` returning control-affine matrix tensor of shape `[B, state_dim, control_dim]`.
4. Control-affine exact identity: $f(x, u) = f_0(x) + G_x(x) u$.
5. Explicit attributes: `Nstates`, `Ncontrols`, `umin`, `umax`, `dt`.
6. Centralized registry in `research/benchmarks/__init__.py`.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `unittest`.

**Spec:** [ROADMAP.md](file:///home/minh/Desktop/Controllable-Koopman/ROADMAP.md) (Phase P2, lines 179-181) and [EXPERIMENT.md](file:///home/minh/Desktop/Controllable-Koopman/EXPERIMENT.md) (§4.2, §4.3, §5 E06-E07).

## Global Constraints

- **No kinematic model substitution:** Unicycle MUST be kinematic unicycle with state $(p_x, p_y, \theta)$ and controls $(v, \omega)$. Do not silently switch to bicycle or steering-angle vehicle models.
- **Orientation-dependent thrust:** Planar Quadrotor MUST be 6D full-state $(y, z, \phi, \dot{y}, \dot{z}, \dot{\phi})$ with controls $(u_1, u_2)$ where horizontal acceleration is explicitly coupled to $-\sin\phi \cdot (u_1 + u_2) / m$.
- **No in-place tensor mutation:** Angle normalization operations must return newly created tensors, never mutating inputs in-place (avoiding autograd bugs like upstream CartPole).
- **Control-affine exactness:** For every system, the continuous dynamics $\dot{x}$ must identically equal $f_0(x) + G_x(x) u$ for all $x, u$.
- **Upstream isolation:** All code resides in `research/benchmarks/` and `tests/`; do not modify upstream `dynamics/`.

## Review Focus

1. **Rotation family at $a=0$:** When rotation parameter $a = 0$, $G_x(x)$ must collapse to constant matrix $[1, 0]^T$ identically for all states $x$.
2. **Rotation family sweep consistency:** As $a$ increases from $0.0$ to $2.0$, the angle of the control direction must rotate monotonically as $a \cdot x_1$.
3. **Unicycle heading wrapping:** State $\theta$ must support transitions across $\pm \pi$ cleanly without breaking autograd gradients.
4. **Planar quadrotor hover equilibrium:** At $\phi = 0$ and $u_1 = u_2 = \frac{mg}{2}$, all accelerations $(\ddot{y}, \ddot{z}, \ddot{\phi})$ must evaluate to $0.0$ within machine precision.
5. **Batch dimension broadcasting:** All vector fields and matrices must broadcast over arbitrary batch dimensions $[B, \dots]$.

---

### Task 1: Rotation Family Dynamics (`research/benchmarks/rotation_family.py`)

**Files:**
- Create: `research/benchmarks/rotation_family.py`
- Test: `tests/test_rotation_family.py`

**Interfaces:**
- Consumes: `torch`, `numpy`
- Produces: Class `RotationFamilyDynamics` with:
  - `__init__(a=1.0, dt=0.02)`
  - `analytic_f0(x) -> Tensor[B, 2]`
  - `analytic_Gx(x) -> Tensor[B, 2, 1]`
  - `dynamics(t, x=None, u=None) -> Tensor[B, 2]`
  - `sample_initial_state(batch_size=1) -> Tensor[B, 2]`
  - Attributes: `Nstates = 2`, `Ncontrols = 1`, `umin = torch.tensor([-2.0])`, `umax = torch.tensor([2.0])`, `dt = 0.02`

- [x] **Step 1: Write failing tests for RotationFamilyDynamics**

```python
# tests/test_rotation_family.py
import unittest
import torch
from research.benchmarks.rotation_family import RotationFamilyDynamics

class TestRotationFamily(unittest.TestCase):
    def test_a0_constant_input_direction(self):
        sys = RotationFamilyDynamics(a=0.0)
        self.assertEqual(sys.Nstates, 2)
        self.assertEqual(sys.Ncontrols, 1)
        x = torch.tensor([[0.5, -1.2], [2.0, 3.0]])
        Gx = sys.analytic_Gx(x)
        self.assertEqual(Gx.shape, (2, 2, 1))
        # When a=0, Gx must be exactly [[1.0], [0.0]] for all states
        expected = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        self.assertTrue(torch.allclose(Gx, expected, atol=1e-6))

    def test_rotation_parameter_sweep(self):
        x = torch.tensor([[1.0, 0.0]])
        for a in [0.0, 0.5, 1.0, 2.0]:
            sys = RotationFamilyDynamics(a=a)
            Gx = sys.analytic_Gx(x)
            expected_angle = a * 1.0
            expected_g = torch.tensor([[[torch.cos(torch.tensor(expected_angle))],
                                        [torch.sin(torch.tensor(expected_angle))]]])
            self.assertTrue(torch.allclose(Gx, expected_g, atol=1e-5))

    def test_control_affine_identity(self):
        sys = RotationFamilyDynamics(a=1.5)
        x = torch.randn(10, 2)
        u = torch.randn(10, 1)
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_call_signature_conventions(self):
        sys = RotationFamilyDynamics(a=1.0)
        x = torch.tensor([[1.0, 0.5]])
        u = torch.tensor([[0.2]])
        res1 = sys.dynamics(0.0, x, u)
        res2 = sys.dynamics(x, u)
        self.assertTrue(torch.allclose(res1, res2))

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_rotation_family.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.benchmarks.rotation_family'

- [x] **Step 3: Implement `RotationFamilyDynamics` in `research/benchmarks/rotation_family.py`**

Formulation:
- Linear damped oscillator drift:
  $$f_0(x) = \begin{bmatrix} x_2 \\ -x_1 - 0.2 x_2 \end{bmatrix}$$
- State-dependent rotating input matrix:
  $$\theta(x) = a \cdot x_1, \quad G_x(x) = \begin{bmatrix} \cos(a \cdot x_1) \\ \sin(a \cdot x_1) \end{bmatrix}$$
- Total dynamics:
  $$\dot{x} = f_0(x) + G_x(x) u$$
- Dual calling convention: if `u is None`, unpack `x, u = t, x; t = 0.0`.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_rotation_family.py`
Expected: PASS (ran 4 tests)

- [x] **Step 5: Commit**

```bash
git add research/benchmarks/rotation_family.py tests/test_rotation_family.py
git commit -m "feat(benchmarks): implement RotationFamilyDynamics with tunable rotation parameter a"
```

---

### Task 2: Unicycle Dynamics (`research/benchmarks/unicycle.py`)

**Files:**
- Create: `research/benchmarks/unicycle.py`
- Test: `tests/test_unicycle.py`

**Interfaces:**
- Consumes: `torch`, `numpy`
- Produces: Class `UnicycleDynamics` with:
  - `__init__(dt=0.02, v_bounds=(-1.0, 1.0), omega_bounds=(-1.0, 1.0))`
  - `analytic_f0(x) -> Tensor[B, 3]`
  - `analytic_Gx(x) -> Tensor[B, 3, 2]`
  - `dynamics(t, x=None, u=None) -> Tensor[B, 3]`
  - `sample_initial_state(batch_size=1) -> Tensor[B, 3]`
  - `angle_normalize(theta) -> Tensor`
  - Attributes: `Nstates = 3`, `Ncontrols = 2`, `umin = torch.tensor([-1.0, -1.0])`, `umax = torch.tensor([1.0, 1.0])`, `dt = 0.02`

- [x] **Step 1: Write failing tests for UnicycleDynamics**

```python
# tests/test_unicycle.py
import unittest
import torch
import numpy as np
from research.benchmarks.unicycle import UnicycleDynamics

class TestUnicycle(unittest.TestCase):
    def test_dimensions_and_controls(self):
        sys = UnicycleDynamics()
        self.assertEqual(sys.Nstates, 3)
        self.assertEqual(sys.Ncontrols, 2)
        self.assertEqual(sys.umin.shape, (2,))
        self.assertEqual(sys.umax.shape, (2,))

    def test_heading_dependent_velocity(self):
        sys = UnicycleDynamics()
        # Test heading 0 -> moves in x
        x_east = torch.tensor([[0.0, 0.0, 0.0]])
        u = torch.tensor([[1.0, 0.0]]) # v=1, omega=0
        dx_east = sys.dynamics(x_east, u)
        self.assertTrue(torch.allclose(dx_east, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5))

        # Test heading pi/2 -> moves in y
        x_north = torch.tensor([[0.0, 0.0, np.pi / 2.0]], dtype=torch.float32)
        dx_north = sys.dynamics(x_north, u)
        self.assertTrue(torch.allclose(dx_north, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5))

        # Test pure rotation
        u_rot = torch.tensor([[0.0, 0.5]]) # v=0, omega=0.5
        dx_rot = sys.dynamics(x_east, u_rot)
        self.assertTrue(torch.allclose(dx_rot, torch.tensor([[0.0, 0.0, 0.5]]), atol=1e-5))

    def test_control_affine_identity(self):
        sys = UnicycleDynamics()
        x = torch.randn(10, 3)
        u = torch.randn(10, 2)
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_no_inplace_mutation(self):
        sys = UnicycleDynamics()
        x = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
        u = torch.tensor([[0.5, 0.1]])
        dx = sys.dynamics(x, u)
        loss = dx.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_unicycle.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.benchmarks.unicycle'

- [x] **Step 3: Implement `UnicycleDynamics` in `research/benchmarks/unicycle.py`**

Formulation:
- State $x = [p_x, p_y, \theta]^T$, controls $u = [v, \omega]^T$.
- Zero drift: $f_0(x) = \mathbf{0}_{3 \times 1}$.
- Heading-dependent input matrix:
  $$G_x(x) = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix}$$
- Safe angle wrap: `((theta + torch.pi) % (2 * torch.pi)) - torch.pi` without in-place tensor modification.
- Initial state sampling: $p_x, p_y \in [-2.0, 2.0]$, $\theta \in [-\pi, \pi]$.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_unicycle.py`
Expected: PASS (ran 4 tests)

- [x] **Step 5: Commit**

```bash
git add research/benchmarks/unicycle.py tests/test_unicycle.py
git commit -m "feat(benchmarks): implement UnicycleDynamics with full state and heading-dependent Gx"
```

---

### Task 3: Planar Quadrotor Dynamics (`research/benchmarks/planar_quadrotor.py`)

**Files:**
- Create: `research/benchmarks/planar_quadrotor.py`
- Create: `research/benchmarks/__init__.py`
- Test: `tests/test_planar_quadrotor.py`

**Interfaces:**
- Consumes: `torch`, `numpy`
- Produces: Class `PlanarQuadrotorDynamics` with:
  - `__init__(m=1.0, g=9.81, I_yy=0.01, L=0.25, dt=0.01, umax=15.0)`
  - `analytic_f0(x) -> Tensor[B, 6]`
  - `analytic_Gx(x) -> Tensor[B, 6, 2]`
  - `dynamics(t, x=None, u=None) -> Tensor[B, 6]`
  - `sample_initial_state(batch_size=1) -> Tensor[B, 6]`
  - Attributes: `Nstates = 6`, `Ncontrols = 2`, `umin = torch.tensor([0.0, 0.0])`, `umax = torch.tensor([15.0, 15.0])`, `dt = 0.01`
- Centralized `BENCHMARK_REGISTRY` in `research/benchmarks/__init__.py`.

- [x] **Step 1: Write failing tests for PlanarQuadrotorDynamics**

```python
# tests/test_planar_quadrotor.py
import unittest
import torch
import numpy as np
from research.benchmarks.planar_quadrotor import PlanarQuadrotorDynamics
from research.benchmarks import BENCHMARK_REGISTRY, load_benchmark

class TestPlanarQuadrotor(unittest.TestCase):
    def test_dimensions_and_hover_equilibrium(self):
        sys = PlanarQuadrotorDynamics()
        self.assertEqual(sys.Nstates, 6)
        self.assertEqual(sys.Ncontrols, 2)
        # Equilibrium state: origin at rest
        x_hover = torch.zeros((1, 6))
        # Hover thrust = m * g / 2 each
        u_hover = torch.tensor([[sys.m * sys.g / 2.0, sys.m * sys.g / 2.0]])
        dx = sys.dynamics(x_hover, u_hover)
        self.assertTrue(torch.allclose(dx, torch.zeros((1, 6)), atol=1e-5))

    def test_orientation_dependent_lateral_acceleration(self):
        sys = PlanarQuadrotorDynamics()
        # When tilted right (phi > 0), positive thrust pushes laterally in negative y direction:
        # y_ddot = -(u1+u2)/m * sin(phi)
        phi = np.pi / 6.0 # 30 deg
        x_tilted = torch.tensor([[0.0, 1.0, phi, 0.0, 0.0, 0.0]], dtype=torch.float32)
        u_thrust = torch.tensor([[5.0, 5.0]]) # 10 N total
        dx = sys.dynamics(x_tilted, u_thrust)
        expected_y_ddot = -(10.0 / sys.m) * np.sin(phi)
        self.assertAlmostEqual(dx[0, 3].item(), expected_y_ddot, places=4)

    def test_differential_thrust_angular_acceleration(self):
        sys = PlanarQuadrotorDynamics()
        x0 = torch.zeros((1, 6))
        # Differential thrust: u2 > u1 -> positive torque -> phi_ddot > 0
        u = torch.tensor([[2.0, 4.0]]) # diff = 2.0
        dx = sys.dynamics(x0, u)
        expected_phi_ddot = sys.L * (4.0 - 2.0) / sys.I_yy
        self.assertAlmostEqual(dx[0, 5].item(), expected_phi_ddot, places=4)

    def test_control_affine_identity(self):
        sys = PlanarQuadrotorDynamics()
        x = torch.randn(10, 6)
        u = torch.rand(10, 2) * 10.0
        f_direct = sys.dynamics(0.0, x, u)
        f0 = sys.analytic_f0(x)
        Gx = sys.analytic_Gx(x)
        f_affine = f0 + torch.bmm(Gx, u.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(f_direct, f_affine, atol=1e-5))

    def test_registry_loader(self):
        sys_dict = load_benchmark("planar_quadrotor")
        self.assertEqual(sys_dict["Nstates"], 6)
        self.assertEqual(sys_dict["Ncontrols"], 2)
        self.assertIn("rotation_family", BENCHMARK_REGISTRY)
        self.assertIn("unicycle", BENCHMARK_REGISTRY)
        self.assertIn("planar_quadrotor", BENCHMARK_REGISTRY)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_planar_quadrotor.py`
Expected: FAIL with ModuleNotFoundError: No module named 'research.benchmarks.planar_quadrotor'

- [x] **Step 3: Implement `PlanarQuadrotorDynamics` and `research/benchmarks/__init__.py`**

Formulation:
- State: $x = [y, z, \phi, \dot{y}, \dot{z}, \dot{\phi}]^T$.
- Thrusts: $u = [u_1, u_2]^T$ in $[0.0, 15.0]$ N.
- Total thrust $F = u_1 + u_2$, torque $\tau = L(u_2 - u_1)$.
- Drift:
  $$f_0(x) = \begin{bmatrix} \dot{y} \\ \dot{z} \\ \dot{\phi} \\ 0 \\ -g \\ 0 \end{bmatrix}$$
- Control-affine input matrix:
  $$G_x(x) = \begin{bmatrix}
  0 & 0 \\
  0 & 0 \\
  0 & 0 \\
  -\frac{\sin\phi}{m} & -\frac{\sin\phi}{m} \\
  \frac{\cos\phi}{m} & \frac{\cos\phi}{m} \\
  -\frac{L}{I_{yy}} & \frac{L}{I_{yy}}
  \end{bmatrix}$$
- In `research/benchmarks/__init__.py`: expose `BENCHMARK_REGISTRY` and `load_benchmark(name, **kwargs)` returning dictionary with `dynamics`, `sample_initial_state`, `analytic_f0`, `analytic_Gx`, `umin`, `umax`, `Nstates`, `Ncontrols`, `dt`.

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/test_planar_quadrotor.py`
Expected: PASS (ran 5 tests)

- [x] **Step 5: Run full benchmark test suite**

Run: `.venv/bin/python -m unittest discover tests`
Expected: PASS (all tests pass, 0 failures)

- [x] **Step 6: Commit**

```bash
git add research/benchmarks/planar_quadrotor.py research/benchmarks/__init__.py tests/test_planar_quadrotor.py
git commit -m "feat(benchmarks): implement PlanarQuadrotorDynamics and benchmark registry"
```

