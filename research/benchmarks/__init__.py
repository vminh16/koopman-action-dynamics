from .rotation_family import RotationFamilyDynamics
from .unicycle import UnicycleDynamics
from .planar_quadrotor import PlanarQuadrotorDynamics

BENCHMARK_REGISTRY = {
    "rotation_family": RotationFamilyDynamics,
    "unicycle": UnicycleDynamics,
    "planar_quadrotor": PlanarQuadrotorDynamics,
}


def load_benchmark(name: str, **kwargs):
    """
    Factory loader for benchmark systems, matching repo's dynamics interface.
    """
    name_key = name.lower()
    if name_key not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}")

    cls = BENCHMARK_REGISTRY[name_key]
    sys = cls(**kwargs)

    return {
        "instance": sys,
        "dynamics": sys.dynamics,
        "analytic_f0": sys.analytic_f0,
        "analytic_Gx": sys.analytic_Gx,
        "sample_initial_state": sys.sample_initial_state,
        "umin": sys.umin,
        "umax": sys.umax,
        "Nstates": sys.Nstates,
        "Ncontrols": sys.Ncontrols,
        "dt": sys.dt,
    }

