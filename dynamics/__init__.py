from .pendulum import PendulumDynamics
from .cartpole import CartPoleDynamics
from .mountaincar import MountainCarDynamics
from .grn import GRNContinuousDynamics

SYSTEM_REGISTRY = {
    "pendulum": PendulumDynamics,
    # "double_pendulum": DoublePendulumDynamics,
    "cartpole": CartPoleDynamics,
    "mountaincar": MountainCarDynamics,
    "grn": GRNContinuousDynamics,
    # ...
}

def load_system(name):
    cls = SYSTEM_REGISTRY[name.lower()]
    sys = cls()
    return {
        "dynamics": sys.dynamics,
        "sample_initial_state": sys.sample_initial_state,
        "umin": sys.umin,
        "umax": sys.umax,
        "Nstates": sys.Nstates,
        "dt": sys.dt
    }
