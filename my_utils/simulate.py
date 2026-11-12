import torch
from torchdiffeq import odeint

def simulate_trajectory(dynamics_fn, x0, u_list, dt=0.02, t0=0.0, device='cpu'):
    """
    Simulate trajectory using a list of control inputs and continuous-time dynamics.

    Args:
        dynamics_fn: function (t, x, u) → dx/dt
        x0: Tensor [state_dim]
        u_list: list of T tensors, each [control_dim]
        dt: float, time step
        t0: float, start time
        device: torch.device

    Returns:
        X: Tensor [T+1, state_dim]  (includes x0)
        U: Tensor [T, control_dim]
        T_arr: Tensor [T+1] (time stamps)
    """
    x = x0.to(device)
    X = [x]
    U = []
    T_arr = [t0]

    for i, u in enumerate(u_list):
        u_t = u.to(device).unsqueeze(0)  # [1, u_dim]
        t_start = t0 + i * dt
        t_end = t0 + (i + 1) * dt

        def dyn(t, x_):
            return dynamics_fn(t, x_.unsqueeze(0), u_t).squeeze(0)

        x = odeint(dyn, x, torch.tensor([t_start, t_end], device=device))[1]
        X.append(x)
        U.append(u)
        T_arr.append(t_end)

    return {
        "x0": x0,
        "u_seq": torch.stack(U),
        "x_seq": torch.stack(X[1:])
    }