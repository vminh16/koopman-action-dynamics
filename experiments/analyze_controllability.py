import argparse
import yaml
import torch

from training.train import KoopmanLitModule
from dynamics import SYSTEM_REGISTRY
from torchdiffeq import odeint
def compute_controllability_matrix(A, B, steps=None):
    n = A.shape[0]
    m = B.shape[1]
    steps = steps or n
    C = B
    AB = B
    for _ in range(1, steps):
        AB = A @ AB
        C = torch.cat((C, AB), dim=1)
    return C

def compute_output_matrix(state_dim, latent_dim):
    C = torch.zeros(state_dim, latent_dim)
    C[:, :state_dim] = torch.eye(state_dim)
    return C

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_A_B(model, koopman_mode):
    if koopman_mode == "free":
        A = model.model.operator.core.A
        B = model.model.operator.core.B
    elif koopman_mode == "controllable":
        A, B = model.model.operator.core.get_transformed_AB()
    else:
        raise ValueError(f"Unknown koopman_mode: {koopman_mode}")
    return A.detach(), B.detach()


def coupled_rhs(t, y, A_matrix, B_matrix, C_output): # Renamed A, B to avoid clash
    # t: current time
    # y: current state (flattened [Phi, W] vector)
    # A_matrix, B_matrix: system matrices
    """
    RHS for coupled ODE system:
    - y = [Phi.flatten(), W.flatten()] ∈ R^{N^2 + N^2}
    - dPhi/dt = A_matrix @ Phi
    - dW/dt = Phi @ B_matrix @ B_matrixᵀ @ Phiᵀ
    """
    N = A_matrix.shape[0]
    n = C_output.shape[0]

    Phi_flat = y[:N*N]
    # W_co_flat = y[N ** 2: N ** 2 + n ** 2]
    # W_flat = y[N*N:] # W_flat is not directly used in this function
    Phi = Phi_flat.view(N, N)
    # W = W_flat.view(N, N) # W is not directly used in this function

    dPhi = A_matrix @ Phi
    Term = C_output @ Phi @ B_matrix
    dW = Term @ Term.T

    return torch.cat([dPhi.reshape(-1), dW.reshape(-1)])

def compute_gramian_coupled(A, B, C_output, T=0.3, dt=0.02): # A, B here are the specific matrices for this computation
    N = A.shape[0]
    n = C_output.shape[0]
    device = A.device

    Phi0 = torch.eye(N, device=device)
    W0 = torch.zeros(n, n, device=device)
    y0 = torch.cat([Phi0.reshape(-1), W0.reshape(-1)])

    # Create time points
    t_eval = torch.arange(0, T + dt/2, dt, device=A.device) # +dt/2 ensures T is likely included

    # Use Euler method and fixed step size
    # Wrap coupled_rhs with a lambda to pass the specific A and B for this call
    # The A and B in the lambda are the A and B passed to compute_gramian_coupled
    rhs_lambda = lambda t_param, y_param: coupled_rhs(t_param, y_param, A, B, C_output)

    traj = odeint(
        rhs_lambda,                 # Pass the wrapped function
        y0,
        t_eval,
        method="euler",
        options={"step_size": dt}
    )

    W_final_flat = traj[-1, N*N : N*N + n*n]
    W_final = W_final_flat.view(n, n)
    return W_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # Load config and system
    config = load_config(args.config)
    koopman_mode = config.get("koopman_mode", "unknown")

    system_class = SYSTEM_REGISTRY[config["system"]]
    system = system_class()
    system_config = {
        "dynamics": system.dynamics,
        "sample_initial_state": system.sample_initial_state,
        "umin": system.umin,
        "umax": system.umax,
        "dt": system.dt,
        "state_dim": system.Nstates
    }

    # Build model and load weights
    model = KoopmanLitModule(system_config, config)
    ckpt = torch.load(config["checkpoint"], map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    # Extract Koopman matrices
    A, B = extract_A_B(model, koopman_mode)
    print("A matrix:\n", A)
    print("\nB matrix:\n", B)
    eigvals = torch.linalg.eigvals(A)
    print("A eigenvalues：", eigvals)
    # Compute controllability
    Cmat = compute_controllability_matrix(A, B)
    rank_C = torch.linalg.matrix_rank(Cmat)
    print(f"\n[Koopman Mode: {koopman_mode}]")
    print("Controllability Matrix Shape:", Cmat.shape)
    print("Rank:", rank_C.item(), "/", A.shape[0])

    # Output controllability
    state_dim = system_config["state_dim"]
    Cout = compute_output_matrix(state_dim, A.shape[0]) @ Cmat
    print(f"Output matrix:\n{Cout}")
    rank_Cout = torch.linalg.matrix_rank(Cout)
    print("Output Controllability Matrix Shape:", Cout.shape)
    print("Rank:", rank_Cout.item(), "/", state_dim)

    T = 4
    dt = 0.02

    C = compute_output_matrix(state_dim, A.shape[0])

    Wc = compute_gramian_coupled(A, B, C, T=T, dt=dt)
    print(f"[Output Gramian, T={T}, dt={dt}] Shape: {Wc.shape}")
    print(Wc)
    eigvals = torch.linalg.eigvalsh(Wc)
    print("[Output Gramian Eigenvalues]:", eigvals.tolist())

    from scipy.linalg import solve_continuous_lyapunov
    Wo = torch.from_numpy(solve_continuous_lyapunov(A.detach().numpy(), -B.detach().numpy() @ B.detach().numpy().T))
    Wci = C @ Wo @ C.T
    eigvals = torch.linalg.eigvalsh(Wci)
    print("\n[Infinite-horizon Output Gramian Eigenvalues]:", eigvals.tolist())

if __name__ == "__main__":
    main()