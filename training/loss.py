import torch
import torch.nn.functional as F
from cvxpy import condition_number
from sympy.strategies import condition
from torchdiffeq import odeint
import math


def prediction_loss(x_pred, x_true, reduction='mean', gamma=0.8):
    """
        Prediction loss with optional gamma-decay weighting across time.

        Args:
            x_pred: [B, T, dx]
            x_true: [B, T, dx]
            gamma: float in (0, 1], decay factor (1.0 means no decay)

        Returns:
            Scalar loss (weighted MSE over time)
        """
    # B, T, dx = x_pred.shape
    # loss = 0.0
    # beta = 1.0
    #
    # for t in range(T):
    #     mse_t = F.mse_loss(x_pred[:, t, :], x_true[:, t, :], reduction='mean')
    #     loss += beta * mse_t
    #     beta *= gamma  # decay weight

    B, T, D = x_pred.shape
    mse_per_t = F.mse_loss(x_pred, x_true, reduction='none').mean(dim=-1)  # [B, T]
    weights = gamma ** torch.arange(T, device=x_pred.device).float()       # [T]
    loss = (mse_per_t * weights).mean()

    return loss

def encoder_loss_from_true(x_true, latent_target, encoder, gamma=0.8):
    """
    Args:
        x_true: [B, T, dx]
        latent_target: [B, T, dz - dx]
        encoder: StructuredEncoder
        gamma: time decay factor
    """
    B, T, dx = x_true.shape
    d_latent = latent_target.shape[-1]

    # flatten
    x_true_flat = x_true.reshape(B*T, dx)
    latent_target_flat = latent_target.reshape(B*T, d_latent)

    # encode x_true into latent prediction
    latent_pred_flat = encoder.extra_net(x_true_flat)  # [B*T, d_latent]

    # loss
    mse_per_t = F.mse_loss(latent_pred_flat, latent_target_flat, reduction='none').mean(dim=-1)  # [B*T]
    weights = gamma ** torch.arange(T, device=x_true.device).float()
    loss = (mse_per_t.reshape(B, T) * weights).mean()

    return loss

def encoder_loss_from_pred(x_pred, latent_target, encoder):
    """
    Encoder consistency loss based on predicted states (only latent parts).
    (NO time decay weighting)

    Args:
        x_pred: [B, T, dx]
        latent_target: [B, T, dz - dx]
        encoder: StructuredEncoder
    """
    B, T, dx = x_pred.shape
    d_latent = latent_target.shape[-1]

    # flatten
    x_pred_flat = x_pred.reshape(B*T, dx)
    latent_target_flat = latent_target.reshape(B*T, d_latent)

    # encode x_pred into latent prediction
    latent_pred_flat = encoder.extra_net(x_pred_flat)  # [B*T, d_latent]

    # plain MSE loss
    loss = F.mse_loss(latent_pred_flat, latent_target_flat, reduction='mean')

    return loss


import torch
# Assuming you are using the torchdiffeq library
# Requires installation: pip install torchdiffeq
from torchdiffeq import odeint

# --- Gramian and Controllability Loss Functions (Corrected) ---

def gramian_rhs_full(t, W_flat, A_matrix, B_matrix, C_output): # Renamed A, B to avoid clash with outer scope A, B if any
    # t: current time
    # W_flat: current state (flattened W matrix)
    # A_matrix, B_matrix: system matrices
    N = A_matrix.shape[0]
    # W = W_flat.view(N, N) # W is not directly used in this function
    Phi = torch.matrix_exp(A_matrix * t)
    Term = C_output @ Phi @ B_matrix
    dW = Term @ Term.T
    return dW.reshape(-1)

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

def compute_gramian_ode_full(A, B, C_output, T=0.3, dt=0.02): # A, B here are the specific matrices for this computation
    N = A.shape[0]
    n = C_output.shape[0]
    W0 = torch.zeros(n, n, device=A.device).reshape(-1)
    # Create time points
    t_eval = torch.arange(0, T + dt/2, dt, device=A.device) # +dt/2 ensures T is likely included

    # Use Euler method and fixed step size
    # Wrap gramian_rhs_full with a lambda to pass the specific A and B for this call
    # The A and B in the lambda are the A and B passed to compute_gramian_ode_full
    rhs_lambda = lambda t_param, W_flat_param: gramian_rhs_full(t_param, W_flat_param, A, B, C_output)

    W_traj = odeint(
        rhs_lambda,                 # Pass the wrapped function
        W0,
        t_eval,
        method='euler',
        options={"step_size": dt}
    )
    W_final = W_traj[-1].view(n, n)
    return W_final

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

def controllability_loss(A, B, state_dim, T=0.3, dt=0.02, method="coupled"):
    """
    Computes -min(eig(Wc)) where Wc is the controllability Gramian of (A, B)
    computed using the specified method ('coupled' or 'full') with Euler integration.

    Args:
        A: [N, N] torch tensor
        B: [N, m] torch tensor
        T: time horizon for integration (default: 0.3)
        dt: time step for Euler integration (default: 0.02)
        method: "coupled" or "full" (integration style)

    Returns:
        Negative of minimum eigenvalue of Wc (smaller => less controllable)
    """
    latent_dim = A.shape[0]
    device = A.device
    dtype = A.dtype
    C_output = torch.zeros(state_dim, latent_dim, device=device, dtype=dtype)
    C_output[:, :state_dim] = torch.eye(state_dim, device=device, dtype=dtype)
    T = 15*dt

    if method == "coupled":
        Wc = compute_gramian_coupled(A, B, C_output, T=T, dt=dt) # A and B are passed through
    elif method == "full":
        Wc = compute_gramian_ode_full(A, B, C_output, T=T, dt=dt) # A and B are passed through
    else:
        raise ValueError("Unknown Gramian computation method. Choose 'coupled' or 'full'.")

    try:
        eigvals = torch.linalg.eigvalsh(Wc)
    except torch._C._LinAlgError as e:
        print(
            f"Warning: eigvalsh failed for W_co ({e}). W_co might be ill-conditioned or not SPD due to numerical issues. Adding small identity perturbation for eigenvalue computation.")
        perturbation = torch.eye(Wc.shape[0], device=Wc.device, dtype=dtype) * 1e-8  # Ensure dtype matches
        try:
            eigvals = torch.linalg.eigvalsh(Wc + perturbation)
        except torch._C._LinAlgError:
            print(
                f"Warning: eigvalsh still failed after perturbation. Using torch.linalg.eigvals and taking real part.")
            eigvals_complex = torch.linalg.eigvals(Wc + perturbation)
            eigvals = eigvals_complex.real

    min_eigval = eigvals.min()
    max_eigval = eigvals.max()
    condition_number = max_eigval/(min_eigval+1e-8)
    log_cond = torch.log(condition_number)
    print("min, max:", min_eigval, max_eigval)
    alpha = 1.0
    beta = 0.
    delta = 0.03
    log_max_cond = math.log(3)
    # eigvals = torch.linalg.eigvalsh(Wc) # Gramian is symmetric
    return F.softplus(alpha*(delta-min_eigval)) + beta * F.softplus(alpha * (log_cond - log_max_cond))
    # return -eigvals.min()

# --- Example Usage ---
# N = 3
# m = 1
# A_test = torch.randn(N, N) * 0.5 # Make the system slightly more stable
# B_test = torch.randn(N, m)
#
# loss_c = controllability_loss(A_test, B_test, method='coupled')
# print(f"Loss (Coupled Euler, T=0.3, dt=0.02): {loss_c.item()}")
#
# loss_f = controllability_loss(A_test, B_test, method='full')
# print(f"Loss (Full Euler, T=0.3, dt=0.02): {loss_f.item()}")