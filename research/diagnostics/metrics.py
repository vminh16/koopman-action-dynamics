import torch
import numpy as np
from scipy.linalg import expm


def compute_continuous_gramian_and_margin(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    T: float,
    num_points: int = 200,
) -> dict:
    """
    Compute continuous-time reachability Gramian Wy(T) and observability Gramian Qy(T)
    using float64 numerical quadrature:
        Wy(T) = int_0^T C exp(As) B B^T exp(A^T s) C^T ds
        Qy(T) = int_0^T C exp(As) exp(A^T s) C^T ds
        rho_T = sqrt(lambda_min(Wy(T))) / sqrt(lambda_max(Qy(T)))
    """
    A_64 = np.asarray(A, dtype=np.float64)
    B_64 = np.asarray(B, dtype=np.float64)
    C_64 = np.asarray(C, dtype=np.float64)

    t_grid = np.linspace(0.0, float(T), num_points)
    dt = t_grid[1] - t_grid[0]

    ny = C_64.shape[0]
    Wy = np.zeros((ny, ny), dtype=np.float64)
    Qy = np.zeros((ny, ny), dtype=np.float64)

    for i, s in enumerate(t_grid):
        weight = dt if (i > 0 and i < num_points - 1) else 0.5 * dt
        e_As = expm(A_64 * s)

        # Integrand for Wy
        M_w = C_64 @ e_As @ B_64
        Wy += weight * (M_w @ M_w.T)

        # Integrand for Qy
        M_q = C_64 @ e_As
        Qy += weight * (M_q @ M_q.T)

    eig_Wy = np.linalg.eigvalsh(Wy)
    eig_Qy = np.linalg.eigvalsh(Qy)

    lambda_min_Wy = float(np.clip(np.min(eig_Wy), a_min=1e-16, a_max=None))
    lambda_max_Wy = float(np.max(eig_Wy))
    lambda_max_Qy = float(np.clip(np.max(eig_Qy), a_min=1e-16, a_max=None))

    rho_T = float(np.sqrt(lambda_min_Wy) / np.sqrt(lambda_max_Qy))
    cond_Wy = float(lambda_max_Wy / lambda_min_Wy)

    return {
        "rho_T": rho_T,
        "lambda_min_Wy": lambda_min_Wy,
        "lambda_max_Wy": lambda_max_Wy,
        "lambda_max_Qy": lambda_max_Qy,
        "cond_Wy": cond_Wy,
        "T": T,
    }


def compute_action_response_error(
    model,
    sys,
    x_samples: torch.Tensor,
    u_samples: torch.Tensor,
    u_ref: torch.Tensor,
) -> float:
    """
    Compute physical same-state action-response error E_response (EXPERIMENT.md §4.2):
        Delta_f(x; u, u0) = f(x, u) - f(x, u0)
        hat_Delta_f(x; u, u0) = C { f_z(phi(x), u) - f_z(phi(x), u0) }
        E_response = (1/K) sum || hat_Delta_f - Delta_f ||^2
    """
    model.eval()
    device = x_samples.device

    with torch.no_grad():
        # Physical truth
        f_true_u = sys.dynamics(0.0, x_samples, u_samples)
        f_true_ref = sys.dynamics(0.0, x_samples, u_ref)
        delta_true = f_true_u - f_true_ref

        # Model predicted response
        if hasattr(model, "forward"):
            f_model_u = model(x_samples, u_samples)
            f_model_ref = model(x_samples, u_ref)
            delta_model = f_model_u - f_model_ref
        elif hasattr(model, "model") and hasattr(model.model, "operator"):
            # Official KoopmanLitModule
            B = model.model.operator.B
            C = torch.eye(sys.Nstates, model.model.operator.latent_dim, device=device)
            # C @ B @ (u - u_ref)
            du = (u_samples - u_ref).unsqueeze(-1)
            delta_model = (C @ B @ du).squeeze(-1)
        else:
            raise AttributeError("Unrecognized model structure for action response computation")

        error_sq = torch.sum((delta_model - delta_true) ** 2, dim=-1)
        return float(torch.mean(error_sq).item())


def compute_vector_field_mismatches(
    model,
    sys,
    x_samples: torch.Tensor,
) -> dict:
    """
    Compute lifted and physical vector field mismatches (EXPERIMENT.md §4.3):
        E_phys = || C B - G_x(x) ||_F
        E_lift = || B - J_phi(x) G_x(x) ||_F
        E_drift = || A phi(x) - J_phi(x) f0(x) ||_2
        d_orient = || (I - Pi_{J_phi G_x}) B ||_2
    """
    device = x_samples.device
    Nstates = sys.Nstates

    # Extract A, B matrices
    if hasattr(model, "A") and hasattr(model, "B"):
        A = model.A
        B = model.B
        encode_fn = model.encode
    elif hasattr(model, "model") and hasattr(model.model, "operator"):
        A = model.model.operator.A
        B = model.model.operator.B
        encode_fn = model.model.encoder
    else:
        raise AttributeError("Unrecognized model structure for vector field extraction")

    latent_dim = A.shape[0]
    C = torch.eye(Nstates, latent_dim, device=device)

    E_phys_list = []
    E_lift_list = []
    E_drift_list = []
    d_orient_list = []

    B_mat = B.detach()
    A_mat = A.detach()

    for i in range(x_samples.shape[0]):
        xi = x_samples[i].clone().requires_grad_(True)
        # Compute Jacobian J_phi(xi) of shape [latent_dim, Nstates]
        phi_i = encode_fn(xi.unsqueeze(0)).squeeze(0)

        # Vectorized Jacobian
        J_phi = torch.zeros(latent_dim, Nstates, device=device)
        for j in range(latent_dim):
            grad_outputs = torch.zeros(latent_dim, device=device)
            grad_outputs[j] = 1.0
            (grad_xi,) = torch.autograd.grad(
                phi_i, xi, grad_outputs=grad_outputs, retain_graph=True
            )
            J_phi[j] = grad_xi

        xi_nograd = xi.detach().unsqueeze(0)
        Gx_i = sys.analytic_Gx(xi_nograd).squeeze(0).detach()     # [Nstates, Ncontrols]
        f0_i = sys.analytic_f0(xi_nograd).squeeze(0).detach()     # [Nstates]

        # 1. Physical error: || C B - G_x ||_F
        CB = C @ B_mat
        E_phys_i = torch.norm(CB - Gx_i, p="fro").item()
        E_phys_list.append(E_phys_i)

        # 2. Lifted input error: || B - J_phi G_x ||_F
        Gz_star = J_phi @ Gx_i  # [latent_dim, Ncontrols]
        E_lift_i = torch.norm(B_mat - Gz_star, p="fro").item()
        E_lift_list.append(E_lift_i)

        # 3. Drift error: || A phi - J_phi f0 ||_2
        drift_model = A_mat @ phi_i.detach()
        drift_true = J_phi @ f0_i
        E_drift_i = torch.norm(drift_model - drift_true, p=2).item()
        E_drift_list.append(E_drift_i)

        # 4. Orientation mismatch: || (I - Pi_{Gz_star}) B ||_2
        # Projector onto Gz_star: Pi = Gz (Gz^T Gz)^(-1) Gz^T
        Gz_np = Gz_star.cpu().numpy()
        U_gz, S_gz, _ = np.linalg.svd(Gz_np, full_matrices=False)
        rank_gz = np.sum(S_gz > 1e-6)
        if rank_gz > 0:
            U_span = U_gz[:, :rank_gz]
            Pi_Gz = torch.from_numpy(U_span @ U_span.T).float().to(device)
            I_mat = torch.eye(latent_dim, device=device)
            B_perp = (I_mat - Pi_Gz) @ B_mat
            d_orient_i = torch.norm(B_perp, p=2).item()
        else:
            d_orient_i = torch.norm(B_mat, p=2).item()
        d_orient_list.append(d_orient_i)

    return {
        "E_phys": float(np.mean(E_phys_list)),
        "E_lift": float(np.mean(E_lift_list)),
        "E_drift": float(np.mean(E_drift_list)),
        "d_orient": float(np.mean(d_orient_list)),
    }


def compute_candidate_regret(
    model,
    sys,
    x_samples: torch.Tensor,
    candidate_actions: torch.Tensor,
    horizon: int = 3,
) -> float:
    """
    Compute Candidate Regret R_cand (EXPERIMENT.md §4.5):
        R_cand = J(arg min_{A in A_pool} hat_J(A)) - min_{A in A_pool} J(A)
    """
    device = x_samples.device
    num_samples = x_samples.shape[0]
    num_candidates = candidate_actions.shape[0]

    regrets = []

    for i in range(num_samples):
        x0 = x_samples[i].clone()
        true_costs = []
        model_costs = []

        for c_idx in range(num_candidates):
            u_cand = candidate_actions[c_idx]

            # Rollout true system for horizon steps with constant candidate action
            x_true = x0.clone()
            cost_true = 0.0
            for _ in range(horizon):
                dx = sys.dynamics(0.0, x_true.unsqueeze(0), u_cand.unsqueeze(0)).squeeze(0)
                x_true = x_true + sys.dt * dx
                cost_true += float(torch.norm(x_true, p=2).item() ** 2 + 0.1 * torch.norm(u_cand, p=2).item() ** 2)
            true_costs.append(cost_true)

            # Rollout model
            x_model = x0.clone()
            cost_model = 0.0
            for _ in range(horizon):
                if hasattr(model, "forward"):
                    dx_m = model(x_model.unsqueeze(0), u_cand.unsqueeze(0)).squeeze(0)
                else:
                    z = model.model.encoder(x_model.unsqueeze(0))
                    dz = model.model.operator(z, u_cand.unsqueeze(0))
                    dx_m = dz[:, :sys.Nstates].squeeze(0)
                x_model = x_model + sys.dt * dx_m
                cost_model += float(torch.norm(x_model, p=2).item() ** 2 + 0.1 * torch.norm(u_cand, p=2).item() ** 2)
            model_costs.append(cost_model)

        best_model_idx = int(np.argmin(model_costs))
        true_cost_of_best_model = true_costs[best_model_idx]
        optimal_true_cost = float(np.min(true_costs))

        regret_i = max(0.0, true_cost_of_best_model - optimal_true_cost)
        regrets.append(regret_i)

    return float(np.mean(regrets))

