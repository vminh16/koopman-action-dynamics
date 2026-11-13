import torch
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F

class BaseKoopmanOperator(nn.Module):
    def dynamics(self, t, z, u):
        raise NotImplementedError("Each KoopmanOperator must implement dynamics(t, z, u)")
    def get_AB(self):
        raise NotImplementedError("Each KoopmanOperator must implement get_AB()")


class FreeKoopmanOperator(BaseKoopmanOperator):
    def __init__(self, latent_dim, control_dim):
        super().__init__()
        # self.A = nn.Parameter(torch.randn(latent_dim, latent_dim))
        # self.B = nn.Parameter(torch.randn(latent_dim, control_dim))
        self.A = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.B = nn.Parameter(torch.empty(latent_dim, control_dim))
        with torch.no_grad():
            self.A.copy_(-1e-2 * torch.eye(latent_dim))
            nn.init.normal_(self.B, mean=0.0, std=1e-3)

    def dynamics(self, t, z, u):
        # assert z.ndim == 2 and u.ndim == 2, f"z: {z.shape}, u: {u.shape}"
        return z @ self.A.T + u @ self.B.T

    def get_AB(self):
        return self.A, self.B

class SingleInputControllableKoopmanOperator(BaseKoopmanOperator):
    def __init__(self, latent_dim, state_dim, eps=1e-2):
        """
        Implements the OOC canonical form for single-input systems.

        latent_dim: N (lifted state dimension)
        state_dim: n (original state dimension)
        """
        super().__init__()
        self.N = latent_dim
        self.n = state_dim
        self.eps = eps

        assert self.n < self.N

        self.A_free = nn.Parameter(torch.randn(self.N - self.n + 1, self.N))
        self.B_free = nn.Parameter(torch.randn(self.N - self.n, 1))


        # Learnable P parameterization: P = H H^T + eps I
        self.H_head = nn.Parameter(torch.randn(self.n, self.n))  # for top-left (n x n)
        k = self.N - self.n
        if k > 0:
            self.H_tail = nn.Parameter(torch.randn(k, k))  # for bottom-right ((N-n) x (N-n))
        else:
            self.H_tail = nn.Parameter(torch.zeros(0, 0), requires_grad=False)
        
        # # initialization
        with torch.no_grad():
            nn.init.normal_(self.A_free, mean=0.0, std=1e-3)  # A
            nn.init.normal_(self.B_free, mean=0.0, std=1e-3)  # B
            self.H_head.mul_(0.01)
            if k > 0: self.H_tail.mul_(0.01)
        self.eps = max(self.eps, 1e-1)
        # Learnable P parameterization: P = H H^T + eps I
        # self.H = nn.Parameter(torch.randn(self.N, self.N))
        # with torch.no_grad():
        #     nn.init.normal_(self.A_free, mean=0.0, std=1e-3)  # A
        #     nn.init.normal_(self.B_free, mean=0.0, std=1e-3)  # B
        #     self.H.mul_(0.01)

    def build_Ac(self):
        Ac = torch.zeros(self.N, self.N, device=self.A_free.device)
        for i in range(self.n - 1):
            Ac[i, i + 1] = 1.0
        Ac[self.n - 1:] = self.A_free
        return Ac

    def build_Bc(self):
        Bc = torch.zeros(self.N, 1, device=self.B_free.device)
        Bc[self.n - 1, 0] = 1.0
        Bc[self.n:] = self.B_free
        # print(B)
        return Bc

    def build_P(self):
        # Ensure strict positive-definite (invertible)

        n, N = self.n, self.N
        device = self.H_head.device
        Px = self.H_head @ self.H_head.T + self.eps * torch.eye(n, device=device)
        k = N - n
        if k == 0:
            return Px
        Pxi = self.H_tail @ self.H_tail.T + self.eps * torch.eye(k, device=device)
        return torch.block_diag(Px, Pxi)
        # eye = torch.eye(self.N, device=self.H.device)
        # P = self.H @ self.H.T + self.eps * eye
        # return P

    def get_transformed_AB(self):
        A_c = self.build_Ac()
        B_c = self.build_Bc()
        P = self.build_P()
        P_inv = torch.linalg.inv(P)
        # Transform to general (A, B) via similarity
        A = P @ A_c @ P_inv
        B = P @ B_c
        return A, B

    def dynamics(self, t, z, u):
        A, B = self.get_transformed_AB()
        return z @ A.T + u @ B.T

    def get_AB(self):
        return self.get_transformed_AB()


# class MultiInputControllableKoopmanOperator(BaseKoopmanOperator):
#     """
#     Multi-input OOC (Form TWO), 'loose' version (free rows may write to ANY columns, i.e., cross-block coupling allowed).
#     - Build canonical (A_c, B_c) by block-diagonally stitching m SISO-OOC blocks.
#     - Block i occupies a contiguous index range [s_i : e_i), size N_i = n_i + t_i.
#       * Top (n_i - 1) rows in the block are shift rows: only A_c[s_i+k, s_i+k+1] = 1, others in that row are 0.
#       * From row (n_i)-th (i.e., index s_i + n_i - 1) to the block end: FREE rows.
#         In this 'loose' version, FREE rows can write to ALL columns (0..N-1), not just this block's columns.
#       * B_c: at row (s_i + n_i - 1), column i, put 1; tail rows are FREE across ALL input columns.
#     - Final (A, B) are obtained via similarity: A = P A_c P^{-1}, B = P B_c, with P = H H^T + eps I.
#     """
#     def __init__(self,
#                  latent_dim: int,
#                  state_dim: int,
#                  control_dim: int,
#                  eps: float = 1e-2,
#                  block_state_sizes: Optional[List[int]] = None,  # [n1,...,nm], sum=n
#                  block_latent_sizes: Optional[List[int]] = None   # [N1,...,Nm], sum = N, each Ni >= ni
#                  ):
#         super().__init__()
#         self.N = latent_dim
#         self.n = state_dim
#         self.m = control_dim
#         self.eps = eps
#
#         assert self.m >= 1, "control_dim must be >= 1"
#         assert self.n < self.N, "Require state_dim < latent_dim to have free rows"
#         assert self.n >= self.m, "Need n >= m so each block has at least one core row"
#
#         # -------- average split helpers --------
#         def _avg_split(total: int, parts: int) -> List[int]:
#             q, r = divmod(total, parts)
#             return [q + (1 if i < r else 0) for i in range(parts)]
#
#         # -------- decide per-block core sizes n_i --------
#         if block_state_sizes is None:
#             self.n_list = _avg_split(self.n, self.m)
#         else:
#             assert len(block_state_sizes) == self.m and sum(block_state_sizes) == self.n
#             assert all(ni > 0 for ni in block_state_sizes), "every n_i must be >= 1"
#             self.n_list = list(block_state_sizes)
#
#         # -------- decide per-block total sizes N_i (thus tails t_i = N_i - n_i) --------
#         if block_latent_sizes is None:
#             tail_total = self.N - self.n
#             t_list = _avg_split(tail_total, self.m)
#             self.N_list = [ni + ti for ni, ti in zip(self.n_list, t_list)]
#         else:
#             assert len(block_latent_sizes) == self.m and sum(block_latent_sizes) == self.N
#             for Ni, ni in zip(block_latent_sizes, self.n_list):
#                 assert Ni >= ni, "Each block must satisfy N_i >= n_i"
#             self.N_list = list(block_latent_sizes)
#
#         # Derived tails and contiguous block ranges
#         self.t_list = [Ni - ni for Ni, ni in zip(self.N_list, self.n_list)]
#         self.block_ranges = []
#         s = 0
#         for Ni in self.N_list:
#             self.block_ranges.append((s, s + Ni))
#             s += Ni
#         assert s == self.N, "Block ranges must cover [0, N)"
#
#         # -------- parameters --------
#         # A_free[i]: FREE rows for block i (from its core-last-row to block end), may write to ALL columns (N).
#         #            shape: (N_i - (n_i - 1), N). Note: if n_i==1, this includes the first row of the block.
#         self.A_free = nn.ParameterList([
#             nn.Parameter(torch.randn(self.N_list[i] - (self.n_list[i] - 1), self.N))
#             for i in range(self.m)
#         ])
#
#         # B_tail[i]: FREE tail rows for block i (rows s_i + n_i : e_i), across ALL input columns (m).
#         #            shape: (t_i, m). If t_i == 0, store a 0xM non-trainable tensor for shape consistency.
#         self.B_tail = nn.ParameterList([
#             nn.Parameter(torch.randn(self.t_list[i], self.m)) if self.t_list[i] > 0
#             else nn.Parameter(torch.zeros(0, self.m), requires_grad=False)
#             for i in range(self.m)
#         ])
#
#         # Similarity transform (full SPD, same style as your SingleInput class)
#         self.H = nn.Parameter(torch.randn(self.N, self.N))
#         # ---- stable init (small free parts + P ~ eps*I) ----
#         with torch.no_grad():
#
#             for p in self.A_free:
#                 nn.init.normal_(p, mean=0.0, std=1e-3)
#
#             for p in self.B_tail:
#                 if isinstance(p, nn.Parameter) and p.numel() > 0:
#                     nn.init.normal_(p, mean=0.0, std=1e-3)
#
#             self.H.mul_(0.01)
#
#
#         # self.eps = max(self.eps, 1e-1)
#
#     # ---------------- canonical builders ----------------
#     def build_Ac(self):
#         device = self.H.device
#         Ac = torch.zeros(self.N, self.N, device=device)
#
#         for i, (s, e) in enumerate(self.block_ranges):
#             ni = self.n_list[i]
#             Ni = self.N_list[i]
#
#             # 1) shift rows: only superdiagonal '1' inside this block, all other entries in that row are 0
#             if ni > 1:
#                 Ac[s : s + ni - 1, s + 1 : s + ni] = torch.eye(ni - 1, device=device)
#
#             # 2) FREE rows from (core-last-row) to block end -> write to ALL columns
#             start = s + (ni - 1)           # index of the core-last-row
#             Afree = self.A_free[i]         # shape: (Ni - (ni - 1), N)
#             assert Afree.shape[0] == (Ni - (ni - 1)) and Afree.shape[1] == self.N
#             Ac[start : e, :] = Afree
#
#         return Ac
#
#     def build_Bc(self):
#         device = self.H.device
#         Bc = torch.zeros(self.N, self.m, device=device)
#
#         # top: place ones at each block's core-last-row, corresponding input column
#         for j, (s, e) in enumerate(self.block_ranges):
#             ni = self.n_list[j]
#             core_last_row = s + (ni - 1)
#             Bc[core_last_row, j] = 1.0
#
#         # bottoms: tail rows FREE across ALL input columns
#         for i, (s, e) in enumerate(self.block_ranges):
#             ni, ti = self.n_list[i], self.t_list[i]
#             if ti > 0:
#                 tail_s = s + ni
#                 tail_e = e
#                 Bt = self.B_tail[i]       # shape: (t_i, m)
#                 assert Bt.shape[0] == ti and Bt.shape[1] == self.m
#                 Bc[tail_s : tail_e, :] = Bt
#
#         return Bc
#
#     def build_P(self):
#         eye = torch.eye(self.N, device=self.H.device)
#         return self.H @ self.H.T + self.eps * eye
#
#     # ---------------- public API ----------------
#     def get_transformed_AB(self):
#         A_c = self.build_Ac()
#         B_c = self.build_Bc()
#         P = self.build_P()
#         P_inv = torch.linalg.inv(P)
#         A = P @ A_c @ P_inv
#         B = P @ B_c
#         return A, B
#
#     def dynamics(self, t, z, u):
#         A, B = self.get_transformed_AB()
#         return z @ A.T + u @ B.T
#
#     def get_AB(self):
#         return self.get_transformed_AB()

class MultiInputControllableKoopmanOperator(BaseKoopmanOperator):
    """
    Multi-input OOC (Form TWO), 'loose' version that preserves "first n entries are the original state".

    - Top-left n×n: Brunovsky core blocks (sizes [n1,...,nm]) on the diagonal with unit superdiagonal shifts.
    - Free rows = (i) each block's core-last row (m rows), plus (ii) all tail rows (N-n rows).
      In the loose version, these rows may write to ANY columns (cross-block coupling allowed).
    - B_c: put 1 at each block's core-last row in its input column; tail rows are free across all inputs.
    - Similarity transform P is block-diagonal:
        P = blkdiag(H_head H_head^T + eps I_n,  H_tail H_tail^T + eps I_{N-n})
      This avoids mixing the original state (first n) with the lifted tail.
    - Final (A,B): A = P A_c P^{-1} (implemented via `solve` for stability), B = P B_c.
    """

    def __init__(self,
                 latent_dim: int,
                 state_dim: int,
                 control_dim: int,
                 eps: float = 1e-2,
                 block_state_sizes: Optional[List[int]] = None  # [n1,...,nm], sum = n
                 ):
        super().__init__()
        self.N = latent_dim
        self.n = state_dim
        self.m = control_dim
        self.eps = eps

        assert self.m >= 1, "control_dim must be >= 1"
        assert self.n < self.N, "Require state_dim < latent_dim (need free rows)"
        assert self.n >= self.m, "Need n >= m so each block has at least one core row"

        # -------- decide core sizes [n1,...,nm] (sum = n) --------
        def _avg_split(total: int, parts: int) -> List[int]:
            q, r = divmod(total, parts)
            return [q + (1 if i < r else 0) for i in range(parts)]

        if block_state_sizes is None:
            self.n_list = _avg_split(self.n, self.m)
        else:
            assert len(block_state_sizes) == self.m and sum(block_state_sizes) == self.n
            assert all(ni > 0 for ni in block_state_sizes)
            self.n_list = list(block_state_sizes)

        # -------- core ranges in [0..n) and their last-row indices --------
        self.core_ranges = []
        s = 0
        for ni in self.n_list:
            self.core_ranges.append((s, s + ni))
            s += ni
        assert s == self.n
        self.core_last_rows = [s0 + (ni - 1) for (s0, _), ni in zip(self.core_ranges, self.n_list)]

        # -------- trainable free rows --------
        # (i) Free rows at each block's core-last row (m rows), can write to all N columns
        self.A_corelast_free = nn.Parameter(torch.randn(self.m, self.N))

        # (ii) Tail free rows (N-n rows), can write to all N columns; and free tail injection into inputs
        tail_dim = self.N - self.n
        if tail_dim > 0:
            self.A_tail_free = nn.Parameter(torch.randn(tail_dim, self.N))
            self.B_tail      = nn.Parameter(torch.randn(tail_dim, self.m))
        else:
            self.A_tail_free = nn.Parameter(torch.zeros(0, self.N), requires_grad=False)
            self.B_tail      = nn.Parameter(torch.zeros(0, self.m), requires_grad=False)

        # -------- block-diagonal similarity P parameters (SPD on each block) --------
        self.H_head = nn.Parameter(torch.randn(self.n, self.n))  # for the top-left n×n block
        if tail_dim > 0:
            self.H_tail = nn.Parameter(torch.randn(tail_dim, tail_dim))  # for the tail (N-n)×(N-n) block
        else:
            self.H_tail = nn.Parameter(torch.zeros(0, 0), requires_grad=False)
        #
        # # -------- stable initialization --------
        with torch.no_grad():
            nn.init.normal_(self.A_corelast_free, mean=0.0, std=1e-3)
            if tail_dim > 0:
                nn.init.normal_(self.A_tail_free, mean=0.0, std=1e-3)
                nn.init.normal_(self.B_tail,      mean=0.0, std=1e-3)
            # make P start near eps*I (good conditioning)
            self.H_head.mul_(0.01)
            if tail_dim > 0:
                self.H_tail.mul_(0.01)
        self.eps = max(self.eps, 1e-1)

        # Legacy dense parameter from an earlier parameterization. It is unused
        # (P is block-diagonal below) but kept as a buffer so checkpoints that
        # contain `operator.core.H` still load with strict=True.
        self.register_buffer("H", torch.zeros(self.N, self.N))

    # ---------------- build canonical A_c, B_c ----------------
    def build_Ac(self):
        device = self.A_corelast_free.device
        dtype = self.A_corelast_free.dtype
        N, n = self.N, self.n
        Ac = torch.zeros(N, N, device=device, dtype=dtype)

        # (1) put Brunovsky shifts inside top-left n×n blocks (unit superdiagonal; rows outside are zero)
        for (s, e) in self.core_ranges:
            ni = e - s
            if ni > 1:
                Ac[s : s + ni - 1, s + 1 : s + ni] = torch.eye(ni - 1, device=device, dtype=dtype)

        # (2) add free rows (differentiable assembly via padding + sum)
        contribs = []
        #   (2.1) each core-last row gets a full free row (size 1×N)
        for i, r in enumerate(self.core_last_rows):
            row = self.A_corelast_free[i].unsqueeze(0)  # (1, N)
            top, bottom = r, (N - r - 1)
            contribs.append(F.pad(row, pad=(0, 0, top, bottom)))  # -> (N, N)

        #   (2.2) all tail rows go to indices [n..N)
        if N - n > 0:
            contribs.append(F.pad(self.A_tail_free, pad=(0, 0, n, 0)))  # -> (N, N)

        if contribs:
            Ac = Ac + torch.stack(contribs, dim=0).sum(dim=0)
        return Ac

    def build_Bc(self):
        device = self.A_corelast_free.device
        dtype = self.A_corelast_free.dtype
        N, n, m = self.N, self.n, self.m
        Bc = torch.zeros(N, m, device=device, dtype=dtype)

        # (1) put 1 at each block's core-last row in its input column
        for j, r in enumerate(self.core_last_rows):
            Bc[r, j] = 1.0

        # (2) tail rows: free across ALL input columns (loose coupling)
        if N - n > 0:
            Bc = Bc + F.pad(self.B_tail, pad=(0, 0, n, 0))  # -> (N, m)
        return Bc

    def build_P(self):
        """Block-diagonal SPD transform: Px on the head (n×n), Pxi on the tail ((N-n)×(N-n))."""
        device = self.H_head.device
        dtype = self.H_head.dtype
        n, N = self.n, self.N
        Px = self.H_head @ self.H_head.T + self.eps * torch.eye(n, device=device, dtype=dtype)
        k = N - n
        if k == 0:
            return Px
        Pxi = self.H_tail @ self.H_tail.T + self.eps * torch.eye(k, device=device, dtype=dtype)
        return torch.block_diag(Px, Pxi)
        # eye = torch.eye(self.N, device=self.H.device)
        # P = self.H @ self.H.T + self.eps * eye
        # return P

    # ---------------- public API ----------------
    def get_transformed_AB(self):
        A_c = self.build_Ac()
        B_c = self.build_Bc()
        P   = self.build_P()
        # stable: avoid explicit inverse; solve P^T U = A_c^T  →  A = P @ U^T
        U = torch.linalg.solve(P.T, A_c.T)
        A = P @ U.T
        B = P @ B_c
        return A, B

    def dynamics(self, t, z, u):
        A, B = self.get_transformed_AB()
        return z @ A.T + u @ B.T

    def get_AB(self):
        return self.get_transformed_AB()


class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim, control_dim, state_dim=None, mode="free", **kwargs):
        super().__init__()
        if mode == "free":
            self.core = FreeKoopmanOperator(latent_dim, control_dim)
        elif mode == "controllable":
            assert state_dim is not None, "state_dim must be specified for controllable mode."
            if control_dim == 1:
                self.core = SingleInputControllableKoopmanOperator(latent_dim, state_dim)
            else:
                # Multi-input controllable
                self.core = MultiInputControllableKoopmanOperator(latent_dim, state_dim, control_dim)
        else:
            raise NotImplementedError(f"Unsupported Koopman mode: {mode}")

    def dynamics(self, t, z, u):
        return self.core.dynamics(t, z, u)

    def get_AB(self):
        return self.core.get_AB()
