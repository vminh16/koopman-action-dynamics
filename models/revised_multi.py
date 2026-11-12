from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # -------- stable initialization --------
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

    # ---------------- build canonical A_c, B_c ----------------
    def build_Ac(self):
        device = self.A_corelast_free.device
        N, n = self.N, self.n
        Ac = torch.zeros(N, N, device=device)

        # (1) put Brunovsky shifts inside top-left n×n blocks (unit superdiagonal; rows outside are zero)
        for (s, e) in self.core_ranges:
            ni = e - s
            if ni > 1:
                Ac[s : s + ni - 1, s + 1 : s + ni] = torch.eye(ni - 1, device=device)

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
        N, n, m = self.N, self.n, self.m
        Bc = torch.zeros(N, m, device=device)

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
        n, N = self.n, self.N
        Px = self.H_head @ self.H_head.T + self.eps * torch.eye(n, device=device)
        k = N - n
        if k == 0:
            return Px
        Pxi = self.H_tail @ self.H_tail.T + self.eps * torch.eye(k, device=device)
        return torch.block_diag(Px, Pxi)

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