import torch

from diff_utils.solve_banded import make_banded_csr, solve_banded


class EigenvectorGateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        phi: torch.Tensor,
        x_star: torch.Tensor,
        d_vals: torch.Tensor,
        e_vals: torch.Tensor,
    ):
        return phi.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi, x_star, d_vals, e_vals = inputs
        ctx.save_for_backward(phi, x_star, d_vals, e_vals)

    @staticmethod
    def backward(ctx, grad_phi):
        phi, x_star, d_vals, e_vals = ctx.saved_tensors

        if phi.dim() == 1:
            _, gx, gd, ge = _backward_single(grad_phi, phi, x_star, d_vals, e_vals)
            return None, gx, gd, ge
        else:
            M, N = phi.shape
            grad_x = torch.zeros_like(x_star)
            grad_d = torch.zeros_like(d_vals)
            grad_e = torch.zeros_like(e_vals)

            for m in range(M):
                _, gx, gd, ge = _backward_single(
                    grad_phi[m], phi[m], x_star[m], d_vals[m], e_vals[m]
                )
                if gx is not None:
                    grad_x[m] = gx
                if gd is not None:
                    grad_d[m] = gd
                if ge is not None:
                    grad_e[m] = ge

            return None, grad_x, grad_d, grad_e


def _backward_single(grad_phi, phi, x_star, d_vals, e_vals):
    N = phi.shape[0]

    if phi.is_complex():
        dot = torch.dot(phi.conj(), grad_phi)
    else:
        dot = torch.dot(phi, grad_phi)
    g = grad_phi - dot * phi

    shifted_d = d_vals - x_star
    diags = {-1: e_vals.clone(), 0: shifted_d.clone(), 1: e_vals.clone()}

    try:
        A_csr = make_banded_csr(diags, N)
        lam = solve_banded(A_csr, g, kl=1, ku=1)
    except Exception:
        return None, None, None, None

    grad_d_out = -lam * phi
    grad_e_out = -lam[:-1] * phi[1:] - lam[1:] * phi[:-1]
    if phi.is_complex():
        grad_x_out = torch.dot(lam.conj(), phi)
    else:
        grad_x_out = torch.dot(lam, phi)

    return None, grad_x_out, grad_d_out, grad_e_out


def _find_clusters(x_star: torch.Tensor, tau: float):
    if tau <= 0.0 or x_star.dim() == 0 or x_star.shape[0] < 2:
        return []
    M = x_star.shape[0]
    sorted_x, sort_idx = x_star.sort()
    clusters = []
    start = 0
    while start < M:
        center = sorted_x[start]
        end = start + 1
        while end < M and (sorted_x[end] - center).abs().item() < tau:
            end += 1
        if end - start > 1:
            clusters.append(sort_idx[start:end].tolist())
        start = end
    return clusters


def _apply_tridiag(d, e, v):
    result = d * v
    result[:-1] += e * v[1:]
    result[1:] += e * v[:-1]
    return result


def eigenvector_gate(
    phi: torch.Tensor,
    x_star: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
) -> torch.Tensor:
    return EigenvectorGateFn.apply(phi, x_star, d_vals, e_vals)


def eigenvector_gate_degpert(
    phi: torch.Tensor,  # [M, N] mode shapes
    x_star: torch.Tensor,  # [M] eigenvalues
    d_vals: torch.Tensor,  # [N] tridiagonal diagonal
    e_vals: torch.Tensor,  # [N-1] tridiagonal off-diagonal
    grad_phi: torch.Tensor,  # [M, N] gradient
    tau: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    gor each cluster of near-degenerate eigenvalues {x_i1, ..., x_ik}:
    - build k×k cluster Hamiltonian: H_ij = phi_iT x A x phi_j
    - diagonalize H with effective eigenvalues x_eff that split degeneracy
    - x_eff in the adjoint solve instead of degenerate x_i.
    - project out within-cluster components from RHS of adjoint solve
    """
    M, N = phi.shape
    dtype = phi.dtype
    device = phi.device
    clusters = _find_clusters(x_star, tau)

    grad_x = torch.zeros(M, dtype=dtype, device=device)
    grad_d = torch.zeros(N, dtype=dtype, device=device)
    grad_e = torch.zeros(N - 1, dtype=dtype, device=device)

    in_cluster = set()
    for cl in clusters:
        in_cluster.update(cl)

    # standard adjoint
    for m in range(M):
        if m in in_cluster:
            continue
        _, gx, gd, ge = _backward_single(grad_phi[m], phi[m], x_star[m], d_vals, e_vals)
        if gx is not None:
            grad_x[m] = gx
        if gd is not None:
            grad_d += gd
        if ge is not None:
            grad_e += ge

    # degenerates
    for cluster_indices in clusters:
        k = len(cluster_indices)
        Phi_c = phi[cluster_indices]  # [k, N]
        grad_phi_c = grad_phi[cluster_indices]  # [k, N]

        A_Phi_c = torch.zeros_like(Phi_c)  # [k, N]
        for i in range(k):
            A_Phi_c[i] = _apply_tridiag(d_vals, e_vals, Phi_c[i])

        if dtype.is_complex:
            H = Phi_c.conj() @ A_Phi_c.T  # [k, k]
        else:
            H = Phi_c @ A_Phi_c.T  # [k, k]

        # eff eigenvalues
        if dtype.is_complex:
            x_eff, R = torch.linalg.eigh(0.5 * (H + H.conj().T))
            x_eff = x_eff.to(dtype)
        else:
            x_eff, R = torch.linalg.eigh(0.5 * (H + H.T))

        # R rotates cluster modes into the right basis for degeneracy
        # Phi_eff = R.conj @ Phi_c gives mode shapesj
        if dtype.is_complex:
            Phi_eff = R.conj().T @ Phi_c  # [k, N]
            grad_phi_eff = R.conj().T @ grad_phi_c  # [k, N]
        else:
            Phi_eff = R.T @ Phi_c
            grad_phi_eff = R.T @ grad_phi_c

        # adjoint sovle
        for i in range(k):
            m = cluster_indices[i]
            _, gx, gd_i, ge_i = _backward_single(
                grad_phi_eff[i], Phi_eff[i], x_eff[i], d_vals, e_vals
            )
            if gx is not None:
                grad_x[m] = gx
            if gd_i is not None:
                grad_d += gd_i
            if ge_i is not None:
                grad_e += ge_i

    return grad_x, grad_d, grad_e


def tridiag_eigvec_adjoint(
    phi: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    grad_phi: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cluster aware degenerate perturbation eigenvector adjoint
    from diff_utils.tridiag_eigh import tridiag_eigh

    N = phi.shape[0]

    # eigendecompose the tridiag
    sigma, Q = tridiag_eigh(d_vals.detach(), e_vals.detach(), eps=eps)

    # which tridiag eigenvalue corresponds to phi
    overlaps = (Q.T @ phi.detach()).abs()
    mode_idx = int(overlaps.argmax().item())
    sigma_m = sigma[mode_idx]

    diffs = (sigma - sigma_m).abs()
    in_cluster = diffs < tau
    in_cluster[mode_idx] = True  # always include the mode itself
    cluster_idx = in_cluster.nonzero(as_tuple=True)[0]

    # project gradient's nullspace
    dot = torch.dot(phi.detach(), grad_phi)
    g = grad_phi - dot * phi.detach()

    g_hat = Q.T @ g  # [N]

    diff_from_mode = sigma - sigma_m
    lorentz = diff_from_mode / (diff_from_mode * diff_from_mode + eps * eps)
    lambda_hat = g_hat * lorentz

    lambda_hat[cluster_idx] = 0.0

    lam = Q @ lambda_hat  # [N]

    lam = lam - torch.dot(phi.detach(), lam) * phi.detach()

    grad_d_out = -lam * phi.detach()
    grad_e_out = -lam[:-1] * phi.detach()[1:] - lam[1:] * phi.detach()[:-1]

    return grad_d_out, grad_e_out


class TridiagEigvecAdjointFn(torch.autograd.Function):
    @staticmethod
    def forward(phi, d_vals, e_vals, tau, eps):
        return phi.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi, d_vals, e_vals, tau, eps = inputs
        ctx.tau = tau
        ctx.eps = eps
        ctx.save_for_backward(phi, d_vals, e_vals)

    @staticmethod
    def backward(ctx, grad_phi):
        phi, d_vals, e_vals = ctx.saved_tensors
        grad_d, grad_e = tridiag_eigvec_adjoint(
            phi,
            d_vals,
            e_vals,
            grad_phi,
            tau=ctx.tau,
            eps=ctx.eps,
        )
        return None, grad_d, grad_e, None, None


def tridiag_eigvec_reattach(
    phi: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
) -> torch.Tensor:
    return TridiagEigvecAdjointFn.apply(phi, d_vals, e_vals, tau, eps)


__all__ = [
    "EigenvectorGateFn",
    "eigenvector_gate",
    "eigenvector_gate_degpert",
    "tridiag_eigvec_adjoint",
    "tridiag_eigvec_reattach",
]
