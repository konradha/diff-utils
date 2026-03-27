import torch

from diff_utils.solve_banded import make_banded_csr, solve_banded
from diff_utils.solve_tridiag import solve_tridiag, solve_tridiag_batch


class EigvecReattachFn(torch.autograd.Function):
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
    if v.dim() == 1:
        result = d * v
        result[:-1] += e * v[1:]
        result[1:] += e * v[:-1]
        return result

    result = v * d.unsqueeze(0)
    result[:, :-1] += v[:, 1:] * e.unsqueeze(0)
    result[:, 1:] += v[:, :-1] * e.unsqueeze(0)
    return result


def _orthonormalize_basis(basis: torch.Tensor) -> torch.Tensor:
    if basis.dim() == 1:
        norm = basis.norm().clamp(min=1e-30)
        return (basis / norm).unsqueeze(0)
    Q, _ = torch.linalg.qr(basis.T, mode="reduced")
    return Q.T.contiguous()


def _project_orthogonal(v, phi):
    return _project_orthogonal_subspace(v, phi.unsqueeze(0))


def _project_orthogonal_subspace(v, basis: torch.Tensor):
    basis_orth = _orthonormalize_basis(basis)
    if v.dim() == 1:
        if basis_orth.is_complex():
            coeff = basis_orth.conj() @ v
        else:
            coeff = basis_orth @ v
        return v - basis_orth.T @ coeff

    if basis_orth.is_complex():
        coeff = basis_orth.conj() @ v.T
    else:
        coeff = basis_orth @ v.T
    projected = v.T - basis_orth.T @ coeff
    return projected.T.contiguous()


def _rayleigh_quotient(phi, d_vals, e_vals):
    A_phi = _apply_tridiag(d_vals, e_vals, phi)
    if phi.is_complex():
        num = torch.dot(phi.conj(), A_phi)
        den = torch.dot(phi.conj(), phi)
    else:
        num = torch.dot(phi, A_phi)
        den = torch.dot(phi, phi)
    return num / den


def _rayleigh_quotient_batch(phi_batch, d_vals, e_vals):
    A_phi = _apply_tridiag(d_vals, e_vals, phi_batch)
    if phi_batch.is_complex():
        num = (phi_batch.conj() * A_phi).sum(dim=1)
        den = (phi_batch.conj() * phi_batch).sum(dim=1)
    else:
        num = (phi_batch * A_phi).sum(dim=1)
        den = (phi_batch * phi_batch).sum(dim=1)
    return num / den


def _cg_solve_normal_equation(
    basis,
    d_vals,
    e_vals,
    sigma,
    rhs,
    *,
    max_iter=None,
    tol=1e-10,
    reg=1e-12,
):
    N = rhs.shape[0]
    if max_iter is None:
        max_iter = max(64, min(8 * N, 4096))

    shifted_d = d_vals - sigma
    basis_orth = _orthonormalize_basis(basis)

    def K(v):
        return _project_orthogonal_subspace(_apply_tridiag(shifted_d, e_vals, v), basis_orth)

    def B(v):
        return _project_orthogonal_subspace(K(K(v)) + reg * v, basis_orth)

    b = _project_orthogonal_subspace(K(rhs), basis_orth)
    x = torch.zeros_like(rhs)
    r = b.clone()
    p = r.clone()

    if rhs.is_complex():
        rs_old = torch.dot(r.conj(), r).real
    else:
        rs_old = torch.dot(r, r)
    b_norm = rs_old.sqrt().clamp(min=1e-30)

    converged = bool((rs_old.sqrt() / b_norm) <= tol)
    n_iter = 0

    for it in range(max_iter):
        Ap = B(p)
        if rhs.is_complex():
            denom = torch.dot(p.conj(), Ap)
        else:
            denom = torch.dot(p, Ap)
        if denom.abs().item() < 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = _project_orthogonal_subspace(r - alpha * Ap, basis_orth)
        if rhs.is_complex():
            rs_new = torch.dot(r.conj(), r).real
        else:
            rs_new = torch.dot(r, r)
        n_iter = it + 1
        if (rs_new.sqrt() / b_norm) <= tol:
            converged = True
            break
        beta = rs_new / rs_old.clamp(min=1e-30)
        p = r + beta * p
        rs_old = rs_new

    x = _project_orthogonal_subspace(x, basis_orth)
    residual = _project_orthogonal_subspace(K(x) - rhs, basis_orth)
    if rhs.is_complex():
        rel_res = torch.dot(residual.conj(), residual).real.sqrt() / rhs.norm().clamp(min=1e-30)
    else:
        rel_res = torch.dot(residual, residual).sqrt() / rhs.norm().clamp(min=1e-30)
    return x, bool(converged), float(rel_res.item()), n_iter


def _project_rows_orthogonal(v_batch, basis_batch):
    if basis_batch.is_complex():
        coeff = (basis_batch.conj() * v_batch).sum(dim=1, keepdim=True)
    else:
        coeff = (basis_batch * v_batch).sum(dim=1, keepdim=True)
    return v_batch - coeff * basis_batch


def _apply_shifted_tridiag_batch(shifted_d_batch, e_vals, v_batch):
    result = shifted_d_batch * v_batch
    result[:, :-1] += v_batch[:, 1:] * e_vals.unsqueeze(0)
    result[:, 1:] += v_batch[:, :-1] * e_vals.unsqueeze(0)
    return result


def _cg_solve_normal_equation_batch(
    phi_batch,
    d_vals,
    e_vals,
    sigma_batch,
    rhs_batch,
    *,
    max_iter=None,
    tol=1e-10,
    reg=1e-12,
):
    M, N = rhs_batch.shape
    if max_iter is None:
        max_iter = max(64, min(8 * N, 4096))

    phi_batch = _orthonormalize_basis(phi_batch) if M == 1 else phi_batch
    shifted_d_batch = d_vals.unsqueeze(0) - sigma_batch.unsqueeze(1)

    def K(v):
        return _project_rows_orthogonal(
            _apply_shifted_tridiag_batch(shifted_d_batch, e_vals, v),
            phi_batch,
        )

    def B(v):
        return _project_rows_orthogonal(K(K(v)) + reg * v, phi_batch)

    b = _project_rows_orthogonal(K(rhs_batch), phi_batch)
    x = torch.zeros_like(rhs_batch)
    r = b.clone()
    p = r.clone()

    if rhs_batch.is_complex():
        rs_old = (r.conj() * r).sum(dim=1).real
    else:
        rs_old = (r * r).sum(dim=1)
    b_norm = rs_old.sqrt().clamp(min=1e-30)
    converged = (rs_old.sqrt() / b_norm) <= tol

    for _ in range(max_iter):
        if bool(converged.all()):
            break
        Ap = B(p)
        if rhs_batch.is_complex():
            denom = (p.conj() * Ap).sum(dim=1)
        else:
            denom = (p * Ap).sum(dim=1)
        safe = (~converged) & (denom.abs() > 1e-30)
        if not bool(safe.any()):
            break

        alpha = torch.zeros_like(denom)
        alpha[safe] = rs_old[safe] / denom[safe]
        x = x + alpha.unsqueeze(1) * p
        r = _project_rows_orthogonal(r - alpha.unsqueeze(1) * Ap, phi_batch)

        if rhs_batch.is_complex():
            rs_new = (r.conj() * r).sum(dim=1).real
        else:
            rs_new = (r * r).sum(dim=1)
        newly_converged = (rs_new.sqrt() / b_norm) <= tol
        beta = torch.zeros_like(rs_new)
        active = safe & (~newly_converged)
        beta[active] = rs_new[active] / rs_old[active].clamp(min=1e-30)
        p = r + beta.unsqueeze(1) * p
        rs_old = rs_new
        converged = converged | newly_converged

    x = _project_rows_orthogonal(x, phi_batch)
    residual = _project_rows_orthogonal(K(x) - rhs_batch, phi_batch)
    if rhs_batch.is_complex():
        rel_res = (residual.conj() * residual).sum(dim=1).real.sqrt() / rhs_batch.norm(dim=1).clamp(
            min=1e-30
        )
    else:
        rel_res = (residual * residual).sum(dim=1).sqrt() / rhs_batch.norm(dim=1).clamp(min=1e-30)
    return x, converged, rel_res


def eigvec_reattach(
    phi: torch.Tensor,
    x_star: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
) -> torch.Tensor:
    return EigvecReattachFn.apply(phi, x_star, d_vals, e_vals)


def eigvec_degpert(
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

        if dtype.is_complex:
            x_eff, R = torch.linalg.eigh(0.5 * (H + H.conj().T))
            x_eff = x_eff.to(dtype)
        else:
            x_eff, R = torch.linalg.eigh(0.5 * (H + H.T))

        if dtype.is_complex:
            Phi_eff = R.conj().T @ Phi_c  # [k, N]
            grad_phi_eff = R.conj().T @ grad_phi_c  # [k, N]
        else:
            Phi_eff = R.T @ Phi_c
            grad_phi_eff = R.T @ grad_phi_c

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


def _tridiag_eigvec_adjoint_dense_oracle(
    phi: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    grad_phi: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    from diff_utils.tridiag_eigh import tridiag_eigh

    N = phi.shape[0]

    sigma, Q = tridiag_eigh(d_vals.detach(), e_vals.detach(), eps=eps)

    overlaps = (Q.T @ phi.detach()).abs()
    mode_idx = int(overlaps.argmax().item())
    sigma_m = sigma[mode_idx]

    diffs = (sigma - sigma_m).abs()
    in_cluster = diffs < tau
    in_cluster[mode_idx] = True  # always include the mode itself
    cluster_idx = in_cluster.nonzero(as_tuple=True)[0]

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


def tridiag_eigvec_adjoint(
    phi: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    grad_phi: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
    max_iter: int | None = None,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi_ref = phi.detach()

    sigma_m = _rayleigh_quotient(phi_ref, d_vals.detach(), e_vals.detach()).real.to(d_vals.dtype)
    g = _project_orthogonal(grad_phi, phi_ref)

    reg = max(eps * eps, 1e-14)
    shifted_d = d_vals.detach() - sigma_m + reg
    lam = solve_tridiag(e_vals.detach(), shifted_d, e_vals.detach(), g)
    lam = _project_orthogonal(lam, phi_ref)

    grad_d_out = -lam * phi_ref
    grad_e_out = -lam[:-1] * phi_ref[1:] - lam[1:] * phi_ref[:-1]
    return grad_d_out, grad_e_out


def tridiag_eigvec_cluster_adjoint(
    phi_cluster: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    grad_phi_cluster: torch.Tensor,
    *,
    eps: float = 1e-10,
    max_iter: int | None = None,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    Phi_c = _orthonormalize_basis(phi_cluster.detach())
    grad_phi_c = grad_phi_cluster

    A_Phi = _apply_tridiag(d_vals, e_vals, Phi_c)
    if Phi_c.is_complex():
        H = Phi_c.conj() @ A_Phi.T
        H = 0.5 * (H + H.conj().T)
        sigma_eff, R = torch.linalg.eigh(H)
        Phi_eff = R.conj().T @ Phi_c
        grad_phi_eff = R.conj().T @ grad_phi_c
    else:
        H = Phi_c @ A_Phi.T
        H = 0.5 * (H + H.T)
        sigma_eff, R = torch.linalg.eigh(H)
        Phi_eff = R.T @ Phi_c
        grad_phi_eff = R.T @ grad_phi_c

    grad_d = torch.zeros_like(d_vals)
    grad_e = torch.zeros_like(e_vals)
    basis_eff = _orthonormalize_basis(Phi_eff)

    reg = max(eps * eps, 1e-14)
    for i in range(Phi_eff.shape[0]):
        rhs = _project_orthogonal_subspace(grad_phi_eff[i], basis_eff)
        shifted_d = d_vals.detach() - sigma_eff[i].real.to(d_vals.dtype) + reg
        lam = solve_tridiag(e_vals.detach(), shifted_d, e_vals.detach(), rhs)
        lam = _project_orthogonal_subspace(lam, basis_eff)
        phi_i = Phi_eff[i]
        grad_d = grad_d - lam * phi_i
        grad_e = grad_e - lam[:-1] * phi_i[1:] - lam[1:] * phi_i[:-1]

    return grad_d, grad_e


def tridiag_eigvec_adjoint_batch(
    phi_batch: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    grad_phi_batch: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
    max_iter: int | None = None,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = phi_batch.shape
    sigma_batch = _rayleigh_quotient_batch(
        phi_batch.detach(), d_vals.detach(), e_vals.detach()
    ).real.to(d_vals.dtype)
    clusters = _find_clusters(sigma_batch, tau)
    in_cluster = set()
    for cl in clusters:
        in_cluster.update(cl)
    isolated = [m for m in range(M) if m not in in_cluster]

    grad_d = torch.zeros_like(d_vals)
    grad_e = torch.zeros_like(e_vals)

    if isolated:
        iso_idx = torch.tensor(isolated, dtype=torch.long, device=phi_batch.device)
        phi_iso = phi_batch.detach()[iso_idx]
        grad_phi_iso = grad_phi_batch[iso_idx]
        rhs_iso = _project_rows_orthogonal(grad_phi_iso, phi_iso)
        lam_iso, _, _ = _cg_solve_normal_equation_batch(
            phi_iso,
            d_vals,
            e_vals,
            sigma_batch[iso_idx],
            rhs_iso,
            max_iter=max_iter,
            tol=tol,
            reg=max(eps * eps, 1e-14),
        )
        grad_d = grad_d - (lam_iso * phi_iso).sum(dim=0)
        grad_e = grad_e - (lam_iso[:, :-1] * phi_iso[:, 1:] + lam_iso[:, 1:] * phi_iso[:, :-1]).sum(
            dim=0
        )

    for cluster_indices in clusters:
        idx = torch.tensor(cluster_indices, dtype=torch.long, device=phi_batch.device)
        gd_i, ge_i = tridiag_eigvec_cluster_adjoint(
            phi_batch[idx],
            d_vals,
            e_vals,
            grad_phi_batch[idx],
            eps=eps,
            max_iter=max_iter,
            tol=tol,
        )
        grad_d = grad_d + gd_i
        grad_e = grad_e + ge_i

    return grad_d, grad_e


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
        if phi.dim() == 1:
            grad_d, grad_e = tridiag_eigvec_adjoint(
                phi,
                d_vals,
                e_vals,
                grad_phi,
                tau=ctx.tau,
                eps=ctx.eps,
            )
        else:
            grad_d, grad_e = tridiag_eigvec_adjoint_batch(
                phi,
                d_vals,
                e_vals,
                grad_phi,
                tau=ctx.tau,
                eps=ctx.eps,
            )
        return None, grad_d, grad_e, None, None


class TridiagEigvecClusterAdjointFn(torch.autograd.Function):
    @staticmethod
    def forward(phi_cluster, d_vals, e_vals, eps):
        return phi_cluster.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi_cluster, d_vals, e_vals, eps = inputs
        ctx.eps = eps
        ctx.save_for_backward(phi_cluster, d_vals, e_vals)

    @staticmethod
    def backward(ctx, grad_phi_cluster):
        phi_cluster, d_vals, e_vals = ctx.saved_tensors
        grad_d, grad_e = tridiag_eigvec_cluster_adjoint(
            phi_cluster,
            d_vals,
            e_vals,
            grad_phi_cluster,
            eps=ctx.eps,
        )
        return None, grad_d, grad_e, None


class TridiagEigvecVaryingBatchAdjointFn(torch.autograd.Function):
    @staticmethod
    def forward(phi_batch, d_batch, e_batch, tau, eps, sigmas=None):
        return phi_batch.detach().clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        phi_batch, d_batch, e_batch, tau, eps, sigmas = inputs
        ctx.tau = tau
        ctx.eps = eps
        if sigmas is not None:
            ctx.save_for_backward(phi_batch, d_batch, e_batch, sigmas)
            ctx.has_sigmas = True
        else:
            ctx.save_for_backward(phi_batch, d_batch, e_batch)
            ctx.has_sigmas = False

    @staticmethod
    def backward(ctx, grad_phi_batch):
        if ctx.has_sigmas:
            phi_batch, d_batch, e_batch, sigmas_saved = ctx.saved_tensors
        else:
            phi_batch, d_batch, e_batch = ctx.saved_tensors
            sigmas_saved = None
        M, N = phi_batch.shape
        reg = max(ctx.eps * ctx.eps, 1e-14)

        # Batched: compute all M projections and tridiag solves together
        if phi_batch.is_complex():
            phi_sq_norm = (phi_batch.conj() * phi_batch).sum(dim=1).real  # [M]
        else:
            phi_sq_norm = (phi_batch * phi_batch).sum(dim=1)
        phi_hat = phi_batch / phi_sq_norm.sqrt().clamp(min=1e-30).unsqueeze(1)  # [M, N]

        # Eigenvalue estimates
        if sigmas_saved is not None:
            sigmas = sigmas_saved  # [M]
        else:
            Tphi = d_batch * phi_hat
            Tphi[:, :-1] += phi_hat[:, 1:] * e_batch
            Tphi[:, 1:] += phi_hat[:, :-1] * e_batch
            if phi_hat.is_complex():
                sigmas = (phi_hat.conj() * Tphi).sum(dim=1).real
            else:
                sigmas = (phi_hat * Tphi).sum(dim=1)

        # Project gradients orthogonal to phi_hat
        if phi_hat.is_complex():
            coeffs = (phi_hat.conj() * grad_phi_batch).sum(dim=1, keepdim=True)
        else:
            coeffs = (phi_hat * grad_phi_batch).sum(dim=1, keepdim=True)
        g_orth = grad_phi_batch - coeffs * phi_hat  # [M, N]

        # Batched tridiag solve: (T - sigma*I + reg) lam = g_orth
        shifted_d = d_batch - sigmas.to(d_batch.dtype).unsqueeze(1) + reg  # [M, N]
        if e_batch.dim() == 2 and M > 1 and torch.allclose(e_batch[0], e_batch[1]):
            # Shared off-diagonal — use batched C++ solve
            lam = solve_tridiag_batch(e_batch[0], shifted_d, e_batch[0], g_orth)
        else:
            # Per-mode off-diagonals — solve individually
            lam = torch.zeros_like(g_orth)
            for m in range(M):
                e_m = e_batch[m] if e_batch.dim() == 2 else e_batch
                lam[m] = solve_tridiag(e_m, shifted_d[m], e_m, g_orth[m])

        # Re-project to remove residual null-space component
        if phi_hat.is_complex():
            coeffs2 = (phi_hat.conj() * lam).sum(dim=1, keepdim=True)
        else:
            coeffs2 = (phi_hat * lam).sum(dim=1, keepdim=True)
        lam = lam - coeffs2 * phi_hat

        grad_d_batch = -lam * phi_batch  # [M, N]
        grad_e_batch = -lam[:, :-1] * phi_batch[:, 1:] - lam[:, 1:] * phi_batch[:, :-1]  # [M, N-1]

        return None, grad_d_batch, grad_e_batch, None, None, None


def tridiag_eigvec_reattach(
    phi: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
) -> torch.Tensor:
    return TridiagEigvecAdjointFn.apply(phi, d_vals, e_vals, tau, eps)


def tridiag_eigvec_cluster_reattach(
    phi_cluster: torch.Tensor,
    d_vals: torch.Tensor,
    e_vals: torch.Tensor,
    *,
    eps: float = 1e-10,
) -> torch.Tensor:
    return TridiagEigvecClusterAdjointFn.apply(phi_cluster, d_vals, e_vals, eps)


def tridiag_eigvec_reattach_varying_batch(
    phi_batch: torch.Tensor,
    d_batch: torch.Tensor,
    e_batch: torch.Tensor,
    *,
    tau: float = 1e-8,
    eps: float = 1e-10,
    sigmas: torch.Tensor | None = None,
) -> torch.Tensor:
    return TridiagEigvecVaryingBatchAdjointFn.apply(phi_batch, d_batch, e_batch, tau, eps, sigmas)


__all__ = [
    "EigvecReattachFn",
    "eigvec_reattach",
    "eigvec_degpert",
    "tridiag_eigvec_adjoint",
    "tridiag_eigvec_adjoint_batch",
    "tridiag_eigvec_cluster_adjoint",
    "tridiag_eigvec_reattach",
    "tridiag_eigvec_cluster_reattach",
    "tridiag_eigvec_reattach_varying_batch",
]
