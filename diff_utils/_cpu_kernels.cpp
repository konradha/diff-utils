#include <torch/extension.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#if defined(__clang__) || defined(__GNUC__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace {

template <typename scalar_t>
inline scalar_t maybe_conj(scalar_t v, bool complex_case) {
  if constexpr (c10::is_complex<scalar_t>::value)
    return complex_case ? std::conj(v) : v;
  else
    return v;
}

template <typename scalar_t>
inline void factorize_inplace(scalar_t *RESTRICT p, int64_t bw, int64_t n,
                              int64_t kl, int64_t ku) {
  if (kl == 1 && ku == 1 && bw >= 3 && n >= 2) {
    scalar_t *RESTRICT sub = p;
    scalar_t *RESTRICT diag = p + n;
    scalar_t *RESTRICT sup = p + 2 * n;
    for (int64_t k = 0; k < n - 1; ++k) {
      scalar_t factor = sub[k + 1] / diag[k];
      sub[k + 1] = factor;
      diag[k + 1] = diag[k + 1] - factor * sup[k];
    }
    return;
  }

  for (int64_t k = 0; k < n; ++k) {
    scalar_t pivot = p[kl * n + k];
    int64_t i_hi = std::min<int64_t>(n - 1, k + kl);
    int64_t d_hi = std::min<int64_t>(ku, n - 1 - k);

    for (int64_t i = k + 1; i <= i_hi; ++i) {
      int64_t li = kl + (k - i);
      scalar_t factor = p[li * n + i] / pivot;
      p[li * n + i] = factor;

      for (int64_t d = 1; d <= d_hi; ++d) {
        int64_t dij = li + d;
        if (dij >= 0 && dij < bw) {
          p[dij * n + i] = p[dij * n + i] - factor * p[(kl + d) * n + k];
        }
      }
    }
  }
}

torch::Tensor csr_values_to_band(torch::Tensor crow, torch::Tensor col,
                                 torch::Tensor values, int64_t kl, int64_t ku) {
  TORCH_CHECK(crow.device().is_cpu(), "crow must be on CPU");
  TORCH_CHECK(col.device().is_cpu(), "col must be on CPU");
  TORCH_CHECK(values.device().is_cpu(), "values must be on CPU");
  TORCH_CHECK(crow.dtype() == torch::kInt64, "crow must be int64");
  TORCH_CHECK(col.dtype() == torch::kInt64, "col must be int64");
  TORCH_CHECK(crow.dim() == 1, "crow must be rank-1");
  TORCH_CHECK(col.dim() == 1, "col must be rank-1");
  TORCH_CHECK(values.dim() == 1, "values must be rank-1");

  const int64_t n = crow.size(0) - 1;
  const int64_t bw = kl + ku + 1;
  auto band = torch::zeros({bw, n}, values.options());

  const int64_t *RESTRICT crowp = crow.data_ptr<int64_t>();
  const int64_t *RESTRICT colp = col.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, values.scalar_type(),
      "csr_values_to_band", [&] {
        const scalar_t *RESTRICT vptr = values.data_ptr<scalar_t>();
        scalar_t *RESTRICT bptr = band.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
          int64_t rs = crowp[i];
          int64_t re = crowp[i + 1];
          for (int64_t p = rs; p < re; ++p) {
            int64_t j = colp[p];
            int64_t r = kl + (j - i);
            if (r >= 0 && r < bw) {
              bptr[r * n + i] = vptr[p];
            }
          }
        }
      });

  return band;
}

torch::Tensor csr_lu_factorize(torch::Tensor crow, torch::Tensor col,
                               torch::Tensor values, int64_t kl, int64_t ku) {
  TORCH_CHECK(crow.device().is_cpu(), "crow must be on CPU");
  TORCH_CHECK(col.device().is_cpu(), "col must be on CPU");
  TORCH_CHECK(values.device().is_cpu(), "values must be on CPU");
  TORCH_CHECK(crow.dtype() == torch::kInt64, "crow must be int64");
  TORCH_CHECK(col.dtype() == torch::kInt64, "col must be int64");
  TORCH_CHECK(crow.dim() == 1, "crow must be rank-1");
  TORCH_CHECK(col.dim() == 1, "col must be rank-1");
  TORCH_CHECK(values.dim() == 1, "values must be rank-1");

  const int64_t n = crow.size(0) - 1;
  const int64_t bw = kl + ku + 1;
  auto lu = torch::zeros({bw, n}, values.options());

  const int64_t *RESTRICT crowp = crow.data_ptr<int64_t>();
  const int64_t *RESTRICT colp = col.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, values.scalar_type(),
      "csr_lu_factorize", [&] {
        const scalar_t *RESTRICT vptr = values.data_ptr<scalar_t>();
        scalar_t *RESTRICT p = lu.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
          int64_t rs = crowp[i];
          int64_t re = crowp[i + 1];
          for (int64_t q = rs; q < re; ++q) {
            int64_t j = colp[q];
            int64_t r = kl + (j - i);
            if (r >= 0 && r < bw) {
              p[r * n + i] = vptr[q];
            }
          }
        }
        factorize_inplace<scalar_t>(p, bw, n, kl, ku);
      });

  return lu;
}

torch::Tensor lu_factorize(torch::Tensor band, int64_t kl, int64_t ku) {
  TORCH_CHECK(band.device().is_cpu(), "band must be on CPU");
  TORCH_CHECK(band.dim() == 2, "band must be rank-2");
  auto lu = band.contiguous().clone();

  const int64_t bw = lu.size(0);
  const int64_t n = lu.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, lu.scalar_type(), "lu_factorize",
      [&] {
        scalar_t *RESTRICT p = lu.data_ptr<scalar_t>();
        factorize_inplace<scalar_t>(p, bw, n, kl, ku);
      });

  return lu;
}

torch::Tensor lu_solve(torch::Tensor lu, torch::Tensor b, int64_t kl,
                       int64_t ku) {
  TORCH_CHECK(lu.device().is_cpu(), "lu must be on CPU");
  TORCH_CHECK(b.device().is_cpu(), "b must be on CPU");
  TORCH_CHECK(lu.dim() == 2, "lu must be rank-2");
  TORCH_CHECK(b.dim() == 3, "b must be rank-3");
  TORCH_CHECK(lu.scalar_type() == b.scalar_type(), "dtype mismatch");

  auto lu_c = lu.contiguous();
  auto b_c = b.contiguous();

  const int64_t bw = lu_c.size(0);
  const int64_t n = lu_c.size(1);
  const int64_t batch = b_c.size(0);
  const int64_t rhs = b_c.size(2);

  auto y = torch::empty_like(b_c);
  auto x = torch::empty_like(b_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, lu_c.scalar_type(), "lu_solve",
      [&] {
        const scalar_t *RESTRICT lptr = lu_c.data_ptr<scalar_t>();
        const scalar_t *RESTRICT bptr = b_c.data_ptr<scalar_t>();
        scalar_t *RESTRICT yptr = y.data_ptr<scalar_t>();
        scalar_t *RESTRICT xptr = x.data_ptr<scalar_t>();

        if (kl == 1 && ku == 1 && bw >= 3) {
          const scalar_t *RESTRICT sub = lptr;
          const scalar_t *RESTRICT diag = lptr + n;
          const scalar_t *RESTRICT sup = lptr + 2 * n;

          for (int64_t bb = 0; bb < batch; ++bb) {
            const scalar_t *RESTRICT b_batch = bptr + bb * n * rhs;
            scalar_t *RESTRICT y_batch = yptr + bb * n * rhs;
            scalar_t *RESTRICT x_batch = xptr + bb * n * rhs;

            {
              const scalar_t *RESTRICT b_row0 = b_batch;
              scalar_t *RESTRICT y_row0 = y_batch;
              for (int64_t r = 0; r < rhs; ++r) {
                y_row0[r] = b_row0[r];
              }
            }
            for (int64_t i = 1; i < n; ++i) {
              const scalar_t coeff = sub[i];
              const scalar_t *RESTRICT b_row = b_batch + i * rhs;
              const scalar_t *RESTRICT y_prev = y_batch + (i - 1) * rhs;
              scalar_t *RESTRICT y_row = y_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                y_row[r] = b_row[r] - coeff * y_prev[r];
              }
            }

            {
              const int64_t i = n - 1;
              const scalar_t d = diag[i];
              const scalar_t *RESTRICT y_row = y_batch + i * rhs;
              scalar_t *RESTRICT x_row = x_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                x_row[r] = y_row[r] / d;
              }
            }
            for (int64_t i = n - 2; i >= 0; --i) {
              const scalar_t d = diag[i];
              const scalar_t u = sup[i];
              const scalar_t *RESTRICT y_row = y_batch + i * rhs;
              const scalar_t *RESTRICT x_next = x_batch + (i + 1) * rhs;
              scalar_t *RESTRICT x_row = x_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                x_row[r] = (y_row[r] - u * x_next[r]) / d;
              }
            }
          }
          return;
        }

        for (int64_t bb = 0; bb < batch; ++bb) {
          const scalar_t *RESTRICT b_batch = bptr + bb * n * rhs;
          scalar_t *RESTRICT y_batch = yptr + bb * n * rhs;
          scalar_t *RESTRICT x_batch = xptr + bb * n * rhs;

          for (int64_t i = 0; i < n; ++i) {
            int64_t tmax = std::min<int64_t>(kl, i);
            const scalar_t *RESTRICT b_row = b_batch + i * rhs;
            scalar_t *RESTRICT y_row = y_batch + i * rhs;
            for (int64_t r = 0; r < rhs; ++r) {
              scalar_t acc = b_row[r];
              for (int64_t t = 1; t <= tmax; ++t) {
                scalar_t coeff = lptr[(kl - t) * n + i];
                const scalar_t *RESTRICT y_prev = y_batch + (i - t) * rhs;
                acc = acc - coeff * y_prev[r];
              }
              y_row[r] = acc;
            }
          }

          for (int64_t i = n - 1; i >= 0; --i) {
            int64_t tmax = std::min<int64_t>(ku, n - 1 - i);
            scalar_t diag = lptr[kl * n + i];
            const scalar_t *RESTRICT y_row = y_batch + i * rhs;
            scalar_t *RESTRICT x_row = x_batch + i * rhs;
            for (int64_t r = 0; r < rhs; ++r) {
              scalar_t acc = y_row[r];
              for (int64_t t = 1; t <= tmax; ++t) {
                scalar_t coeff = lptr[(kl + t) * n + i];
                const scalar_t *RESTRICT x_next = x_batch + (i + t) * rhs;
                acc = acc - coeff * x_next[r];
              }
              x_row[r] = acc / diag;
            }
          }
        }
      });

  return x;
}

torch::Tensor lu_solve_adjoint(torch::Tensor lu, torch::Tensor g, int64_t kl,
                               int64_t ku, bool complex_case) {
  TORCH_CHECK(lu.device().is_cpu(), "lu must be on CPU");
  TORCH_CHECK(g.device().is_cpu(), "g must be on CPU");
  TORCH_CHECK(lu.dim() == 2, "lu must be rank-2");
  TORCH_CHECK(g.dim() == 3, "g must be rank-3");
  TORCH_CHECK(lu.scalar_type() == g.scalar_type(), "dtype mismatch");

  auto lu_c = lu.contiguous();
  auto g_c = g.contiguous();

  const int64_t bw = lu_c.size(0);
  const int64_t n = lu_c.size(1);
  const int64_t batch = g_c.size(0);
  const int64_t rhs = g_c.size(2);

  auto y = torch::empty_like(g_c);
  auto out = torch::empty_like(g_c);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, lu_c.scalar_type(),
      "lu_solve_adjoint", [&] {
        const scalar_t *RESTRICT lptr = lu_c.data_ptr<scalar_t>();
        const scalar_t *RESTRICT gptr = g_c.data_ptr<scalar_t>();
        scalar_t *RESTRICT yptr = y.data_ptr<scalar_t>();
        scalar_t *RESTRICT optr = out.data_ptr<scalar_t>();

        if (kl == 1 && ku == 1 && bw >= 3) {
          const scalar_t *RESTRICT sub = lptr;
          const scalar_t *RESTRICT diag = lptr + n;
          const scalar_t *RESTRICT sup = lptr + 2 * n;

          for (int64_t bb = 0; bb < batch; ++bb) {
            const scalar_t *RESTRICT g_batch = gptr + bb * n * rhs;
            scalar_t *RESTRICT y_batch = yptr + bb * n * rhs;
            scalar_t *RESTRICT o_batch = optr + bb * n * rhs;

            {
              const int64_t i = 0;
              scalar_t d = maybe_conj(diag[i], complex_case);
              const scalar_t *RESTRICT g_row = g_batch + i * rhs;
              scalar_t *RESTRICT y_row = y_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                y_row[r] = g_row[r] / d;
              }
            }
            for (int64_t i = 1; i < n; ++i) {
              scalar_t d = maybe_conj(diag[i], complex_case);
              scalar_t u_prev = maybe_conj(sup[i - 1], complex_case);
              const scalar_t *RESTRICT g_row = g_batch + i * rhs;
              const scalar_t *RESTRICT y_prev = y_batch + (i - 1) * rhs;
              scalar_t *RESTRICT y_row = y_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                y_row[r] = (g_row[r] - u_prev * y_prev[r]) / d;
              }
            }

            {
              const int64_t i = n - 1;
              const scalar_t *RESTRICT y_row = y_batch + i * rhs;
              scalar_t *RESTRICT o_row = o_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                o_row[r] = y_row[r];
              }
            }
            for (int64_t i = n - 2; i >= 0; --i) {
              scalar_t l_next = maybe_conj(sub[i + 1], complex_case);
              const scalar_t *RESTRICT y_row = y_batch + i * rhs;
              const scalar_t *RESTRICT o_next = o_batch + (i + 1) * rhs;
              scalar_t *RESTRICT o_row = o_batch + i * rhs;
              for (int64_t r = 0; r < rhs; ++r) {
                o_row[r] = y_row[r] - l_next * o_next[r];
              }
            }
          }
          return;
        }

        for (int64_t bb = 0; bb < batch; ++bb) {
          const scalar_t *RESTRICT g_batch = gptr + bb * n * rhs;
          scalar_t *RESTRICT y_batch = yptr + bb * n * rhs;
          scalar_t *RESTRICT o_batch = optr + bb * n * rhs;

          for (int64_t i = 0; i < n; ++i) {
            int64_t tmax = std::min<int64_t>(ku, i);
            scalar_t diag_c = maybe_conj(lptr[kl * n + i], complex_case);
            const scalar_t *RESTRICT g_row = g_batch + i * rhs;
            scalar_t *RESTRICT y_row = y_batch + i * rhs;
            for (int64_t r = 0; r < rhs; ++r) {
              scalar_t acc = g_row[r];
              for (int64_t t = 1; t <= tmax; ++t) {
                scalar_t coeff =
                    maybe_conj(lptr[(kl + t) * n + (i - t)], complex_case);
                const scalar_t *RESTRICT y_prev = y_batch + (i - t) * rhs;
                acc = acc - coeff * y_prev[r];
              }
              y_row[r] = acc / diag_c;
            }
          }

          for (int64_t i = n - 1; i >= 0; --i) {
            int64_t tmax = std::min<int64_t>(kl, n - 1 - i);
            const scalar_t *RESTRICT y_row = y_batch + i * rhs;
            scalar_t *RESTRICT o_row = o_batch + i * rhs;
            for (int64_t r = 0; r < rhs; ++r) {
              scalar_t acc = y_row[r];
              for (int64_t t = 1; t <= tmax; ++t) {
                scalar_t coeff =
                    maybe_conj(lptr[(kl - t) * n + (i + t)], complex_case);
                const scalar_t *RESTRICT o_next = o_batch + (i + t) * rhs;
                acc = acc - coeff * o_next[r];
              }
              o_row[r] = acc;
            }
          }
        }
      });

  return out;
}

// three-term recurrence p2 = (h2k2[m] - B1[j]) * p1 - p0
// sweeps j from loc_end down to loc_start.
// B1: [N], h2k2: [M], p1_init: [M], p2_init: [M]
// POST: f_num[M], g_val[M], p_history[M, sweep_len] (p1 for each step)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
acoustic_recurrence_fwd(torch::Tensor B1, torch::Tensor h2k2, int64_t loc_start,
                        int64_t loc_end, torch::Tensor p1_init,
                        torch::Tensor p2_init) {
  TORCH_CHECK(B1.device().is_cpu(), "B1 must be on CPU");
  TORCH_CHECK(h2k2.device().is_cpu(), "h2k2 must be on CPU");
  TORCH_CHECK(B1.dim() == 1, "B1 must be 1D");
  TORCH_CHECK(h2k2.dim() == 1, "h2k2 must be 1D");
  TORCH_CHECK(p1_init.dim() == 1 && p2_init.dim() == 1, "p_init must be 1D");
  TORCH_CHECK(loc_end >= loc_start, "loc_end must be >= loc_start");

  const int64_t M = h2k2.size(0);
  const int64_t sweep_len = loc_end - loc_start + 1;

  auto f_num = torch::empty({M}, h2k2.options());
  auto g_val = torch::empty({M}, h2k2.options());
  auto p_history = torch::empty({M, sweep_len}, h2k2.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, h2k2.scalar_type(),
      "acoustic_recurrence_fwd", [&] {
        const scalar_t *RESTRICT b1 = B1.data_ptr<scalar_t>();
        const scalar_t *RESTRICT hk = h2k2.data_ptr<scalar_t>();
        const scalar_t *RESTRICT p1i = p1_init.data_ptr<scalar_t>();
        const scalar_t *RESTRICT p2i = p2_init.data_ptr<scalar_t>();
        scalar_t *RESTRICT fn = f_num.data_ptr<scalar_t>();
        scalar_t *RESTRICT gv = g_val.data_ptr<scalar_t>();
        scalar_t *RESTRICT ph = p_history.data_ptr<scalar_t>();

        for (int64_t m = 0; m < M; ++m) {
            scalar_t p0 = p1i[m];
            scalar_t p1 = p1i[m];
            scalar_t p2 = p2i[m];
            scalar_t hk_m = hk[m];

            for (int64_t s = 0; s < sweep_len; ++s) {
              int64_t jj = loc_end - s;
              p0 = p1;
              p1 = p2;
              p2 = (hk_m - b1[jj]) * p1 - p0;
              ph[m * sweep_len + s] = p1;
            }
            fn[m] = -(p2 - p0);
            gv[m] = -p1;
        }
      });

  return std::make_tuple(f_num, g_val, p_history);
}

// adjoint recurrence
// grad_f_num[M], grad_g_val[M] -> grad_B1[N], grad_h2k2[M],
// grad_p1_init[M], grad_p2_init[M]
//  grad_z* = conj(J^T @ conj(grad_w))
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
acoustic_recurrence_bwd(torch::Tensor grad_f_num, torch::Tensor grad_g_val,
                        torch::Tensor B1, torch::Tensor h2k2,
                        torch::Tensor p_history, int64_t loc_start,
                        int64_t loc_end, torch::Tensor p1_init,
                        torch::Tensor p2_init, bool complex_case) {
  const int64_t M = h2k2.size(0);
  const int64_t N = B1.size(0);
  const int64_t sweep_len = loc_end - loc_start + 1;

  auto grad_B1 = torch::zeros({N}, B1.options());
  auto grad_h2k2 = torch::zeros({M}, h2k2.options());
  auto grad_p1_init = torch::zeros({M}, h2k2.options());
  auto grad_p2_init = torch::zeros({M}, h2k2.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, h2k2.scalar_type(),
      "acoustic_recurrence_bwd", [&] {
        const scalar_t *RESTRICT b1 = B1.data_ptr<scalar_t>();
        const scalar_t *RESTRICT hk = h2k2.data_ptr<scalar_t>();
        const scalar_t *RESTRICT ph = p_history.data_ptr<scalar_t>();
        const scalar_t *RESTRICT gfn = grad_f_num.data_ptr<scalar_t>();
        const scalar_t *RESTRICT ggv = grad_g_val.data_ptr<scalar_t>();
        scalar_t *RESTRICT gb1 = grad_B1.data_ptr<scalar_t>();
        scalar_t *RESTRICT ghk = grad_h2k2.data_ptr<scalar_t>();
        scalar_t *RESTRICT gp1i = grad_p1_init.data_ptr<scalar_t>();
        scalar_t *RESTRICT gp2i = grad_p2_init.data_ptr<scalar_t>();

        for (int64_t m = 0; m < M; ++m) {
          scalar_t hk_m = maybe_conj(hk[m], complex_case);

          scalar_t d_p2 = -gfn[m];
          scalar_t d_p1 = -ggv[m];
          scalar_t d_p0 = gfn[m];

          for (int64_t s = sweep_len - 1; s >= 0; --s) {
            int64_t jj = loc_end - s;
            scalar_t p1_at_step_conj =
                maybe_conj(ph[m * sweep_len + s], complex_case);

            scalar_t coeff = hk_m - maybe_conj(b1[jj], complex_case);
            scalar_t d_p1_from_p2 = d_p2 * coeff;
            scalar_t d_p0_from_p2 = -d_p2;

            gb1[jj] += -d_p2 * p1_at_step_conj;
            ghk[m] += d_p2 * p1_at_step_conj;

            scalar_t new_d_p2 = d_p1 + d_p1_from_p2;
            scalar_t new_d_p1 = d_p0 + d_p0_from_p2;
            d_p2 = new_d_p2;
            d_p1 = new_d_p1;
            d_p0 = scalar_t(0);
          }

          gp1i[m] = d_p1 + d_p0;
          gp2i[m] = d_p2;
        }
      });

  return std::make_tuple(grad_B1, grad_h2k2, grad_p1_init, grad_p2_init);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
searchsorted_lerp_fwd(torch::Tensor z_knots, torch::Tensor values,
                      torch::Tensor z_query) {
  TORCH_CHECK(z_knots.device().is_cpu(), "z_knots must be on CPU");
  TORCH_CHECK(z_knots.dim() == 1, "z_knots must be 1D");
  TORCH_CHECK(values.dim() == 1, "values must be 1D");
  TORCH_CHECK(z_query.dim() == 1, "z_query must be 1D");

  const int64_t n = z_knots.size(0);
  const int64_t m = z_query.size(0);

  auto out = torch::empty({m}, values.options());
  auto idx = torch::empty({m}, torch::kInt64);
  auto weights = torch::empty({m}, torch::kFloat64);

  const double *RESTRICT zk = z_knots.data_ptr<double>();
  const double *RESTRICT zq = z_query.data_ptr<double>();
  int64_t *RESTRICT idxp = idx.data_ptr<int64_t>();
  double *RESTRICT wp = weights.data_ptr<double>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, values.scalar_type(),
      "searchsorted_lerp_fwd", [&] {
        const scalar_t *RESTRICT vp = values.data_ptr<scalar_t>();
        scalar_t *RESTRICT op = out.data_ptr<scalar_t>();

        for (int64_t i = 0; i < m; ++i) {
          double q = zq[i];
          // clamp to [0, n-2]
          int64_t lo = 0, hi = n - 2;
          while (lo < hi) {
            int64_t mid = (lo + hi + 1) / 2;
            if (zk[mid] <= q)
              lo = mid;
            else
              hi = mid - 1;
          }
          if (lo < 0)
            lo = 0;
          if (lo > n - 2)
            lo = n - 2;

          double dz = zk[lo + 1] - zk[lo];
          double w = (dz == 0.0) ? 0.0 : (q - zk[lo]) / dz;
          if (w < 0.0)
            w = 0.0;
          if (w > 1.0)
            w = 1.0;

          idxp[i] = lo;
          wp[i] = w;
          op[i] = vp[lo] * scalar_t(1.0 - w) + vp[lo + 1] * scalar_t(w);
        }
      });

  return std::make_tuple(out, idx, weights);
}

torch::Tensor searchsorted_lerp_bwd(torch::Tensor grad_out, torch::Tensor idx,
                                    torch::Tensor weights, int64_t n_knots) {
  const int64_t m = grad_out.size(0);
  auto grad_values = torch::zeros({n_knots}, grad_out.options());

  const int64_t *RESTRICT idxp = idx.data_ptr<int64_t>();
  const double *RESTRICT wp = weights.data_ptr<double>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, grad_out.scalar_type(),
      "searchsorted_lerp_bwd", [&] {
        const scalar_t *RESTRICT gop = grad_out.data_ptr<scalar_t>();
        scalar_t *RESTRICT gvp = grad_values.data_ptr<scalar_t>();

        for (int64_t i = 0; i < m; ++i) {
          int64_t lo = idxp[i];
          double w = wp[i];
          gvp[lo] += gop[i] * scalar_t(1.0 - w);
          gvp[lo + 1] += gop[i] * scalar_t(w);
        }
      });

  return grad_values;
}

torch::Tensor solve_tridiag(torch::Tensor dl, torch::Tensor d, torch::Tensor du,
                            torch::Tensor b) {
  TORCH_CHECK(dl.device().is_cpu() && d.device().is_cpu() &&
                  du.device().is_cpu() && b.device().is_cpu(),
              "all inputs must be on CPU");
  const int64_t N = d.size(0);
  TORCH_CHECK(dl.size(0) == N - 1, "dl must have N-1 elements");
  TORCH_CHECK(du.size(0) == N - 1, "du must have N-1 elements");
  TORCH_CHECK(b.size(0) == N, "b must have N elements");

  auto d_w = d.contiguous().clone();
  auto x = b.contiguous().clone();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, d.scalar_type(), "solve_tridiag",
      [&] {
        scalar_t *RESTRICT dw = d_w.data_ptr<scalar_t>();
        scalar_t *RESTRICT xp = x.data_ptr<scalar_t>();
        const scalar_t *RESTRICT dlp = dl.data_ptr<scalar_t>();
        const scalar_t *RESTRICT dup = du.data_ptr<scalar_t>();

        for (int64_t i = 1; i < N; ++i) {
          scalar_t w = dlp[i - 1] / dw[i - 1];
          dw[i] -= w * dup[i - 1];
          xp[i] -= w * xp[i - 1];
        }
        xp[N - 1] /= dw[N - 1];
        for (int64_t i = N - 2; i >= 0; --i) {
          xp[i] = (xp[i] - dup[i] * xp[i + 1]) / dw[i];
        }
      });

  return x;
}

torch::Tensor solve_tridiag_batch(torch::Tensor dl, torch::Tensor d_batch,
                                  torch::Tensor du, torch::Tensor b_batch) {
  TORCH_CHECK(d_batch.dim() == 2 && b_batch.dim() == 2, "d_batch/b_batch 2D");
  const int64_t M = d_batch.size(0);
  const int64_t N = d_batch.size(1);
  TORCH_CHECK(dl.size(0) == N - 1 && du.size(0) == N - 1, "dl/du size");
  TORCH_CHECK(b_batch.size(0) == M && b_batch.size(1) == N, "b_batch size");

  auto d_w = d_batch.contiguous().clone();
  auto x = b_batch.contiguous().clone();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, d_batch.scalar_type(),
      "solve_tridiag_batch", [&] {
        scalar_t *RESTRICT dw = d_w.data_ptr<scalar_t>();
        scalar_t *RESTRICT xp = x.data_ptr<scalar_t>();
        const scalar_t *RESTRICT dlp = dl.data_ptr<scalar_t>();
        const scalar_t *RESTRICT dup = du.data_ptr<scalar_t>();

        for (int64_t m = 0; m < M; ++m) {
            scalar_t *RESTRICT dm = dw + m * N;
            scalar_t *RESTRICT xm = xp + m * N;
            for (int64_t i = 1; i < N; ++i) {
              scalar_t w = dlp[i - 1] / dm[i - 1];
              dm[i] -= w * dup[i - 1];
              xm[i] -= w * xm[i - 1];
            }
            xm[N - 1] /= dm[N - 1];
            for (int64_t i = N - 2; i >= 0; --i) {
              xm[i] = (xm[i] - dup[i] * xm[i + 1]) / dm[i];
            }
        }
      });

  return x;
}

std::tuple<double, double, int64_t>
acoustic_recurrence_scalar_counted(torch::Tensor B1, double h2k2,
                                   int64_t loc_start, int64_t loc_end,
                                   double p1_init, double p2_init) {
  TORCH_CHECK(B1.device().is_cpu() && B1.dim() == 1, "B1 must be 1D CPU");
  TORCH_CHECK(loc_end >= loc_start, "loc_end >= loc_start");

  const int64_t sweep_len = loc_end - loc_start + 1;
  const double *RESTRICT b1 = B1.data_ptr<double>();

  double p0 = p1_init;
  double p1 = p1_init;
  double p2 = p2_init;
  int64_t mode_count = 0;

  for (int64_t s = 0; s < sweep_len; ++s) {
    int64_t jj = loc_end - s;
    p0 = p1;
    p1 = p2;
    p2 = (h2k2 - b1[jj]) * p1 - p0;
    if (p0 * p1 <= 0.0 && p0 != 0.0) {
      ++mode_count;
    }
  }

  double f_num = -(p2 - p0);
  double g_val = -p1;
  return std::make_tuple(f_num, g_val, mode_count);
}

struct AcousticBC {
  //  0=vacuum, 1=rigid, 2=acoustic halfspace, 3=elastic halfspace
  int type;
  double cp, cs, rho;
};

struct DispResult {
  double Delta;
  int64_t iPower;
  int64_t mode_count;
};

static void bc_impedance(double x, double omega2, const AcousticBC &bc,
                         bool is_top, double &f, double &g, int64_t &iPower,
                         int64_t &mc) {
  iPower = 0;
  mc = 0;
  if (bc.type == 1 || bc.rho >= 1e10) {
    f = 0.0;
    g = 1.0;
    return;
  }
  if (bc.type == 0 || bc.rho == 0.0) {
    f = 1.0;
    g = 0.0;
    return;
  }
  if (bc.cs > 0.0) {
    double gammaS2 = x - omega2 / (bc.cs * bc.cs);
    double gammaP2 = x - omega2 / (bc.cp * bc.cp);
    double gammaS = gammaS2 >= 0 ? std::sqrt(gammaS2) : 0.0;
    double gammaP = gammaP2 >= 0 ? std::sqrt(gammaP2) : 0.0;
    double mu = bc.rho * bc.cs * bc.cs;
    f = omega2 * gammaP * (x - gammaS2);
    g = ((gammaS2 + x) * (gammaS2 + x) - 4.0 * gammaS * gammaP * x) * mu;
    if (g > 0.0)
      mc = 1;
  } else { // acoustic halfspace
    double gamma2 = x - omega2 / (bc.cp * bc.cp);
    f = gamma2 >= 0 ? std::sqrt(gamma2) : 0.0;
    g = bc.rho;
  }
  if (is_top)
    g = -g;
}

static const double FLOOR_V = 1e-50;
static const double ROOF_V = 1e50;
static const int64_t IPOWER_R_V = 50;
static const int64_t IPOWER_F_V = -50;

static DispResult dispersion_eval(
    double x, const double *b1, int64_t n_layers, const int64_t *layer_loc,
    const int64_t *layer_n, const double *layer_h, const double *layer_rho,
    double omega2, const AcousticBC &bc_bot, const AcousticBC &bc_top,
    bool count_modes, int64_t mode, const double *prev_eigenvalues) {

  double f, g;
  int64_t iPower = 0, mc_bc = 0;
  bc_impedance(x, omega2, bc_bot, false, f, g, iPower, mc_bc);
  int64_t mode_count = mc_bc;

  for (int64_t li = n_layers - 1; li >= 0; --li) {
    double h = layer_h[li];
    double h2k2 = h * h * x;
    double rho = layer_rho[li];
    int64_t loc_s = layer_loc[li];
    int64_t loc_e = loc_s + layer_n[li] - 1;

    double p1v = -2.0 * g;
    double p2v = (b1[loc_e] - h2k2) * g - 2.0 * h * f * rho;
    double p0v = p1v;

    int64_t sweep_len = loc_e - 1 - loc_s + 1;
    for (int64_t s = 0; s < sweep_len; ++s) {
      int64_t jj = (loc_e - 1) - s;
      p0v = p1v;
      p1v = p2v;
      p2v = (h2k2 - b1[jj]) * p1v - p0v;
      if (count_modes && p0v * p1v <= 0.0 && p0v != 0.0)
        ++mode_count;
    }

    double rho_top = layer_rho[li];
    f = -(p2v - p0v) / (2.0 * h * rho_top);
    g = -p1v;
  }

  double f_top, g_top;
  int64_t ip_top = 0, mc_top = 0;
  bc_impedance(x, omega2, bc_top, true, f_top, g_top, ip_top, mc_top);
  iPower += ip_top;
  mode_count += mc_top;

  double Delta = f * g_top - g * f_top;

  if (!count_modes && g * Delta > 0.0)
    mode_count += 1;

  if (mode > 0 && prev_eigenvalues != nullptr) {
    for (int64_t j = 0; j < mode; ++j) {
      double denom = x - prev_eigenvalues[j];
      if (std::abs(denom) > 0.0)
        Delta /= denom;
      if (!std::isfinite(Delta)) {
        Delta = ROOF_V;
        iPower = 0;
        break;
      }
      while (0.0 < std::abs(Delta) && std::abs(Delta) < FLOOR_V) {
        Delta *= ROOF_V;
        iPower -= IPOWER_R_V;
      }
      while (std::abs(Delta) > ROOF_V) {
        Delta *= FLOOR_V;
        iPower -= IPOWER_F_V;
      }
    }
  }

  return {Delta, iPower, mode_count};
}

static double zbrent(const double *b1, int64_t n_layers,
                     const int64_t *layer_loc, const int64_t *layer_n,
                     const double *layer_h, const double *layer_rho,
                     double omega2, const AcousticBC &bc_bot,
                     const AcousticBC &bc_top, double x1, double x2, double tol,
                     int64_t mode, const double *prev_eigenvalues) {

  auto r1 =
      dispersion_eval(x1, b1, n_layers, layer_loc, layer_n, layer_h, layer_rho,
                      omega2, bc_bot, bc_top, false, mode, prev_eigenvalues);
  auto r2 =
      dispersion_eval(x2, b1, n_layers, layer_loc, layer_n, layer_h, layer_rho,
                      omega2, bc_bot, bc_top, false, mode, prev_eigenvalues);

  int64_t ip_ref = std::max(r1.iPower, r2.iPower);
  double f1 = r1.Delta * std::pow(10.0, (double)(r1.iPower - ip_ref));
  double f2 = r2.Delta * std::pow(10.0, (double)(r2.iPower - ip_ref));

  if (f1 * f2 > 0.0)
    return 0.5 * (x1 + x2);

  double a = x1, b = x2, fa = f1, fb = f2;
  double c = b, fc = fb;
  double d = b - a, e = d;

  for (int iter = 0; iter < 200; ++iter) {
    if (fb * fc > 0.0) {
      c = a;
      fc = fa;
      d = e = b - a;
    }
    if (std::abs(fc) < std::abs(fb)) {
      a = b;
      fa = fb;
      b = c;
      fb = fc;
      c = a;
      fc = fa;
    }

    double tol1 = 2.0e-16 * std::abs(b) + 0.5 * tol;
    double xm = 0.5 * (c - b);

    if (std::abs(xm) <= tol1 || fb == 0.0)
      return b;

    if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
      double s = fb / fa;
      double p, q;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        double r = fb / fc;
        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0.0)
        q = -q;
      p = std::abs(p);
      if (2.0 * p <
          std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }

    a = b;
    fa = fb;
    if (std::abs(d) > tol1)
      b += d;
    else
      b += (xm >= 0.0 ? tol1 : -tol1);

    auto rb =
        dispersion_eval(b, b1, n_layers, layer_loc, layer_n, layer_h, layer_rho,
                        omega2, bc_bot, bc_top, false, mode, prev_eigenvalues);
    fb = rb.Delta * std::pow(10.0, (double)(rb.iPower - ip_ref));
  }
  return b;
}

static double secant(const double *b1, int64_t n_layers,
                     const int64_t *layer_loc, const int64_t *layer_n,
                     const double *layer_h, const double *layer_rho,
                     double omega2, const AcousticBC &bc_bot,
                     const AcousticBC &bc_top, double x_init, double tol,
                     int64_t mode, const double *prev_eigenvalues) {

  double x2 = x_init;
  double x1 = x2 + 10.0 * tol;
  auto r1 =
      dispersion_eval(x1, b1, n_layers, layer_loc, layer_n, layer_h, layer_rho,
                      omega2, bc_bot, bc_top, false, mode, prev_eigenvalues);
  double f1 = r1.Delta;
  int64_t ip1 = r1.iPower;

  for (int iter = 0; iter < 2000; ++iter) {
    double x0 = x1, f0 = f1;
    int64_t ip0 = ip1;
    x1 = x2;
    auto r = dispersion_eval(x1, b1, n_layers, layer_loc, layer_n, layer_h,
                             layer_rho, omega2, bc_bot, bc_top, false, mode,
                             prev_eigenvalues);
    f1 = r.Delta;
    ip1 = r.iPower;

    double c_num = f1 * (x1 - x0);
    double c_den = f1 - f0 * std::pow(10.0, (double)(ip0 - ip1));
    double shift;
    if (std::abs(c_num) >= std::abs(c_den * x1))
      shift = 0.1 * tol;
    else
      shift = c_num / c_den;

    x2 = x1 - shift;
    if (std::abs(x2 - x1) + std::abs(x2 - x0) < tol)
      return x2;
  }
  return x2;
}

std::tuple<torch::Tensor, int64_t> acoustic_solve1(
    torch::Tensor B1, torch::Tensor layer_loc_t, torch::Tensor layer_n_t,
    torch::Tensor layer_h_t, torch::Tensor layer_rho_t, double omega2,
    int64_t bc_bot_type, double bc_bot_cp, double bc_bot_cs, double bc_bot_rho,
    int64_t bc_top_type, double bc_top_cp, double bc_top_cs, double bc_top_rho,
    double x_min, double x_max, double precision, int64_t max_modes) {

  const double *b1 = B1.data_ptr<double>();
  const int64_t n_layers = layer_loc_t.size(0);
  const int64_t *ll = layer_loc_t.data_ptr<int64_t>();
  const int64_t *ln = layer_n_t.data_ptr<int64_t>();
  const double *lh = layer_h_t.data_ptr<double>();
  const double *lr = layer_rho_t.data_ptr<double>();

  AcousticBC bc_bot = {(int)bc_bot_type, bc_bot_cp, bc_bot_cs, bc_bot_rho};
  AcousticBC bc_top = {(int)bc_top_type, bc_top_cp, bc_top_cs, bc_top_rho};

  auto r_min = dispersion_eval(x_min, b1, n_layers, ll, ln, lh, lr, omega2,
                               bc_bot, bc_top, true, 0, nullptr);
  auto r_max = dispersion_eval(x_max, b1, n_layers, ll, ln, lh, lr, omega2,
                               bc_bot, bc_top, true, 0, nullptr);

  int64_t M = r_min.mode_count - r_max.mode_count;
  if (M <= 0 || M > max_modes)
    M = std::min(M, max_modes);
  if (M <= 0) {
    return std::make_tuple(torch::empty({0}, torch::kFloat64), (int64_t)0);
  }

  std::vector<double> x_l(M, x_min), x_r(M, x_max);
  std::unordered_map<int64_t, int64_t> mc_cache;

  auto get_mc = [&](double x) -> int64_t {
    int64_t key = (int64_t)(x * 1e15);
    auto it = mc_cache.find(key);
    if (it != mc_cache.end())
      return it->second;
    auto r = dispersion_eval(x, b1, n_layers, ll, ln, lh, lr, omega2, bc_bot,
                             bc_top, true, 0, nullptr);
    mc_cache[key] = r.mode_count;
    return r.mode_count;
  };

  int64_t mc_min_val = r_min.mode_count;

  for (int64_t iter = 0; iter < 50 * M; ++iter) {
    bool all_isolated = true;
    for (int64_t j = 0; j < M; ++j) {
      int64_t target = mc_min_val - j - 1;
      if (x_r[j] - x_l[j] < 1e-20 * std::abs(x_l[j]))
        continue;
      double x_mid = 0.5 * (x_l[j] + x_r[j]);
      int64_t mc_mid = get_mc(x_mid);
      if (mc_mid > target)
        x_l[j] = x_mid;
      else
        x_r[j] = x_mid;
      if (x_r[j] - x_l[j] > 1e-14 * std::abs(x_l[j]))
        all_isolated = false;
    }
    if (all_isolated)
      break;
  }

  auto eigenvalues = torch::empty({M}, torch::kFloat64);
  double *ev = eigenvalues.data_ptr<double>();

  for (int64_t mode = 0; mode < M; ++mode) {
    double eps = std::abs(x_r[mode]) * std::pow(10.0, 2.0 - precision);
    ev[mode] = zbrent(b1, n_layers, ll, ln, lh, lr, omega2, bc_bot, bc_top,
                      x_l[mode], x_r[mode], eps, mode, ev);
  }

  return std::make_tuple(eigenvalues, M);
}

std::tuple<torch::Tensor, int64_t>
acoustic_solve2(torch::Tensor B1, torch::Tensor layer_loc_t,
                torch::Tensor layer_n_t, torch::Tensor layer_h_t,
                torch::Tensor layer_rho_t, double omega2, int64_t bc_bot_type,
                double bc_bot_cp, double bc_bot_cs, double bc_bot_rho,
                int64_t bc_top_type, double bc_top_cp, double bc_top_cs,
                double bc_top_rho, torch::Tensor prev_eigenvalues, int64_t M,
                double precision, double c_high) {

  const double *b1 = B1.data_ptr<double>();
  const int64_t n_layers = layer_loc_t.size(0);
  const int64_t *ll = layer_loc_t.data_ptr<int64_t>();
  const int64_t *ln = layer_n_t.data_ptr<int64_t>();
  const double *lh = layer_h_t.data_ptr<double>();
  const double *lr = layer_rho_t.data_ptr<double>();

  AcousticBC bc_bot = {(int)bc_bot_type, bc_bot_cp, bc_bot_cs, bc_bot_rho};
  AcousticBC bc_top = {(int)bc_top_type, bc_top_cp, bc_top_cs, bc_top_rho};

  auto eigenvalues = torch::empty({M}, torch::kFloat64);
  double *ev = eigenvalues.data_ptr<double>();
  const double *prev_ev = prev_eigenvalues.data_ptr<double>();

  int64_t n_total = 0;
  for (int64_t i = 0; i < n_layers; ++i)
    n_total += ln[i];

  for (int64_t mode = 0; mode < M; ++mode) {
    double x_init = prev_ev[mode];
    double tol =
        std::abs(x_init) * (double)n_total * std::pow(10.0, 1.0 - precision);

    ev[mode] = secant(b1, n_layers, ll, ln, lh, lr, omega2, bc_bot, bc_top,
                      x_init, tol, mode, ev);

    if (omega2 / (c_high * c_high) > ev[mode]) {
      return std::make_tuple(eigenvalues.slice(0, 0, mode), mode);
    }
  }

  return std::make_tuple(eigenvalues, M);
}

// --- Batched inverse iteration (per-mode local buffer, cache-friendly) ---

torch::Tensor tridiag_inverse_iteration_batch_cpp(
    torch::Tensor d_batch,   // [M, N]
    torch::Tensor e,         // [N-1]
    int64_t n_iter
) {
  TORCH_CHECK(d_batch.dim() == 2, "d_batch must be 2D");
  TORCH_CHECK(e.dim() == 1, "e must be 1D");
  const int64_t M = d_batch.size(0);
  const int64_t N = d_batch.size(1);
  TORCH_CHECK(e.size(0) == N - 1, "e must have N-1 elements");

  if (N < 3) {
    double inv_sqrt_n = 1.0 / std::sqrt((double)N);
    return torch::full({M, N}, inv_sqrt_n, d_batch.options());
  }

  auto phi = torch::full({M, N}, 1e-10, d_batch.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kComplexFloat, at::kComplexDouble, d_batch.scalar_type(),
      "tridiag_inverse_iteration_batch_inner", [&] {
        const scalar_t *RESTRICT d_src = d_batch.data_ptr<scalar_t>();
        const scalar_t *RESTRICT ep = e.data_ptr<scalar_t>();
        scalar_t *RESTRICT pp = phi.data_ptr<scalar_t>();

        // Single pre-allocated work buffer, reused across modes
        std::vector<scalar_t> dw_local(N);
        for (int64_t m = 0; m < M; ++m) {
            const scalar_t *RESTRICT d_m = d_src + m * N;
            scalar_t *RESTRICT pm = pp + m * N;

            for (int64_t iter = 0; iter < n_iter; ++iter) {
              std::memcpy(dw_local.data(), d_m, N * sizeof(scalar_t));
              for (int64_t i = 1; i < N; ++i) {
                scalar_t w = ep[i - 1] / dw_local[i - 1];
                dw_local[i] -= w * ep[i - 1];
                pm[i] -= w * pm[i - 1];
              }
              pm[N - 1] /= dw_local[N - 1];
              for (int64_t i = N - 2; i >= 0; --i) {
                pm[i] = (pm[i] - ep[i] * pm[i + 1]) / dw_local[i];
              }
              double max_abs = 0.0;
              for (int64_t i = 0; i < N; ++i) {
                double a = std::abs(pm[i]);
                if (a > max_abs) max_abs = a;
              }
              if (max_abs > 0.0) {
                scalar_t inv = scalar_t(1.0 / max_abs);
                for (int64_t i = 0; i < N; ++i) pm[i] *= inv;
              }
            }
            double sq_sum = 0.0;
            for (int64_t i = 0; i < N; ++i) {
              double a = std::abs(pm[i]);
              sq_sum += a * a;
            }
            double norm = std::sqrt(sq_sum);
            if (norm > 0.0) {
              scalar_t inv_norm = scalar_t(1.0 / norm);
              for (int64_t i = 0; i < N; ++i) pm[i] *= inv_norm;
            }
        }
      });

  return phi;
}

// --- Fused assembly + inverse iteration (no d_batch allocation) ---

torch::Tensor fused_assembly_inverse_iteration_cpp(
    torch::Tensor base_d,       // [N] complex128 — shared base diagonal
    torch::Tensor h2_factor,    // [N] complex128 — shared h^2/(h*rho) factor
    torch::Tensor x_batch,      // [M] complex128 — eigenvalues
    torch::Tensor e,            // [N-1] complex128 — shared off-diagonal
    torch::Tensor bc_top_ratio, // [M] complex128 — f/g at top per mode
    torch::Tensor bc_bot_ratio, // [M] complex128 — f/g at bottom per mode
    torch::Tensor bc_top_vacuum,// [M] bool
    torch::Tensor bc_bot_vacuum,// [M] bool
    int64_t n_iter
) {
  using C = c10::complex<double>;
  const int64_t M = x_batch.size(0);
  const int64_t N = base_d.size(0);

  if (N < 3) {
    double inv_sqrt_n = 1.0 / std::sqrt((double)N);
    return torch::full({M, N}, inv_sqrt_n, base_d.options());
  }

  auto phi = torch::full({M, N}, C(1e-10), base_d.options());

  const C *RESTRICT bd = reinterpret_cast<const C*>(base_d.data_ptr());
  const C *RESTRICT hf = reinterpret_cast<const C*>(h2_factor.data_ptr());
  const C *RESTRICT xp = reinterpret_cast<const C*>(x_batch.data_ptr());
  const C *RESTRICT ep = reinterpret_cast<const C*>(e.data_ptr());
  const C *RESTRICT tr = reinterpret_cast<const C*>(bc_top_ratio.data_ptr());
  const C *RESTRICT br = reinterpret_cast<const C*>(bc_bot_ratio.data_ptr());
  const bool *RESTRICT tv = bc_top_vacuum.data_ptr<bool>();
  const bool *RESTRICT bv = bc_bot_vacuum.data_ptr<bool>();
  C *RESTRICT pp = reinterpret_cast<C*>(phi.data_ptr());

  {
    std::vector<C> dw(N);
    for (int64_t m = 0; m < M; ++m) {
      C x_m = xp[m];
      C *RESTRICT pm = pp + m * N;

      // Assemble diagonal: d[i] = base_d[i] - x_m * h2_factor[i]
      for (int64_t i = 0; i < N; ++i)
        dw[i] = bd[i] - x_m * hf[i];

      // BC injection
      if (tv[m]) { dw[0] = C(1.0); /* e[0] already 0 in e */ }
      else       { dw[0] = dw[0] / C(2.0) + tr[m]; }
      if (bv[m]) { dw[N-1] = C(1.0); }
      else       { dw[N-1] = dw[N-1] / C(2.0) - br[m]; }

      for (int64_t iter = 0; iter < n_iter; ++iter) {
        // Restore diagonal (assembly is cheap, avoids copy)
        if (iter > 0) {
          for (int64_t i = 0; i < N; ++i)
            dw[i] = bd[i] - x_m * hf[i];
          if (tv[m]) dw[0] = C(1.0); else dw[0] = dw[0]/C(2.0) + tr[m];
          if (bv[m]) dw[N-1] = C(1.0); else dw[N-1] = dw[N-1]/C(2.0) - br[m];
        }

        // Thomas forward elimination
        for (int64_t i = 1; i < N; ++i) {
          C w = ep[i-1] / dw[i-1];
          dw[i] -= w * ep[i-1];
          pm[i] -= w * pm[i-1];
        }
        pm[N-1] /= dw[N-1];
        for (int64_t i = N-2; i >= 0; --i)
          pm[i] = (pm[i] - ep[i] * pm[i+1]) / dw[i];

        // Normalize
        double mx = 0.0;
        for (int64_t i = 0; i < N; ++i) { double a = std::abs(pm[i]); if (a > mx) mx = a; }
        if (mx > 0.0) { C inv(1.0/mx); for (int64_t i = 0; i < N; ++i) pm[i] *= inv; }
      }

      // Final L2 norm
      double sq = 0.0;
      for (int64_t i = 0; i < N; ++i) { double a = std::abs(pm[i]); sq += a*a; }
      double nrm = std::sqrt(sq);
      if (nrm > 0.0) { C inv(1.0/nrm); for (int64_t i = 0; i < N; ++i) pm[i] *= inv; }
    }
  }
  return phi;
}

// --- Complex batched dispersion (no p_history allocation) ---

static inline void bc_impedance_complex(
    c10::complex<double> x, double omega2,
    double cp_r, double c_imag_p, double cs_r, double c_imag_s,
    double rho, int bc_type, bool is_top,
    c10::complex<double> branch_sign,
    c10::complex<double> &f, c10::complex<double> &g) {

  using C = c10::complex<double>;
  if (bc_type == 1 || rho >= 1e10) { f = C(0.0, 0.0); g = C(1.0, 0.0); }
  else if (bc_type == 0 || rho == 0.0) { f = C(1.0, 0.0); g = C(0.0, 0.0); }
  else if (cs_r > 0.0) {
    C cp_c(cp_r, c_imag_p), cs_c(cs_r, c_imag_s);
    C gammaS2 = x - omega2 / (cs_c * cs_c);
    C gammaP2 = x - omega2 / (cp_c * cp_c);
    C gammaS = std::sqrt(gammaS2);
    C gammaP = std::sqrt(gammaP2);
    C mu = rho * cs_c * cs_c;
    f = omega2 * gammaP * (x - gammaS2);
    g = ((gammaS2 + x) * (gammaS2 + x) - C(4.0) * gammaS * gammaP * x) * mu;
  } else {
    C cp_c(cp_r, c_imag_p);
    C gamma2 = x - omega2 / (cp_c * cp_c);
    f = branch_sign * std::sqrt(gamma2);
    g = C(rho, 0.0);
  }
  if (is_top) g = -g;
}

torch::Tensor dispersion_complex_batch_cpp(
    torch::Tensor x_batch,      // [M] complex128
    torch::Tensor B1,            // [N] float64 or complex128
    torch::Tensor layer_loc,     // [n_layers] int64
    torch::Tensor layer_n,       // [n_layers] int64
    torch::Tensor layer_h,       // [n_layers] float64
    torch::Tensor layer_rho,     // [n_layers] float64
    double omega2,
    // bottom BC params
    int64_t bc_bot_type, double bot_cp, double bot_c_imag_p,
    double bot_cs, double bot_c_imag_s, double bot_rho,
    // top BC params
    int64_t bc_top_type, double top_cp, double top_c_imag_p,
    double top_cs, double top_c_imag_s, double top_rho,
    // branch signs per mode
    torch::Tensor branch_signs   // [M] float64
) {
  using C = c10::complex<double>;
  const int64_t M = x_batch.size(0);
  const int64_t n_layers = layer_loc.size(0);

  auto delta_out = torch::empty({M}, torch::dtype(torch::kComplexDouble));
  C *RESTRICT delta_p = reinterpret_cast<C *>(delta_out.data_ptr());
  const C *RESTRICT xp = reinterpret_cast<const C *>(x_batch.data_ptr());
  const double *RESTRICT b1 = B1.data_ptr<double>();
  const int64_t *RESTRICT ll = layer_loc.data_ptr<int64_t>();
  const int64_t *RESTRICT ln = layer_n.data_ptr<int64_t>();
  const double *RESTRICT lh = layer_h.data_ptr<double>();
  const double *RESTRICT lr = layer_rho.data_ptr<double>();
  const double *RESTRICT bsigns = branch_signs.data_ptr<double>();

  for (int64_t m = 0; m < M; ++m) {
      C x = xp[m];
      C bsign(bsigns[m], 0.0);
      C f, g;
      bc_impedance_complex(x, omega2, bot_cp, bot_c_imag_p,
                           bot_cs, bot_c_imag_s, bot_rho,
                           bc_bot_type, false, bsign, f, g);

      for (int64_t li = n_layers - 1; li >= 0; --li) {
        double h = lh[li];
        C h2k2 = h * h * x;
        double rho = lr[li];
        int64_t loc_s = ll[li];
        int64_t loc_e = loc_s + ln[li] - 1;

        C p1 = C(-2.0) * g;
        C p2 = (C(b1[loc_e]) - h2k2) * g - C(2.0 * h) * f * C(rho);
        C p0 = p1;

        int64_t sweep_len = loc_e - 1 - loc_s + 1;
        for (int64_t s = 0; s < sweep_len; ++s) {
          int64_t jj = (loc_e - 1) - s;
          p0 = p1;
          p1 = p2;
          p2 = (h2k2 - C(b1[jj])) * p1 - p0;
        }
        f = -(p2 - p0) / C(2.0 * h * rho);
        g = -p1;
      }

      C f_top, g_top;
      bc_impedance_complex(x, omega2, top_cp, top_c_imag_p,
                           top_cs, top_c_imag_s, top_rho,
                           bc_top_type, true, C(1.0), f_top, g_top);
      delta_p[m] = f * g_top - g * f_top;
  }
  return delta_out;
}

// Single-x wrapper for scalar dispersion_complex (avoids tensor overhead)
c10::complex<double> dispersion_complex_scalar_cpp(
    c10::complex<double> x,
    torch::Tensor B1,
    torch::Tensor layer_loc,
    torch::Tensor layer_n,
    torch::Tensor layer_h,
    torch::Tensor layer_rho,
    double omega2,
    int64_t bc_bot_type, double bot_cp, double bot_c_imag_p,
    double bot_cs, double bot_c_imag_s, double bot_rho,
    int64_t bc_top_type, double top_cp, double top_c_imag_p,
    double top_cs, double top_c_imag_s, double top_rho,
    double branch_sign
) {
  using C = c10::complex<double>;
  const int64_t n_layers = layer_loc.size(0);
  const double *RESTRICT b1 = B1.data_ptr<double>();
  const int64_t *RESTRICT ll = layer_loc.data_ptr<int64_t>();
  const int64_t *RESTRICT ln = layer_n.data_ptr<int64_t>();
  const double *RESTRICT lh = layer_h.data_ptr<double>();
  const double *RESTRICT lr = layer_rho.data_ptr<double>();

  C bsign(branch_sign, 0.0);
  C f, g;
  bc_impedance_complex(x, omega2, bot_cp, bot_c_imag_p,
                       bot_cs, bot_c_imag_s, bot_rho,
                       bc_bot_type, false, bsign, f, g);

  for (int64_t li = n_layers - 1; li >= 0; --li) {
    double h = lh[li];
    C h2k2 = h * h * x;
    double rho = lr[li];
    int64_t loc_s = ll[li];
    int64_t loc_e = loc_s + ln[li] - 1;

    C p1 = C(-2.0) * g;
    C p2 = (C(b1[loc_e]) - h2k2) * g - C(2.0 * h) * f * C(rho);
    C p0 = p1;

    int64_t sweep_len = loc_e - 1 - loc_s + 1;
    for (int64_t s = 0; s < sweep_len; ++s) {
      int64_t jj = (loc_e - 1) - s;
      p0 = p1;
      p1 = p2;
      p2 = (h2k2 - C(b1[jj])) * p1 - p0;
    }
    f = -(p2 - p0) / C(2.0 * h * rho);
    g = -p1;
  }

  C f_top, g_top;
  bc_impedance_complex(x, omega2, top_cp, top_c_imag_p,
                       top_cs, top_c_imag_s, top_rho,
                       bc_top_type, true, C(1.0), f_top, g_top);
  return f * g_top - g * f_top;
}

// --- Inline dispersion for the refinement loop ---
struct DispEnv {
  const double *b1;
  int64_t n_layers;
  const int64_t *ll;
  const int64_t *ln;
  const double *lh;
  const double *lr;
  double omega2;
  int64_t bc_bot_type; double bot_cp, bot_ci, bot_cs, bot_csi, bot_rho;
  int64_t bc_top_type; double top_cp, top_ci, top_cs, top_csi, top_rho;
};

// Batched dispersion: evaluate disp(x[k], sign[k]) for k=0..K-1 in one
// mesh sweep. Reads B1 once; updates K modes per mesh point.
static void disp_eval_batch(
    const c10::complex<double> *x, const double *bsigns,
    int64_t K, const DispEnv &E, c10::complex<double> *out) {
  using C = c10::complex<double>;
  // Per-mode state: f[K], g[K]
  std::vector<C> f(K), g(K);
  for (int64_t k = 0; k < K; ++k) {
    C bs(bsigns[k], 0.0);
    bc_impedance_complex(x[k], E.omega2, E.bot_cp, E.bot_ci, E.bot_cs, E.bot_csi,
                         E.bot_rho, E.bc_bot_type, false, bs, f[k], g[k]);
  }
  for (int64_t li = E.n_layers - 1; li >= 0; --li) {
    double h = E.lh[li];
    double rho = E.lr[li];
    int64_t loc_s = E.ll[li], loc_e = loc_s + E.ln[li] - 1;
    int64_t sweep_len = loc_e - 1 - loc_s + 1;

    // Init p0, p1, p2 per mode
    std::vector<C> p0(K), p1(K), p2(K);
    for (int64_t k = 0; k < K; ++k) {
      C h2k2 = h * h * x[k];
      p1[k] = C(-2.0) * g[k];
      p2[k] = (C(E.b1[loc_e]) - h2k2) * g[k] - C(2.0*h) * f[k] * C(rho);
      p0[k] = p1[k];
    }

    // Sweep mesh: one B1 read per step, K mode updates
    for (int64_t s = 0; s < sweep_len; ++s) {
      double b1_val = E.b1[(loc_e-1) - s];
      for (int64_t k = 0; k < K; ++k) {
        C h2k2 = h * h * x[k];
        p0[k] = p1[k]; p1[k] = p2[k];
        p2[k] = (h2k2 - C(b1_val)) * p1[k] - p0[k];
      }
    }
    C inv_2hr(1.0 / (2.0 * h * rho));
    for (int64_t k = 0; k < K; ++k) {
      f[k] = -(p2[k] - p0[k]) * inv_2hr;
      g[k] = -p1[k];
    }
  }
  for (int64_t k = 0; k < K; ++k) {
    C ft, gt;
    bc_impedance_complex(x[k], E.omega2, E.top_cp, E.top_ci, E.top_cs, E.top_csi,
                         E.top_rho, E.bc_top_type, true, C(1.0), ft, gt);
    out[k] = f[k]*gt - g[k]*ft;
  }
}

static inline c10::complex<double> disp_eval(
    c10::complex<double> x, const DispEnv &E, double branch_sign) {
  using C = c10::complex<double>;
  C bsign(branch_sign, 0.0);
  C f, g;
  bc_impedance_complex(x, E.omega2, E.bot_cp, E.bot_ci, E.bot_cs, E.bot_csi,
                       E.bot_rho, E.bc_bot_type, false, bsign, f, g);
  for (int64_t li = E.n_layers - 1; li >= 0; --li) {
    double h = E.lh[li];
    C h2k2 = h * h * x;
    double rho = E.lr[li];
    int64_t loc_s = E.ll[li], loc_e = loc_s + E.ln[li] - 1;
    C p1 = C(-2.0)*g, p2 = (C(E.b1[loc_e])-h2k2)*g - C(2.0*h)*f*C(rho), p0 = p1;
    for (int64_t s = 0; s < loc_e-1-loc_s+1; ++s) {
      p0=p1; p1=p2; p2 = (h2k2-C(E.b1[(loc_e-1)-s]))*p1-p0;
    }
    f = -(p2-p0)/C(2.0*h*rho); g = -p1;
  }
  C ft, gt;
  bc_impedance_complex(x, E.omega2, E.top_cp, E.top_ci, E.top_cs, E.top_csi,
                       E.top_rho, E.bc_top_type, true, C(1.0), ft, gt);
  return f*gt - g*ft;
}

// Refine M roots: secant + polish + branch selection, all in C++
std::tuple<torch::Tensor, torch::Tensor> refine_roots_cpp(
    torch::Tensor x_init_t,    // [M] complex128 — initial guesses from real KRAKEN
    torch::Tensor k_init_t,    // [M] complex128 — initial k from real KRAKEN
    torch::Tensor B1,
    torch::Tensor layer_loc, torch::Tensor layer_n,
    torch::Tensor layer_h, torch::Tensor layer_rho,
    double omega2,
    int64_t bc_bot_type, double bot_cp, double bot_ci,
    double bot_cs, double bot_csi, double bot_rho,
    int64_t bc_top_type, double top_cp, double top_ci,
    double top_cs, double top_csi, double top_rho,
    double real_k_min, int64_t max_iter
) {
  using C = c10::complex<double>;
  const int64_t M = x_init_t.size(0);

  DispEnv E;
  E.b1 = B1.data_ptr<double>();
  E.n_layers = layer_loc.size(0);
  E.ll = layer_loc.data_ptr<int64_t>();
  E.ln = layer_n.data_ptr<int64_t>();
  E.lh = layer_h.data_ptr<double>();
  E.lr = layer_rho.data_ptr<double>();
  E.omega2 = omega2;
  E.bc_bot_type = bc_bot_type; E.bot_cp = bot_cp; E.bot_ci = bot_ci;
  E.bot_cs = bot_cs; E.bot_csi = bot_csi; E.bot_rho = bot_rho;
  E.bc_top_type = bc_top_type; E.top_cp = top_cp; E.top_ci = top_ci;
  E.top_cs = top_cs; E.top_csi = top_csi; E.top_rho = top_rho;

  const C *xi = reinterpret_cast<const C*>(x_init_t.data_ptr());
  const C *ki = reinterpret_cast<const C*>(k_init_t.data_ptr());

  auto x_out = torch::empty({M}, torch::dtype(torch::kComplexDouble));
  auto signs_out = torch::empty({M}, torch::dtype(torch::kDouble));
  C *xo = reinterpret_cast<C*>(x_out.data_ptr());
  double *so = signs_out.data_ptr<double>();

  // Determine branch signs for all modes (batch: 2 dispersion calls per mode)
  // Then refine each mode independently
  for (int64_t m = 0; m < M; ++m) {
      C x0 = xi[m];
      C k_init = ki[m];

      // Branch sign selection
      double resid_pos = std::abs(disp_eval(x0, E, 1.0));
      double resid_neg = std::abs(disp_eval(x0, E, -1.0));
      double default_sign = (resid_neg < resid_pos) ? -1.0 : 1.0;

      // Refine with both branches, pick best
      auto do_refine = [&](double bsign) -> C {
        C x_a = x0;
        double sc = std::max(std::abs(x_a), 1.0);
        C x_b = x_a * C(1.0+1e-8, 1e-8) + C(0.0, 1e-10*sc);
        C f_a = disp_eval(x_a, E, bsign);
        C f_b = disp_eval(x_b, E, bsign);
        C best = x_a;
        double best_r = std::abs(f_a);
        if (std::abs(f_b) < best_r) { best = x_b; best_r = std::abs(f_b); }

        for (int64_t it = 0; it < max_iter; ++it) {
          C denom = f_b - f_a;
          if (std::abs(denom) < 1e-20) break;
          C x_c = x_b - f_b * (x_b - x_a) / denom;
          if (!std::isfinite(x_c.real()) || !std::isfinite(x_c.imag())) break;
          if (x_c.real() <= 0.0) break;
          C f_c = disp_eval(x_c, E, bsign);
          if (std::abs(f_c) < best_r) { best = x_c; best_r = std::abs(f_c); }
          if (std::abs(x_c - x_b) <= 1e-12 * (1.0 + std::abs(x_c))) return best;
          x_a = x_b; f_a = f_b; x_b = x_c; f_b = f_c;
          if (std::abs(f_b) <= 1e-10 * sc) return best;
        }

        // Polish with Newton (8 iterations)
        C xp = best;
        double bp = best_r;
        for (int i = 0; i < 8; ++i) {
          double sloc = std::max(std::abs(xp), 1.0);
          C h(1e-8*sloc, 1e-8*sloc);
          C fc = disp_eval(xp, E, bsign);
          C fp = disp_eval(xp+h, E, bsign);
          C fm = disp_eval(xp-h, E, bsign);
          C dfdx = (fp-fm)/(C(2.0)*h);
          if (std::abs(dfdx) < 1e-20) break;
          C xn = xp - fc/dfdx;
          if (!std::isfinite(xn.real()) || !std::isfinite(xn.imag())) break;
          if (xn.real() <= 0.0 || std::sqrt(xn.real()) < real_k_min) break;
          C fn = disp_eval(xn, E, bsign);
          if (std::abs(fn) < bp) { xp = xn; bp = std::abs(fn); }
          else xp = xn;  // continue Newton anyway
        }
        if (bp < best_r) best = xp;
        return best;
      };

      C x_def = do_refine(default_sign);
      C x_alt = do_refine(-default_sign);

      // Candidate scoring: 4 candidates (default±conj, alt±conj)
      struct Cand { double sign; C x; C k; };
      auto make_k = [](C x) -> C {
        C k = std::sqrt(x); return k.real() < 0 ? -k : k;
      };
      Cand cands[4] = {
        {default_sign, x_def, make_k(x_def)},
        {default_sign, std::conj(x_def), make_k(std::conj(x_def))},
        {-default_sign, x_alt, make_k(x_alt)},
        {-default_sign, std::conj(x_alt), make_k(std::conj(x_alt))},
      };

      auto score = [&](const Cand &c) -> std::tuple<int,double,double> {
        double r = std::abs(disp_eval(c.x, E, c.sign));
        int pos_imag = c.k.imag() > 1e-12 ? 1 : 0;
        return {pos_imag, r, std::abs(c.k - k_init)};
      };

      int best_idx = 0;
      auto best_score = score(cands[0]);
      for (int i = 1; i < 4; ++i) {
        auto s = score(cands[i]);
        if (s < best_score) { best_score = s; best_idx = i; }
      }

      C k_best = cands[best_idx].k;
      C x_best = cands[best_idx].x;
      double sign_best = cands[best_idx].sign;

      // Guard: near real axis → keep perturbative
      if (std::abs(k_best.imag()) < std::max(1e-8, 1e-2*std::abs(k_init.imag()))) {
        x_best = x0; k_best = k_init;
      }
      if (k_best.real() <= 0.0 || k_best.real() < real_k_min) {
        x_best = x0; k_best = k_init;
      }
      if (k_best.imag() > 0.0 && std::abs(k_best.imag()) < 1e-6) {
        x_best = x0; k_best = k_init;
      }

      xo[m] = x_best;
      so[m] = sign_best;
  }

  return std::make_tuple(x_out, signs_out);
}

// ---------------------------------------------------------------------------
// Elastic compound-matrix propagation (5-component Porter formulation).
// Propagates yV[5] through elastic layers using the standard finite-difference
// leapfrog scheme with iPower normalization on yV[1].
// ---------------------------------------------------------------------------

static void elastic_compound_propagate(
    const double *b1, const double *b2, const double *b3, const double *b4,
    const double *rho_arr, double x, double yV[5], double h, int64_t n_steps,
    int64_t loc_start, bool going_up, int64_t &iPower) {

  double two_x = 2.0 * x;
  double two_h = 2.0 * h;
  double four_h_x = 4.0 * h * x;

  int64_t j = loc_start + (going_up ? n_steps : 0);
  double xB3 = x * b3[j] - rho_arr[j];

  // Half-step: z = y - 0.5 * M * y
  double z[5];
  z[0] = yV[0] - 0.5 * (b1[j] * yV[3] - b2[j] * yV[4]);
  z[1] = yV[1] - 0.5 * (-rho_arr[j] * yV[3] - xB3 * yV[4]);
  z[2] = yV[2] - 0.5 * (two_h * yV[3] + b4[j] * yV[4]);
  z[3] = yV[3] - 0.5 * (xB3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2]);
  z[4] = yV[4] - 0.5 * (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2]);

  for (int64_t step = 0; step < n_steps; ++step) {
    if (going_up)
      --j;
    else
      ++j;

    // Swap: y_old = y; y = z; z = y_old
    double tmp[5];
    for (int i = 0; i < 5; ++i) tmp[i] = yV[i];
    for (int i = 0; i < 5; ++i) yV[i] = z[i];
    for (int i = 0; i < 5; ++i) z[i] = tmp[i];

    xB3 = x * b3[j] - rho_arr[j];

    z[0] -= (b1[j] * yV[3] - b2[j] * yV[4]);
    z[1] -= (-rho_arr[j] * yV[3] - xB3 * yV[4]);
    z[2] -= (two_h * yV[3] + b4[j] * yV[4]);
    z[3] -= (xB3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2]);
    z[4] -= (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2]);

    // iPower normalization on z[1]
    if (step < n_steps - 1) {
      double scale_val = std::abs(z[1]);
      if (scale_val < FLOOR_V) {
        for (int i = 0; i < 5; ++i) { z[i] *= ROOF_V; yV[i] *= ROOF_V; }
        iPower -= IPOWER_R_V;
      } else if (scale_val > ROOF_V) {
        for (int i = 0; i < 5; ++i) { z[i] *= FLOOR_V; yV[i] *= FLOOR_V; }
        iPower += IPOWER_R_V;
      }
    }
  }

  // Output is z (the "newest" value)
  for (int i = 0; i < 5; ++i) yV[i] = z[i];
}

// Full elastic dispersion evaluation: BC -> elastic layers -> acoustic recurrence -> match
static DispResult dispersion_elastic_eval(
    double x, const double *b1, const double *b2, const double *b3,
    const double *b4, const double *rho_full, int64_t n_layers,
    const int64_t *layer_loc, const int64_t *layer_n, const double *layer_h,
    const double *layer_rho, double omega2, const AcousticBC &bc_bot,
    const AcousticBC &bc_top, int64_t first_acoustic, int64_t last_acoustic) {

  // Bottom BC
  double f, g;
  int64_t iPower = 0, mc = 0;
  bc_impedance(x, omega2, bc_bot, false, f, g, iPower, mc);

  // Propagate through elastic layers below the acoustic region (bottom -> up)
  if (last_acoustic < n_layers - 1) {
    // Build initial yV from f, g (acoustic-to-elastic interface)
    // For elastic half-space bottom BC, bc_impedance already set f,g correctly
    // (f = omega2 * yV[3], g = yV[1])
    // We need the full 5-component vector
    double gammaS2 = x - omega2 / (bc_bot.cs * bc_bot.cs);
    double gammaP2 = x - omega2 / (bc_bot.cp * bc_bot.cp);
    double gammaS = gammaS2 >= 0 ? std::sqrt(gammaS2) : 0.0;
    double gammaP = gammaP2 >= 0 ? std::sqrt(gammaP2) : 0.0;
    double mu = bc_bot.rho * bc_bot.cs * bc_bot.cs;
    double yV[5];
    yV[0] = mu != 0.0 ? (gammaS * gammaP - x) / mu : 0.0;
    yV[1] = ((gammaS2 + x) * (gammaS2 + x) - 4.0 * gammaS * gammaP * x) * mu;
    yV[2] = 2.0 * gammaS * gammaP - gammaS2 - x;
    yV[3] = gammaP * (x - gammaS2);
    yV[4] = gammaS * (gammaS2 - x);

    for (int64_t med = n_layers - 1; med > last_acoustic; --med) {
      int64_t ns = layer_n[med] - 1;
      elastic_compound_propagate(b1, b2, b3, b4, rho_full, x, yV,
                                 layer_h[med], ns, layer_loc[med], true, iPower);
    }
    f = omega2 * yV[3];
    g = yV[1];
    if (g > 0.0) mc = 1; else mc = 0;
  }

  // Acoustic recurrence through acoustic layers
  for (int64_t li = last_acoustic; li >= first_acoustic; --li) {
    double h = layer_h[li];
    double h2k2 = h * h * x;
    double rho = layer_rho[li];
    int64_t loc_s = layer_loc[li];
    int64_t loc_e = loc_s + layer_n[li] - 1;

    double p1v = -2.0 * g;
    double p2v = (b1[loc_e] - h2k2) * g - 2.0 * h * f * rho;
    double p0v = p1v;

    int64_t sweep_len = loc_e - 1 - loc_s + 1;
    for (int64_t s = 0; s < sweep_len; ++s) {
      int64_t jj = (loc_e - 1) - s;
      p0v = p1v;
      p1v = p2v;
      p2v = (h2k2 - b1[jj]) * p1v - p0v;
    }

    f = -(p2v - p0v) / (2.0 * h * rho);
    g = -p1v;
  }

  // Top BC
  double f_top, g_top;
  int64_t ip_top = 0, mc_top = 0;

  if (first_acoustic > 0) {
    // Elastic layers above the acoustic region
    bc_impedance(x, omega2, bc_top, true, f_top, g_top, ip_top, mc_top);

    // Build yV from top BC
    double yV_top[5];
    if (bc_top.type == 0) { // vacuum
      yV_top[0] = 1.0; yV_top[1] = 0.0; yV_top[2] = 0.0;
      yV_top[3] = 0.0; yV_top[4] = 0.0;
    } else if (bc_top.type == 1) { // rigid
      yV_top[0] = 0.0; yV_top[1] = -1.0; yV_top[2] = 0.0;
      yV_top[3] = 0.0; yV_top[4] = 0.0;
    } else {
      // Reconstruct full yV from half-space BC
      double gS2 = x - omega2 / (bc_top.cs * bc_top.cs);
      double gP2 = x - omega2 / (bc_top.cp * bc_top.cp);
      double gS = gS2 >= 0 ? std::sqrt(gS2) : 0.0;
      double gP = gP2 >= 0 ? std::sqrt(gP2) : 0.0;
      double mu_t = bc_top.rho * bc_top.cs * bc_top.cs;
      yV_top[0] = mu_t != 0.0 ? (gS * gP - x) / mu_t : 0.0;
      yV_top[1] = ((gS2 + x) * (gS2 + x) - 4.0 * gS * gP * x) * mu_t;
      yV_top[2] = 2.0 * gS * gP - gS2 - x;
      yV_top[3] = gP * (x - gS2);
      yV_top[4] = gS * (gS2 - x);
    }

    // Propagate down through elastic layers above acoustic region
    for (int64_t med = 0; med < first_acoustic; ++med) {
      int64_t ns = layer_n[med] - 1;
      elastic_compound_propagate(b1, b2, b3, b4, rho_full, x, yV_top,
                                 layer_h[med], ns, layer_loc[med], false, ip_top);
    }

    f_top = omega2 * yV_top[3];
    g_top = -yV_top[1];  // negate for top BC convention
    iPower += ip_top;
  } else {
    bc_impedance(x, omega2, bc_top, true, f_top, g_top, ip_top, mc_top);
    iPower += ip_top;
  }

  double Delta = f * g_top - g * f_top;

  if (g * Delta > 0.0)
    mc += 1;

  return {Delta, iPower, mc};
}

// elastic_solve1: grid scan + bisection for elastic eigenvalue problems.
// Returns eigenvalues as a tensor.
torch::Tensor elastic_solve1_cpp(
    torch::Tensor B1, torch::Tensor B2, torch::Tensor B3, torch::Tensor B4,
    torch::Tensor rho_full, int64_t n_layers, torch::Tensor layer_loc_t,
    torch::Tensor layer_n_t, torch::Tensor layer_h_t, torch::Tensor layer_rho_t,
    double omega2, int64_t bc_bot_type, double bc_bot_cp, double bc_bot_cs,
    double bc_bot_rho, int64_t bc_top_type, double bc_top_cp, double bc_top_cs,
    double bc_top_rho, int64_t first_acoustic, int64_t last_acoustic,
    double x_min, double x_max, int64_t n_grid) {

  const double *b1 = B1.data_ptr<double>();
  const double *b2 = B2.data_ptr<double>();
  const double *b3 = B3.data_ptr<double>();
  const double *b4 = B4.data_ptr<double>();
  const double *rho_f = rho_full.data_ptr<double>();
  const int64_t *lloc = layer_loc_t.data_ptr<int64_t>();
  const int64_t *ln = layer_n_t.data_ptr<int64_t>();
  const double *lh = layer_h_t.data_ptr<double>();
  const double *lrho = layer_rho_t.data_ptr<double>();

  AcousticBC bc_bot{(int)bc_bot_type, bc_bot_cp, bc_bot_cs, bc_bot_rho};
  AcousticBC bc_top{(int)bc_top_type, bc_top_cp, bc_top_cs, bc_top_rho};

  // Grid scan for sign changes
  std::vector<double> delta_signs(n_grid);
  double dx = (x_max - x_min) / (double)(n_grid - 1);

  for (int64_t i = 0; i < n_grid; ++i) {
    double xi = x_min + i * dx;
    auto r = dispersion_elastic_eval(xi, b1, b2, b3, b4, rho_f, n_layers,
                                     lloc, ln, lh, lrho, omega2, bc_bot, bc_top,
                                     first_acoustic, last_acoustic);
    delta_signs[i] = r.Delta > 0 ? 1.0 : (r.Delta < 0 ? -1.0 : 0.0);
  }

  // Find sign changes and bisect
  std::vector<double> roots;
  roots.reserve(512);

  for (int64_t i = 0; i < n_grid - 1; ++i) {
    if (delta_signs[i] * delta_signs[i + 1] < 0) {
      double a = x_min + i * dx;
      double b = x_min + (i + 1) * dx;
      double sa = delta_signs[i];

      for (int iter = 0; iter < 60; ++iter) {
        double c = 0.5 * (a + b);
        auto rc = dispersion_elastic_eval(c, b1, b2, b3, b4, rho_f, n_layers,
                                          lloc, ln, lh, lrho, omega2, bc_bot,
                                          bc_top, first_acoustic, last_acoustic);
        double sc = rc.Delta > 0 ? 1.0 : (rc.Delta < 0 ? -1.0 : 0.0);
        if (sa * sc < 0) {
          b = c;
        } else {
          a = c;
          sa = sc;
        }
        if (std::abs(b - a) < 1e-15 * std::abs(a + b))
          break;
      }
      double root = 0.5 * (a + b);
      if (roots.empty() || std::abs(root - roots.back()) > 1e-12 * std::abs(root)) {
        roots.push_back(root);
      }
    }
  }

  auto result = torch::empty({(int64_t)roots.size()}, torch::kFloat64);
  double *rp = result.data_ptr<double>();
  for (size_t i = 0; i < roots.size(); ++i)
    rp[i] = roots[i];
  return result;
}

// ---------------------------------------------------------------------------
// Fused KRAKENC solve: refine roots + build tridiag + inverse iteration.
// Eliminates Python round-trips between refine_roots and extract_modes.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
krakenc_fused_cpp(
    torch::Tensor x_init_t,     // [M] complex128 — real KRAKEN eigenvalues
    torch::Tensor k_init_t,     // [M] complex128 — real KRAKEN wavenumbers
    torch::Tensor B1,           // [N_total] float64
    torch::Tensor B1C,          // [N_total] float64 (complex part)
    torch::Tensor layer_loc,    // [n_ac_layers] int64
    torch::Tensor layer_n,      // [n_ac_layers] int64
    torch::Tensor layer_h,      // [n_ac_layers] float64
    torch::Tensor layer_rho,    // [n_ac_layers] float64
    double omega2,
    int64_t bc_bot_type, double bot_cp, double bot_ci,
    double bot_cs, double bot_csi, double bot_rho,
    int64_t bc_top_type, double top_cp, double top_ci,
    double top_cs, double top_csi, double top_rho,
    double real_k_min, int64_t max_iter, int64_t n_inv_iter
) {
  using C = c10::complex<double>;
  const int64_t M = x_init_t.size(0);
  const int64_t n_layers = layer_loc.size(0);

  // --- Step 1: Build DispEnv for root refinement ---
  DispEnv E;
  E.b1 = B1.data_ptr<double>();
  E.n_layers = n_layers;
  E.ll = layer_loc.data_ptr<int64_t>();
  E.ln = layer_n.data_ptr<int64_t>();
  E.lh = layer_h.data_ptr<double>();
  E.lr = layer_rho.data_ptr<double>();
  E.omega2 = omega2;
  E.bc_bot_type = bc_bot_type; E.bot_cp = bot_cp; E.bot_ci = bot_ci;
  E.bot_cs = bot_cs; E.bot_csi = bot_csi; E.bot_rho = bot_rho;
  E.bc_top_type = bc_top_type; E.top_cp = top_cp; E.top_ci = top_ci;
  E.top_cs = top_cs; E.top_csi = top_csi; E.top_rho = top_rho;

  // --- Step 2: Batched branch-sign check + selective refinement ---
  const C *xi = reinterpret_cast<const C*>(x_init_t.data_ptr());
  const C *ki = reinterpret_cast<const C*>(k_init_t.data_ptr());

  std::vector<C> x_refined(M);
  std::vector<double> signs(M);

  // Warm-start all modes from perturbative k²
  // Pre-filter: modes with negligible |Im(k)| don't need refinement at all.
  // The perturbative correction from KRAKEN's extract_modes is already exact
  // for these (same as Porter's Tindle-Guthrie formula).
  std::vector<C> x0_all(M);
  std::vector<int64_t> refine_list;
  refine_list.reserve(M);
  for (int64_t m = 0; m < M; ++m) {
    x0_all[m] = ki[m] * ki[m];
    double k_imag_ratio = std::abs(ki[m].imag()) / std::max(1e-30, std::abs(ki[m].real()));
    if (k_imag_ratio < 1e-10) {
      // Negligible attenuation — perturbative k is exact
      x_refined[m] = x0_all[m];
      signs[m] = 1.0;
    } else {
      refine_list.push_back(m);
    }
  }

  // Batched branch-sign check only for modes that need refinement
  if (!refine_list.empty()) {
    int64_t K = (int64_t)refine_list.size();
    std::vector<C> x_check(K);
    std::vector<double> pos_s(K, 1.0), neg_s(K, -1.0);
    std::vector<C> rp(K), rn(K);
    for (int64_t i = 0; i < K; ++i)
      x_check[i] = x0_all[refine_list[i]];
    disp_eval_batch(x_check.data(), pos_s.data(), K, E, rp.data());
    disp_eval_batch(x_check.data(), neg_s.data(), K, E, rn.data());

    // Check if any can be skipped based on residual
    std::vector<int64_t> refine_list2;
    refine_list2.reserve(K);
    for (int64_t i = 0; i < K; ++i) {
      int64_t m = refine_list[i];
      double r_pos = std::abs(rp[i]), r_neg = std::abs(rn[i]);
      double default_sign = (r_neg < r_pos) ? -1.0 : 1.0;
      double best_r = std::min(r_pos, r_neg);
      double scale = std::max(1.0, std::abs(x0_all[m]));
      if (best_r < 1e-6 * scale) {
        x_refined[m] = x0_all[m];
        signs[m] = default_sign;
      } else {
        refine_list2.push_back(m);
        signs[m] = default_sign;
      }
    }
    refine_list = std::move(refine_list2);
  }

  // Batched secant refinement: advance all active modes together per iteration.
  // Two parallel tracks: default_sign and alt_sign (-default_sign).
  if (!refine_list.empty()) {
    int64_t K = (int64_t)refine_list.size();

    // Per-mode secant state for both branch signs
    struct SecState {
      C x_a, x_b, f_a, f_b, best;
      double best_r, bsign;
      bool done;
    };
    std::vector<SecState> tracks(2 * K);  // [0..K) = default, [K..2K) = alt

    // Initialize: x_a = x0, x_b = perturbed x0
    std::vector<C> eval_x(2 * K);
    std::vector<double> eval_s(2 * K);
    for (int64_t i = 0; i < K; ++i) {
      int64_t m = refine_list[i];
      C x0 = x0_all[m];
      double sc = std::max(std::abs(x0), 1.0);
      C x_b = x0 * C(1.0+1e-8, 1e-8) + C(0.0, 1e-10*sc);
      double ds = signs[m];  // default_sign from branch check

      tracks[i]     = {x0, x_b, C(0), C(0), x0, 1e30, ds, false};
      tracks[K + i] = {x0, x_b, C(0), C(0), x0, 1e30, -ds, false};
      eval_x[i]     = x0;     eval_s[i]     = ds;
      eval_x[K + i] = x0;     eval_s[K + i] = -ds;
    }

    // Initial f_a evaluation (batched, one sweep for all 2K modes)
    std::vector<C> f_out(2 * K);
    disp_eval_batch(eval_x.data(), eval_s.data(), 2 * K, E, f_out.data());
    for (int64_t i = 0; i < 2 * K; ++i) {
      tracks[i].f_a = f_out[i];
      tracks[i].best_r = std::abs(f_out[i]);
    }

    // Initial f_b evaluation
    for (int64_t i = 0; i < 2 * K; ++i) eval_x[i] = tracks[i].x_b;
    disp_eval_batch(eval_x.data(), eval_s.data(), 2 * K, E, f_out.data());
    for (int64_t i = 0; i < 2 * K; ++i) {
      tracks[i].f_b = f_out[i];
      if (std::abs(f_out[i]) < tracks[i].best_r) {
        tracks[i].best = tracks[i].x_b;
        tracks[i].best_r = std::abs(f_out[i]);
      }
    }

    // Secant iterations (batched)
    for (int64_t it = 0; it < max_iter; ++it) {
      // Compute next x_c for all active tracks
      int64_t n_active = 0;
      std::vector<int64_t> active_idx;
      active_idx.reserve(2 * K);
      for (int64_t i = 0; i < 2 * K; ++i) {
        if (tracks[i].done) continue;
        C denom = tracks[i].f_b - tracks[i].f_a;
        if (std::abs(denom) < 1e-20) { tracks[i].done = true; continue; }
        C x_c = tracks[i].x_b - tracks[i].f_b * (tracks[i].x_b - tracks[i].x_a) / denom;
        if (!std::isfinite(x_c.real()) || !std::isfinite(x_c.imag()) || x_c.real() <= 0.0) {
          tracks[i].done = true; continue;
        }
        eval_x[n_active] = x_c;
        eval_s[n_active] = tracks[i].bsign;
        active_idx.push_back(i);
        ++n_active;
      }
      if (n_active == 0) break;

      // Batched evaluation
      disp_eval_batch(eval_x.data(), eval_s.data(), n_active, E, f_out.data());

      for (int64_t j = 0; j < n_active; ++j) {
        int64_t i = active_idx[j];
        C x_c = eval_x[j];
        C f_c = f_out[j];
        if (std::abs(f_c) < tracks[i].best_r) { tracks[i].best = x_c; tracks[i].best_r = std::abs(f_c); }
        if (std::abs(x_c - tracks[i].x_b) <= 1e-12 * (1.0 + std::abs(x_c))) tracks[i].done = true;
        tracks[i].x_a = tracks[i].x_b; tracks[i].f_a = tracks[i].f_b;
        tracks[i].x_b = x_c; tracks[i].f_b = f_c;
      }
    }

    // Score 4 candidates per mode (batched)
    auto make_k = [](C x) -> C { C k = std::sqrt(x); return k.real() < 0 ? -k : k; };
    std::vector<C> score_x(4 * K);
    std::vector<double> score_s(4 * K);
    for (int64_t i = 0; i < K; ++i) {
      C x_def = tracks[i].best, x_alt = tracks[K + i].best;
      double ds = tracks[i].bsign;
      score_x[4*i+0] = x_def;             score_s[4*i+0] = ds;
      score_x[4*i+1] = std::conj(x_def);  score_s[4*i+1] = ds;
      score_x[4*i+2] = x_alt;             score_s[4*i+2] = -ds;
      score_x[4*i+3] = std::conj(x_alt);  score_s[4*i+3] = -ds;
    }
    std::vector<C> score_d(4 * K);
    disp_eval_batch(score_x.data(), score_s.data(), 4 * K, E, score_d.data());

    for (int64_t i = 0; i < K; ++i) {
      int64_t m = refine_list[i];
      C k_init = ki[m];

      int best_idx = 0;
      auto mk = [&](int j) -> std::tuple<int,double,double> {
        C k = make_k(score_x[4*i+j]);
        return {k.imag() > 1e-12 ? 1 : 0, std::abs(score_d[4*i+j]), std::abs(k - k_init)};
      };
      auto best_sc = mk(0);
      for (int j = 1; j < 4; ++j) {
        auto s = mk(j);
        if (s < best_sc) { best_sc = s; best_idx = j; }
      }

      C x_best = score_x[4*i + best_idx];
      C k_best = make_k(x_best);
      double sign_best = score_s[4*i + best_idx];
      C x0 = x0_all[m];

      if (std::abs(k_best.imag()) < std::max(1e-8, 1e-2*std::abs(k_init.imag()))) {
        x_best = x0; k_best = k_init;
      }
      if (k_best.real() <= 0.0 || k_best.real() < real_k_min) {
        x_best = x0; k_best = k_init;
      }
      if (k_best.imag() > 0.0 && std::abs(k_best.imag()) < 1e-6) {
        x_best = x0; k_best = k_init;
      }
      x_refined[m] = x_best;
      signs[m] = sign_best;
    }
  }

  // --- Step 3: Build base_d, h2_factor, e from workspace ---
  const double *b1_ptr = B1.data_ptr<double>();
  const double *b1c_ptr = B1C.data_ptr<double>();
  const int64_t *ll = layer_loc.data_ptr<int64_t>();
  const int64_t *ln_p = layer_n.data_ptr<int64_t>();
  const double *lh_p = layer_h.data_ptr<double>();
  const double *lr_p = layer_rho.data_ptr<double>();

  int64_t N_total1 = 0;
  for (int64_t li = 0; li < n_layers; ++li)
    N_total1 += (li == 0) ? ln_p[li] : ln_p[li] - 1;

  std::vector<C> base_d(N_total1);
  std::vector<C> h2_fac(N_total1);
  std::vector<C> e_vec(N_total1);
  int64_t j = 0;
  for (int64_t li = 0; li < n_layers; ++li) {
    int64_t loc = ll[li];
    int64_t n = ln_p[li];
    double h = lh_p[li];
    double h_rho = h * lr_p[li];
    double inv_hrho = 1.0 / h_rho;
    double xcoeff = (h * h) / h_rho;

    if (li == 0) {
      e_vec[j] = C(0.0);
      for (int64_t i = 0; i < n; ++i) {
        C b1_val = C(b1_ptr[loc + i]) + C(0.0, h * h) * C(b1c_ptr[loc + i]);
        base_d[j + i] = b1_val / h_rho;
        h2_fac[j + i] = C(xcoeff);
        if (i > 0) e_vec[j + i] = C(inv_hrho);
      }
      j += n;
    } else {
      // Interface averaging at j-1
      C b1_val_0 = C(b1_ptr[loc]) + C(0.0, h * h) * C(b1c_ptr[loc]);
      base_d[j - 1] = (base_d[j - 1] + b1_val_0 / h_rho) / C(2.0);
      h2_fac[j - 1] = (h2_fac[j - 1] + C(xcoeff)) / C(2.0);
      for (int64_t i = 1; i < n; ++i) {
        C b1_val = C(b1_ptr[loc + i]) + C(0.0, h * h) * C(b1c_ptr[loc + i]);
        base_d[j + i - 1] = b1_val / h_rho;
        h2_fac[j + i - 1] = C(xcoeff);
        e_vec[j + i - 1] = C(inv_hrho);
      }
      j += n - 1;
    }
  }

  // --- Step 4: Compute BC ratios + inverse iteration (only for refined modes) ---
  auto phi = torch::full({M, N_total1}, C(1e-10), torch::dtype(torch::kComplexDouble));
  C *pp = reinterpret_cast<C*>(phi.data_ptr());

  auto x_out = torch::empty({M}, torch::dtype(torch::kComplexDouble));
  auto signs_out = torch::empty({M}, torch::dtype(torch::kDouble));
  // needs_phi[m] = 1 if mode m needs inverse iteration, 0 if perturbative (caller fills phi)
  auto needs_phi_t = torch::zeros({M}, torch::dtype(torch::kBool));
  bool *needs_phi = needs_phi_t.data_ptr<bool>();
  C *xo = reinterpret_cast<C*>(x_out.data_ptr());
  double *so = signs_out.data_ptr<double>();

  // Mark which modes were refined (need new phi) vs perturbative (reuse existing)
  for (int64_t m = 0; m < M; ++m) {
    double k_imag_ratio = std::abs(ki[m].imag()) / std::max(1e-30, std::abs(ki[m].real()));
    needs_phi[m] = (k_imag_ratio >= 1e-10);
  }

  std::vector<C> dw(N_total1);

  for (int64_t m = 0; m < M; ++m) {
    C x_m = x_refined[m];
    xo[m] = x_m;
    so[m] = signs[m];

    if (!needs_phi[m]) continue;  // skip inverse iteration for perturbative modes

    // BC impedance for this mode
    C bsign(signs[m], 0.0);
    C f_bot, g_bot, f_top, g_top;
    bc_impedance_complex(x_m, omega2, bot_cp, bot_ci, bot_cs, bot_csi, bot_rho,
                         bc_bot_type, false, bsign, f_bot, g_bot);
    bc_impedance_complex(x_m, omega2, top_cp, top_ci, top_cs, top_csi, top_rho,
                         bc_top_type, true, C(1.0), f_top, g_top);

    bool top_vac = (std::abs(g_top) == 0.0);
    bool bot_vac = (std::abs(g_bot) == 0.0);
    C top_ratio = top_vac ? C(0.0) : f_top / g_top;
    C bot_ratio = bot_vac ? C(0.0) : f_bot / g_bot;

    C *pm = pp + m * N_total1;

    for (int64_t iter = 0; iter < n_inv_iter; ++iter) {
      // Assemble diagonal
      for (int64_t i = 0; i < N_total1; ++i)
        dw[i] = base_d[i] - x_m * h2_fac[i];

      // BC injection
      if (top_vac) dw[0] = C(1.0);
      else         dw[0] = dw[0] / C(2.0) + top_ratio;
      if (bot_vac) dw[N_total1-1] = C(1.0);
      else         dw[N_total1-1] = dw[N_total1-1] / C(2.0) - bot_ratio;

      // Thomas forward
      for (int64_t i = 1; i < N_total1; ++i) {
        C w = e_vec[i] / dw[i-1];
        dw[i] -= w * e_vec[i];
        pm[i] -= w * pm[i-1];
      }
      pm[N_total1-1] /= dw[N_total1-1];
      for (int64_t i = N_total1-2; i >= 0; --i)
        pm[i] = (pm[i] - e_vec[i+1] * pm[i+1]) / dw[i];

      // Max-norm
      double mx = 0.0;
      for (int64_t i = 0; i < N_total1; ++i) { double a = std::abs(pm[i]); if (a > mx) mx = a; }
      if (mx > 0.0) { C inv(1.0/mx); for (int64_t i = 0; i < N_total1; ++i) pm[i] *= inv; }
    }

    // Final L2 norm
    double sq = 0.0;
    for (int64_t i = 0; i < N_total1; ++i) { double a = std::abs(pm[i]); sq += a*a; }
    double nrm = std::sqrt(sq);
    if (nrm > 0.0) { C inv(1.0/nrm); for (int64_t i = 0; i < N_total1; ++i) pm[i] *= inv; }
  }

  return std::make_tuple(x_out, signs_out, phi, needs_phi_t);
}

// ---------------------------------------------------------------------------
// FIELD3D GBT: Gaussian Beam Tracing kernel.
// Traces beams through triangular mesh, deposits contributions on receiver
// radials. All beam stepping + influence in one C++ call.
// ---------------------------------------------------------------------------

torch::Tensor field3d_gbt_cpp(
    // Mesh: nodes and connectivity
    torch::Tensor node_x,       // [N_nodes] float64 — x coordinates (meters)
    torch::Tensor node_y,       // [N_nodes] float64 — y coordinates
    torch::Tensor elements,     // [N_elt, 3] int64 — 0-based node indices
    torch::Tensor adjacency,    // [N_elt, 3] int64 — neighbor per side (-1 = boundary)
    // Per-node modal data
    torch::Tensor node_k,       // [N_nodes, M_max] complex128 — wavenumbers
    torch::Tensor node_phi_s,   // [N_nodes, M_max] complex128 — phi at source depth
    torch::Tensor node_phi_r,   // [N_nodes, M_max] complex128 — phi at receiver depth
    torch::Tensor node_M,       // [N_nodes] int64 — mode count per node
    // Source
    double sx, double sy,
    int64_t ie_src,             // source element index
    // Beam params
    double alpha1_deg, double alpha2_deg, int64_t n_alpha,
    double step_size, int64_t n_steps, double eps_mult,
    // Receiver grid
    torch::Tensor rad_cos,      // [N_theta] float64
    torch::Tensor rad_sin,      // [N_theta] float64
    double r_min, double r_max, int64_t n_r,
    // Limits
    int64_t M_limit,
    char beam_type              // 'F' or 'M'
) {
  using C = c10::complex<double>;
  const double PI = 3.141592653589793;
  const double BEAM_WINDOW = 5.0;

  int64_t n_theta = rad_cos.size(0);
  int64_t n_nodes = node_x.size(0);
  int64_t n_elt = elements.size(0);
  int64_t M_max = node_k.size(1);
  double delta_r = (r_max - r_min) / std::max(n_r - 1, (int64_t)1);

  double alpha1 = alpha1_deg * PI / 180.0;
  double alpha2 = alpha2_deg * PI / 180.0;
  double d_alpha = (alpha2 - alpha1) / std::max(n_alpha - 1, (int64_t)1);

  const double *nx = node_x.data_ptr<double>();
  const double *ny = node_y.data_ptr<double>();
  const int64_t *elt = elements.data_ptr<int64_t>();
  const int64_t *adj = adjacency.data_ptr<int64_t>();
  const C *nk = reinterpret_cast<const C*>(node_k.data_ptr());
  const C *nps = reinterpret_cast<const C*>(node_phi_s.data_ptr());
  const C *npr = reinterpret_cast<const C*>(node_phi_r.data_ptr());
  const int64_t *nM = node_M.data_ptr<int64_t>();
  const double *rc = rad_cos.data_ptr<double>();
  const double *rs = rad_sin.data_ptr<double>();

  auto P_t = torch::zeros({n_theta, n_r}, torch::dtype(torch::kComplexDouble));
  C *P = reinterpret_cast<C*>(P_t.data_ptr());

  // Helper: get element info for a mode
  auto get_elt = [&](int64_t ie, int64_t mode,
      double &x1, double &y1, double &x2, double &y2, double &x3, double &y3,
      double &D12, double &D13, double &D23, double &delta,
      int64_t &s1, int64_t &s2, int64_t &s3, int64_t &Mprop,
      C &cx_out, C &cy_out) -> bool {
    s1 = elt[ie*3]; s2 = elt[ie*3+1]; s3 = elt[ie*3+2];
    Mprop = std::min({nM[s1], nM[s2], nM[s3], M_max});
    if (mode >= Mprop) return false;
    x1 = nx[s1]; y1 = ny[s1]; x2 = nx[s2]; y2 = ny[s2]; x3 = nx[s3]; y3 = ny[s3];
    D12 = x1*y2 - y1*x2; D13 = x1*y3 - y1*x3; D23 = x2*y3 - y2*x3;
    delta = D23 - D13 + D12;
    if (std::abs(delta) < 1e-30) return false;
    C k1 = nk[s1*M_max+mode], k2 = nk[s2*M_max+mode], k3 = nk[s3*M_max+mode];
    double A1x=-y3+y2, A2x=y3-y1, A3x=-y2+y1;
    double A1y=x3-x2, A2y=-x3+x1, A3y=x2-x1;
    cx_out = (A1x/k1 + A2x/k2 + A3x/k3) / delta;
    cy_out = (A1y/k1 + A2y/k2 + A3y/k3) / delta;
    return true;
  };

  for (int64_t ia = 0; ia < n_alpha; ++ia) {
    double alpha = alpha1 + ia * d_alpha;
    double tsx = std::cos(alpha), tsy = std::sin(alpha);

    for (int64_t mode = 0; mode < M_limit; ++mode) {
      double x1,y1,x2,y2,x3,y3,D12,D13,D23,delta;
      int64_t s1,s2,s3,Mprop;
      C cx, cy;
      int64_t ie = ie_src;

      if (!get_elt(ie, mode, x1,y1,x2,y2,x3,y3,D12,D13,D23,delta,s1,s2,s3,Mprop,cx,cy))
        continue;
      if (mode >= Mprop) continue;

      // Barycentric at source
      double DB1=sx*y1-sy*x1, DB2=sx*y2-sy*x2, DB3=sx*y3-sy*x3;
      double A1=(D23-DB3+DB2)/delta, A2=(DB3-D13-DB1)/delta, A3=(-DB2+DB1+D12)/delta;

      C cA = A1/nk[s1*M_max+mode] + A2/nk[s2*M_max+mode] + A3/nk[s3*M_max+mode];
      C phi_src = A1*nps[s1*M_max+mode] + A2*nps[s2*M_max+mode] + A3*nps[s3*M_max+mode];

      double Hwidth;
      if (beam_type == 'F') Hwidth = 2.0 / (std::real(C(1.0)/cA) * d_alpha);
      else Hwidth = std::sqrt(2.0 * std::real(cA) * r_max);
      double EpsOpt = 0.5 * Hwidth * Hwidth;
      C eps = C(0.0, eps_mult * EpsOpt);
      C cnst = phi_src * std::sqrt(eps / cA) * d_alpha;

      double xA=sx, yA=sy;
      double xiA=tsx/std::real(cA), etaA=tsy/std::real(cA);
      C pA(1.0), qA=eps, tauA(0.0);
      int KMAHA = 1;
      C phiA = A1*npr[s1*M_max+mode] + A2*npr[s2*M_max+mode] + A3*npr[s3*M_max+mode];
      bool exited = false;

      for (int64_t istep = 0; istep < n_steps; ++istep) {
        double cA_r = std::real(cA);
        double xB = xA + step_size * cA_r * xiA;
        double yB = yA + step_size * cA_r * etaA;
        double xiB = xiA - step_size * std::real(cx / (cA*cA));
        double etaB = etaA - step_size * std::real(cy / (cA*cA));
        C pB = pA;
        C qB = qA + step_size * cA_r * pA;
        C tauB = tauA + step_size / cA;

        int KMAHB = KMAHA;
        if (std::real(qB) < 0.0) {
          if ((std::imag(qA) < 0 && std::imag(qB) >= 0) ||
              (std::imag(qA) > 0 && std::imag(qB) <= 0)) KMAHB = -KMAHA;
        }

        C cB = cA, phiB = phiA;
        if (!exited) {
          DB1=xB*y1-yB*x1; DB2=xB*y2-yB*x2; DB3=xB*y3-yB*x3;
          A1=(D23-DB3+DB2)/delta; A2=(DB3-D13-DB1)/delta; A3=(-DB2+DB1+D12)/delta;

          for (int cross = 0; cross < 100 && (A1<0||A2<0||A3<0); ++cross) {
            int side = (A3<0) ? 0 : (A1<0 ? 1 : 2);
            int64_t next = adj[ie*3+side];
            if (next < 0) { exited = true; cx=C(0); cy=C(0); break; }
            ie = next;
            if (!get_elt(ie,mode,x1,y1,x2,y2,x3,y3,D12,D13,D23,delta,s1,s2,s3,Mprop,cx,cy))
              { exited = true; break; }
            if (mode >= Mprop) { exited = true; break; }
            DB1=xB*y1-yB*x1; DB2=xB*y2-yB*x2; DB3=xB*y3-yB*x3;
            A1=(D23-DB3+DB2)/delta; A2=(DB3-D13-DB1)/delta; A3=(-DB2+DB1+D12)/delta;
          }
          if (!exited) {
            cB = A1/nk[s1*M_max+mode]+A2/nk[s2*M_max+mode]+A3/nk[s3*M_max+mode];
            phiB = A1*npr[s1*M_max+mode]+A2*npr[s2*M_max+mode]+A3*npr[s3*M_max+mode];
          }
        }

        // Influence on receivers
        double _xA=xA-sx, _yA=yA-sy, _xB=xB-sx, _yB=yB-sy;
        for (int64_t it = 0; it < n_theta; ++it) {
          double rv1=rc[it], rv2=rs[it];
          double dA = rv1*xiA + rv2*etaA;
          double dB = rv1*xiB + rv2*etaB;
          if (std::abs(dA)<1e-30 || std::abs(dB)<1e-30 || dA*dB<=0) continue;
          double rA = (_yA*etaA+_xA*xiA)/dA;
          double rB = (_yB*etaB+_xB*xiB)/dB;
          int64_t ir1 = std::max(std::min((int64_t)((rA-r_min)/delta_r)+1, n_r), (int64_t)0);
          int64_t ir2 = std::max(std::min((int64_t)((rB-r_min)/delta_r)+1, n_r), (int64_t)1);
          if (ir2 <= ir1) continue;
          double nA_v = (_xA*rv2-_yA*rv1)/(cA_r*dA);
          double nB_v = (_xB*rv2-_yB*rv1)/(std::real(cB)*dB);
          double rBA = rB - rA;
          if (std::abs(rBA) < 1e-30) continue;
          for (int64_t ir = ir1+1; ir <= std::min(ir2, n_r); ++ir) {
            double W = (r_min + (ir-1)*delta_r - rA) / rBA;
            C pM = pA + W*(pB-pA);
            C qM = qA + W*(qB-qA);
            double n2 = (nA_v + W*(nB_v-nA_v));
            n2 = n2*n2;
            if (qM != C(0) && -0.5*std::imag(pM/qM)*n2 < BEAM_WINDOW) {
              C cM = cA + W*(cB-cA);
              C tM = tauA + W*(tauB-tauA);
              C phM = phiA + W*(phiB-phiA);
              int km = KMAHA;
              if (std::real(qM)<0) {
                if ((std::imag(qA)<0 && std::imag(qM)>=0)||(std::imag(qA)>0 && std::imag(qM)<=0))
                  km = -km;
              }
              C contrib = cnst * phM * std::sqrt(cM/qM) * std::exp(C(0,-1)*(tM + C(0.5)*pM/qM*n2));
              if (km < 0) contrib = -contrib;
              P[it*n_r + ir-1] += contrib;
            }
          }
        }

        xA=xB; yA=yB; xiA=xiB; etaA=etaB;
        pA=pB; qA=qB; tauA=tauB; cA=cB;
        KMAHA=KMAHB; phiA=phiB;
      }
    }
  }
  return P_t;
}

// ---------------------------------------------------------------------------
// FIELD3D GBT backward: re-execute forward, accumulate dP/d(node_k) and dP/d(node_phi).
// Uses the same beam tracing as forward but instead of accumulating P,
// propagates grad_P back to node_k and node_phi_r via the contribution formula.
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> field3d_gbt_backward_cpp(
    torch::Tensor grad_P,       // [N_theta, N_r] complex128
    // Same inputs as forward:
    torch::Tensor node_x, torch::Tensor node_y,
    torch::Tensor elements, torch::Tensor adjacency,
    torch::Tensor node_k, torch::Tensor node_phi_s, torch::Tensor node_phi_r,
    torch::Tensor node_M_t,
    double sx, double sy, int64_t ie_src,
    double alpha1_deg, double alpha2_deg, int64_t n_alpha,
    double step_size, int64_t n_steps, double eps_mult,
    torch::Tensor rad_cos, torch::Tensor rad_sin,
    double r_min, double r_max, int64_t n_r,
    int64_t M_limit, char beam_type
) {
  using C = c10::complex<double>;
  const double PI = 3.141592653589793;
  const double BEAM_WINDOW = 5.0;

  int64_t n_theta = rad_cos.size(0);
  int64_t n_nodes = node_x.size(0);
  int64_t M_max = node_k.size(1);
  double delta_r = (r_max - r_min) / std::max(n_r - 1, (int64_t)1);
  double alpha1 = alpha1_deg * PI / 180.0;
  double alpha2 = alpha2_deg * PI / 180.0;
  double d_alpha = (alpha2 - alpha1) / std::max(n_alpha - 1, (int64_t)1);

  const double *nx_p = node_x.data_ptr<double>();
  const double *ny_p = node_y.data_ptr<double>();
  const int64_t *elt = elements.data_ptr<int64_t>();
  const int64_t *adj_p = adjacency.data_ptr<int64_t>();
  const C *nk = reinterpret_cast<const C*>(node_k.data_ptr());
  const C *nps = reinterpret_cast<const C*>(node_phi_s.data_ptr());
  const C *npr = reinterpret_cast<const C*>(node_phi_r.data_ptr());
  const int64_t *nM = node_M_t.data_ptr<int64_t>();
  const double *rc = rad_cos.data_ptr<double>();
  const double *rs = rad_sin.data_ptr<double>();
  const C *gP = reinterpret_cast<const C*>(grad_P.data_ptr());

  // Output gradients
  auto grad_k = torch::zeros_like(node_k);
  auto grad_phi_r = torch::zeros_like(node_phi_r);
  C *gk = reinterpret_cast<C*>(grad_k.data_ptr());
  C *gpr = reinterpret_cast<C*>(grad_phi_r.data_ptr());

  // Re-use get_elt from forward (same lambda)
  auto get_elt = [&](int64_t ie, int64_t mode,
      double &x1, double &y1, double &x2, double &y2, double &x3, double &y3,
      double &D12, double &D13, double &D23, double &delta,
      int64_t &s1, int64_t &s2, int64_t &s3, int64_t &Mprop,
      C &cx_out, C &cy_out) -> bool {
    s1 = elt[ie*3]; s2 = elt[ie*3+1]; s3 = elt[ie*3+2];
    Mprop = std::min({nM[s1], nM[s2], nM[s3], M_max});
    if (mode >= Mprop) return false;
    x1 = nx_p[s1]; y1 = ny_p[s1]; x2 = nx_p[s2]; y2 = ny_p[s2]; x3 = nx_p[s3]; y3 = ny_p[s3];
    D12 = x1*y2 - y1*x2; D13 = x1*y3 - y1*x3; D23 = x2*y3 - y2*x3;
    delta = D23 - D13 + D12;
    if (std::abs(delta) < 1e-30) return false;
    C k1 = nk[s1*M_max+mode], k2 = nk[s2*M_max+mode], k3 = nk[s3*M_max+mode];
    double A1x=-y3+y2, A2x=y3-y1, A3x=-y2+y1;
    double A1y=x3-x2, A2y=-x3+x1, A3y=x2-x1;
    cx_out = (A1x/k1 + A2x/k2 + A3x/k3) / delta;
    cy_out = (A1y/k1 + A2y/k2 + A3y/k3) / delta;
    return true;
  };

  // Re-execute forward beam tracing, accumulate gradients at each hit
  for (int64_t ia = 0; ia < n_alpha; ++ia) {
    double alpha = alpha1 + ia * d_alpha;
    double tsx = std::cos(alpha), tsy = std::sin(alpha);

    for (int64_t mode = 0; mode < M_limit; ++mode) {
      double x1,y1,x2,y2,x3,y3,D12,D13,D23,delta;
      int64_t s1,s2,s3,Mprop;
      C cx, cy;
      int64_t ie = ie_src;

      if (!get_elt(ie, mode, x1,y1,x2,y2,x3,y3,D12,D13,D23,delta,s1,s2,s3,Mprop,cx,cy))
        continue;

      double DB1=sx*y1-sy*x1, DB2=sx*y2-sy*x2, DB3=sx*y3-sy*x3;
      double A1=(D23-DB3+DB2)/delta, A2=(DB3-D13-DB1)/delta, A3=(-DB2+DB1+D12)/delta;

      C cA = A1/nk[s1*M_max+mode] + A2/nk[s2*M_max+mode] + A3/nk[s3*M_max+mode];
      C phi_src = A1*nps[s1*M_max+mode] + A2*nps[s2*M_max+mode] + A3*nps[s3*M_max+mode];

      double Hwidth;
      if (beam_type == 'F') Hwidth = 2.0 / (std::real(C(1.0)/cA) * d_alpha);
      else Hwidth = std::sqrt(2.0 * std::real(cA) * r_max);
      double EpsOpt = 0.5 * Hwidth * Hwidth;
      C eps = C(0.0, eps_mult * EpsOpt);
      C cnst = phi_src * std::sqrt(eps / cA) * d_alpha;

      double xA=sx, yA=sy;
      double xiA=tsx/std::real(cA), etaA=tsy/std::real(cA);
      C pA(1.0), qA=eps, tauA(0.0);
      int KMAHA = 1;
      C phiA = A1*npr[s1*M_max+mode] + A2*npr[s2*M_max+mode] + A3*npr[s3*M_max+mode];

      // Track current element's node indices for gradient accumulation
      int64_t cur_s1=s1, cur_s2=s2, cur_s3=s3;
      double cur_A1=A1, cur_A2=A2, cur_A3=A3;
      bool exited = false;

      for (int64_t istep = 0; istep < n_steps; ++istep) {
        double cA_r = std::real(cA);
        double xB = xA + step_size * cA_r * xiA;
        double yB = yA + step_size * cA_r * etaA;
        double xiB = xiA - step_size * std::real(cx / (cA*cA));
        double etaB = etaA - step_size * std::real(cy / (cA*cA));
        C pB = pA;
        C qB = qA + step_size * cA_r * pA;
        C tauB = tauA + step_size / cA;

        int KMAHB = KMAHA;
        if (std::real(qB) < 0.0) {
          if ((std::imag(qA) < 0 && std::imag(qB) >= 0) ||
              (std::imag(qA) > 0 && std::imag(qB) <= 0)) KMAHB = -KMAHA;
        }

        C cB = cA, phiB = phiA;
        int64_t B_s1=cur_s1, B_s2=cur_s2, B_s3=cur_s3;
        double B_A1=cur_A1, B_A2=cur_A2, B_A3=cur_A3;

        if (!exited) {
          DB1=xB*y1-yB*x1; DB2=xB*y2-yB*x2; DB3=xB*y3-yB*x3;
          A1=(D23-DB3+DB2)/delta; A2=(DB3-D13-DB1)/delta; A3=(-DB2+DB1+D12)/delta;

          for (int cross = 0; cross < 100 && (A1<0||A2<0||A3<0); ++cross) {
            int side = (A3<0) ? 0 : (A1<0 ? 1 : 2);
            int64_t next = adj_p[ie*3+side];
            if (next < 0) { exited = true; cx=C(0); cy=C(0); break; }
            ie = next;
            if (!get_elt(ie,mode,x1,y1,x2,y2,x3,y3,D12,D13,D23,delta,s1,s2,s3,Mprop,cx,cy))
              { exited = true; break; }
            if (mode >= Mprop) { exited = true; break; }
            DB1=xB*y1-yB*x1; DB2=xB*y2-yB*x2; DB3=xB*y3-yB*x3;
            A1=(D23-DB3+DB2)/delta; A2=(DB3-D13-DB1)/delta; A3=(-DB2+DB1+D12)/delta;
          }
          if (!exited) {
            cB = A1/nk[s1*M_max+mode]+A2/nk[s2*M_max+mode]+A3/nk[s3*M_max+mode];
            phiB = A1*npr[s1*M_max+mode]+A2*npr[s2*M_max+mode]+A3*npr[s3*M_max+mode];
            B_s1=s1; B_s2=s2; B_s3=s3;
            B_A1=A1; B_A2=A2; B_A3=A3;
          }
        }

        // Backward influence: same loop as forward but accumulate gradients
        double _xA=xA-sx, _yA=yA-sy, _xB=xB-sx, _yB=yB-sy;
        for (int64_t it = 0; it < n_theta; ++it) {
          double rv1=rc[it], rv2=rs[it];
          double dA = rv1*xiA + rv2*etaA;
          double dB = rv1*xiB + rv2*etaB;
          if (std::abs(dA)<1e-30 || std::abs(dB)<1e-30 || dA*dB<=0) continue;
          double rA_v = (_yA*etaA+_xA*xiA)/dA;
          double rB_v = (_yB*etaB+_xB*xiB)/dB;
          int64_t ir1 = std::max(std::min((int64_t)((rA_v-r_min)/delta_r)+1, n_r), (int64_t)0);
          int64_t ir2 = std::max(std::min((int64_t)((rB_v-r_min)/delta_r)+1, n_r), (int64_t)1);
          if (ir2 <= ir1) continue;
          double nA_v = (_xA*rv2-_yA*rv1)/(cA_r*dA);
          double nB_v = (_xB*rv2-_yB*rv1)/(std::real(cB)*dB);
          double rBA = rB_v - rA_v;
          if (std::abs(rBA) < 1e-30) continue;
          for (int64_t ir = ir1+1; ir <= std::min(ir2, n_r); ++ir) {
            double W = (r_min + (ir-1)*delta_r - rA_v) / rBA;
            C pM = pA + W*(pB-pA);
            C qM = qA + W*(qB-qA);
            double n2 = (nA_v + W*(nB_v-nA_v)); n2 = n2*n2;
            if (qM != C(0) && -0.5*std::imag(pM/qM)*n2 < BEAM_WINDOW) {
              C cM = cA + W*(cB-cA);
              C tM = tauA + W*(tauB-tauA);
              C phM = phiA + W*(phiB-phiA);
              int km = KMAHA;
              if (std::real(qM)<0) {
                if ((std::imag(qA)<0 && std::imag(qM)>=0)||(std::imag(qA)>0 && std::imag(qM)<=0))
                  km = -km;
              }
              C contrib = cnst * phM * std::sqrt(cM/qM) * std::exp(C(0,-1)*(tM + C(0.5)*pM/qM*n2));
              if (km < 0) contrib = -contrib;

              // grad_P contribution
              C gp = gP[it*n_r + ir-1];
              if (std::abs(gp) < 1e-30) continue;

              // d(contrib)/d(phiB) = W * contrib/phM (phi enters linearly)
              if (std::abs(phM) > 1e-30) {
                C dcdphi = W * contrib / phM;
                C grad_phi_contrib = gp * dcdphi;
                // Distribute to nodes at B via barycentric
                gpr[B_s1*M_max+mode] += std::conj(grad_phi_contrib) * C(B_A1);
                gpr[B_s2*M_max+mode] += std::conj(grad_phi_contrib) * C(B_A2);
                gpr[B_s3*M_max+mode] += std::conj(grad_phi_contrib) * C(B_A3);
              }

              // d(contrib)/d(cB) → d(cB)/d(ki) = -Ai/ki²
              if (std::abs(cM) > 1e-30) {
                C dcdc = W * contrib * C(0.5) / cM;  // from sqrt(cM/qM) term
                C grad_c_contrib = gp * dcdc;
                C k1 = nk[B_s1*M_max+mode], k2 = nk[B_s2*M_max+mode], k3 = nk[B_s3*M_max+mode];
                gk[B_s1*M_max+mode] += std::conj(grad_c_contrib) * C(-B_A1) / (k1*k1);
                gk[B_s2*M_max+mode] += std::conj(grad_c_contrib) * C(-B_A2) / (k2*k2);
                gk[B_s3*M_max+mode] += std::conj(grad_c_contrib) * C(-B_A3) / (k3*k3);
              }
            }
          }
        }

        xA=xB; yA=yB; xiA=xiB; etaA=etaB;
        pA=pB; qA=qB; tauA=tauB; cA=cB;
        KMAHA=KMAHB; phiA=phiB;
        cur_s1=B_s1; cur_s2=B_s2; cur_s3=B_s3;
        cur_A1=B_A1; cur_A2=B_A2; cur_A3=B_A3;
      }
    }
  }
  return std::make_tuple(grad_k, grad_phi_r);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csr_values_to_band", &csr_values_to_band);
  m.def("csr_lu_factorize", &csr_lu_factorize);
  m.def("lu_factorize", &lu_factorize);
  m.def("lu_solve", &lu_solve);
  m.def("lu_solve_adjoint", &lu_solve_adjoint);
  m.def("acoustic_recurrence_fwd", &acoustic_recurrence_fwd);
  m.def("acoustic_recurrence_bwd", &acoustic_recurrence_bwd);
  m.def("searchsorted_lerp_fwd", &searchsorted_lerp_fwd);
  m.def("searchsorted_lerp_bwd", &searchsorted_lerp_bwd);
  m.def("solve_tridiag", &solve_tridiag);
  m.def("solve_tridiag_batch", &solve_tridiag_batch);
  m.def("acoustic_recurrence_scalar_counted",
        &acoustic_recurrence_scalar_counted);
  m.def("acoustic_solve1", &acoustic_solve1);
  m.def("acoustic_solve2", &acoustic_solve2);
  m.def("dispersion_complex_batch", &dispersion_complex_batch_cpp);
  m.def("dispersion_complex_scalar", &dispersion_complex_scalar_cpp);
  m.def("tridiag_inverse_iteration_batch", &tridiag_inverse_iteration_batch_cpp);
  m.def("fused_assembly_inverse_iteration", &fused_assembly_inverse_iteration_cpp);
  m.def("refine_roots", &refine_roots_cpp);
  m.def("elastic_solve1", &elastic_solve1_cpp);
  m.def("krakenc_fused", &krakenc_fused_cpp);
  m.def("field3d_gbt", &field3d_gbt_cpp);
  m.def("field3d_gbt_backward", &field3d_gbt_backward_cpp);
}
