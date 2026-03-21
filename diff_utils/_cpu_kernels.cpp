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
  return v;
}

template <>
inline c10::complex<float> maybe_conj(c10::complex<float> v,
                                      bool complex_case) {
  return complex_case ? std::conj(v) : v;
}

template <>
inline c10::complex<double> maybe_conj(c10::complex<double> v,
                                       bool complex_case) {
  return complex_case ? std::conj(v) : v;
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

        at::parallel_for(0, M, 1, [&](int64_t m_begin, int64_t m_end) {
          for (int64_t m = m_begin; m < m_end; ++m) {
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
      });

  return std::make_tuple(f_num, g_val, p_history);
}

// adjoint recurrence
// grad_f_num[M], grad_g_val[M] → grad_B1[N], grad_h2k2[M],
// grad_p1_init[M], grad_p2_init[M]
//  grad_z* = conj(J^T @ conj(grad_w)).
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

// binary search + linear interpolation
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

        at::parallel_for(0, M, 1, [&](int64_t m0, int64_t m1) {
          for (int64_t m = m0; m < m1; ++m) {
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
}
