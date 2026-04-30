// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file discretization.hpp
 * @brief Point-wise spatial-operator abstraction for heat3d.
 *
 * Splits the spatial discretization from the physics (RHS) and the time
 * integrator. The user-facing form is point-wise:
 *
 *     du[{i,j,k}] = rhs(t, grad_evaluator(i,j,k))
 *
 * where `grad_evaluator(i,j,k)` returns a `GradPoint` carrying the partial
 * derivatives the model needs at one local interior cell. The same point-wise
 * form works for any backend that can produce a `GradPoint`:
 *
 *  - **`FdGradient`** (this header): on-the-fly FD stencils, reads `core` only
 *    (face halos are not needed in the interior `[hw, n-hw)` because, with
 *    `halo_width >= order/2`, all stencil neighbours stay in the local core).
 *  - A spectral evaluator (future): pre-materializes derivative fields with
 *    one forward FFT plus one inverse FFT per requested derivative; the
 *    `operator()(i,j,k)` then reads from those arrays.
 *
 * Today's `GradPoint` carries only what `heat3d` actually consumes:
 * `u`, `uxx`, `uyy`, `uzz`. Mixed terms (`uxy`, `uxz`, `uyz`) require corner
 * halos on the FD side and are intentionally out of scope of this header.
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

namespace heat3d {

/**
 * @brief Local partial derivatives of a scalar field at one interior cell.
 *
 * Extend with `ux, uy, uz, uxy, ...` when a model needs them; the FD and
 * spectral backends are responsible for filling whichever fields exist.
 */
struct GradPoint {
  double u{};
  double uxx{};
  double uyy{};
  double uzz{};
};

/**
 * @brief Point-wise FD evaluator for `GradPoint{u, uxx, uyy, uzz}`.
 *
 * Wraps the same even-order central second-derivative stencil used by
 * `pfc::field::fd::laplacian_even_order_interior_separated`, but exposes the
 * three axes separately so a model can combine them as it likes.
 *
 * Construction binds to a contiguous local `core` array of shape `nx*ny*nz`
 * (x varies fastest) and the per-axis grid metric. `prepare()` is a no-op for
 * FD: the application is responsible for any halo exchange before iterating
 * (face halos are unused in the interior; see file-level docs).
 *
 * @note `operator()` is `const`/`noexcept` and intentionally inlines the
 *       stencil arithmetic so the surrounding `for_each_interior` loop fuses
 *       into a single tight kernel.
 */
class FdGradient {
public:
  FdGradient(const double *core, int nx, int ny, int nz, double inv_dx2,
             double inv_dy2, double inv_dz2, int halo_width, int order)
      : m_core(core), m_nx(nx), m_ny(ny), m_nz(nz),
        m_sxy(static_cast<std::ptrdiff_t>(nx) * static_cast<std::ptrdiff_t>(ny)),
        m_hw(halo_width), m_inv_dx2(inv_dx2), m_inv_dy2(inv_dy2),
        m_inv_dz2(inv_dz2) {
    pfc::field::fd::detail::EvenFdStencil1d st{};
    if (pfc::field::fd::detail::fd_even_order_lookup(order, &st)) {
      m_stencil = st;
      const double inv_den = 1.0 / static_cast<double>(st.denom);
      m_sx = inv_dx2 * inv_den;
      m_sy = inv_dy2 * inv_den;
      m_sz = inv_dz2 * inv_den;
    }
  }

  void prepare() noexcept {}

  int imin() const noexcept { return m_hw; }
  int imax() const noexcept { return m_nx - m_hw; }
  int jmin() const noexcept { return m_hw; }
  int jmax() const noexcept { return m_ny - m_hw; }
  int kmin() const noexcept { return m_hw; }
  int kmax() const noexcept { return m_nz - m_hw; }

  inline std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_nx) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_sxy);
  }

  inline GradPoint operator()(int ix, int iy, int iz) const noexcept {
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(idx(ix, iy, iz));
    const double uc = m_core[c];
    const int M = m_stencil.half_width;
    const std::int64_t *coeff = m_stencil.coeffs;

    double dxx = static_cast<double>(coeff[0]) * uc;
    double dyy = static_cast<double>(coeff[0]) * uc;
    double dzz = static_cast<double>(coeff[0]) * uc;
    for (int k = 1; k <= M; ++k) {
      const double ck = static_cast<double>(coeff[k]);
      const std::ptrdiff_t kx = static_cast<std::ptrdiff_t>(k);
      const std::ptrdiff_t ky = kx * static_cast<std::ptrdiff_t>(m_nx);
      const std::ptrdiff_t kz = kx * m_sxy;
      dxx += ck * (m_core[c - kx] + m_core[c + kx]);
      dyy += ck * (m_core[c - ky] + m_core[c + ky]);
      dzz += ck * (m_core[c - kz] + m_core[c + kz]);
    }
    GradPoint g;
    g.u = uc;
    g.uxx = m_sx * dxx;
    g.uyy = m_sy * dyy;
    g.uzz = m_sz * dzz;
    return g;
  }

private:
  const double *m_core{nullptr};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::ptrdiff_t m_sxy{0};
  int m_hw{0};
  double m_inv_dx2{0.0};
  double m_inv_dy2{0.0};
  double m_inv_dz2{0.0};
  double m_sx{0.0};
  double m_sy{0.0};
  double m_sz{0.0};
  pfc::field::fd::detail::EvenFdStencil1d m_stencil{};
};

/**
 * @brief Point-wise spectral evaluator for `GradPoint{u, uxx, uyy, uzz}`.
 *
 * Materializes the second-derivative fields once per call to `prepare()` using
 * one forward FFT of the bound input field plus one inverse FFT per axis
 * (multiplying by the spectral symbols \f$-k_x^2, -k_y^2, -k_z^2\f$). After
 * `prepare()`, `operator()(ix,iy,iz)` is just three reads + one read of the
 * input field for `g.u`.
 *
 * Trade-offs vs. the implicit-Fourier path used by `run_spectral` today:
 *  - This evaluator supports an arbitrary point-wise RHS (any model that fits
 *    `GradPoint`), but uses **explicit** time stepping (CFL-limited).
 *  - The implicit-Fourier path is **2 FFTs/step** and unconditionally stable
 *    for the linear heat operator, but is specific to a constant-coefficient
 *    diffusion and cannot accommodate non-linear point-wise terms.
 *
 * Memory: one inbox-sized real buffer per derivative + one outbox-sized
 * complex buffer for the forward transform + one for the per-axis multiply.
 */
class SpectralGradient {
public:
  /**
   * @param fft         FFT plan (caller-owned; must outlive this object).
   * @param u_in        Bound input field; the FFT API requires a `std::vector`,
   *                    and the bound vector must outlive this object and not
   *                    be reassigned/resized between `prepare()` calls.
   * @param global_size Global grid size `{Nx, Ny, Nz}`.
   * @param spacing     Grid spacing `{dx, dy, dz}`.
   * @param inbox       Local real-space box for this rank (from
   * `IFFT::get_inbox_bounds()`).
   * @param outbox      Local Fourier-space box for this rank (from
   * `IFFT::get_outbox_bounds()`).
   */
  SpectralGradient(pfc::fft::IFFT &fft, const pfc::fft::RealVector &u_in,
                   std::array<int, 3> global_size, std::array<double, 3> spacing,
                   pfc::fft::Box3i inbox, pfc::fft::Box3i outbox)
      : m_fft(&fft), m_u_in(&u_in), m_inbox(inbox), m_outbox(outbox),
        m_nx(inbox.size[0]), m_ny(inbox.size[1]), m_nz(inbox.size[2]) {
    const std::size_t in_n = fft.size_inbox();
    const std::size_t out_n = fft.size_outbox();
    m_uxx.assign(in_n, 0.0);
    m_uyy.assign(in_n, 0.0);
    m_uzz.assign(in_n, 0.0);
    m_u_F.assign(out_n, std::complex<double>{});
    m_tmp_F.assign(out_n, std::complex<double>{});
    m_op_xx.assign(out_n, 0.0);
    m_op_yy.assign(out_n, 0.0);
    m_op_zz.assign(out_n, 0.0);

    const double fx = 2.0 * pfc::constants::pi /
                      (spacing[0] * static_cast<double>(global_size[0]));
    const double fy = 2.0 * pfc::constants::pi /
                      (spacing[1] * static_cast<double>(global_size[1]));
    const double fz = 2.0 * pfc::constants::pi /
                      (spacing[2] * static_cast<double>(global_size[2]));

    std::size_t idx = 0;
    for (int kk = outbox.low[2]; kk <= outbox.high[2]; ++kk) {
      const double kz = (kk <= global_size[2] / 2)
                            ? static_cast<double>(kk) * fz
                            : static_cast<double>(kk - global_size[2]) * fz;
      for (int jj = outbox.low[1]; jj <= outbox.high[1]; ++jj) {
        const double ky = (jj <= global_size[1] / 2)
                              ? static_cast<double>(jj) * fy
                              : static_cast<double>(jj - global_size[1]) * fy;
        for (int ii = outbox.low[0]; ii <= outbox.high[0]; ++ii) {
          const double kx = (ii <= global_size[0] / 2)
                                ? static_cast<double>(ii) * fx
                                : static_cast<double>(ii - global_size[0]) * fx;
          m_op_xx[idx] = -kx * kx;
          m_op_yy[idx] = -ky * ky;
          m_op_zz[idx] = -kz * kz;
          ++idx;
        }
      }
    }
  }

  void prepare() {
    m_fft->forward(*m_u_in, m_u_F);
    fft_axis(m_op_xx, m_uxx);
    fft_axis(m_op_yy, m_uyy);
    fft_axis(m_op_zz, m_uzz);
  }

  int imin() const noexcept { return 0; }
  int imax() const noexcept { return m_nx; }
  int jmin() const noexcept { return 0; }
  int jmax() const noexcept { return m_ny; }
  int kmin() const noexcept { return 0; }
  int kmax() const noexcept { return m_nz; }

  inline std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_nx) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_nx) *
               static_cast<std::size_t>(m_ny);
  }

  inline GradPoint operator()(int ix, int iy, int iz) const noexcept {
    const std::size_t c = idx(ix, iy, iz);
    GradPoint g;
    g.u = (*m_u_in)[c];
    g.uxx = m_uxx[c];
    g.uyy = m_uyy[c];
    g.uzz = m_uzz[c];
    return g;
  }

private:
  void fft_axis(const std::vector<double> &op, std::vector<double> &out) {
    const std::size_t n = m_u_F.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_tmp_F[i] = m_u_F[i] * op[i];
    }
    m_fft->backward(m_tmp_F, out);
  }

  pfc::fft::IFFT *m_fft{nullptr};
  const pfc::fft::RealVector *m_u_in{nullptr};
  pfc::fft::Box3i m_inbox{};
  pfc::fft::Box3i m_outbox{};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::vector<double> m_uxx;
  std::vector<double> m_uyy;
  std::vector<double> m_uzz;
  std::vector<std::complex<double>> m_u_F;
  std::vector<std::complex<double>> m_tmp_F;
  std::vector<double> m_op_xx;
  std::vector<double> m_op_yy;
  std::vector<double> m_op_zz;
};

/**
 * @brief Apply a point-wise RHS to every interior cell of a discretization.
 *
 * `eval` provides the interior bounds (`imin/imax`, `jmin/jmax`, `kmin/kmax`),
 * the linear index `idx(ix,iy,iz)`, an optional `prepare()` (e.g. FFTs for a
 * spectral evaluator), and `operator()(ix,iy,iz) -> GradPoint`. `rhs(t, g)`
 * returns the time derivative for that cell. Cells outside the interior are
 * left untouched (`du` is not cleared here).
 *
 * Parallelized with `#pragma omp parallel for collapse(2)` over `(iz, iy)`,
 * matching the strategy used by `heat3d` today.
 */
template <class Eval, class Rhs>
inline void for_each_interior(Eval &eval, double *du, double t, Rhs &&rhs) {
  eval.prepare();
  const int kmin = eval.kmin();
  const int kmax = eval.kmax();
  const int jmin = eval.jmin();
  const int jmax = eval.jmax();
  const int imin = eval.imin();
  const int imax = eval.imax();
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const GradPoint g = eval(ix, iy, iz);
        du[eval.idx(ix, iy, iz)] = rhs(t, g);
      }
    }
  }
}

} // namespace heat3d
