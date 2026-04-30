// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_gradient.hpp
 * @brief Point-wise spectral evaluator that fills `pfc::field::GradPoint`.
 *
 * @details
 * `SpectralGradient` materializes the second-derivative fields once per call
 * to `prepare()` using one forward FFT of the bound input field plus one
 * inverse FFT per axis (multiplying by the spectral symbols
 * \f$-k_x^2, -k_y^2, -k_z^2\f$). After `prepare()`, `operator()(ix,iy,iz)`
 * is just three reads + one read of the input field for `g.u`.
 *
 * Trade-offs vs. an implicit-Fourier path that solves
 * \f$\hat u^{n+1} = \hat u^n / (1 - \Delta t \, D \, k_{\mathrm{lap}})\f$:
 *  - This evaluator supports an arbitrary point-wise RHS (any model that
 *    fits `GradPoint`), but uses **explicit** time stepping (CFL-limited).
 *  - The implicit-Fourier path is **2 FFTs/step** and unconditionally stable
 *    for the linear heat operator, but is specific to a constant-coefficient
 *    diffusion and cannot accommodate non-linear point-wise terms.
 *
 * Memory: one inbox-sized real buffer per derivative + one outbox-sized
 * complex buffer for the forward transform + one for the per-axis multiply.
 *
 * @see grad_point.hpp for the interface struct
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 * @see openpfc/kernel/fft/fft_interface.hpp for the FFT contract
 */

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/grad_point.hpp>

namespace pfc::field {

class SpectralGradient {
public:
  /**
   * @param fft         FFT plan (caller-owned; must outlive this object).
   * @param u_in        Bound input field; the FFT API requires a `std::vector`,
   *                    and the bound vector must outlive this object and not
   *                    be reassigned/resized between `prepare()` calls.
   * @param global_size Global grid size `{Nx, Ny, Nz}`.
   * @param spacing     Grid spacing `{dx, dy, dz}`.
   * @param inbox       Local real-space box for this rank
   *                    (from `IFFT::get_inbox_bounds()`).
   * @param outbox      Local Fourier-space box for this rank
   *                    (from `IFFT::get_outbox_bounds()`).
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

} // namespace pfc::field
