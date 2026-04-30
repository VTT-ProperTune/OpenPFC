// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_gradient.hpp
 * @brief Point-wise spectral evaluator parameterized on a model-owned grads type.
 *
 * @details
 * `SpectralGradient<G>` materializes the partial-derivative fields requested
 * by the model-owned grads aggregate `G`. Once per call to `prepare()` it
 * runs one forward FFT of the bound input field plus one inverse FFT per
 * requested derivative member (with the appropriate spectral symbol):
 *
 *  - Second derivatives: real symbol \f$-k_i^2\f$ (one inverse FFT per
 *    `xx`, `yy`, `zz` member declared by `G`).
 *  - First derivatives: imaginary symbol \f$\mathrm{i}\,k_i\f$ (handled via
 *    a complex spectral multiply followed by an inverse FFT to a real
 *    buffer).
 *  - Mixed second derivatives: real symbol \f$-k_i\,k_j\f$ (one inverse FFT
 *    per `xy`, `xz`, `yz` member declared by `G`).
 *
 * After `prepare()`, `operator()(ix,iy,iz)` is a straight read from the
 * materialized buffers (and from the input field for `g.value`). Buffers
 * for derivatives `G` does not declare are never allocated and never
 * computed.
 *
 * Trade-offs vs. an implicit-Fourier path that solves
 * \f$\hat u^{n+1} = \hat u^n / (1 - \Delta t \, D \, k_{\mathrm{lap}})\f$:
 *  - This evaluator supports an arbitrary point-wise RHS (any model that
 *    fits a per-point grads contract), but uses **explicit** time stepping
 *    (CFL-limited).
 *  - The implicit-Fourier path is **2 FFTs/step** and unconditionally
 *    stable for the linear heat operator, but is specific to a
 *    constant-coefficient diffusion and cannot accommodate non-linear
 *    point-wise terms.
 *
 * @see grad_concepts.hpp for the per-member detection concepts
 * @see grad_point.hpp for the convenience default catalog struct
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
#include <openpfc/kernel/field/grad_concepts.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace pfc::field {

/**
 * @brief Spectral point evaluator templated on a model-owned grads aggregate.
 *
 * @tparam G Per-point grads aggregate. The constructor allocates a real
 *           inbox-sized buffer per declared derivative member, plus the
 *           shared forward/temp complex spectra. The cost of `prepare()`
 *           is therefore proportional to the number of derivative members
 *           in `G`.
 */
template <class G> class SpectralGradient {
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
    m_u_F.assign(out_n, std::complex<double>{});
    m_tmp_F.assign(out_n, std::complex<double>{});

    if constexpr (has_x<G>) m_dx.assign(in_n, 0.0);
    if constexpr (has_y<G>) m_dy.assign(in_n, 0.0);
    if constexpr (has_z<G>) m_dz.assign(in_n, 0.0);
    if constexpr (has_xx<G>) m_dxx.assign(in_n, 0.0);
    if constexpr (has_yy<G>) m_dyy.assign(in_n, 0.0);
    if constexpr (has_zz<G>) m_dzz.assign(in_n, 0.0);
    if constexpr (has_xy<G>) m_dxy.assign(in_n, 0.0);
    if constexpr (has_xz<G>) m_dxz.assign(in_n, 0.0);
    if constexpr (has_yz<G>) m_dyz.assign(in_n, 0.0);

    if constexpr (has_x<G>) m_op_x.assign(out_n, std::complex<double>{});
    if constexpr (has_y<G>) m_op_y.assign(out_n, std::complex<double>{});
    if constexpr (has_z<G>) m_op_z.assign(out_n, std::complex<double>{});
    if constexpr (has_xx<G>) m_op_xx.assign(out_n, 0.0);
    if constexpr (has_yy<G>) m_op_yy.assign(out_n, 0.0);
    if constexpr (has_zz<G>) m_op_zz.assign(out_n, 0.0);
    if constexpr (has_xy<G>) m_op_xy.assign(out_n, 0.0);
    if constexpr (has_xz<G>) m_op_xz.assign(out_n, 0.0);
    if constexpr (has_yz<G>) m_op_yz.assign(out_n, 0.0);

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
          if constexpr (has_x<G>) m_op_x[idx] = std::complex<double>(0.0, kx);
          if constexpr (has_y<G>) m_op_y[idx] = std::complex<double>(0.0, ky);
          if constexpr (has_z<G>) m_op_z[idx] = std::complex<double>(0.0, kz);
          if constexpr (has_xx<G>) m_op_xx[idx] = -kx * kx;
          if constexpr (has_yy<G>) m_op_yy[idx] = -ky * ky;
          if constexpr (has_zz<G>) m_op_zz[idx] = -kz * kz;
          if constexpr (has_xy<G>) m_op_xy[idx] = -kx * ky;
          if constexpr (has_xz<G>) m_op_xz[idx] = -kx * kz;
          if constexpr (has_yz<G>) m_op_yz[idx] = -ky * kz;
          ++idx;
        }
      }
    }
  }

  void prepare() {
    m_fft->forward(*m_u_in, m_u_F);
    if constexpr (has_x<G>) invert_complex_op(m_op_x, m_dx);
    if constexpr (has_y<G>) invert_complex_op(m_op_y, m_dy);
    if constexpr (has_z<G>) invert_complex_op(m_op_z, m_dz);
    if constexpr (has_xx<G>) invert_real_op(m_op_xx, m_dxx);
    if constexpr (has_yy<G>) invert_real_op(m_op_yy, m_dyy);
    if constexpr (has_zz<G>) invert_real_op(m_op_zz, m_dzz);
    if constexpr (has_xy<G>) invert_real_op(m_op_xy, m_dxy);
    if constexpr (has_xz<G>) invert_real_op(m_op_xz, m_dxz);
    if constexpr (has_yz<G>) invert_real_op(m_op_yz, m_dyz);
  }

  int imin() const noexcept { return 0; }
  int imax() const noexcept { return m_nx; }
  int jmin() const noexcept { return 0; }
  int jmax() const noexcept { return m_ny; }
  int kmin() const noexcept { return 0; }
  int kmax() const noexcept { return m_nz; }

  [[nodiscard]] std::size_t idx(int ix, int iy, int iz) const noexcept {
    return static_cast<std::size_t>(ix) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_nx) +
           static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_nx) *
               static_cast<std::size_t>(m_ny);
  }

  [[nodiscard]] G operator()(int ix, int iy, int iz) const noexcept {
    G g{};
    const std::size_t c = idx(ix, iy, iz);
    if constexpr (has_value<G>) g.value = (*m_u_in)[c];
    if constexpr (has_x<G>) g.x = m_dx[c];
    if constexpr (has_y<G>) g.y = m_dy[c];
    if constexpr (has_z<G>) g.z = m_dz[c];
    if constexpr (has_xx<G>) g.xx = m_dxx[c];
    if constexpr (has_yy<G>) g.yy = m_dyy[c];
    if constexpr (has_zz<G>) g.zz = m_dzz[c];
    if constexpr (has_xy<G>) g.xy = m_dxy[c];
    if constexpr (has_xz<G>) g.xz = m_dxz[c];
    if constexpr (has_yz<G>) g.yz = m_dyz[c];
    return g;
  }

private:
  void invert_real_op(const std::vector<double> &op, std::vector<double> &out) {
    const std::size_t n = m_u_F.size();
    for (std::size_t i = 0; i < n; ++i) m_tmp_F[i] = m_u_F[i] * op[i];
    m_fft->backward(m_tmp_F, out);
  }

  void invert_complex_op(const std::vector<std::complex<double>> &op,
                         std::vector<double> &out) {
    const std::size_t n = m_u_F.size();
    for (std::size_t i = 0; i < n; ++i) m_tmp_F[i] = m_u_F[i] * op[i];
    m_fft->backward(m_tmp_F, out);
  }

  pfc::fft::IFFT *m_fft{nullptr};
  const pfc::fft::RealVector *m_u_in{nullptr};
  pfc::fft::Box3i m_inbox{};
  pfc::fft::Box3i m_outbox{};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};

  std::vector<std::complex<double>> m_u_F;
  std::vector<std::complex<double>> m_tmp_F;

  std::vector<double> m_dx, m_dy, m_dz;
  std::vector<double> m_dxx, m_dyy, m_dzz;
  std::vector<double> m_dxy, m_dxz, m_dyz;

  std::vector<std::complex<double>> m_op_x, m_op_y, m_op_z;
  std::vector<double> m_op_xx, m_op_yy, m_op_zz;
  std::vector<double> m_op_xy, m_op_xz, m_op_yz;
};

/**
 * @brief Free-function factory: build a `SpectralGradient<G>` from an FFT
 *        plan and a `LocalField`.
 *
 * Mirrors the `world::create`, `decomposition::create`, `fft::create` family:
 * derives the local inbox bounds and the local Fourier outbox bounds from
 * `fft`, and the global grid size and grid spacing from `u`. The caller
 * supplies the model-owned grads type as an explicit template argument and
 * the two pieces that are not derivable from the field (the FFT plan and
 * the field itself).
 *
 * Example:
 * @code
 * struct HeatGrads { double xx{}, yy{}, zz{}; };
 * auto grad = pfc::field::create<HeatGrads>(stack.u(), stack.fft());
 * @endcode
 *
 * @tparam G    Model-owned grads aggregate.
 * @param u    Local field bound to the FFT inbox layout (must outlive the
 *             returned evaluator and not be reassigned/resized between
 *             `prepare()` calls; the evaluator stores a pointer to
 *             `u.vec()`).
 * @param fft  FFT plan (caller-owned; must outlive the returned evaluator).
 *
 * @return A `SpectralGradient<G>` ready to be passed to
 *         `pfc::sim::for_each_interior` (or `pfc::sim::steppers::create`).
 */
template <class G>
[[nodiscard]] inline SpectralGradient<G> create(LocalField<double> &u,
                                                pfc::fft::IFFT &fft) {
  return SpectralGradient<G>(fft, u.vec(), u.global_size(), u.spacing(),
                             fft.get_inbox_bounds(), fft.get_outbox_bounds());
}

} // namespace pfc::field
