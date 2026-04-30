// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_heat_propagator.hpp
 * @brief Implicit-Euler-in-Fourier-space propagator for the heat equation.
 *
 * @details
 * One step of \f$\partial_t u = D\nabla^2 u\f$ in Fourier space with
 * implicit Euler is the elementwise multiply
 * \f$\hat u^{n+1}(\mathbf k) = \hat u^{n}(\mathbf k)\,/\,
 *    \bigl(1 - \Delta t\, D\, k_\mathrm{lap}(\mathbf k)\bigr)\f$
 * with \f$k_\mathrm{lap} = -(k_x^2 + k_y^2 + k_z^2)\f$.
 *
 * The wavenumber lookup-table \f$1/(1 - \Delta t\,D\,k_\mathrm{lap})\f$
 * is heat-specific (it embeds the operator's symbol), so this lives in
 * the heat3d application rather than in OpenPFC. The constructor builds
 * the table from the FFT layout once; `step` is a fwd FFT, an
 * elementwise multiply, and an inv FFT.
 *
 * Backend-agnostic: holds a reference to the abstract `pfc::fft::IFFT`,
 * so the same class works with CPU FFTW, cuFFT, ROCm — whatever
 * `fft::create` returns.
 *
 * Lifetime contract: the propagator borrows the FFT by reference, so the
 * `IFFT` instance must outlive the propagator (same pattern as
 * `pfc::field::SpectralGradient`).
 */

#include <complex>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace heat3d {

/**
 * @brief Implicit-Euler propagator for \f$\partial_t u = D\nabla^2 u\f$.
 *
 * Construct once from the FFT and the field's geometry; call `step(u)`
 * each time-step to advance `u` by `dt` in place.
 */
class SpectralHeatPropagator {
public:
  /**
   * @param fft FFT plan to reuse (borrowed; must outlive the propagator).
   * @param u   Field that defines the global grid size + spacing used to
   *            build the wavenumber table. The propagator does not
   *            retain a reference to this field — only its geometry is
   *            sampled at construction.
   * @param D   Diffusion coefficient (heat-equation parameter).
   * @param dt  Time-step size.
   */
  SpectralHeatPropagator(pfc::fft::IFFT &fft,
                         const pfc::field::LocalField<double> &u, double D,
                         double dt)
      : m_fft(fft), m_psi_F(fft.size_outbox()), m_opL(fft.size_outbox()) {
    const auto size = u.global_size();
    const auto spacing = u.spacing();
    const auto ob = fft.get_outbox_bounds();
    const double fx =
        2.0 * pfc::constants::pi / (spacing[0] * static_cast<double>(size[0]));
    const double fy =
        2.0 * pfc::constants::pi / (spacing[1] * static_cast<double>(size[1]));
    const double fz =
        2.0 * pfc::constants::pi / (spacing[2] * static_cast<double>(size[2]));
    std::size_t idx = 0;
    for (int k = ob.low[2]; k <= ob.high[2]; ++k) {
      for (int j = ob.low[1]; j <= ob.high[1]; ++j) {
        for (int i = ob.low[0]; i <= ob.high[0]; ++i) {
          const double ki = (i <= size[0] / 2)
                                ? static_cast<double>(i) * fx
                                : static_cast<double>(i - size[0]) * fx;
          const double kj = (j <= size[1] / 2)
                                ? static_cast<double>(j) * fy
                                : static_cast<double>(j - size[1]) * fy;
          const double kk = (k <= size[2] / 2)
                                ? static_cast<double>(k) * fz
                                : static_cast<double>(k - size[2]) * fz;
          const double k_lap = -(ki * ki + kj * kj + kk * kk);
          m_opL[idx++] = 1.0 / (1.0 - dt * D * k_lap);
        }
      }
    }
  }

  /** Advance `u` by one implicit-Euler step (1 fwd FFT + 1 inv FFT). */
  void step(pfc::field::LocalField<double> &u) {
    m_fft.forward(u.vec(), m_psi_F);
    for (std::size_t k = 0; k < m_psi_F.size(); ++k) m_psi_F[k] *= m_opL[k];
    m_fft.backward(m_psi_F, u.vec());
  }

private:
  pfc::fft::IFFT &m_fft;
  std::vector<std::complex<double>> m_psi_F;
  std::vector<double> m_opL;
};

} // namespace heat3d
