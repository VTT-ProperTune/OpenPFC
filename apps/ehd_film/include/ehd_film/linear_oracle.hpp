// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file linear_oracle.hpp
 * @brief Exact constant-mobility forced bending+tension solution (`#610`).
 *
 * @details
 * Linearize the lubrication plate about \f$h=h_0\f$ with constant mobility
 * \f$M_0\f$, no adhesion (\f$A=0\f$), and deflection \f$u=h-h_0\f$:
 *
 * \f[
 *   p = B\nabla^4 u - \gamma\nabla^2 u + p_{\mathrm{ext}},
 *   \qquad
 *   \partial_t u = M_0\nabla^2 p.
 * \f]
 *
 * Each nonzero Fourier mode then satisfies
 *
 * \f[
 *   \partial_t\hat u_k = -\lambda_k\hat u_k - M_0 k^2\hat p_{\mathrm{ext},k},
 *   \qquad
 *   \lambda_k = M_0(B k^6 + \gamma k^4).
 * \f]
 *
 * For a time-independent load on \f$0\le t<T\f$ and \f$\hat u_k(0)=0\f$,
 *
 * \f[
 *   \hat u_k(t)
 *   = -\frac{\hat p_{\mathrm{ext},k}}{k^2(B k^2+\gamma)}
 *     \bigl(1-e^{-\lambda_k t}\bigr),
 *   \qquad k\neq 0.
 * \f]
 *
 * After switch-off at \f$T\f$,
 *
 * \f[
 *   \hat u_k(t)=\hat u_k(T)\,e^{-\lambda_k(t-T)}.
 * \f]
 *
 * The zero mode is unchanged because the load enters through
 * \f$\nabla^2 p_{\mathrm{ext}}\f$. Use the DFT of the same sampled load and
 * the same periodic wavenumbers as the solver; do not replace it with a
 * continuum Gaussian transform.
 */

#include <algorithm>
#include <cmath>
#include <complex>

namespace ehd_film {

/// Linear decay rate \f$\lambda(k^2)=M_0(B k^6+\gamma k^4)\f$.
[[nodiscard]] inline double forced_linear_decay_rate(double k2, double B,
                                                     double gamma, double M0) {
  const double k4 = k2 * k2;
  return M0 * (B * k4 * k2 + gamma * k4);
}

/**
 * @brief Exact Fourier coefficient of deflection for a gated constant load.
 *
 * @param p_hat  DFT coefficient of the sampled load (same convention as
 *               the solver's forward transform)
 * @param k2     \f$|k|^2\f$ of this mode
 * @param t      observation time
 * @param t_load load is on for \f$t\in[0,t_{\mathrm{load}})\f$
 */
[[nodiscard]] inline std::complex<double>
forced_linear_u_hat(std::complex<double> p_hat, double k2, double B, double gamma,
                    double M0, double t, double t_load) {
  if (!(k2 > 0.0) || t <= 0.0) {
    return {0.0, 0.0};
  }
  const double lambda = forced_linear_decay_rate(k2, B, gamma, M0);
  const double stiff = k2 * (B * k2 + gamma);
  const auto drive = -p_hat / stiff;
  const double t_on = std::min(t, t_load);
  auto u = drive * (1.0 - std::exp(-lambda * t_on));
  if (t > t_load) {
    u *= std::exp(-lambda * (t - t_load));
  }
  return u;
}

} // namespace ehd_film
