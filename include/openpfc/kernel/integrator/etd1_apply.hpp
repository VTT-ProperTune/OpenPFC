// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file etd1_apply.hpp
 * @brief Host ETD1 candidate combine: `out = exp_Ldt * u + phi1_L * N`
 *
 * @details
 * Applies already-built diagonal ETD1 coefficients to a field (or Fourier
 * buffer). `phi1_L` is `(exp(L*dt)-1)/L` and already includes the factor of
 * `dt`; do not multiply by `dt` again.
 *
 * Device-resident combine lives in `runtime/gpu/etd1_apply_gpu.hpp` and uses
 * the generic two-term kernel (`combine_two_term_*`).
 *
 * @see spectral_exp_coefficients.hpp
 * @see runtime/gpu/etd1_apply_gpu.hpp
 */

#include <span>
#include <stdexcept>
#include <string>

namespace pfc::integrator {

/**
 * @brief Combine ETD1 weights on the host.
 *
 * @tparam Scalar Field element (`double` or `std::complex<double>`).
 * @tparam Real   Coefficient type (default `double`).
 *
 * @throws std::invalid_argument if any span length differs from `u.size()`.
 */
template <class Scalar, class Real = double>
inline void apply_etd1_update(std::span<const Real> exp_Ldt,
                              std::span<const Real> phi1_L,
                              std::span<const Scalar> u,
                              std::span<const Scalar> n_of_u,
                              std::span<Scalar> candidate) {
  const std::size_t n = u.size();
  if (exp_Ldt.size() != n || phi1_L.size() != n || n_of_u.size() != n ||
      candidate.size() != n) {
    throw std::invalid_argument(
        "apply_etd1_update: span sizes must match u.size() (" +
        std::to_string(n) + ")");
  }
  for (std::size_t i = 0; i < n; ++i) {
    candidate[i] =
        Scalar(exp_Ldt[i]) * u[i] + Scalar(phi1_L[i]) * n_of_u[i];
  }
}

} // namespace pfc::integrator
