// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file affine_laplacian.hpp
 * @brief Fourier Laplacian on an affine image of the Cartesian reference cell.
 *
 * Wavevectors are the reference-cell components from `for_each_kpoint`
 * (`k_ξ`), not physical Cartesian components. An identity Jacobian calls
 * `k_laplacian_value` and does not form `A^{-1} A^{-T}`. Any other constant
 * nonsingular `A` multiplies each mode by `-k_ξ^T (A^{-1} A^{-T}) k_ξ`.
 * Existing Cartesian solvers keep calling `k_laplacian_value` directly.
 */

#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/geometry/affine_map.hpp>

#include <cstddef>
#include <stdexcept>

namespace pfc::fft {

[[nodiscard]] inline double laplacian_symbol(const geometry::AffineMap &map,
                                             double kx, double ky,
                                             double kz) noexcept {
  if (map.uses_cartesian_laplacian()) {
    return kspace::k_laplacian_value(kx, ky, kz);
  }
  return map.metric_laplacian_symbol(kx, ky, kz);
}

/**
 * @brief Multiply each local outbox coefficient by the affine Laplacian symbol.
 *
 * `coefficients` is the unscaled real-to-complex spectrum in outbox order.
 * The identity Jacobian uses `k_laplacian_value` at every mode. A size other
 * than the outbox count throws.
 */
template <class Complex>
void apply_affine_laplacian(const geometry::AffineMap &map, const Box3i &outbox,
                            std::array<int, 3> global_size,
                            std::array<double, 3> spacing, Complex *coefficients,
                            std::size_t count) {
  if (static_cast<long long>(count) != outbox.count() ||
      (coefficients == nullptr && count != 0)) {
    throw std::invalid_argument(
        "affine Laplacian needs one coefficient per FFT outbox point");
  }
  if (map.uses_cartesian_laplacian()) {
    kspace::for_each_kpoint(
        outbox, global_size, spacing,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          coefficients[idx] *= kspace::k_laplacian_value(kx, ky, kz);
        });
    return;
  }
  kspace::for_each_kpoint(
      outbox, global_size, spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        coefficients[idx] *= map.metric_laplacian_symbol(kx, ky, kz);
      });
}

} // namespace pfc::fft
