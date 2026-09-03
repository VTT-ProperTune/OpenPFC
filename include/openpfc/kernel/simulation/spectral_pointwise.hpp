// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_pointwise.hpp
 * @brief Per-cell nonlinearity contract for spectral-ETD physics.
 *
 * @details
 * A spectral-ETD physics splits into two parts:
 *
 * - **k-space symbols** (`linear_symbol`, optional `filter_mf`,
 *   `correlation_kernel`, `nonlinear_symbol`): host functions evaluated once
 *   per operator preparation.
 * - **the pointwise part**: a small trivially-copyable functor returned by
 *   `physics.pointwise()`. `SpectralETDSystem` evaluates it once per cell per
 *   step, on the host or inside a GPU kernel, so it must be device-callable
 *   (`OPENPFC_HD`) and must carry every constant it needs by value.
 *
 * The functor receives a `SpectralCell` holding the primary field value
 * and, when the physics declares the corresponding k-space operator, the
 * mean-field filtered value and the correlation-kernel convolution, plus the
 * cell coordinates and time. This header is deliberately light so a device
 * translation unit can include it without pulling in `SimulationState`.
 *
 * @see spectral_etd_system.hpp
 * @see runtime/gpu/spectral_pointwise_gpu.hpp (device evaluation)
 */

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::sim {

/**
 * @brief Inputs to a pointwise spectral-ETD nonlinearity at one cell.
 *
 * `psi_mf` is zero unless the physics has `filter_mf(k)`; `p_star` is zero
 * unless it has `correlation_kernel(k)`.
 */
struct SpectralCell {
  double psi{};    ///< primary field value
  double psi_mf{}; ///< mean-field filtered value  (\f$\chi * \psi\f$)
  double p_star{}; ///< correlation-kernel convolution (\f$P * \psi\f$)
  double x{};      ///< physical coordinates of the cell
  double y{};
  double z{};
  double t{}; ///< current time
};

/**
 * @brief Local owned-box geometry used to form cell coordinates.
 *
 * Coordinates are formed as `origin + (low + i) * spacing` exactly as
 * `pfc::data::Field::coords` does, so host and device agree bit-for-bit.
 */
struct PointwiseGeometry {
  int nx{};
  int ny{};
  int nz{};
  int low_x{};
  int low_y{};
  int low_z{};
  double origin_x{};
  double origin_y{};
  double origin_z{};
  double dx{};
  double dy{};
  double dz{};

  OPENPFC_HD std::size_t volume() const noexcept {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz);
  }

  /// Physical coordinates of the x-fastest linear index @p idx.
  OPENPFC_HD void coords(std::size_t idx, double &x, double &y,
                         double &z) const noexcept {
    const std::size_t nxy = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const std::size_t k = idx / nxy;
    const std::size_t rem = idx - k * nxy;
    const std::size_t j = rem / static_cast<std::size_t>(nx);
    const std::size_t i = rem - j * static_cast<std::size_t>(nx);
    x = origin_x + static_cast<double>(low_x + static_cast<int>(i)) * dx;
    y = origin_y + static_cast<double>(low_y + static_cast<int>(j)) * dy;
    z = origin_z + static_cast<double>(low_z + static_cast<int>(k)) * dz;
  }
};

/**
 * @brief Device-capable pointwise nonlinearity: `nonlinearity(cell)`.
 *
 * Trivially copyable so it can be passed by value to a GPU kernel.
 */
template <class F>
concept SpectralPointwise =
    std::is_trivially_copyable_v<F> &&
    requires(const F &f, const SpectralCell &c) {
      { f.nonlinearity(c) } -> std::convertible_to<double>;
    };

/**
 * @brief Optional per-cell free-energy density observable.
 */
template <class F>
concept HasFreeEnergyDensity = requires(const F &f, const SpectralCell &c) {
  { f.free_energy_density(c) } -> std::convertible_to<double>;
};

/**
 * @brief Host evaluation of a pointwise functor over an owned box.
 *
 * @param g       Local geometry (x-fastest, halo 0).
 * @param t       Current time.
 * @param psi     Primary field (length `g.volume()`).
 * @param psi_mf  Mean-field values or `nullptr`.
 * @param p_star  Correlation convolution or `nullptr`.
 * @param n_out   Nonlinearity output (length `g.volume()`).
 * @param fe_out  Free-energy density output, or `nullptr`; written only when
 *                `F` models `HasFreeEnergyDensity`.
 */
template <class F>
  requires SpectralPointwise<F>
inline void for_each_spectral_cell(const PointwiseGeometry &g, double t,
                                   const double *psi, const double *psi_mf,
                                   const double *p_star, double *n_out,
                                   double *fe_out, const F &f) {
  const std::size_t n = g.volume();
  for (std::size_t idx = 0; idx < n; ++idx) {
    SpectralCell c;
    c.psi = psi[idx];
    c.psi_mf = (psi_mf != nullptr) ? psi_mf[idx] : 0.0;
    c.p_star = (p_star != nullptr) ? p_star[idx] : 0.0;
    g.coords(idx, c.x, c.y, c.z);
    c.t = t;
    n_out[idx] = f.nonlinearity(c);
    if constexpr (HasFreeEnergyDensity<F>) {
      if (fe_out != nullptr) {
        fe_out[idx] = f.free_energy_density(c);
      }
    } else {
      (void)fe_out;
    }
  }
}

} // namespace pfc::sim
