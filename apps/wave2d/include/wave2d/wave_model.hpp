// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_model.hpp
 * @brief 2D wave equation as coupled first-order system — physics only.
 *
 * \f$\partial_t u = v\f$, \f$\partial_t v = c^2 \Delta u\f$ on a 2D slab
 * (`nz == 1` in OpenPFC indexing). Laplacian entries are **unscaled** FD
 * sums (same convention as `FdGradient`): multiply by `inv_dx2` / `inv_dy2`
 * outside or fold into `c`.
 */

#include <tuple>

namespace wave2d {

/** Wave speed \f$c\f$ shared by all wave2d binaries (change here for experiments).
 */
inline constexpr double kC = 1.0;

/** Per-point Laplacian aggregate (unscaled 3-point sums along x and y). */
struct WaveLaplacian {
  double lxx = 0.0;
  double lyy = 0.0;
};

/** Increments \f$(du, dv)\f$ for `MultiEulerStepper` tuple protocol. */
struct WaveIncrements {
  double du = 0.0;
  double dv = 0.0;
  auto as_tuple() { return std::tie(du, dv); }
  auto as_tuple() const { return std::tie(du, dv); }
};

/**
 * @brief Point-wise RHS: \f$du = v\f$, \f$dv = c^2 (l_{xx} + l_{yy})\f$
 *        with metric scaling.
 */
struct WaveModel {
  double inv_dx2 = 1.0;
  double inv_dy2 = 1.0;

  [[nodiscard]] WaveIncrements rhs(double /*t*/, double v_val,
                                   const WaveLaplacian &lap) const noexcept {
    const double lap_u = inv_dx2 * lap.lxx + inv_dy2 * lap.lyy;
    return WaveIncrements{v_val, kC * kC * lap_u};
  }
};

} // namespace wave2d
