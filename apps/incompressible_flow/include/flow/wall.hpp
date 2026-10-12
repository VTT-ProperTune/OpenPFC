// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wall.hpp
 * @brief Impermeable projection for one Fourier × Fourier × Chebyshev channel.
 *
 * The Chebyshev layer owns the Neumann Poisson solve and the mixed
 * gradient. This application composes them so the pressure gradient
 * is removed and the normal acceleration vanishes at both Chebyshev
 * ends. No-slip stays Dirichlet data on the velocity.
 */

#include <span>
#include <vector>

namespace flow {

struct WallAcceleration {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
};

/// Remove the pressure gradient from a momentum tendency.
/// The acceleration approximates the continuous divergence-free projection.
/// Its sampled divergence converges with polynomial degree; it is not an
/// exact discrete Leray projector on an underresolved pressure space.
/// Its wall-normal component
/// vanishes at both Chebyshev ends. The pressure solve is the
/// Neumann Poisson problem on a degree-N+2 pressure grid: dp/dz equals the normal
/// tendency at each end. No-slip is the Dirichlet data on the
/// velocity, not a penalty on a periodic mode. The tendency polynomial is
/// prolonged, pressure is differentiated before restriction to degree N,
/// and the caller's velocity grid is unchanged.
[[nodiscard]] WallAcceleration
impermeable_acceleration(std::span<const double> tendency_x,
                         std::span<const double> tendency_y,
                         std::span<const double> tendency_z, int nx, int ny,
                         double period_x, double period_y);

} // namespace flow
