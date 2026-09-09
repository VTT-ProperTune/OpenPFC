// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nonlinear.hpp
 * @brief Full lubrication thin film: \f$h^3\f$ mobility, rupture, dewetting.
 *
 * @details
 * The shipped spectral preset freezes the hydrodynamic mobility at
 * \f$M_0=M(h_0)\f$, which is exact for the linear band and wrong for
 * everything after it. For a Newtonian no-slip film the lubrication mobility
 * is cubic in thickness,
 *
 * \f[
 *   \partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],
 *   \qquad
 *   p = -\gamma\nabla^2 h - \Pi(h),
 *   \qquad
 *   M(h) = M_0\left(\frac{h}{h_0}\right)^{3},
 * \f]
 *
 * and that cubic factor is what decides where the film ruptures: a thinning
 * region loses mobility as \f$h^3\f$, so drainage stalls and the depression
 * sharpens into a hole instead of relaxing. Constant mobility cannot produce
 * that, which is why the linear preset is a verifier and not the science case.
 *
 * \f$M_0(h/h_0)^3\f$ is written in units of \f$M_0\f$ so the nonlinear model
 * reduces to the linear one at \f$h=h_0\f$; in dimensional terms
 * \f$M=h^3/(3\eta)\f$ with \f$M_0=h_0^3/(3\eta)\f$.
 *
 * The flux is evaluated with `pfc::apps::FluxETD`; see
 * `openpfc_apps/spectral_flux.hpp` for why a state-dependent mobility cannot
 * be written as a reciprocal-space symbol.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>

#include <thin_film/thin_film_pointwise.hpp>

namespace thin_film {

/// Cubic lubrication mobility, normalised so that \f$M(h_0)=M_0\f$.
///
/// Trivially copyable and `OPENPFC_HD` so it can run through
/// `pfc::apps::MobilityGradPointwise` on host or device (see
/// `apps/thin_film/src/gpu/thin_film_pointwise.inc`).
struct CubicMobility {
  double M0{1.0};
  double h0{1.0};
  /// Films are clamped away from zero; a ruptured cell must not give M < 0.
  static constexpr double kMinH = 1.0e-6;

  [[nodiscard]] OPENPFC_HD double operator()(double h) const {
    const double u = (h > kMinH ? h : kMinH) / h0;
    return M0 * u * u * u;
  }
};

/**
 * @brief Device-capable adapter: \f$\Pi(h)\f$ as a `SpectralPointwise`
 *        functor, for computing the full disjoining potential (not the
 *        ETD remainder `ThinFilmPointwise::nonlinearity`).
 */
struct PotentialPointwise {
  ThinFilmPointwise pw{};

  [[nodiscard]] OPENPFC_HD double
  nonlinearity(const pfc::sim::SpectralCell &cell) const {
    return pw.Pi(cell.psi);
  }
};

/// Observables that describe a dewetting film rather than a decaying mode.
struct FilmSample {
  double min_h{}, max_h{}, mean_h{}, volume{};
  double hole_area_fraction{}; ///< fraction of cells below `h0/2`
  double dominant_spacing{};   ///< from the structure factor, in code lengths
  bool ruptured{false};        ///< min thickness fell below the rupture cut
};

/**
 * @brief Collective film observables.
 *
 * @param h            thickness field
 * @param domain       grid geometry
 * @param h0           reference thickness, for the hole criterion
 * @param rupture_frac rupture when `min_h < rupture_frac * h0`
 * @param comm         communicator to reduce over
 */
template <class MemorySpace = pfc::HostSpace>
[[nodiscard]] inline FilmSample
sample_film(pfc::data::Field<double, MemorySpace> &h, const pfc::Domain &domain,
           double h0, double rupture_frac, MPI_Comm comm) {
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  double local_sum = 0.0, local_holes = 0.0, local_cells = 0.0;
  h.with_host_view([&](const double *d, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      lo = std::min(lo, d[i]);
      hi = std::max(hi, d[i]);
      local_sum += d[i];
      if (d[i] < 0.5 * h0) local_holes += 1.0;
      local_cells += 1.0;
    }
  });
  double g_lo = 0, g_hi = 0, g[3]{}, l[3]{local_sum, local_holes, local_cells};
  MPI_Allreduce(&lo, &g_lo, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&hi, &g_hi, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(l, g, 3, MPI_DOUBLE, MPI_SUM, comm);

  const auto dx = pfc::domain::get_spacing(domain);
  const double cell = dx[0] * dx[1] * dx[2];
  FilmSample s;
  s.min_h = g_lo;
  s.max_h = g_hi;
  s.mean_h = (g[2] > 0.0) ? g[0] / g[2] : 0.0;
  s.volume = g[0] * cell;
  s.hole_area_fraction = (g[2] > 0.0) ? g[1] / g[2] : 0.0;
  s.ruptured = g_lo < rupture_frac * h0;
  return s;
}

/**
 * @brief A localized thickness depression, for defect-triggered dewetting.
 *
 * Returns the *relative* depression \f$-a\exp(-r^2/2\sigma^2)\f$ at a point,
 * so the caller composes it into one initial condition rather than writing the
 * field twice:
 *
 * \f[
 *   h(\mathbf x,0) = h_0\bigl[1 + \epsilon\,\xi(\mathbf x)
 *                              + g(\mathbf x)\bigr].
 * \f]
 *
 * Comparing this against a homogeneous-noise run is the engineering question:
 * does the film fail where it is disturbed, or where the instability wants?
 */
struct GaussianDefect {
  double amplitude{0.0}; ///< depth as a fraction of h0
  double sigma{1.0};     ///< width in code lengths
  double cx{0.0}, cy{0.0};

  [[nodiscard]] double operator()(double x, double y) const {
    if (amplitude == 0.0) return 0.0;
    const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    return -amplitude * std::exp(-r2 / (2.0 * sigma * sigma));
  }

  /// Centre the defect in @p domain.
  static GaussianDefect centred(const pfc::Domain &domain, double amplitude,
                                double sigma) {
    const auto size = pfc::domain::get_size(domain);
    const auto dx = pfc::domain::get_spacing(domain);
    return GaussianDefect{amplitude, sigma, 0.5 * size[0] * dx[0],
                          0.5 * size[1] * dx[1]};
  }
};

} // namespace thin_film
