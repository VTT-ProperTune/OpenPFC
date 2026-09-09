// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nonlinear.hpp
 * @brief Nonlinear compliant lubrication: \f$h^3\f$ mobility, load, adhesion (`#116`).
 *
 * @details
 * The shipped `ehd_film` preset freezes the hydrodynamic mobility at
 * \f$M_0=M(h_0)\f$: exact for the linear \f$k^6\f$ bending-relaxation band,
 * but not an elastohydrodynamic lubrication problem. This header adds the
 * pieces the science case in `#116` needs on top of it:
 *
 * \f[
 *   p = B\nabla^4h - \gamma\nabla^2h - \Pi(h) + p_{\mathrm{ext}}(x,y,t),
 *   \qquad
 *   \partial_t h = \nabla\cdot\bigl[M(h)\nabla p\bigr],
 *   \qquad
 *   M(h) = M_0\left(\frac{h}{h_0}\right)^{3}.
 * \f]
 *
 * ## Sign conventions
 *
 * * `p` follows `ehd_film_physics.hpp`: \f$p=B\nabla^4h-\gamma\nabla^2h
 *   -\Pi(h)\f$, flux down the pressure gradient
 *   (\f$\partial_t h=\nabla\cdot[M\nabla p]\f$). With OpenPFC's
 *   \f$k_{\mathrm{lap}}=-|k|^2\f$ this gives pure-bending decay
 *   \f$\lambda(k)=-M_0Bk^6<0\f$ — the existing verifier.
 * * \f$M(h)=M_0(h/h_0)^3\f$ is written in units of \f$M_0\f$, so it reduces
 *   to the constant-mobility linear model at \f$h=h_0\f$; in dimensional
 *   lubrication terms \f$M=h^3/(3\eta)\f$, \f$M_0=h_0^3/(3\eta)\f$ (same
 *   normalisation as `thin_film::CubicMobility`).
 * * \f$p_{\mathrm{ext}}>0\f$ is a **load pushing the plate down onto the
 *   film** (a positive applied pressure). A localized positive
 *   \f$p_{\mathrm{ext}}\f$ has \f$\nabla^2p_{\mathrm{ext}}<0\f$ at its
 *   centre, so \f$\partial_th=\nabla\cdot[M\nabla p]\approx M\nabla^2
 *   p_{\mathrm{ext}}<0\f$ there: the gap thins under the load and thickens
 *   in the surrounding annulus where fluid is displaced to — squeeze flow,
 *   not suction. This sign is asserted directly in
 *   `test_ehd_film.cpp` ("p_ext enters..."), not merely assumed.
 * * Adhesion is the same disjoining pressure as `thin_film`/`ehd_film`:
 *   \f$\Pi(h)=A[(h_0/h)^3-(h_0/h)^9]\f$ (or the `h_star` precursor form,
 *   see `ehd_film_pointwise.hpp`). \f$A>0\f$ is attractive van der Waals at
 *   \f$h_0\f$; \f$-\Pi(h)\f$ enters `p` with the same sign as bending and
 *   tension.
 *
 * ## Numerics
 *
 * The flux is evaluated with `pfc::apps::FluxETD`
 * (`openpfc_apps/spectral_flux.hpp`): state-dependent \f$M\f$ cannot be
 * written as a reciprocal-space symbol, so it is applied in real space every
 * step with Orszag 2/3 dealiasing on the flux transform. Host (CPU) only —
 * see that header for why. The existing constant-mobility `ehd_film_hip`
 * binary is untouched; the nonlinear science preset has no GPU twin.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

#include <ehd_film/ehd_film_pointwise.hpp>

namespace ehd_film {

/// Cubic lubrication mobility, normalised so that \f$M(h_0)=M_0\f$. Same law
/// as `thin_film::CubicMobility`; copied rather than shared (app convention).
struct CubicMobility {
  double M0{1.0};
  double h0{1.0};
  /// A plate pinned to near-zero gap must not give M <= 0.
  static constexpr double kMinH = 1.0e-6;

  [[nodiscard]] double operator()(double h) const {
    const double u = std::max(h, kMinH) / h0;
    return M0 * u * u * u;
  }
};

/**
 * @brief Localized external load, applied and then removed.
 *
 * \f[
 *   p_{\mathrm{ext}}(x,y,t) =
 *     \begin{cases}
 *       p_0\exp\!\bigl(-r^2/2a^2\bigr) & 0\le t<t_{\mathrm{load}} \\
 *       0 & t\ge t_{\mathrm{load}}
 *     \end{cases}
 * \f]
 *
 * models a stamp or roller pressing the plate onto the film for a finite
 * dwell time, then lifting off so the run shows loading followed by
 * recovery/redistribution (`#116`, computational experiment B).
 */
struct GaussianLoad {
  double p0{0.0};      ///< peak applied pressure
  double a{1.0};       ///< Gaussian width (code length units)
  double t_load{0.0};  ///< dwell time; load is on for t in [0, t_load)
  double cx{0.0}, cy{0.0};

  [[nodiscard]] double operator()(double x, double y, double t) const {
    if (p0 == 0.0 || !(t < t_load)) return 0.0;
    const double dx = x - cx, dy = y - cy;
    const double r2 = dx * dx + dy * dy;
    return p0 * std::exp(-r2 / (2.0 * a * a));
  }

  /// Centre the load in @p domain.
  static GaussianLoad centred(const pfc::Domain &domain, double p0, double a,
                              double t_load) {
    const auto size = pfc::domain::get_size(domain);
    const auto dx = pfc::domain::get_spacing(domain);
    return GaussianLoad{p0, a, t_load, 0.5 * size[0] * dx[0],
                        0.5 * size[1] * dx[1]};
  }
};

/// Observables a compliant-lubrication load/relaxation experiment reports.
struct EhdFilmSample {
  double h_center{};        ///< thickness at the domain centre
  double p_max{}, p_min{};  ///< pressure extrema
  double spreading_radius{};///< RMS radius of the disturbance about centre
  double displaced_volume{};///< volume squeezed out from under the load
  double volume{};          ///< total fluid volume (conservation check)
};

/**
 * @brief Collective EHD-film observables at the domain centre and overall.
 *
 * @param h      thickness field
 * @param p      pressure field, evaluated at the same instant as @p h
 * @param domain grid geometry
 * @param h0     reference (unloaded) thickness
 * @param comm   communicator to reduce over
 *
 * `h_center` is read at the grid point nearest the domain centre — exact
 * when the grid size is even, so each science/verifier case uses an even
 * `Lx`, `Ly`. `spreading_radius` is the \f$(h-h_0)^2\f$-weighted RMS radius
 * of the disturbance, the natural second-moment analogue of a spreading
 * length for a localized depression/bump. `displaced_volume` is
 * \f$\int\max(0,h_0-h)\,\mathrm dA\f$, the volume squeezed out from under
 * the load; by exact volume conservation it must equal the volume gained in
 * the surrounding annulus, up to the reported round-off drift in `volume`.
 */
[[nodiscard]] inline EhdFilmSample
sample_ehd_film(pfc::data::Field<double> &h, pfc::data::Field<double> &p,
                const pfc::Domain &domain, double h0, MPI_Comm comm) {
  const auto size = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);
  const double cx = 0.5 * static_cast<double>(size[0]) * dx[0];
  const double cy = 0.5 * static_cast<double>(size[1]) * dx[1];
  const double cell = dx[0] * dx[1] * dx[2];

  double local_h_center = 0.0;
  double local_pmax = -std::numeric_limits<double>::infinity();
  double local_pmin = std::numeric_limits<double>::infinity();
  double local_num = 0.0, local_den = 0.0; // spreading-radius second moment
  double local_displaced = 0.0;
  double local_volume = 0.0;

  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = h.coords(i, j, k);
        const double hv = h(i, j, k);
        const double pv = p(i, j, k);
        local_volume += hv * cell;
        local_displaced += std::max(0.0, h0 - hv) * cell;
        local_pmax = std::max(local_pmax, pv);
        local_pmin = std::min(local_pmin, pv);
        const double dev = hv - h0;
        const double w = dev * dev;
        const double rdx = x[0] - cx, rdy = x[1] - cy;
        local_num += w * (rdx * rdx + rdy * rdy);
        local_den += w;
        if (std::abs(x[0] - cx) <= 0.5 * dx[0] + 1.0e-9 &&
            std::abs(x[1] - cy) <= 0.5 * dx[1] + 1.0e-9) {
          local_h_center = hv; // exactly one grid point qualifies globally
        }
      }
    }
  }

  double g_hcenter = 0.0, g_pmax = 0.0, g_pmin = 0.0;
  double lsum[4]{local_num, local_den, local_displaced, local_volume};
  double gsum[4]{};
  MPI_Allreduce(&local_h_center, &g_hcenter, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_pmax, &g_pmax, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_pmin, &g_pmin, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(lsum, gsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  EhdFilmSample s;
  s.h_center = g_hcenter;
  s.p_max = g_pmax;
  s.p_min = g_pmin;
  s.spreading_radius = (gsum[1] > 0.0) ? std::sqrt(gsum[0] / gsum[1]) : 0.0;
  s.displaced_volume = gsum[2];
  s.volume = gsum[3];
  return s;
}

} // namespace ehd_film
