// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <span>
#include <string>
#include <vector>

#include <allen_cahn/rank_report.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

namespace allen_cahn {

/**
 * @brief Shipped preset and the thresholds that keep it honest.
 *
 * The preset moved in 0.2 from `64^2, dt=9e-5, eps=0.19, F=10` to the values
 * below, for two coupled reasons measured in
 * `out/report/data/allen_cahn_resolution_margin.csv`:
 *
 * 1. **The old interface was sub-grid.** `eps*sqrt(2M) = 0.76` cells, so the
 *    front was pinned by the lattice rather than set by the continuum
 *    physics. Against the curvature-corrected sharp-interface law it ran
 *    `-32%` slow at a comfortable driving force.
 * 2. **The old driving force was 6% under the bistability ceiling** — and
 *    that was load-bearing, not incidental. Backing `F` off at
 *    `eps = 0.19` walks the error from `+17%` (at the ceiling) to `-65%` (a
 *    quarter of it): at a 0.76-cell interface the *only* driving force that
 *    reproduces the continuum law is one so large that the double well has
 *    almost stopped being a double well. The margin cannot be fixed on its
 *    own; the resolution has to be fixed first.
 *
 * At `eps = 0.75` the interface is 3.0 cells wide, `F` sits a factor 2.7
 * under the ceiling, and the measured `dR/dt` matches the curvature-corrected
 * law to 2.3% on every grid from `128^2` to `512^2`.
 */
struct RunConfig {
  /**
   * `256^2`, not `64^2`. A resolved interface forces a much larger critical
   * nucleus: `R* = M/v = delta/(3 F eps^2)`, which is 7.1 cells here against
   * 0.70 for the old preset. The seed (`seed_radius_cells`) has to clear
   * `R*` by a healthy factor or it dissolves instead of growing, and at
   * `64^2` it does not — the seed is 4.1 cells and vanishes. `256^2` puts a
   * 16.7-cell seed against a 7.1-cell critical radius.
   */
  int nx_glob = 256;
  int ny_glob = 256;
  /**
   * Long enough for the front to outrun the Gaussian initial condition and
   * settle into steady propagation, which is what the built-in check measures.
   * Below `kMinStepsForKinetics` the check reports SKIPPED rather than a
   * number the transient dominates.
   */
  int n_steps = 5000;
  /**
   * 16% of the explicit-Euler diffusive limit `dx^2/(4M) = 0.031` (the
   * reaction limit `~eps^2 = 0.56` is far looser at this `eps`). The
   * superlevel counts are unchanged from `dt/8`, and `dt = 0.031` blows up.
   */
  double dt = 0.005;
  double M = 8.0;
  /** Sets the interface width `eps*sqrt(2M) = 3.0` cells at `dx = 1`. */
  double epsilon = 0.75;
  /**
   * Positive bulk driving term that favors the φ≈+1 seed over the φ≈-1
   * matrix. `F eps^2 = 0.141` against the bistability ceiling `0.385`, i.e.
   * a factor 2.7 of margin (37% of the way to the ceiling).
   */
  double driving_force = 0.25;
  /** If non-empty, gather the final scalar field on rank 0 and write a grayscale
   * PNG. */
  std::string png_output;
  /** If non-empty, write the field right after IC, before time stepping. */
  std::string png_output_initial;
  /** Opt in to letting a failed physics check set the process exit status. */
  bool strict = false;
  /**
   * Periodic pair of flat cubic-heteroclinic fronts instead of the Gaussian
   * seed. The disc `physics_check` is not applied to this geometry. The
   * cell-count columns still print, and the front position is extra output.
   */
  bool two_front = false;
  static constexpr int kHaloWidth = 1;
  /** Superlevel for the seed-area metric: φ > 0 matches the visible seed in PNGs. */
  static constexpr double kLevelSetThreshold = 0.0;
  /**
   * Band around the curvature-corrected sharp-interface prediction that
   * `v_late` must land in.
   *
   * ±25%, where it used to be a factor of two either way. The factor of two
   * was the price of a 0.76-cell interface compared against a *flat*-front
   * law; both halves of that are now fixed, so the band can describe the
   * physics instead of the discretisation. What ±25% has to cover:
   *
   * - the finite-tilt correction, `<=7%` at the shipped margin — the
   *   `(3/2) F eps sqrt(2M)` law is the `F eps^2 -> 0` limit, and the exact
   *   travelling wave of the tilted cubic is faster than it by
   *   `1 + O((F eps^2)^2)`;
   * - the lattice correction, `<=3%` for `eps*sqrt(2M) >= 2` cells;
   * - the quantisation of a superlevel *cell count* into a radius.
   *
   * Measured residual at the shipped preset: `1.6%`, and at most `2.3%` on
   * any grid from `128^2` to `512^2`.
   */
  static constexpr double kVelocityBandLo = 0.75;
  static constexpr double kVelocityBandHi = 1.25;
  /**
   * Interface width, in cells, below which the front is lattice-limited.
   *
   * At `eps*sqrt(2M) = 0.76` the measured speed is `-32%` off the
   * curvature-corrected law; at `1.5` cells it is `+0.05%`, at `2.0` cells
   * `+1.2%`. Two cells is the first round number on the flat part of that
   * curve. The app warns below it rather than refusing to run — an
   * under-resolved run is a legitimate thing to ask for, it just is not a
   * measurement of the continuum physics.
   */
  static constexpr double kMinInterfaceWidthCells = 2.0;
  /**
   * How far towards the bistability ceiling a preset may sit.
   *
   * `F eps^2` must stay under `2/(3 sqrt 3)` for the unfavoured phase to
   * exist at all (`max_bistable_driving_force`). Half of that leaves the
   * matrix well at `phi = -0.88` with 34% of the symmetric barrier depth,
   * and leaves room for a user to double `F` or add 40% to `eps` before the
   * box flips. The shipped preset sits at `0.37`, comfortably inside.
   */
  static constexpr double kMaxDrivingForceFraction = 0.5;
  /** Quarter-to-quarter agreement required before `v_late` counts as steady. */
  static constexpr double kSteadyTolerance = 0.25;
  /** Below this the last-half displacement is swamped by area quantisation. */
  static constexpr double kMinMeasurableAdvanceCells = 1.0;
  /** Fewer steps than this cannot outrun the initial-condition transient. */
  static constexpr int kMinStepsForKinetics = 400;
};

/**
 * @brief Strip `--strict` out of `argv`, leaving the positional layout intact.
 *
 * The rest of the CLI is positional by index, so a flag can only be added by
 * removing it before those indices are read. Returns the new `argc`; rewrites
 * `argv` in place.
 */
inline int extract_flags(int argc, char **argv, RunConfig *c) {
  int out = 0;
  for (int i = 0; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--strict") {
      c->strict = true;
      continue;
    }
    if (arg == "--two-front") {
      c->two_front = true;
      continue;
    }
    argv[out++] = argv[i];
  }
  return out;
}

inline RunConfig parse_args(int argc, char **argv) {
  RunConfig c;
  argc = extract_flags(argc, argv, &c);
  if (argc > 1) {
    c.nx_glob = std::atoi(argv[1]);
  }
  if (argc > 2) {
    c.ny_glob = std::atoi(argv[2]);
  }
  if (argc > 3) {
    c.n_steps = std::atoi(argv[3]);
  }
  if (argc > 4) {
    c.dt = std::atof(argv[4]);
  }
  if (argc > 5) {
    c.M = std::atof(argv[5]);
  }
  if (argc > 6) {
    c.epsilon = std::atof(argv[6]);
  }
  if (argc > 7) {
    char *end = nullptr;
    const double parsed = std::strtod(argv[7], &end);
    int png_arg = 7;
    if (end != argv[7] && *end == '\0') {
      c.driving_force = parsed;
      png_arg = 8;
    }
    if (argc > png_arg) {
      if (argc > png_arg + 1) {
        c.png_output_initial = argv[png_arg];
        c.png_output = argv[png_arg + 1];
      } else {
        c.png_output = argv[png_arg];
      }
    }
  }
  return c;
}

/** Gaussian width of the seed, in cells: 5.5% of the shorter side, min 2. */
[[nodiscard]] inline double seed_sigma_cells(int nx, int ny) noexcept {
  return std::max(2.0, 0.055 * static_cast<double>(std::min(nx, ny)));
}

/**
 * @brief Radius of the visible seed at t=0, in cells.
 *
 * The IC is `phi = -1 + 2 exp(-r^2/(2 sigma^2))`, so the `phi > 0` contour
 * the app measures sits at `r = sigma sqrt(2 ln 2)`. Worth having as a
 * function rather than a comment: it has to be compared with
 * `critical_radius_cells` before a preset can be said to demonstrate growth.
 */
[[nodiscard]] inline double seed_radius_cells(int nx, int ny) noexcept {
  return seed_sigma_cells(nx, ny) * std::sqrt(2.0 * std::log(2.0));
}

/**
 * @brief Phase field at t=0: one "grain" as a Gaussian bump on a φ≈-1 matrix.
 *
 * φ(g) = -1 + 2 exp( -r² / (2σ²) ), with r measured from the domain center in
 * index space. At the center φ→+1; far from the center φ→-1.
 */
inline void fill_initial_condition(std::vector<double> *u,
                                   const pfc::decomposition::Decomposition &decomp,
                                   int rank) {
  const auto &gw = pfc::decomposition::domain(decomp);
  auto gsz = pfc::domain::get_size(gw);
  const auto &local = pfc::decomposition::local_box(decomp, rank);
  auto lo = local.low;
  auto sz = local.size;
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int sxy = nx * ny;
  const double cx = 0.5 * static_cast<double>(gsz[0] - 1);
  const double cy = 0.5 * static_cast<double>(gsz[1] - 1);
  const double sigma = seed_sigma_cells(gsz[0], gsz[1]);
  const double denom = 2.0 * sigma * sigma;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = lo[0] + ix;
        const int gy = lo[1] + iy;
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const double dxg = static_cast<double>(gx) - cx;
        const double dyg = static_cast<double>(gy) - cy;
        const double r2 = dxg * dxg + dyg * dyg;
        (*u)[idx] = -1.0 + 2.0 * std::exp(-r2 / denom);
      }
    }
  }
}

/** Local count of cells with φ strictly above @p threshold. */
inline std::int64_t count_cells_above(const double *u, std::size_t n_cells,
                                      double threshold) {
  std::int64_t n = 0;
  for (std::size_t i = 0; i < n_cells; ++i) {
    if (u[i] > threshold) {
      ++n;
    }
  }
  return n;
}

/** Local count of cells with φ strictly above @p threshold. */
inline std::int64_t count_cells_above(const std::vector<double> &u,
                                      double threshold) {
  return count_cells_above(u.data(), u.size(), threshold);
}

/** Global superlevel cell count, summed across `comm` and known to every rank. */
inline std::int64_t global_area_cells(MPI_Comm comm, std::int64_t n_local) {
  std::int64_t n = 0;
  MPI_Allreduce(&n_local, &n, 1, MPI_INT64_T, MPI_SUM, comm);
  return n;
}

/** Radius of the disc with the same area as @p area_cells cells of size `dx`. */
[[nodiscard]] inline double equivalent_radius(std::int64_t area_cells,
                                              double dx) noexcept {
  if (area_cells <= 0) {
    return 0.0;
  }
  const double area = static_cast<double>(area_cells) * dx * dx;
  return std::sqrt(area / 3.14159265358979323846);
}

/**
 * @brief Disc radius for a fractional cell area.
 *
 * Named apart from `equivalent_radius(std::int64_t, double)` so a call with
 * an untyped `0` still selects the integer count.
 */
[[nodiscard]] inline double equivalent_radius_from_area(double area_cells,
                                                        double dx) noexcept {
  if (!(area_cells > 0.0)) {
    return 0.0;
  }
  return std::sqrt(area_cells * dx * dx / 3.14159265358979323846);
}

/**
 * @brief Which way a sampled edge crosses `phi = 0`.
 *
 * `Rising` is the edge from a non-positive sample to a positive one, which is
 * the left side of a `phi > 0` slab. `Falling` is the opposite edge.
 */
enum class Crossing { Rising, Falling };

/** Linear `phi = 0` position on one periodic row, in sample-index units. */
struct FrontHit {
  double position = 0.0;
  bool found = false;
};

/**
 * @brief Shortest signed step from @p x0 to @p x1 on a circle of length @p period.
 */
[[nodiscard]] inline double signed_displacement(double x0, double x1,
                                                double period) noexcept {
  double d = x1 - x0;
  const double half = 0.5 * period;
  if (d > half) {
    d -= period;
  }
  if (d < -half) {
    d += period;
  }
  return d;
}

/**
 * @brief First `phi = 0` crossing of @p sense along a periodic 1D row.
 *
 * The position is the linear interpolant between the two samples that straddle
 * zero. It lies strictly closer to the true root than the index of the first
 * positive sample whenever the root is not on a sample.
 */
[[nodiscard]] inline FrontHit periodic_zero_crossing(const double *row, int n,
                                                     Crossing sense) noexcept {
  FrontHit hit;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1 == n) ? 0 : i + 1;
    const double a = row[i];
    const double b = row[j];
    const bool rise = a <= 0.0 && b > 0.0;
    const bool fall = a > 0.0 && b <= 0.0;
    const bool match = (sense == Crossing::Rising) ? rise : fall;
    if (!match) {
      continue;
    }
    hit.found = true;
    hit.position = static_cast<double>(i) + (0.0 - a) / (b - a);
    return hit;
  }
  return hit;
}

/** Rising and falling planar fronts, averaged over rows. */
struct PlanarFronts {
  FrontHit rising;
  FrontHit falling;
};

/**
 * @brief Mean sub-cell position of the first rising and falling fronts.
 *
 * Each row is unwrapped against the first row that has that front, so a front
 * that sits on the periodic seam is not averaged with its image `nx` away.
 */
[[nodiscard]] inline PlanarFronts planar_fronts(const double *u, int nx, int ny) {
  PlanarFronts out;
  double sum_rising = 0.0;
  double sum_falling = 0.0;
  int n_rising = 0;
  int n_falling = 0;
  double ref_rising = 0.0;
  double ref_falling = 0.0;
  const double period = static_cast<double>(nx);
  for (int iy = 0; iy < ny; ++iy) {
    const double *row =
        u + static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx);
    const FrontHit rising = periodic_zero_crossing(row, nx, Crossing::Rising);
    const FrontHit falling = periodic_zero_crossing(row, nx, Crossing::Falling);
    if (rising.found) {
      const double position =
          (n_rising == 0)
              ? rising.position
              : ref_rising +
                    signed_displacement(ref_rising, rising.position, period);
      if (n_rising == 0) {
        ref_rising = position;
      }
      sum_rising += position;
      ++n_rising;
    }
    if (falling.found) {
      const double position =
          (n_falling == 0)
              ? falling.position
              : ref_falling +
                    signed_displacement(ref_falling, falling.position, period);
      if (n_falling == 0) {
        ref_falling = position;
      }
      sum_falling += position;
      ++n_falling;
    }
  }
  auto wrap = [period](double position) {
    double wrapped = std::fmod(position, period);
    if (wrapped < 0.0) {
      wrapped += period;
    }
    return wrapped;
  };
  if (n_rising > 0) {
    out.rising.found = true;
    out.rising.position = wrap(sum_rising / static_cast<double>(n_rising));
  }
  if (n_falling > 0) {
    out.falling.found = true;
    out.falling.position = wrap(sum_falling / static_cast<double>(n_falling));
  }
  return out;
}

/**
 * @brief Area of `{phi > 0}` on one triangle of a unit-square split.
 *
 * The triangle has area `1/2`. A vertex contributes only when its sample is
 * strictly positive, matching `count_cells_above`.
 */
[[nodiscard]] inline double triangle_positive_area(double p, double q,
                                                   double r) noexcept {
  const int n = static_cast<int>(p > 0.0) + static_cast<int>(q > 0.0) +
                static_cast<int>(r > 0.0);
  if (n == 0) {
    return 0.0;
  }
  if (n == 3) {
    return 0.5;
  }
  // Area of the corner at `pos` cut off by the zeros on the two adjacent edges.
  auto corner = [](double pos, double a, double b) noexcept {
    return 0.5 * (pos / (pos - a)) * (pos / (pos - b));
  };
  if (n == 1) {
    if (p > 0.0) {
      return corner(p, q, r);
    }
    if (q > 0.0) {
      return corner(q, p, r);
    }
    return corner(r, p, q);
  }
  if (!(p > 0.0)) {
    return 0.5 - corner(p, q, r);
  }
  if (!(q > 0.0)) {
    return 0.5 - corner(q, p, r);
  }
  return 0.5 - corner(r, p, q);
}

/**
 * @brief Cell-units area of the periodic piecewise-linear set `{phi > 0}`.
 *
 * Each quad of neighboring samples is split into two triangles. A field that
 * is positive at every sample has area `nx * ny`, the same number the integer
 * superlevel count reports. A front that crosses between samples contributes
 * the interpolated fraction instead of a whole cell.
 */
[[nodiscard]] inline double periodic_positive_area(const double *u, int nx,
                                                   int ny) noexcept {
  double area = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    const int jy = (iy + 1 == ny) ? 0 : iy + 1;
    for (int ix = 0; ix < nx; ++ix) {
      const int jx = (ix + 1 == nx) ? 0 : ix + 1;
      const auto at = [u, nx](int x, int y) noexcept {
        return u[static_cast<std::size_t>(x) +
                 static_cast<std::size_t>(nx) * static_cast<std::size_t>(y)];
      };
      const double a = at(ix, iy);
      const double b = at(jx, iy);
      const double c = at(ix, jy);
      const double d = at(jx, jy);
      area += triangle_positive_area(a, b, c);
      area += triangle_positive_area(b, d, c);
    }
  }
  return area;
}

/**
 * @brief Equilibrium interface half-thickness `eps*sqrt(2M)`, in cells.
 *
 * The travelling-wave profile of `phi_t = M phi_xx - (phi^3 - phi)/eps^2` is
 * `tanh(x / (eps*sqrt(2M)))`. `dx` is hard-coded to 1 in this app, so this is
 * the only knob that resolves the interface: refining the grid cannot. Below
 * roughly 1.5 cells the front is pinned by the lattice rather than set by the
 * continuum physics, and every kinetic number the app reports acquires a
 * systematic error of tens of percent — measured `-32%` at 0.76 cells,
 * `+0.05%` at 1.5 and `+1.2%` at 2.0
 * (`out/report/data/allen_cahn_resolution_margin.csv`). See
 * `RunConfig::kMinInterfaceWidthCells`.
 */
[[nodiscard]] inline double interface_width_cells(double M, double epsilon,
                                                  double dx) noexcept {
  return epsilon * std::sqrt(2.0 * M) / dx;
}

/**
 * @brief Two flat fronts from the cubic heteroclinic, periodic in x.
 *
 * The stationary kink at zero driving force is `tanh(x / (eps sqrt(2M)))`.
 * A `phi ≈ +1` slab from `L/4` to `3L/4` is that kink minus the kink centred
 * on the other front. Both gaps are a quarter of the box, so on the shipped
 * grid the fronts are many interface widths apart and one of them can be
 * timed before they meet. The profile does not depend on y.
 */
inline void
fill_two_front_initial_condition(std::vector<double> *u,
                                 const pfc::decomposition::Decomposition &decomp,
                                 int rank, double M, double epsilon) {
  const auto &gw = pfc::decomposition::domain(decomp);
  auto gsz = pfc::domain::get_size(gw);
  const auto &local = pfc::decomposition::local_box(decomp, rank);
  auto lo = local.low;
  auto sz = local.size;
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int sxy = nx * ny;
  const double width = interface_width_cells(M, epsilon, 1.0);
  const double delta = width > 0.0 ? width : 1.0;
  const double x_left = 0.25 * static_cast<double>(gsz[0]);
  const double x_right = 0.75 * static_cast<double>(gsz[0]);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const double x = static_cast<double>(lo[0] + ix);
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        (*u)[idx] =
            std::tanh((x - x_left) / delta) - std::tanh((x - x_right) / delta) - 1.0;
      }
    }
  }
}

/**
 * @brief Sharp-interface normal velocity of a flat front.
 *
 * For `phi_t = M phi_xx - (phi^3 - phi)/eps^2 + F`, projecting the travelling
 * wave onto `phi'` gives `-v \int phi'^2 = F \int phi'`. With
 * `phi = -tanh(xi/delta)`, `delta = eps*sqrt(2M)`: `\int phi'^2 = 4/(3 delta)`
 * and `\int phi' = -2`, hence
 *
 *     v = (3/2) F eps sqrt(2 M)
 *
 * — grid-size independent, which is the whole point: it is a property of the
 * material parameters, not of how many cells the box happens to have.
 */
[[nodiscard]] inline double sharp_interface_velocity(double M, double epsilon,
                                                     double F) noexcept {
  return 1.5 * F * epsilon * std::sqrt(2.0 * M);
}

/**
 * @brief Largest `F` for which the unfavoured phase is still metastable.
 *
 * `-(phi^3 - phi)/eps^2 + F` has three roots only while `F eps^2` stays below
 * `max |phi^3 - phi|` on `[-1, 1]`, i.e. `2/(3 sqrt 3)`. Above it there is no
 * `phi < 0` phase to grow into: the whole domain decays to `phi > 0` and there
 * is no front at all.
 */
[[nodiscard]] inline double max_bistable_driving_force(double epsilon) noexcept {
  return 2.0 / (3.0 * std::sqrt(3.0)) / (epsilon * epsilon);
}

/**
 * @brief How far towards the bistability ceiling a parameter set sits, in [0,1).
 *
 * `F eps^2 / (2/(3 sqrt 3))`. One number instead of two, so a preset can be
 * compared with `RunConfig::kMaxDrivingForceFraction` without repeating the
 * constant. At 1 the double well has degenerated into a single well.
 */
[[nodiscard]] inline double driving_force_fraction(double epsilon,
                                                   double F) noexcept {
  return F * epsilon * epsilon / (2.0 / (3.0 * std::sqrt(3.0)));
}

/**
 * @brief Critical nucleus radius `R* = M / v_flat`, in cells at `dx`.
 *
 * A circular seed obeys `dR/dt = v_flat - M/R` (see
 * `curvature_corrected_velocity`), so a seed smaller than `R*` shrinks and
 * vanishes however favourable the bulk driving force is. This is the
 * constraint that sets the default grid: `R*` scales as `1/F`, so buying
 * bistability margin by lowering `F` makes the critical nucleus bigger, and
 * the seed — which scales with the box — has to keep up.
 */
[[nodiscard]] inline double critical_radius_cells(double M, double epsilon, double F,
                                                  double dx) noexcept {
  const double v = sharp_interface_velocity(M, epsilon, F);
  if (v == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return M / v / dx;
}

/**
 * @brief Normal velocity of a *circular* front: `v_flat - M/R`.
 *
 * Allen–Cahn moves an interface by mean curvature as well as by the bulk
 * driving force, `v_n = -M kappa + v_flat`, and for a disc `kappa = 1/R`.
 * The app measures `dR/dt` of a disc, so this — not the flat-front law — is
 * what it has to be compared with.
 *
 * Ignoring it was harmless while `F = 10` made `R_crit/R` about 8%; at a driving
 * force far enough under the bistability ceiling to be safe, `R_crit/R` is 20%
 * at the shipped grid and *does not shrink with the grid* (the seed scales
 * with the box, so `R/R*` is nearly grid-independent). Comparing against the
 * flat law would leave a 20% bias that no amount of refinement removes.
 */
[[nodiscard]] inline double curvature_corrected_velocity(double v_flat, double M,
                                                         double radius) noexcept {
  if (radius <= 0.0) {
    return v_flat;
  }
  return v_flat - M / radius;
}

/** Superlevel areas sampled at t=0, t/2, 3t/4 and the end of the run. */
struct AreaSamples {
  std::int64_t initial = 0;
  std::int64_t half = 0;
  std::int64_t three_quarter = 0;
  std::int64_t final_ = 0;
};

/**
 * @brief Fractional `{phi > 0}` area and, for a flat pair, the two fronts.
 *
 * Printed beside the integer cell-count radii. Not an input to
 * `analyse_interface_kinetics`, so it cannot change `physics_check`.
 */
struct SubcellSamples {
  double area_initial = 0.0;
  double area_half = 0.0;
  double area_three_quarter = 0.0;
  double area_final = 0.0;
  PlanarFronts fronts_initial{};
  PlanarFronts fronts_half{};
  PlanarFronts fronts_three_quarter{};
  PlanarFronts fronts_final{};
  /** Set for `--two-front`. The radial seed leaves this false. */
  bool planar = false;
};

enum class SampleWhen { Initial, Half, ThreeQuarter, Final };

/**
 * @brief Assemble the owned slabs into one periodic `nx * ny` field.
 *
 * Every rank returns the same array. The diagnostic grids are a few hundred
 * cells on a side, so the copy is not part of the timestep cost that matters.
 */
[[nodiscard]] inline std::vector<double>
gather_xy_samples(MPI_Comm comm, const pfc::decomposition::Decomposition &decomp,
                  int rank, const double *local, int nx, int ny) {
  const auto &box = pfc::decomposition::local_box(decomp, rank);
  const int gx0 = box.low[0];
  const int gy0 = box.low[1];
  const auto gsz = pfc::domain::get_size(pfc::decomposition::domain(decomp));
  const int nxg = gsz[0];
  const int nyg = gsz[1];

  int nproc = 1;
  MPI_Comm_size(comm, &nproc);
  const std::array<int, 4> meta{gx0, gy0, nx, ny};
  std::vector<int> all_meta(static_cast<std::size_t>(4 * nproc));
  MPI_Allgather(meta.data(), 4, MPI_INT, all_meta.data(), 4, MPI_INT, comm);

  std::vector<int> counts(static_cast<std::size_t>(nproc));
  std::vector<int> displs(static_cast<std::size_t>(nproc));
  int total = 0;
  for (int r = 0; r < nproc; ++r) {
    const int rnx = all_meta[static_cast<std::size_t>(4 * r + 2)];
    const int rny = all_meta[static_cast<std::size_t>(4 * r + 3)];
    counts[static_cast<std::size_t>(r)] = rnx * rny;
    displs[static_cast<std::size_t>(r)] = total;
    total += rnx * rny;
  }
  std::vector<double> packed(static_cast<std::size_t>(total));
  MPI_Allgatherv(local, nx * ny, MPI_DOUBLE, packed.data(), counts.data(),
                 displs.data(), MPI_DOUBLE, comm);

  std::vector<double> global(static_cast<std::size_t>(nxg) *
                             static_cast<std::size_t>(nyg));
  for (int r = 0; r < nproc; ++r) {
    const int rgx = all_meta[static_cast<std::size_t>(4 * r)];
    const int rgy = all_meta[static_cast<std::size_t>(4 * r + 1)];
    const int rnx = all_meta[static_cast<std::size_t>(4 * r + 2)];
    const int rny = all_meta[static_cast<std::size_t>(4 * r + 3)];
    const double *src = packed.data() + displs[static_cast<std::size_t>(r)];
    for (int iy = 0; iy < rny; ++iy) {
      for (int ix = 0; ix < rnx; ++ix) {
        global[static_cast<std::size_t>(rgx + ix) +
               static_cast<std::size_t>(nxg) * static_cast<std::size_t>(rgy + iy)] =
            src[static_cast<std::size_t>(ix) +
                static_cast<std::size_t>(rnx) * static_cast<std::size_t>(iy)];
      }
    }
  }
  return global;
}

/** Record one sub-cell sample. Every rank must call this together. */
inline void sample_subcell(MPI_Comm comm,
                           const pfc::decomposition::Decomposition &decomp, int rank,
                           const double *local, int nx, int ny,
                           SubcellSamples *samples, SampleWhen when) {
  const std::vector<double> field =
      gather_xy_samples(comm, decomp, rank, local, nx, ny);
  const auto gsz = pfc::domain::get_size(pfc::decomposition::domain(decomp));
  const double area = periodic_positive_area(field.data(), gsz[0], gsz[1]);
  const PlanarFronts fronts =
      samples->planar ? planar_fronts(field.data(), gsz[0], gsz[1]) : PlanarFronts{};
  double *slot = &samples->area_initial;
  PlanarFronts *dest = &samples->fronts_initial;
  switch (when) {
  case SampleWhen::Initial: break;
  case SampleWhen::Half:
    slot = &samples->area_half;
    dest = &samples->fronts_half;
    break;
  case SampleWhen::ThreeQuarter:
    slot = &samples->area_three_quarter;
    dest = &samples->fronts_three_quarter;
    break;
  case SampleWhen::Final:
    slot = &samples->area_final;
    dest = &samples->fronts_final;
    break;
  }
  *slot = area;
  if (samples->planar) {
    *dest = fronts;
  }
}

/**
 * @brief Speed of the left front into the matrix over the last half of the run.
 *
 * Positive when the `phi > 0` slab expands. Zero when either sample is missing.
 */
[[nodiscard]] inline double rising_front_speed(const SubcellSamples &samples,
                                               const RunConfig &cfg) noexcept {
  const double dt_half = 0.5 * static_cast<double>(cfg.n_steps) * cfg.dt;
  if (!(dt_half > 0.0) || !samples.fronts_half.rising.found ||
      !samples.fronts_final.rising.found) {
    return 0.0;
  }
  const double moved = signed_displacement(samples.fronts_half.rising.position,
                                           samples.fronts_final.rising.position,
                                           static_cast<double>(cfg.nx_glob));
  return -moved / dt_half;
}

enum class CheckVerdict { Pass, Fail, Skipped };

/**
 * @brief Interface kinetics derived from four superlevel-area samples.
 *
 * `v_late` is the rate of change of the *equivalent radius* over the second
 * half of the run. Radius, not area: the area ratio `N1/N0` that this app used
 * to check grows like `((R0 + vt)/R0)^2`, so the same interface speed reports a
 * different number for every seed size — and the seed size scales with the
 * grid. `dR/dt` does not.
 *
 * The second half, not the whole run: the Gaussian initial condition is far
 * from the equilibrium `tanh`, and the front's first job is to sharpen. That
 * transient moves the contour by a distance proportional to the seed width, so
 * a whole-run average is grid-dependent for exactly the same reason the area
 * ratio is.
 *
 * `v_late` is compared with `v_predicted = v_theory - M/R`, not with
 * `v_theory`: the measured object is a growing *disc*, and Allen–Cahn moves
 * an interface by curvature as well as by the bulk driving force. With that
 * term in, 128^2 / 192^2 / 256^2 / 384^2 / 512^2 agree with the prediction to
 * 2.3%; without it the same runs sit 41% / 25% / 20% / 15% / 12% below the
 * flat-front law — a spread that looks like a grid dependence and is not one.
 */
struct InterfaceKinetics {
  double r_initial = 0.0;
  double r_half = 0.0;
  double r_three_quarter = 0.0;
  double r_final = 0.0;
  /// dR/dt over [t/2, t].
  double v_late = 0.0;
  /// dR/dt over [t/2, 3t/4] and [3t/4, t]; equal means steady propagation.
  double v_third_quarter = 0.0;
  double v_fourth_quarter = 0.0;
  /// Flat-front law `(3/2) F eps sqrt(2M)`; reported, but not what is checked.
  double v_theory = 0.0;
  /// `v_theory - M/R` at the mean radius of the last half: what *is* checked.
  double v_predicted = 0.0;
  /// `M / v_theory`: a seed below this shrinks whatever the driving force.
  double r_critical = 0.0;
  double interface_width = 0.0;
  bool steady = false;
  CheckVerdict verdict = CheckVerdict::Skipped;
  std::string reason;
};

[[nodiscard]] inline InterfaceKinetics
analyse_interface_kinetics(const AreaSamples &a, const RunConfig &cfg, double dx) {
  InterfaceKinetics k;
  k.r_initial = equivalent_radius(a.initial, dx);
  k.r_half = equivalent_radius(a.half, dx);
  k.r_three_quarter = equivalent_radius(a.three_quarter, dx);
  k.r_final = equivalent_radius(a.final_, dx);
  k.v_theory = sharp_interface_velocity(cfg.M, cfg.epsilon, cfg.driving_force);
  // In the same length units as r_half / r_final, hence the dx back-multiply.
  k.r_critical =
      critical_radius_cells(cfg.M, cfg.epsilon, cfg.driving_force, dx) * dx;
  k.interface_width = interface_width_cells(cfg.M, cfg.epsilon, dx);
  // The prediction is evaluated at the mean radius over the interval the
  // measurement covers, which is the last half of the run. Using R_final
  // instead would compare a mean velocity with an instantaneous one.
  k.v_predicted =
      curvature_corrected_velocity(k.v_theory, cfg.M, 0.5 * (k.r_half + k.r_final));

  const double dt_half = 0.5 * static_cast<double>(cfg.n_steps) * cfg.dt;
  const double dt_quarter = 0.5 * dt_half;
  if (dt_half > 0.0) {
    k.v_late = (k.r_final - k.r_half) / dt_half;
  }
  if (dt_quarter > 0.0) {
    k.v_third_quarter = (k.r_three_quarter - k.r_half) / dt_quarter;
    k.v_fourth_quarter = (k.r_final - k.r_three_quarter) / dt_quarter;
  }
  const double v_scale =
      std::max(std::abs(k.v_third_quarter), std::abs(k.v_fourth_quarter));
  k.steady = (v_scale <= 0.0) || (std::abs(k.v_fourth_quarter - k.v_third_quarter) <=
                                  RunConfig::kSteadyTolerance * v_scale);

  if (a.initial <= 0) {
    k.verdict = CheckVerdict::Fail;
    k.reason = "no seed at t=0 (N0 == 0)";
    return k;
  }
  if (a.final_ <= 0) {
    // Distinguished from "the front is slow": a seed under the critical
    // radius dissolves no matter how long the run is, and the fix is a
    // bigger box or a bigger driving force, not more steps.
    k.verdict = CheckVerdict::Fail;
    k.reason = k.r_initial < k.r_critical
                   ? "the seed dissolved: R0 is below the critical radius "
                     "M/v, so curvature beats the driving force"
                   : "the seed dissolved (N1 == 0)";
    return k;
  }
  if (cfg.n_steps < RunConfig::kMinStepsForKinetics) {
    k.verdict = CheckVerdict::Skipped;
    k.reason = "run shorter than " +
               std::to_string(RunConfig::kMinStepsForKinetics) +
               " steps: the initial-condition transient dominates";
    return k;
  }
  const double floor_distance = RunConfig::kMinMeasurableAdvanceCells * dx;
  if (std::abs(k.r_final - k.r_half) < floor_distance) {
    // Below one cell of travel the superlevel *count* has not changed enough
    // for dR/dt to mean anything. Whether that is the run's fault or the
    // physics' is settled by asking how far the front was supposed to go: if
    // even the prediction is sub-cell there is nothing to measure, otherwise
    // the front should have moved and did not.
    if (std::abs(k.v_predicted) * dt_half < floor_distance) {
      k.verdict = CheckVerdict::Skipped;
      k.reason = "predicted advance over the last half of the run is under one "
                 "cell: nothing measurable at this dt/n_steps";
    } else {
      k.verdict = CheckVerdict::Fail;
      k.reason = "interface did not move a whole cell over the last half of "
                 "the run, but was predicted to";
    }
    return k;
  }
  if (!k.steady) {
    k.verdict = CheckVerdict::Skipped;
    k.reason = "interface speed still changing between the last two quarters "
               "(not yet steady); run longer";
    return k;
  }
  // min/max rather than lo/hi directly: a negative driving force, or a
  // subcritical seed, predicts a *shrinking* seed and the band has to
  // bracket that too.
  const double bound_a = RunConfig::kVelocityBandLo * k.v_predicted;
  const double bound_b = RunConfig::kVelocityBandHi * k.v_predicted;
  if (k.v_late < std::min(bound_a, bound_b) ||
      k.v_late > std::max(bound_a, bound_b)) {
    k.verdict = CheckVerdict::Fail;
    k.reason = "steady interface speed outside the sharp-interface band";
    return k;
  }
  k.verdict = CheckVerdict::Pass;
  k.reason = "steady interface speed consistent with (3/2) F eps sqrt(2M) - M/R";
  return k;
}

[[nodiscard]] inline const char *to_string(CheckVerdict v) noexcept {
  switch (v) {
  case CheckVerdict::Pass: return "PASS";
  case CheckVerdict::Fail: return "FAIL";
  case CheckVerdict::Skipped: return "SKIPPED";
  }
  return "SKIPPED";
}

/**
 * @brief The disc criterion does not describe a flat front pair.
 *
 * The cell-count columns are still computed and printed. The verdict is
 * `SKIPPED`, which does not set the exit status even with `--strict`.
 */
inline void skip_disc_check_for_flat_fronts(InterfaceKinetics *k) {
  k->verdict = CheckVerdict::Skipped;
  k->reason = "two-front initial condition: the disc criterion is not applied";
}

/** Rank-0 report of the kinetics block. Silent on every other rank. */
inline void report_interface_kinetics(int rank, const AreaSamples &a,
                                      const RunConfig &cfg,
                                      const InterfaceKinetics &k,
                                      const SubcellSamples *sub = nullptr,
                                      double dx = 1.0,
                                      std::ostream &os = std::cout) {
  if (rank != 0) {
    return;
  }
  auto with_subcell = [&](double cell_radius, double area_cells) {
    os << cell_radius;
    if (sub != nullptr) {
      os << " (subcell " << equivalent_radius_from_area(area_cells, dx) << ")";
    }
  };
  auto front_position = [&](const FrontHit &hit) {
    if (hit.found) {
      os << hit.position;
    } else {
      os << "missing";
    }
  };
  os << "Superlevel area (cells with phi > " << RunConfig::kLevelSetThreshold
     << "): N0=" << a.initial << ", N_half=" << a.half
     << ", N_3q=" << a.three_quarter << ", N1=" << a.final_ << "\n";
  os << "Equivalent radius: R0=";
  with_subcell(k.r_initial, sub == nullptr ? 0.0 : sub->area_initial);
  os << ", R_half=";
  with_subcell(k.r_half, sub == nullptr ? 0.0 : sub->area_half);
  os << ", R_3q=";
  with_subcell(k.r_three_quarter, sub == nullptr ? 0.0 : sub->area_three_quarter);
  os << ", R1=";
  with_subcell(k.r_final, sub == nullptr ? 0.0 : sub->area_final);
  os << "\n";
  if (sub != nullptr && sub->planar) {
    os << "Planar fronts (subcell): x_left=";
    front_position(sub->fronts_initial.rising);
    os << ", ";
    front_position(sub->fronts_half.rising);
    os << ", ";
    front_position(sub->fronts_three_quarter.rising);
    os << ", ";
    front_position(sub->fronts_final.rising);
    os << "; x_right=";
    front_position(sub->fronts_initial.falling);
    os << ", ";
    front_position(sub->fronts_half.falling);
    os << ", ";
    front_position(sub->fronts_three_quarter.falling);
    os << ", ";
    front_position(sub->fronts_final.falling);
    os << "; v_front=" << rising_front_speed(*sub, cfg)
       << " (left front into the matrix, last half)\n";
  }
  os << "Interface velocity dR/dt (last half): v_late=" << k.v_late
     << ", per quarter " << k.v_third_quarter << " / " << k.v_fourth_quarter
     << " (steady=" << (k.steady ? "yes" : "no") << ")\n";
  os << "Sharp-interface prediction (3/2) F eps sqrt(2M): v_theory=" << k.v_theory
     << "\n";
  os << "Critical radius M/v_theory = " << k.r_critical << " (R0=" << k.r_initial
     << ")";
  if (k.r_initial < k.r_critical) {
    os << "  [WARNING: the seed is subcritical — curvature beats the "
          "driving force and it will dissolve]";
  }
  os << "\n";
  os << "Curvature-corrected prediction v_theory - M/R: v_predicted="
     << k.v_predicted << ", accepted band ["
     << std::min(RunConfig::kVelocityBandLo * k.v_predicted,
                 RunConfig::kVelocityBandHi * k.v_predicted)
     << ", "
     << std::max(RunConfig::kVelocityBandLo * k.v_predicted,
                 RunConfig::kVelocityBandHi * k.v_predicted)
     << "]\n";
  os << "Interface width eps*sqrt(2M) = " << k.interface_width << " cells";
  if (k.interface_width < RunConfig::kMinInterfaceWidthCells) {
    os << "  [WARNING: under " << RunConfig::kMinInterfaceWidthCells
       << " cells — the front is lattice-limited, not "
          "continuum-limited, and the speed can be tens of percent off]";
  }
  os << "\n";
  const double f_max = max_bistable_driving_force(cfg.epsilon);
  const double fraction = driving_force_fraction(cfg.epsilon, cfg.driving_force);
  os << "Bistability limit: driving_force=" << cfg.driving_force << " vs "
     << "2/(3 sqrt 3)/eps^2 = " << f_max << " (at " << 100.0 * fraction
     << "% of it)";
  if (cfg.driving_force >= f_max) {
    os << "  [VIOLATED: phi<0 is not metastable, the whole domain decays "
          "and there is no front]";
  } else if (fraction > RunConfig::kMaxDrivingForceFraction) {
    os << "  [WARNING: over " << 100.0 * RunConfig::kMaxDrivingForceFraction
       << "% of the ceiling — the matrix well is shallow and the "
          "sharp-interface law is no longer asymptotic]";
  }
  os << "\n";
  os << "physics_check=" << to_string(k.verdict) << " (" << k.reason << ")\n";
}

inline void
step_explicit_euler_cpu(std::vector<double> *u, std::vector<double> *lap,
                        std::array<std::vector<double>, 6> *face_halos,
                        pfc::comm::SparseExchange<pfc::HostSpace, double> *exchanger,
                        int nx, int ny, int nz, double inv_dx2, double inv_dy2,
                        double dt, double M, double inv_eps2, double driving_force) {
  constexpr int hw = RunConfig::kHaloWidth;
  exchanger->exchange(std::span<double>(*u));
  pfc::halo::copy_to_face_layout(exchanger->halos(), *face_halos);
  std::fill(lap->begin(), lap->end(), 0.0);
  std::array<const double *, 6> face_ptrs{};
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<std::size_t>(i)] =
        (*face_halos)[static_cast<std::size_t>(i)].data();
  }
  pfc::field::fd::laplacian2d_xy_periodic_separated<2>(
      u->data(), face_ptrs, lap->data(), nx, ny, nz, inv_dx2, inv_dy2, hw);
  for (std::size_t i = 0; i < u->size(); ++i) {
    const double p = (*u)[i];
    (*u)[i] += dt * (M * (*lap)[i] - inv_eps2 * (p * p * p - p) + driving_force);
  }
}

} // namespace allen_cahn
