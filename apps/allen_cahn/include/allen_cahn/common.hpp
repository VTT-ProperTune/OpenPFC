// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>
#include <openpfc_apps/mpi_report.hpp>

namespace allen_cahn {

struct RunConfig {
  int nx_glob = 64;
  int ny_glob = 64;
  /**
   * Long enough for the front to outrun the Gaussian initial condition and
   * settle into steady propagation, which is what the built-in check measures.
   * Below `kMinStepsForKinetics` the check reports SKIPPED rather than a
   * number the transient dominates.
   */
  int n_steps = 5000;
  double dt = 0.00009;
  double M = 8.0;
  double epsilon = 0.19;
  /** Positive bulk driving term that favors the φ≈+1 seed over the φ≈-1 matrix. */
  double driving_force = 10.0;
  /** If non-empty, gather the final scalar field on rank 0 and write a grayscale
   * PNG. */
  std::string png_output;
  /** If non-empty, write the field right after IC, before time stepping. */
  std::string png_output_initial;
  /** Opt in to letting a failed physics check set the process exit status. */
  bool strict = false;
  static constexpr int kHaloWidth = 1;
  /** Superlevel for the seed-area metric: φ > 0 matches the visible seed in PNGs. */
  static constexpr double kLevelSetThreshold = 0.0;
  /**
   * Band around the sharp-interface prediction that `v_late` must land in.
   *
   * Deliberately a factor of two either way, not a percentage. The shipped
   * preset has an interface only `eps*sqrt(2M) = 0.76` cells wide (see
   * `interface_width_cells`), so the front is lattice-limited rather than
   * asymptotic and the measured speed sits a few tens of percent off the
   * continuum law — measured `+11%` at the default parameters and `-35%` at
   * `driving_force = 5`. A factor-of-two band still separates "a front
   * propagates at roughly the predicted speed" from every failure mode that
   * matters: no growth, the wrong sign, or the whole domain flipping.
   */
  static constexpr double kVelocityBandLo = 0.5;
  static constexpr double kVelocityBandHi = 2.0;
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
    if (std::string(argv[i]) == "--strict") {
      c->strict = true;
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
  const int gmin = std::min(gsz[0], gsz[1]);
  const double sigma = std::max(2.0, 0.055 * static_cast<double>(gmin));
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
 * @brief Equilibrium interface half-thickness `eps*sqrt(2M)`, in cells.
 *
 * The travelling-wave profile of `phi_t = M phi_xx - (phi^3 - phi)/eps^2` is
 * `tanh(x / (eps*sqrt(2M)))`. Below one cell the front is pinned by the
 * lattice rather than set by the continuum physics, and every kinetic number
 * this app reports acquires a systematic error of tens of percent.
 */
[[nodiscard]] inline double interface_width_cells(double M, double epsilon,
                                                  double dx) noexcept {
  return epsilon * std::sqrt(2.0 * M) / dx;
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

/** Superlevel areas sampled at t=0, t/2, 3t/4 and the end of the run. */
struct AreaSamples {
  std::int64_t initial = 0;
  std::int64_t half = 0;
  std::int64_t three_quarter = 0;
  std::int64_t final_ = 0;
};

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
 * ratio is. Measured over the last half instead, 64^2 / 128^2 / 256^2 agree to
 * 5%.
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
  double v_theory = 0.0;
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
  k.interface_width = interface_width_cells(cfg.M, cfg.epsilon, dx);

  const double dt_half = 0.5 * static_cast<double>(cfg.n_steps) * cfg.dt;
  const double dt_quarter = 0.5 * dt_half;
  if (dt_half > 0.0) {
    k.v_late = (k.r_final - k.r_half) / dt_half;
  }
  if (dt_quarter > 0.0) {
    k.v_third_quarter = (k.r_three_quarter - k.r_half) / dt_quarter;
    k.v_fourth_quarter = (k.r_final - k.r_three_quarter) / dt_quarter;
  }
  const double v_scale = std::max(std::abs(k.v_third_quarter),
                                  std::abs(k.v_fourth_quarter));
  k.steady = (v_scale <= 0.0) ||
             (std::abs(k.v_fourth_quarter - k.v_third_quarter) <=
              RunConfig::kSteadyTolerance * v_scale);

  if (a.initial <= 0) {
    k.verdict = CheckVerdict::Fail;
    k.reason = "no seed at t=0 (N0 == 0)";
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
    if (std::abs(k.v_theory) * dt_half < floor_distance) {
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
  // min/max rather than lo/hi directly: a negative driving force predicts a
  // *shrinking* seed, and the band has to bracket that too.
  const double bound_a = RunConfig::kVelocityBandLo * k.v_theory;
  const double bound_b = RunConfig::kVelocityBandHi * k.v_theory;
  if (k.v_late < std::min(bound_a, bound_b) ||
      k.v_late > std::max(bound_a, bound_b)) {
    k.verdict = CheckVerdict::Fail;
    k.reason = "steady interface speed outside the sharp-interface band";
    return k;
  }
  k.verdict = CheckVerdict::Pass;
  k.reason = "steady interface speed consistent with (3/2) F eps sqrt(2M)";
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

/** Rank-0 report of the kinetics block. Silent on every other rank. */
inline void report_interface_kinetics(int rank, const AreaSamples &a,
                                      const RunConfig &cfg,
                                      const InterfaceKinetics &k) {
  if (rank != 0) {
    return;
  }
  std::cout << "Superlevel area (cells with phi > " << RunConfig::kLevelSetThreshold
            << "): N0=" << a.initial << ", N_half=" << a.half
            << ", N_3q=" << a.three_quarter << ", N1=" << a.final_ << "\n";
  std::cout << "Equivalent radius: R0=" << k.r_initial << ", R_half=" << k.r_half
            << ", R_3q=" << k.r_three_quarter << ", R1=" << k.r_final << "\n";
  std::cout << "Interface velocity dR/dt (last half): v_late=" << k.v_late
            << ", per quarter " << k.v_third_quarter << " / "
            << k.v_fourth_quarter << " (steady=" << (k.steady ? "yes" : "no")
            << ")\n";
  std::cout << "Sharp-interface prediction (3/2) F eps sqrt(2M): v_theory="
            << k.v_theory << ", accepted band ["
            << std::min(RunConfig::kVelocityBandLo * k.v_theory,
                        RunConfig::kVelocityBandHi * k.v_theory)
            << ", "
            << std::max(RunConfig::kVelocityBandLo * k.v_theory,
                        RunConfig::kVelocityBandHi * k.v_theory)
            << "]\n";
  std::cout << "Interface width eps*sqrt(2M) = " << k.interface_width << " cells";
  if (k.interface_width < 1.0) {
    std::cout << "  [WARNING: below one cell — the front is lattice-limited, "
                 "not continuum-limited]";
  }
  std::cout << "\n";
  const double f_max = max_bistable_driving_force(cfg.epsilon);
  std::cout << "Bistability limit: driving_force=" << cfg.driving_force << " vs "
            << "2/(3 sqrt 3)/eps^2 = " << f_max;
  if (cfg.driving_force >= f_max) {
    std::cout << "  [VIOLATED: phi<0 is not metastable, the whole domain decays "
                 "and there is no front]";
  }
  std::cout << "\n";
  std::cout << "physics_check=" << to_string(k.verdict) << " (" << k.reason
            << ")\n";
}

inline void report_step_timing(MPI_Comm comm, int rank, int n_steps,
                               double elapsed_local_s) {
  pfc::apps::report_step_timing(comm, rank, n_steps, elapsed_local_s);
}

inline void
step_explicit_euler_cpu(std::vector<double> *u, std::vector<double> *lap,
                        std::array<std::vector<double>, 6> *face_halos,
                        pfc::comm::SparseExchange<pfc::HostSpace, double> *exchanger,
                        int nx, int ny, int nz, double inv_dx2, double inv_dy2,
                        double dt, double M, double inv_eps2, double driving_force) {
  constexpr int hw = RunConfig::kHaloWidth;
  exchanger->exchange(u->data(), u->size());
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
