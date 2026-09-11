// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cases.hpp
 * @brief The two shipped cases -- Stage-1 planar verification and the
 *        Stage-2 deterministic 2-D dendrite -- as callable functions.
 *
 * @details
 * Both cases are functions rather than `main`s so that the ctest suite runs
 * *the shipped physics* at a small size instead of a re-implementation of it.
 * A verification harness that is not the code it verifies is a second thing
 * to keep in sync, and it is always the one that rots.
 *
 * ## Stage 1: the planar case and why it is built the way it is
 *
 * A single planar front in a periodic box is impossible -- it would run into
 * its own solute boundary layer. So the initial condition is a **solid slab
 * centred in a periodic box**, giving two fronts that move apart. This is
 * not a workaround, it is the cheapest way to get all four Stage-1 checks at
 * once:
 *
 *  - the domain is closed, so total solute is conserved and any drift is a
 *    bug rather than an outflow;
 *  - a periodic grid is where the central-difference divergence telescopes
 *    exactly, which is what makes "conserved to round-off" a meaningful
 *    statement rather than a tolerance;
 *  - the two fronts are symmetric, so the transverse-mean profile is a clean
 *    1-D profile with no boundary treatment anywhere.
 *
 * The transverse direction has a handful of cells (`ny = 4` by default), not
 * one. That is on purpose: it exercises the 2-D code path -- gradients,
 * anisotropy, both flux components, the `y` halo exchange -- on a problem
 * whose answer is known in 1-D. A genuine `ny = 1` run would not test any of
 * the machinery the dendrite case depends on.
 *
 * The supersaturation is derived from a *target velocity*, not the other way
 * round, because it is `V` that has to stay inside the thin-interface window
 * `W0 V / D_l << 1` while the boundary layer `D_l / V` stays much larger than
 * `W0` and much smaller than the box. Asking for `V` and computing
 * `Omega = 1 + k beta V` keeps those three constraints visible.
 *
 * ## Stage 2: the dendrite case
 *
 * Deterministic: one seed, no noise, fixed geometry, so two runs of the same
 * binary produce the same CSV. Four-fold anisotropy on a periodic square with
 * the seed at the centre gives four equivalent `<100>` arms; the `+x` arm is
 * the one measured. Latent heat is on and feeds back through `M_c`, so
 * equation (4) is actually coupled rather than merely integrated -- which is
 * the point of a *thermo*-solutal core.
 *
 * @see diagnostics.hpp for what the reported numbers mean, precisely
 * @see parameters.hpp for the predictions Stage 1 is measured against
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>

#include <alloy_dendrite/diagnostics.hpp>
#include <alloy_dendrite/parameters.hpp>
#include <alloy_dendrite/step.hpp>

namespace alloy_dendrite {

// ===========================================================================
// Stage 1 -- planar interface, isothermal
// ===========================================================================

/// Inputs of @ref run_planar. Everything a Stage-1 run needs, and nothing a
/// Stage-1 run does not: no output fields, no visualisation.
struct PlanarConfig {
  ModelParams model{};
  int nx = 640;
  /// Transverse cells. Four is enough to exercise the 2-D path; the answer
  /// must not depend on it, which is itself a cheap regression check.
  int ny = 4;
  double dx = 0.8;
  int fd_order = 4;
  /// Explicit step. `<= 0` selects `dt_safety * explicit_dt_limit`.
  double dt = 0.0;
  double dt_safety = 0.25;
  /// Target steady velocity. `Omega` follows from `1 + k beta V`.
  double velocity_target = 0.1;
  /// Overrides @ref velocity_target when positive.
  double omega = 0.0;
  double t_end = 1200.0;
  int n_sample = 240;
  /// Trailing fraction of samples used for the velocity fit and for the
  /// steady-state averages of `U_i`, `U_s`, `k_eff` and `ell`.
  double fit_fraction = 0.4;
  /// Half-thickness of the initial solid slab, in `W0`.
  double seed_half_width = 12.0;
  /**
   * @brief Start from the analytic steady-state solute profile.
   *
   * A uniform `U = -Omega` melt is a legitimate initial condition but a very
   * expensive one: the front starts at `V ~ Omega / beta`, roughly sixteen
   * times its steady value here, and only relaxes onto the attractor after
   * `~ D_l / V^2` of time, by which point it has crossed most of the box.
   * Seeding the exponential boundary layer that belongs to the *predicted*
   * velocity skips that transient.
   *
   * This does not prejudge the answer. The steady planar state is an
   * attractor -- too fast a front rejects less solute than the far field
   * supplies, the layer thickens, `U_i` rises and the front slows -- so a
   * model whose kinetics disagree with the prediction simply walks off the
   * seeded profile and settles somewhere else. Running the same case with
   * `analytic_ic = false` in a long enough box must give the same velocity,
   * and that is worth doing once per parameter set rather than every run.
   */
  bool analytic_ic = true;
  /**
   * @brief Seed the analytic profile for *this* velocity instead of the
   *        predicted one. Zero means "use the prediction".
   *
   * The control that makes @ref analytic_ic honest. Seed the boundary layer
   * of a front moving at twice, or half, the predicted speed and the run
   * must still settle on the same velocity -- if it settled wherever it was
   * put, the initial condition would be the answer and the test would be
   * vacuous. Costs one extra run per parameter set and is much cheaper than
   * the uniform-melt transient it replaces.
   */
  double ic_velocity = 0.0;
  PlanarWindows windows{};
  std::string csv_timeseries;
  std::string csv_summary;
  std::string run_id = "planar";
  bool quiet = false;
};

/// Everything @ref run_planar measures, plus the predictions it is compared
/// against, so a caller can assert on ratios without recomputing anything.
struct PlanarResult {
  bool valid{false};
  double omega{0.0};
  double dt{0.0};
  int n_steps{0};
  double v_measured{std::numeric_limits<double>::quiet_NaN()};
  double v_predicted{std::numeric_limits<double>::quiet_NaN()};
  double beta_measured{std::numeric_limits<double>::quiet_NaN()};
  double beta_theory{std::numeric_limits<double>::quiet_NaN()};
  double ell_measured{std::numeric_limits<double>::quiet_NaN()};
  double ell_predicted{std::numeric_limits<double>::quiet_NaN()};
  double k_eff{std::numeric_limits<double>::quiet_NaN()};
  double u_interface{std::numeric_limits<double>::quiet_NaN()};
  double u_solid{std::numeric_limits<double>::quiet_NaN()};
  double u_far{std::numeric_limits<double>::quiet_NaN()};
  /**
   * @brief Residual of the steady-state mass balance `U_inf = k U_s - 1`.
   *
   * Independent of the kinetic relation and of the boundary-layer fit: it
   * says the freshly formed solid carries exactly the far-field composition,
   * which is what "steady" means for a planar front. Three of the four
   * Stage-1 relations (`U_i = -beta V`, `ell = D_l/V`, `U_s = U_i`) can be
   * argued to be partly seeded by the analytic initial condition; this one
   * is a statement about the solid the model laid down during the run.
   */
  double stefan_residual{std::numeric_limits<double>::quiet_NaN()};
  double fit_r2{std::numeric_limits<double>::quiet_NaN()};
  /// `|solute(t_end) - solute(0)| / |solute(0)|`.
  double solute_drift_rel{std::numeric_limits<double>::quiet_NaN()};
  /// `|balance(t_end) - balance(0)|` scaled by the largest term in it.
  double heat_drift_rel{std::numeric_limits<double>::quiet_NaN()};
  double peclet{std::numeric_limits<double>::quiet_NaN()};
  double phi_min{0.0};
  double phi_max{0.0};
};

/// Mean of the trailing @p fraction of the finite entries of @p v.
[[nodiscard]] inline double trailing_mean(const std::vector<double> &v,
                                          double fraction) {
  if (v.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t want = std::max<std::size_t>(
      1, static_cast<std::size_t>(fraction * static_cast<double>(v.size())));
  const std::size_t begin = (want >= v.size()) ? 0 : (v.size() - want);
  double acc = 0.0;
  double m = 0.0;
  for (std::size_t q = begin; q < v.size(); ++q) {
    if (std::isfinite(v[q])) {
      acc += v[q];
      m += 1.0;
    }
  }
  return (m > 0.0) ? acc / m : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Run the Stage-1 planar verification and return every measured and
 *        predicted quantity.
 *
 * Writes one time-series row per sample and one summary row per run, both in
 * append mode (see @ref CsvAppender).
 */
[[nodiscard]] inline PlanarResult run_planar(const PlanarConfig &cfg, int rank,
                                             int nproc, MPI_Comm comm) {
  PlanarResult res;
  const ModelParams p = cfg.model;

  const double beta = kinetic_coefficient(p);
  res.beta_theory = beta;
  if (!(beta > 0.0)) {
    // Not a numerical guard -- a statement about what Stage 1 can measure.
    // At beta = 0 the steady-state condition collapses to Omega = 1 with V
    // undetermined, the classical degeneracy of planar one-sided growth, so
    // the velocity oracle has no content. Refusing here is better than
    // reporting a division by zero three lines further down.
    throw std::invalid_argument(
        "run_planar: beta = " + std::to_string(beta) +
        " is not positive (lambda = " + std::to_string(p.lambda) +
        " against the vanishing-kinetics value D_l tau0 / (a2 W0^2) = " +
        std::to_string(p.D_l * p.tau0 / (kA2 * p.W0 * p.W0)) +
        "). A planar isothermal front is velocity-degenerate at beta = 0: "
        "Omega = 1 admits any V, so there is nothing to measure. Use a "
        "lambda well below the vanishing-kinetics value for Stage 1.");
  }
  res.omega =
      (cfg.omega > 0.0) ? cfg.omega : planar_supersaturation(p, cfg.velocity_target);
  const double v_pred = planar_steady_velocity(p, res.omega);
  res.v_predicted = v_pred;
  res.ell_predicted = boundary_layer_width(p, v_pred);
  res.peclet = interface_peclet(p, v_pred);

  const double dt_lim = explicit_dt_limit(p, cfg.dx, 2);
  res.dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;
  if (res.dt > dt_lim) {
    throw std::invalid_argument("run_planar: dt " + std::to_string(res.dt) +
                                " exceeds the explicit limit " +
                                std::to_string(dt_lim));
  }
  // Interface CFL: the front must not cross a cell in one step, or the
  // discrete d_t phi that feeds the anti-trapping current is meaningless.
  // PR #103 hit this the hard way; it is cheap to assert here.
  if (v_pred > 0.0 && res.dt > 0.8 * cfg.dx / v_pred) {
    throw std::invalid_argument("run_planar: dt violates the interface CFL dx/V");
  }
  res.n_steps = std::max(1, static_cast<int>(std::llround(cfg.t_end / res.dt)));
  const int sample_every = std::max(1, res.n_steps / std::max(1, cfg.n_sample));

  // Box-size guard. Two fronts move apart in a periodic box, so each one's
  // exponential tail decays toward the seam where the other one's tail is
  // arriving. `U_far` is read at the seam, so if the half-box is not many
  // boundary layers long the two tails overlap there and the fit's constant
  // -- hence `U_i`, hence `k_eff` -- is biased by the residual. At
  // `V = 0.05, D_l = 2` a 512 W0 box puts `exp(-half/ell) ~ 2%` of the
  // profile amplitude at the seam, which is the same size as `U_i` itself
  // and shifts the measured `k_eff` by 4%. Eight layers keeps that under
  // 0.1%; the run is allowed to proceed but the caller is told.
  const double half_box = 0.5 * static_cast<double>(cfg.nx) * cfg.dx;
  if (rank == 0 && !cfg.quiet && half_box < 8.0 * res.ell_predicted) {
    std::cerr << "run_planar: WARNING half-box " << half_box << " W0 is only "
              << half_box / res.ell_predicted
              << " boundary layers; the two fronts' tails overlap at the "
                 "periodic seam and U_far (hence k_eff) is biased. Use "
                 "nx >= "
              << static_cast<int>(std::ceil(16.0 * res.ell_predicted / cfg.dx))
              << ".\n";
  }

  auto domain = pfc::domain::create(pfc::GridSize({cfg.nx, cfg.ny, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  pfc::comm::HaloExchangeOptions opt;
  opt.directions = Stepper<2>::directions();
  pfc::sim::stacks::FDPaddedCPUStack stack(domain, cfg.fd_order / 2, rank, nproc,
                                           comm, opt);
  Stepper<2> st(stack, p, cfg.fd_order);

  // --- initial condition -------------------------------------------------
  // Solid slab of half-width `seed_half_width` centred in the box, with the
  // equilibrium tanh profile so the run does not spend its first hundred
  // steps relaxing an interface it was handed as a step function.
  const double xc = 0.5 * static_cast<double>(cfg.nx) * cfg.dx;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  const double u_inf = -res.omega;
  const double v_ic = (cfg.ic_velocity > 0.0) ? cfg.ic_velocity : v_pred;
  const double u_i0 = -beta * v_ic;  // Gibbs-Thomson at the seeded V
  const double decay = v_ic / p.D_l; // inverse boundary-layer width
  const double half = cfg.seed_half_width * p.W0;
  st.phi().for_each_owned([&](int i, int j, int kk) {
    const double x = st.phi().coords(i, j, kk)[0];
    const double d = std::fabs(x - xc) - half;
    st.phi()(i, j, kk) = std::tanh(-d * inv_w);
    if (cfg.analytic_ic) {
      st.solute()(i, j, kk) =
          (d <= 0.0) ? u_i0 : (u_inf + (u_i0 - u_inf) * std::exp(-decay * d));
    } else {
      st.solute()(i, j, kk) = u_inf;
    }
    st.temperature()(i, j, kk) = 0.0;
  });
  st.seed_conserved_solute();

  const auto cons0 =
      measure_conservation(st.phi(), st.solute(), st.temperature(), p, comm);

  CsvAppender ts_csv;
  CsvAppender sum_csv;
  if (!cfg.csv_timeseries.empty()) {
    ts_csv = CsvAppender(cfg.csv_timeseries,
                         "run_id,step,t,x_if,u_far,u_interface,u_solid,ell,k_eff,"
                         "fit_r2,fit_points,solute_total,solute_drift_rel,"
                         "heat_balance,heat_drift_abs,phi_min,phi_max,u_min,u_max",
                         rank);
  }

  std::vector<double> t_s, x_s, ui_s, us_s, uf_s, ke_s, ell_s, r2_s;
  Conservation cons = cons0;
  double t = 0.0;
  for (int step = 1; step <= res.n_steps; ++step) {
    st.step(res.dt);
    t = static_cast<double>(step) * res.dt;
    if (step % sample_every != 0 && step != res.n_steps) {
      continue;
    }
    const auto phi_prof = transverse_mean_profile(st.phi(), comm);
    const auto u_prof = transverse_mean_profile(st.solute(), comm);
    const auto front =
        measure_planar_front(phi_prof, u_prof, cfg.dx, p, cfg.windows);
    cons = measure_conservation(st.phi(), st.solute(), st.temperature(), p, comm);

    if (front.valid) {
      t_s.push_back(t);
      x_s.push_back(front.x_if);
      ui_s.push_back(front.u_interface);
      us_s.push_back(front.u_solid);
      uf_s.push_back(front.u_far);
      ke_s.push_back(front.k_eff);
      ell_s.push_back(front.ell);
      r2_s.push_back(front.fit_r2);
    }
    const double sol_drift = std::fabs(cons.solute_total - cons0.solute_total) /
                             std::fabs(cons0.solute_total);
    const double heat_drift = std::fabs(cons.heat_balance - cons0.heat_balance);
    if (ts_csv.active()) {
      ts_csv.row(format("%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d,"
                        "%.17g,%.6g,%.17g,%.6g,%.10g,%.10g,%.10g,%.10g",
                        cfg.run_id.c_str(), step, t, front.x_if, front.u_far,
                        front.u_interface, front.u_solid, front.ell, front.k_eff,
                        front.fit_r2, front.fit_points, cons.solute_total, sol_drift,
                        cons.heat_balance, heat_drift, cons.phi_min, cons.phi_max,
                        cons.u_min, cons.u_max));
    }
    if (!std::isfinite(cons.phi_max) || cons.phi_max > 1.5 || cons.phi_min < -1.5) {
      if (rank == 0) {
        std::cerr << "run_planar: phi left [-1.5, 1.5] at step " << step
                  << " -- the run is unstable, not slow.\n";
      }
      break;
    }
  }

  res.v_measured = trailing_slope(t_s, x_s, cfg.fit_fraction);
  res.u_interface = trailing_mean(ui_s, cfg.fit_fraction);
  res.u_solid = trailing_mean(us_s, cfg.fit_fraction);
  res.u_far = trailing_mean(uf_s, cfg.fit_fraction);
  res.k_eff = trailing_mean(ke_s, cfg.fit_fraction);
  res.stefan_residual = res.u_far - (p.k * res.u_solid - 1.0);
  res.ell_measured = trailing_mean(ell_s, cfg.fit_fraction);
  res.fit_r2 = trailing_mean(r2_s, cfg.fit_fraction);
  res.beta_measured = (std::isfinite(res.v_measured) && res.v_measured != 0.0)
                          ? -res.u_interface / res.v_measured
                          : std::numeric_limits<double>::quiet_NaN();
  res.solute_drift_rel = std::fabs(cons.solute_total - cons0.solute_total) /
                         std::fabs(cons0.solute_total);
  const double heat_scale =
      std::max({std::fabs(0.5 * cons.phi_total), std::fabs(cons.theta_total), 1.0});
  res.heat_drift_rel =
      std::fabs(cons.heat_balance - cons0.heat_balance) / heat_scale;
  res.phi_min = cons.phi_min;
  res.phi_max = cons.phi_max;
  res.valid = std::isfinite(res.v_measured) && std::isfinite(res.k_eff);

  if (!cfg.csv_summary.empty()) {
    sum_csv = CsvAppender(
        cfg.csv_summary,
        "run_id,nx,ny,dx,dx_over_W0,fd_order,dt,t_end,lambda,k,D_l,W0,tau0,eps4,"
        "at_scale,spec_source,omega,peclet,beta_theory,beta_measured,v_predicted,"
        "v_measured,v_rel_err,ell_predicted,ell_measured,ell_rel_err,u_interface,"
        "u_solid,u_far,stefan_residual,k_eff,k_eff_rel_err,fit_r2,"
        "solute_drift_rel,heat_drift_rel,phi_min,phi_max",
        rank);
    const double v_err = (std::isfinite(res.v_measured) && v_pred != 0.0)
                             ? (res.v_measured - v_pred) / v_pred
                             : std::numeric_limits<double>::quiet_NaN();
    const double ell_pred_meas =
        (std::isfinite(res.v_measured) && res.v_measured > 0.0)
            ? p.D_l / res.v_measured
            : res.ell_predicted;
    const double ell_err = (std::isfinite(res.ell_measured) && ell_pred_meas != 0.0)
                               ? (res.ell_measured - ell_pred_meas) / ell_pred_meas
                               : std::numeric_limits<double>::quiet_NaN();
    sum_csv.row(format(
        "%s,%d,%d,%.10g,%.10g,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,"
        "%.10g,%d,%.12g,%.6g,%.10g,%.10g,%.10g,%.10g,%.6g,%.10g,%.10g,%.6g,%.10g,"
        "%.10g,%.10g,%.3e,%.10g,%.6g,%.6g,%.3e,%.3e,%.6g,%.6g",
        cfg.run_id.c_str(), cfg.nx, cfg.ny, cfg.dx, cfg.dx / p.W0, cfg.fd_order,
        res.dt, cfg.t_end, p.lambda, p.k, p.D_l, p.W0, p.tau0, p.eps4, p.at_scale,
        p.spec_source ? 1 : 0, res.omega, res.peclet, res.beta_theory,
        res.beta_measured, v_pred, res.v_measured, v_err, ell_pred_meas,
        res.ell_measured, ell_err, res.u_interface, res.u_solid, res.u_far,
        res.stefan_residual, res.k_eff, (res.k_eff - p.k) / p.k, res.fit_r2,
        res.solute_drift_rel, res.heat_drift_rel, res.phi_min, res.phi_max));
  }
  return res;
}

// ===========================================================================
// Stage 2 -- deterministic 2-D dendrite
// ===========================================================================

/// Inputs of @ref run_dendrite_2d.
struct DendriteConfig {
  ModelParams model{};
  int nx = 300;
  int ny = 300;
  /// `1` selects the 2-D slab (`Stepper<2>`); anything larger selects the
  /// full 3-D brick (`Stepper<3>`), with a spherical seed, `<100>` cubic
  /// anisotropy including `n_z`, and the tip measured on the mid-`z` plane.
  int nz = 1;
  double dx = 0.8;
  int fd_order = 4;
  double dt = 0.0;
  double dt_safety = 0.2;
  /// Initial supersaturation of the melt; `U = -omega` everywhere at `t = 0`.
  double omega = 0.55;
  /// Seed radius in `W0`. Must clear the critical nucleus `~ d0 / omega`.
  double seed_radius = 8.0;
  double t_end = 400.0;
  int n_sample = 200;
  /// Rows either side of the tip used for the parabola fit, in cells. The
  /// default corresponds to about `3 W0` at `dx = 0.8`.
  int tip_fit_halfwidth = 4;
  /// Trailing fraction of samples used for the tip-velocity fit.
  double fit_fraction = 0.3;
  std::string csv_timeseries;
  std::string csv_summary;
  std::string run_id = "dendrite";
  bool quiet = false;
};

/// Steady-state summary of a dendrite run.
struct DendriteResult {
  bool valid{false};
  double dt{0.0};
  int n_steps{0};
  double v_tip{std::numeric_limits<double>::quiet_NaN()};
  double rho_tip{std::numeric_limits<double>::quiet_NaN()};
  double x_tip{std::numeric_limits<double>::quiet_NaN()};
  /// `V rho^2 / (D_l d0)`, the selection parameter `1/sigma*` is built from.
  double selection{std::numeric_limits<double>::quiet_NaN()};
  double solute_drift_rel{std::numeric_limits<double>::quiet_NaN()};
  double heat_drift_rel{std::numeric_limits<double>::quiet_NaN()};
  double phi_min{0.0};
  double phi_max{0.0};
};

/**
 * @brief Deterministic 2-D thermo-solutal dendrite with tip diagnostics.
 *
 * Periodic square, one seed at the centre, four-fold anisotropy, no noise.
 * The `+x` arm is measured every `n_sample`-th step and written to CSV.
 */
template <int Dim>
[[nodiscard]] inline DendriteResult run_dendrite(const DendriteConfig &cfg, int rank,
                                                 int nproc, MPI_Comm comm) {
  DendriteResult res;
  const ModelParams p = cfg.model;

  const double dt_lim = explicit_dt_limit(p, cfg.dx, Dim);
  res.dt = (cfg.dt > 0.0) ? cfg.dt : cfg.dt_safety * dt_lim;
  if (res.dt > dt_lim) {
    throw std::invalid_argument("run_dendrite: dt exceeds the explicit limit");
  }
  if constexpr (Dim == 2) {
    if (cfg.nz != 1) {
      throw std::invalid_argument("run_dendrite<2>: nz must be 1");
    }
  } else {
    if (cfg.nz < 2) {
      throw std::invalid_argument("run_dendrite<3>: nz must be > 1");
    }
  }
  res.n_steps = std::max(1, static_cast<int>(std::llround(cfg.t_end / res.dt)));
  const int sample_every = std::max(1, res.n_steps / std::max(1, cfg.n_sample));

  auto domain = pfc::domain::create(pfc::GridSize({cfg.nx, cfg.ny, cfg.nz}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({cfg.dx, cfg.dx, cfg.dx}));
  pfc::comm::HaloExchangeOptions opt;
  opt.directions = Stepper<Dim>::directions();
  pfc::sim::stacks::FDPaddedCPUStack stack(domain, cfg.fd_order / 2, rank, nproc,
                                           comm, opt);
  Stepper<Dim> st(stack, p, cfg.fd_order);

  const int i_seed = cfg.nx / 2;
  const int j_seed = cfg.ny / 2;
  const int k_seed = cfg.nz / 2;
  const double xc = static_cast<double>(i_seed) * cfg.dx;
  const double yc = static_cast<double>(j_seed) * cfg.dx;
  const double zc = static_cast<double>(k_seed) * cfg.dx;
  const double inv_w = 1.0 / (std::sqrt(2.0) * p.W0);
  const double u0 = -cfg.omega;
  st.phi().for_each_owned([&](int i, int j, int kk) {
    const auto c = st.phi().coords(i, j, kk);
    const double dz = (Dim == 3) ? (c[2] - zc) : 0.0;
    const double r =
        std::sqrt((c[0] - xc) * (c[0] - xc) + (c[1] - yc) * (c[1] - yc) + dz * dz);
    st.phi()(i, j, kk) = std::tanh((cfg.seed_radius * p.W0 - r) * inv_w);
    st.solute()(i, j, kk) = u0;
    st.temperature()(i, j, kk) = 0.0;
  });
  st.seed_conserved_solute();

  const auto cons0 =
      measure_conservation(st.phi(), st.solute(), st.temperature(), p, comm);

  CsvAppender ts_csv;
  if (!cfg.csv_timeseries.empty()) {
    ts_csv = CsvAppender(cfg.csv_timeseries,
                         "run_id,step,t,x_tip,y_tip,v_tip,rho_tip,fit_rms,fit_rows,"
                         "solute_total,solute_drift_rel,heat_balance,heat_drift_abs,"
                         "theta_total,phi_total,phi_min,phi_max,u_min,u_max",
                         rank);
  }

  std::vector<double> t_s, x_s, rho_s;
  Conservation cons = cons0;
  double t = 0.0;
  for (int step = 1; step <= res.n_steps; ++step) {
    st.step(res.dt);
    t = static_cast<double>(step) * res.dt;
    if (step % sample_every != 0 && step != res.n_steps) {
      continue;
    }
    cons = measure_conservation(st.phi(), st.solute(), st.temperature(), p, comm);
    // The tip is measured on the mid-`z` plane, which is `k = 0` in 2-D and
    // the seed plane in 3-D: for `<100>` cubic anisotropy the `+x` arm grows
    // in that plane, so the 2-D and 3-D measurements are the same quantity.
    const auto plane = global_xy_plane(st.phi(), k_seed, comm);
    const DendriteTip tip = measure_tip(plane, cfg.nx, cfg.ny, cfg.dx, cfg.dx,
                                        i_seed, j_seed, cfg.tip_fit_halfwidth);
    if (tip.valid) {
      t_s.push_back(t);
      x_s.push_back(tip.x_tip);
      rho_s.push_back(tip.rho);
    }
    const double sol_drift = std::fabs(cons.solute_total - cons0.solute_total) /
                             std::fabs(cons0.solute_total);
    const double heat_drift = std::fabs(cons.heat_balance - cons0.heat_balance);
    const double v_now = trailing_slope(t_s, x_s, 0.25);
    if (ts_csv.active()) {
      ts_csv.row(format("%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%d,%.17g,%.6g,"
                        "%.17g,%.6g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g",
                        cfg.run_id.c_str(), step, t, tip.x_tip, tip.y_tip, v_now,
                        tip.rho, tip.fit_rms, tip.fit_rows, cons.solute_total,
                        sol_drift, cons.heat_balance, heat_drift, cons.theta_total,
                        cons.phi_total, cons.phi_min, cons.phi_max, cons.u_min,
                        cons.u_max));
    }
    // The dendrite must not touch its periodic image: past that point the
    // tip is growing into its own solute field and no measurement is valid.
    if (std::isfinite(tip.x_tip) &&
        tip.x_tip > xc + 0.42 * static_cast<double>(cfg.nx) * cfg.dx) {
      if (rank == 0 && !cfg.quiet) {
        std::cout << "run_dendrite: tip reached 84% of the half-box at t=" << t
                  << "; stopping before it meets its periodic image.\n";
      }
      break;
    }
  }

  res.v_tip = trailing_slope(t_s, x_s, cfg.fit_fraction);
  res.rho_tip = trailing_mean(rho_s, cfg.fit_fraction);
  res.x_tip = x_s.empty() ? std::numeric_limits<double>::quiet_NaN() : x_s.back();
  {
    const double d0 = capillary_length(p);
    res.selection = (std::isfinite(res.v_tip) && std::isfinite(res.rho_tip))
                        ? res.v_tip * res.rho_tip * res.rho_tip / (p.D_l * d0)
                        : std::numeric_limits<double>::quiet_NaN();
  }
  res.solute_drift_rel = std::fabs(cons.solute_total - cons0.solute_total) /
                         std::fabs(cons0.solute_total);
  const double heat_scale =
      std::max({std::fabs(0.5 * cons.phi_total), std::fabs(cons.theta_total), 1.0});
  res.heat_drift_rel =
      std::fabs(cons.heat_balance - cons0.heat_balance) / heat_scale;
  res.phi_min = cons.phi_min;
  res.phi_max = cons.phi_max;
  res.valid = std::isfinite(res.v_tip);

  if (rank == 0 && !cfg.csv_summary.empty()) {
    CsvAppender sum(cfg.csv_summary,
                    "run_id,nx,ny,nz,dx,fd_order,dt,t_end,lambda,k,D_l,D_th,M_c,"
                    "eps4,omega,seed_radius,d0,v_tip,rho_tip,x_tip,selection,"
                    "solute_drift_rel,heat_drift_rel,phi_min,phi_max",
                    rank);
    sum.row(format("%s,%d,%d,%d,%.10g,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,"
                   "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.3e,%.3e,"
                   "%.6g,%.6g",
                   cfg.run_id.c_str(), cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.fd_order,
                   res.dt, cfg.t_end, p.lambda, p.k, p.D_l, p.D_th, p.M_c, p.eps4,
                   cfg.omega, cfg.seed_radius, capillary_length(p), res.v_tip,
                   res.rho_tip, res.x_tip, res.selection, res.solute_drift_rel,
                   res.heat_drift_rel, res.phi_min, res.phi_max));
  }
  return res;
}

/// Dimension dispatcher: `nz == 1` runs the 2-D slab, anything else the 3-D
/// brick. Drivers call this so that neither of them has to know that `Dim` is
/// a template parameter.
[[nodiscard]] inline DendriteResult
run_dendrite_case(const DendriteConfig &cfg, int rank, int nproc, MPI_Comm comm) {
  return (cfg.nz == 1) ? run_dendrite<2>(cfg, rank, nproc, comm)
                       : run_dendrite<3>(cfg, rank, nproc, comm);
}

} // namespace alloy_dendrite
