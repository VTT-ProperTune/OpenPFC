// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_planar.cpp
 * @brief Stage-1 driver: isothermal planar front, measured against the
 *        thin-interface prediction.
 *
 * @details
 * This is the acceptance gate for the solidification core, so the binary's
 * job is to print numbers that can be compared with theory, not to print
 * "PASS". It reports, on one line each:
 *
 *  - the steady front velocity against `V = (Omega - 1) / (k beta)`;
 *  - the kinetic coefficient recovered as `-U_i / V` against
 *    `beta = a1 (tau0 / (lambda W0)) (1 - a2 lambda W0^2 / (tau0 D_l))`;
 *  - the solute boundary-layer width against `D_l / V`;
 *  - the effective partition coefficient against the input `k`;
 *  - the drift of total solute and of the latent-heat balance.
 *
 * Run it three times with `--at-scale=1`, `--at-scale=0` and
 * `--at-scale=-1` and the `k_eff` column says, without argument, whether the
 * anti-trapping current is right: only the first is flat in velocity.
 *
 * The temperature field is integrated (`--evolve-theta=1`, the default) but
 * does not feed back (`--Mc=0`), which is what "isothermal" means here: the
 * phase field sees no thermal driving force, while equation (4) still
 * releases latent heat so its balance can be checked in the same run.
 *
 * Usage: `alloy_dendrite_planar [--key=value ...]`, `--help` for the list.
 */

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <mpi.h>

#include <openpfc/runtime/common/mpi_main.hpp>

#include <alloy_dendrite/cases.hpp>
#include <alloy_dendrite/cli.hpp>

namespace {

void print_usage(std::ostream &os, const char *exe) {
  alloy_dendrite::PlanarConfig d;
  os << "Usage: " << exe << " [--key=value ...]\n\n"
     << "Stage-1 planar-interface verification of the dilute-alloy phase field.\n"
     << "A solid slab in a periodic box gives two symmetric fronts; the +x one\n"
     << "is measured. Every option below is optional.\n\n"
     << "Grid and integration\n"
     << "  --nx=N              cells along x                    (" << d.nx << ")\n"
     << "  --ny=N              transverse cells, >=4 recommended (" << d.ny << ")\n"
     << "  --dx=X              spacing in W0                     (" << d.dx << ")\n"
     << "  --fd-order=N        even central FD order, 2..14      (" << d.fd_order
     << ")\n"
     << "  --dt=X              explicit step; 0 = auto           (" << d.dt << ")\n"
     << "  --dt-safety=X       fraction of the explicit limit    (" << d.dt_safety
     << ")\n"
     << "  --t-end=X           end time in tau0                  (" << d.t_end
     << ")\n"
     << "  --samples=N         diagnostic samples                (" << d.n_sample
     << ")\n"
     << "  --fit-fraction=X    trailing fraction used for fits   (" << d.fit_fraction
     << ")\n"
     << "  --seed-half-width=X initial slab half-width in W0     ("
     << d.seed_half_width << ")\n"
     << "  --analytic-ic=0|1   seed the steady solute profile     (" << d.analytic_ic
     << ")\n"
     << "  --ic-velocity=X     seed the profile of a front at this speed\n"
     << "                      instead of the predicted one; 0 = predicted\n\n"
     << "Model (equations (1)-(4))\n"
     << "  --lambda=X          coupling constant                 (" << d.model.lambda
     << ")\n"
     << "  --k=X               partition coefficient             (" << d.model.k
     << ")\n"
     << "  --Dl=X              solute diffusivity in W0^2/tau0   (" << d.model.D_l
     << ")\n"
     << "  --Dth=X             thermal diffusivity               (" << d.model.D_th
     << ")\n"
     << "  --Mc=X              thermal coupling in eq. (2)       (" << d.model.M_c
     << ")\n"
     << "  --eps4=X            cubic anisotropy strength         (" << d.model.eps4
     << ")\n"
     << "  --W0=X, --tau0=X    length and time units             (" << d.model.W0
     << ", " << d.model.tau0 << ")\n"
     << "  --evolve-theta=0|1  integrate equation (4)            ("
     << d.model.evolve_theta << ")\n\n"
     << "Verification knobs\n"
     << "  --velocity=X        target steady velocity; sets Omega ("
     << d.velocity_target << ")\n"
     << "  --omega=X           set Omega directly, overrides --velocity\n"
     << "  --at-scale=X        anti-trapping multiplier: 1 physical, 0 off,\n"
     << "                      -1 wrong sign                     ("
     << d.model.at_scale << ")\n"
     << "  --spec-source=0|1   use MODEL_SPEC eq. (3)'s inconsistent source "
        "term\n\n"
     << "Output (CSV is appended, never truncated)\n"
     << "  --csv=PATH          per-sample time series\n"
     << "  --summary=PATH      one row per run\n"
     << "  --run-id=NAME       identifier written into both CSVs (" << d.run_id
     << ")\n"
     << "  --quiet=1           suppress the human-readable report\n";
}

int run(int argc, char **argv, int rank, int nproc) {
  alloy_dendrite::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) {
      print_usage(std::cout, argv[0]);
    }
    return EXIT_SUCCESS;
  }

  alloy_dendrite::PlanarConfig cfg;
  cfg.nx = opt.integer("nx", cfg.nx);
  cfg.ny = opt.integer("ny", cfg.ny);
  cfg.dx = opt.real("dx", cfg.dx);
  cfg.fd_order = opt.integer("fd-order", cfg.fd_order);
  cfg.dt = opt.real("dt", cfg.dt);
  cfg.dt_safety = opt.real("dt-safety", cfg.dt_safety);
  cfg.t_end = opt.real("t-end", cfg.t_end);
  cfg.n_sample = opt.integer("samples", cfg.n_sample);
  cfg.fit_fraction = opt.real("fit-fraction", cfg.fit_fraction);
  cfg.seed_half_width = opt.real("seed-half-width", cfg.seed_half_width);
  cfg.analytic_ic = opt.flag("analytic-ic", cfg.analytic_ic);
  cfg.ic_velocity = opt.real("ic-velocity", cfg.ic_velocity);
  cfg.velocity_target = opt.real("velocity", cfg.velocity_target);
  cfg.omega = opt.real("omega", cfg.omega);
  cfg.model.lambda = opt.real("lambda", cfg.model.lambda);
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.W0 = opt.real("W0", cfg.model.W0);
  cfg.model.tau0 = opt.real("tau0", cfg.model.tau0);
  cfg.model.at_scale = opt.real("at-scale", cfg.model.at_scale);
  cfg.model.spec_source = opt.flag("spec-source", cfg.model.spec_source);
  cfg.model.evolve_theta = opt.flag("evolve-theta", cfg.model.evolve_theta);
  cfg.csv_timeseries = opt.text("csv", cfg.csv_timeseries);
  cfg.csv_summary = opt.text("summary", cfg.csv_summary);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  cfg.quiet = opt.flag("quiet", cfg.quiet);
  opt.require_all_consumed();

  const auto res = alloy_dendrite::run_planar(cfg, rank, nproc, MPI_COMM_WORLD);

  if (rank != 0 || cfg.quiet) {
    return res.valid ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  const double d0 = alloy_dendrite::capillary_length(cfg.model);
  std::cout << std::setprecision(6);
  std::cout << "\n=== alloy_dendrite planar, Stage 1 ===\n"
            << "  grid          " << cfg.nx << " x " << cfg.ny
            << "  dx/W0=" << cfg.dx / cfg.model.W0 << "  fd_order=" << cfg.fd_order
            << "  ranks=" << nproc << "\n"
            << "  model         " << alloy_dendrite::derived_summary(cfg.model)
            << "\n"
            << "  d0/W0         " << d0 / cfg.model.W0 << "\n"
            << "  dt            " << res.dt << " over " << res.n_steps
            << " steps to t=" << cfg.t_end << "\n"
            << "  Omega         " << res.omega
            << "   (Peclet W0 V/D_l = " << res.peclet << ")\n"
            << "  ------------------------------------------------------------\n"
            << "  velocity      measured " << res.v_measured << "  predicted "
            << res.v_predicted << "  rel.err "
            << (res.v_measured - res.v_predicted) / res.v_predicted << "\n"
            << "  beta          measured " << res.beta_measured << "  theory   "
            << res.beta_theory << "  rel.err "
            << (res.beta_measured - res.beta_theory) / res.beta_theory << "\n"
            << "  layer D_l/V   measured " << res.ell_measured << "  predicted "
            << cfg.model.D_l / res.v_measured << "  rel.err "
            << (res.ell_measured * res.v_measured / cfg.model.D_l - 1.0)
            << "   (fit R^2 " << res.fit_r2 << ")\n"
            << "  k_eff         measured " << res.k_eff << "  input k  "
            << cfg.model.k << "  rel.err " << (res.k_eff - cfg.model.k) / cfg.model.k
            << "\n"
            << "  U_i           " << res.u_interface << "   U_solid " << res.u_solid
            << "   (equal iff k_eff == k)\n"
            << "  mass balance  U_far " << res.u_far
            << "  vs k U_s - 1 = " << (cfg.model.k * res.u_solid - 1.0)
            << "  residual " << res.stefan_residual << "\n"
            << "  ------------------------------------------------------------\n"
            << "  solute drift  " << res.solute_drift_rel << " (relative)\n"
            << "  heat balance  " << res.heat_drift_rel
            << " (relative drift of "
               "sum(theta) - sum(phi)/2)\n"
            << "  phi range     [" << res.phi_min << ", " << res.phi_max << "]\n";
  if (res.peclet > 0.1) {
    std::cout << "  WARNING: interface Peclet " << res.peclet
              << " > 0.1; the thin-interface prediction is outside its range "
                 "of validity here.\n";
  }
  std::cout << std::endl;
  return res.valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        try {
          return run(app_argc, app_argv, rank, nproc);
        } catch (const std::exception &e) {
          if (rank == 0) {
            std::cerr << "alloy_dendrite_planar: " << e.what() << "\n";
            print_usage(std::cerr, app_argv[0]);
          }
          return EXIT_FAILURE;
        }
      });
}
