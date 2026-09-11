// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_growth.cpp
 * @brief Stage-2/3 driver: deterministic thermo-solutal dendrite, 2-D or
 *        3-D, with tip-velocity and tip-radius diagnostics.
 *
 * @details
 * Everything about this case is fixed so that the CSV it writes is a
 * reproducible artefact: one seed at the centre of a periodic square, no
 * noise, no random initial perturbation, a fixed sample schedule. Two runs
 * of the same command line produce the same rows. That is the property a
 * later agent doing science runs needs, and it is easy to lose by accident.
 *
 * The measurements written per sample are `x_tip`, `v_tip`, `rho_tip` and
 * the parabola-fit residual, plus the conservation invariants. Their
 * definitions -- which crossing, which rows, which fit -- are in
 * `diagnostics.hpp` next to the code that computes them, deliberately, so
 * that a plot made from this CSV can be traced to an operational definition
 * rather than to a convention.
 *
 * Latent heat is on and coupled: `--Dth` and `--Mc` are nonzero by default,
 * so the growing tip warms its surroundings and the thermal field feeds back
 * into equation (2). Setting `--Mc=0` reduces the case to a purely solutal
 * dendrite, which is a useful control.
 *
 * `--nz=N` with `N > 1` runs exactly the same model as a 3-D brick: the
 * seed becomes a sphere, the cubic anisotropy of equation (1) picks up its
 * `n_z^4` term, and the tip is measured on the mid-`z` plane, which is where
 * the `+x` `<100>` arm grows. There is no second code path -- the dimension
 * is a template parameter on the stepper, so a 2-D result is a 3-D result
 * with one axis switched off rather than a separate implementation that can
 * drift.
 *
 * Usage: `alloy_dendrite_growth [--key=value ...]`, `--help` for the list.
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

/**
 * @brief The shipped case, and why each number is what it is.
 *
 * `lambda = D_l / a2` puts the kinetic coefficient at zero, which is the
 * standard choice for a *dendrite*: the tip should be selected by capillarity
 * and diffusion, not by interface attachment kinetics. It also fixes
 * `d0 / W0 = a1 a2 / D_l = 0.277`, the same ratio as the Karma 2001
 * benchmark that PR #104 reproduces, so the two are comparable.
 *
 * `Omega = 0.55` is that benchmark's supersaturation.
 *
 * **`eps4 = 0.2` is not an anisotropy of 20%.** With the spec's
 * un-normalised `a_s = 1 + eps4 (n_x^4 + n_y^4)` and the 2-D identity
 * `n_x^4 + n_y^4 = (3 + cos 4 theta) / 4`,
 *
 *     a_s = (1 + 0.75 eps4) [ 1 + eps_eff cos 4 theta ],
 *     eps_eff = (eps4 / 4) / (1 + 0.75 eps4)
 *
 * so `eps4 = 0.2` is `eps_eff = 0.043`, which is the usual dendritic range,
 * and the `eps4 = 0.02` of the Karma 2001 benchmark would be `eps_eff =
 * 0.005` here -- far too weak to select a tip. The factor of roughly four
 * between the two conventions is the single easiest way to run this model
 * and get a growing blob instead of a dendrite; it is a consequence of the
 * spec's choice of `a_s`, discussed in `step.hpp`.
 *
 * **`D_th = 2` is a compromise and the most important caveat in this file.**
 * A metal's Lewis number `D_th / D_l` is `10^3` to `10^4`. An explicit
 * scheme's step is set by the *fastest* diffusivity, so a realistic Lewis
 * number would make the thermal field alone cost three to four orders of
 * magnitude more than the solute field it is coupled to. `Le = 1` keeps
 * equation (4) genuinely coupled -- latent heat measurably retards the tip
 * through `M_c` -- at a cost a login-node core can pay in minutes. A
 * quantitative thermo-solutal dendrite needs an implicit or spectral thermal
 * solve; that is not in scope here and is called out in the README.
 *
 * `240^2` at `dx = 0.8 W0` is a `192 W0` box: the arm reaches about `60 W0`
 * in the shipped `t_end`, against a tip radius near `10 W0`, so there is a
 * genuine tip-scale separation and the diffusion field still fits. The run
 * takes about three minutes on one login-node core.
 */
alloy_dendrite::DendriteConfig preset() {
  alloy_dendrite::DendriteConfig c;
  c.model.D_l = 2.0;
  c.model.k = 0.15;
  c.model.lambda = c.model.D_l / alloy_dendrite::kA2; // beta = 0
  c.model.D_th = 2.0;
  c.model.M_c = 0.5;
  c.model.eps4 = 0.2;
  c.model.evolve_theta = true;
  c.nx = 240;
  c.ny = 240;
  c.dx = 0.8;
  c.t_end = 400.0;
  c.seed_radius = 8.0;
  c.tip_fit_halfwidth = 5;
  return c;
}

void print_usage(std::ostream &os, const char *exe) {
  const auto d = preset();
  os << "Usage: " << exe << " [--key=value ...]\n\n"
     << "Deterministic thermo-solutal dendrite, equations (1)-(4).\n"
     << "2-D by default; --nz=N > 1 runs the same model as a 3-D brick.\n\n"
     << "Grid and integration\n"
     << "  --nx=N --ny=N       grid                    (" << d.nx << ", " << d.ny
     << ")\n"
     << "  --nz=N              1 = 2-D slab, >1 = 3-D brick (" << d.nz << ")\n"
     << "  --dx=X              spacing in W0           (" << d.dx << ")\n"
     << "  --fd-order=N        even FD order, 2..14    (" << d.fd_order << ")\n"
     << "  --dt=X              step; 0 = auto          (" << d.dt << ")\n"
     << "  --dt-safety=X       fraction of the limit   (" << d.dt_safety << ")\n"
     << "  --t-end=X           end time in tau0        (" << d.t_end << ")\n"
     << "  --samples=N         diagnostic samples      (" << d.n_sample << ")\n"
     << "  --fit-fraction=X    trailing fit fraction   (" << d.fit_fraction << ")\n"
     << "  --tip-halfwidth=N   rows each side of the tip used for the\n"
     << "                      parabola fit, in cells  (" << d.tip_fit_halfwidth
     << ")\n\n"
     << "Model\n"
     << "  --lambda=X --k=X --Dl=X --Dth=X --Mc=X --eps4=X --W0=X --tau0=X\n"
     << "  --at-scale=X        anti-trapping multiplier (" << d.model.at_scale
     << ")\n"
     << "  --evolve-theta=0|1  integrate equation (4)   (" << d.model.evolve_theta
     << ")\n\n"
     << "Case\n"
     << "  --omega=X           initial supersaturation (" << d.omega << ")\n"
     << "  --seed-radius=X     seed radius in W0       (" << d.seed_radius << ")\n\n"
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

  auto cfg = preset();
  cfg.nx = opt.integer("nx", cfg.nx);
  cfg.ny = opt.integer("ny", cfg.ny);
  cfg.nz = opt.integer("nz", cfg.nz);
  cfg.dx = opt.real("dx", cfg.dx);
  cfg.fd_order = opt.integer("fd-order", cfg.fd_order);
  cfg.dt = opt.real("dt", cfg.dt);
  cfg.dt_safety = opt.real("dt-safety", cfg.dt_safety);
  cfg.t_end = opt.real("t-end", cfg.t_end);
  cfg.n_sample = opt.integer("samples", cfg.n_sample);
  cfg.fit_fraction = opt.real("fit-fraction", cfg.fit_fraction);
  cfg.tip_fit_halfwidth = opt.integer("tip-halfwidth", cfg.tip_fit_halfwidth);
  cfg.omega = opt.real("omega", cfg.omega);
  cfg.seed_radius = opt.real("seed-radius", cfg.seed_radius);
  cfg.model.lambda = opt.real("lambda", cfg.model.lambda);
  cfg.model.k = opt.real("k", cfg.model.k);
  cfg.model.D_l = opt.real("Dl", cfg.model.D_l);
  cfg.model.D_th = opt.real("Dth", cfg.model.D_th);
  cfg.model.M_c = opt.real("Mc", cfg.model.M_c);
  cfg.model.eps4 = opt.real("eps4", cfg.model.eps4);
  cfg.model.W0 = opt.real("W0", cfg.model.W0);
  cfg.model.tau0 = opt.real("tau0", cfg.model.tau0);
  cfg.model.at_scale = opt.real("at-scale", cfg.model.at_scale);
  cfg.model.evolve_theta = opt.flag("evolve-theta", cfg.model.evolve_theta);
  cfg.csv_timeseries = opt.text("csv", cfg.csv_timeseries);
  cfg.csv_summary = opt.text("summary", cfg.csv_summary);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  cfg.quiet = opt.flag("quiet", cfg.quiet);
  opt.require_all_consumed();

  const auto res =
      alloy_dendrite::run_dendrite_case(cfg, rank, nproc, MPI_COMM_WORLD);

  if (rank != 0 || cfg.quiet) {
    return res.valid ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  std::cout << std::setprecision(6);
  std::cout << "\n=== alloy_dendrite dendrite growth ===\n"
            << "  grid          " << cfg.nx << " x " << cfg.ny << " x " << cfg.nz
            << "  dx/W0=" << cfg.dx / cfg.model.W0 << "  fd_order=" << cfg.fd_order
            << "  ranks=" << nproc << "\n"
            << "  model         " << alloy_dendrite::derived_summary(cfg.model)
            << "  D_th=" << cfg.model.D_th << " M_c=" << cfg.model.M_c << "\n"
            << "  dt            " << res.dt << " over " << res.n_steps
            << " steps to t=" << cfg.t_end << "\n"
            << "  Omega         " << cfg.omega << "\n"
            << "  ------------------------------------------------------------\n"
            << "  tip position  " << res.x_tip << "\n"
            << "  tip velocity  " << res.v_tip << "  (W0/tau0)\n"
            << "  tip radius    " << res.rho_tip << "  (W0)\n"
            << "  selection     V rho^2 / (D_l d0) = " << res.selection << "\n"
            << "  ------------------------------------------------------------\n"
            << "  solute drift  " << res.solute_drift_rel << " (relative)\n"
            << "  heat balance  " << res.heat_drift_rel << " (relative)\n"
            << "  phi range     [" << res.phi_min << ", " << res.phi_max << "]\n"
            << std::endl;
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
            std::cerr << "alloy_dendrite_growth: " << e.what() << "\n";
            print_usage(std::cerr, app_argv[0]);
          }
          return EXIT_FAILURE;
        }
      });
}
