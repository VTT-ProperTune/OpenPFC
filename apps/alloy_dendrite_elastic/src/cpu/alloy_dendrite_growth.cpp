// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file alloy_dendrite_growth.cpp
 * @brief Stage-2/3/4 driver: deterministic dendrite, 2-D or 3-D, with tip
 *        diagnostics and the optional elastic coupling of equations (5)-(7).
 *
 * @details
 * Everything about this case is fixed so that the CSV it writes is a
 * reproducible artefact: one seed at the centre of a periodic square, no
 * noise, no random initial perturbation, a fixed sample schedule. Two runs
 * of the same command line produce the same rows. That is the property a
 * science comparison needs, and it is easy to lose by accident -- it is also
 * what makes `--elastic=0` versus `--elastic=1` a controlled experiment
 * rather than two runs that merely look similar.
 *
 * The measurements written per sample are `x_tip`, `v_tip`, `rho_tip` at
 * four independent fit windows, and the parabola-fit residual, plus the
 * conservation invariants and (when the coupling is on) the elastic
 * iteration count, energy and driving force. Their definitions -- which
 * crossing, which rows, which fit -- are in `diagnostics.hpp` next to the
 * code that computes them, deliberately, so that a plot made from this CSV
 * can be traced to an operational definition rather than to a convention.
 *
 * `--nz=N` with `N > 1` runs exactly the same model as a 3-D brick: the seed
 * becomes a sphere, the cubic anisotropy of equation (1) picks up its
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
#if ALLOY_DENDRITE_HAVE_ELASTICITY
#include <alloy_dendrite/material.hpp>
#endif

namespace {

/**
 * @brief The shipped case, and why each number is what it is.
 *
 * `lambda = D_l / a2` puts the kinetic coefficient at zero, which is the
 * standard choice for a *dendrite*: the tip should be selected by capillarity
 * and diffusion, not by interface attachment kinetics. It also fixes
 * `d0 / W0 = a1 a2 / D_l = 0.277`, the same ratio as the Karma 2001
 * dilute-alloy benchmark, so the two are comparable.
 *
 * `Omega = 0.55`, `k = 0.15`, `D_l = 2` are that benchmark's parameters.
 *
 * **`eps4 = 0.02` now means 2% anisotropy, and it did not used to.** Equation
 * (1) is the *normalised* Karma-Rappel form since the 2026-09-11 spec
 * correction, so in 2-D `a_s = 1 + eps4 cos 4 theta` exactly and literature
 * `eps4` transfer directly. The previous revision of this application used
 * the un-normalised `a_s = 1 + eps4 sum n_i^4`, whose effective strength is
 * `(eps4/4)/(1 + 0.75 eps4)` -- about a quarter of nominal -- and which
 * therefore needed `eps4 = 0.2` to behave like 4.3%. Anyone comparing a new
 * CSV against an old one has to know that: **the same `eps4` is not the same
 * physics across that change.**
 *
 * **Latent heat is off by default, and that is a deliberate reversal.**
 * `M_c = 0`, `--evolve-theta=0`: the shipped case is the *isothermal*
 * dilute-alloy dendrite. The reason is that a steady tip is the whole point
 * of Stage 2 and a closed periodic box with latent-heat release **does not
 * have one**: equation (4) has no sink, so `sum theta` grows monotonically,
 * `M_c theta` eats the driving force, and the tip decelerates for as long as
 * the run lasts. The previous revision shipped exactly that case and
 * correctly reported that it never reached a steady tip. Keeping the thermal
 * coupling would have made a selection measurement impossible, so it is one
 * flag away (`--Mc=0.5 --evolve-theta=1 --Dth=2`) rather than on, and the
 * README shows what it does to the plateau.
 *
 * `600^2` at `dx = 0.8 W0` is a `480 W0` box, which is what it takes for the
 * tip velocity to reach a plateau *and* for the plateau to be independent of
 * the box: the tip radius is about `10 W0` and the solute diffusion length
 * `D_l/V` about `20 W0`, but the four arms keep rejecting solute into a
 * closed domain and it is the *reservoir*, not the diffusion length, that
 * sets the box size. `omega_eff` in the CSV is the far-field supersaturation
 * the tip actually sees, and watching it is how the reservoir argument is
 * checked rather than asserted.
 */
alloy_dendrite::DendriteConfig preset() {
  alloy_dendrite::DendriteConfig c;
  c.model.D_l = 2.0;
  c.model.k = 0.15;
  c.model.lambda = c.model.D_l / alloy_dendrite::kA2; // beta = 0
  c.model.D_th = 2.0;
  c.model.M_c = 0.0;
  c.model.eps4 = 0.02;
  c.model.evolve_theta = false;
  c.nx = 600;
  c.ny = 600;
  c.dx = 0.8;
  c.t_end = 2000.0;
  c.n_sample = 400;
  c.seed_radius = 8.0;
  c.tip_fit_halfwidth = 6;
  c.fit_fraction = 0.3;
  return c;
}

void print_usage(std::ostream &os, const char *exe) {
  const auto d = preset();
  os << "Usage: " << exe << " [--key=value ...]\n\n"
     << "Deterministic dilute-alloy dendrite, equations (1)-(4), optionally\n"
     << "coupled to the eigenstrain microelasticity of equations (5)-(7).\n"
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
     << "  --tip-halfwidth=N   rows each side of the tip used for the primary\n"
     << "                      parabola fit, in cells  (" << d.tip_fit_halfwidth
     << ")\n"
     << "  --tip-windows=A:B:C:D  the four fit half-widths reported side by\n"
     << "                      side, in cells          ("
     << d.tip_windows[0] << ":" << d.tip_windows[1] << ":" << d.tip_windows[2]
     << ":" << d.tip_windows[3] << ")\n\n"
     << "Model\n"
     << "  --lambda=X --k=X --Dl=X --Dth=X --Mc=X --eps4=X --W0=X --tau0=X\n"
     << "  --at-scale=X        anti-trapping multiplier (" << d.model.at_scale
     << ")\n"
     << "  --evolve-theta=0|1  integrate equation (4)   " << "("
     << d.model.evolve_theta << ")\n"
     << "                      NOTE: a closed periodic box with latent heat\n"
     << "                      has no steady tip; see the file comment.\n\n"
     << "Case\n"
     << "  --omega=X           initial supersaturation (" << d.omega << ")\n"
     << "  --seed-radius=X     seed radius in W0       (" << d.seed_radius
     << ")\n\n"
#if ALLOY_DENDRITE_HAVE_ELASTICITY
     << "Elasticity, equations (5)-(7) (defaults: Al-4.5wt%Cu, see "
        "material.hpp)\n"
     << "  --elastic=0|1       solve equations (5)-(7)  (0)\n"
     << "  --lambda-el=X       coupling in equation (2); default is lambda\n"
     << "                      itself, which is the calibrated value when the\n"
     << "                      stiffnesses are in units of f_ref = L dT0/T_M\n"
     << "  --eps-c=X           d eps*/dU               (" << alloy_dendrite::material::kEpsC
     << ")\n"
     << "  --eps-T=X           d eps*/dtheta           (" << alloy_dendrite::material::kEpsT
     << ")\n"
     << "  --u-ref=X           unstrained U; default -omega\n"
     << "  --theta-ref=X       unstrained theta        (0)\n"
     << "  --el-c11=X --el-c12=X --el-c44=X   solid stiffness in units of\n"
     << "                      f_ref; default Al softened by --el-soften\n"
     << "  --el-soften=X       fraction of the 300 K constants ("
     << alloy_dendrite::material::kSofteningAtTm << ")\n"
     << "  --el-mu-liquid=X    mu_l/mu_s regularisation (0.05)\n"
     << "  --el-bulk-liquid=X  K_l/K_s                  (1)\n"
     << "  --el-macro=free|clamped  how eps_hat(0) is fixed (free)\n"
     << "  --el-scheme=em|basic     fixed point         (em)\n"
     << "  --el-tol=X          relative polarisation tol (1e-6)\n"
     << "  --el-iter=N         cap on Green applications (50)\n"
     << "  --n-el-substep=N    solve every N-th step    (1)\n"
     << "  --el-warm-start=0|1 reuse the previous strain (1)\n\n"
#endif
     << "Output (CSV is appended, never truncated)\n"
     << "  --fields-dir=DIR    raw-brick snapshots of phi, U, theta (and,\n"
     << "                      with --elastic=1, f_el, dfel_dphi and the two\n"
     << "                      stress invariants) plus a JSON manifest\n"
     << "  --fields-every=N    snapshot every N-th diagnostic sample   (1)\n"
     << "  --csv=PATH          per-sample time series\n"
     << "  --summary=PATH      one row per run\n"
     << "  --run-id=NAME       identifier written into both CSVs (" << d.run_id
     << ")\n"
     << "  --quiet=1           suppress the human-readable report\n";
}

/// Parse `A:B:C:D` into the four tip-radius fit half-widths.
void parse_tip_windows(const std::string &spec, int (&out)[4]) {
  int n = 0;
  std::size_t pos = 0;
  while (n < 4 && pos <= spec.size()) {
    const std::size_t colon = spec.find(':', pos);
    const std::string tok = spec.substr(
        pos, (colon == std::string::npos) ? std::string::npos : colon - pos);
    if (tok.empty()) {
      break;
    }
    out[n++] = std::atoi(tok.c_str());
    if (colon == std::string::npos) {
      break;
    }
    pos = colon + 1;
  }
  if (n != 4) {
    throw std::invalid_argument(
        "--tip-windows needs exactly four colon-separated half-widths, e.g. "
        "3:5:8:12");
  }
  for (int q = 0; q < 4; ++q) {
    if (out[q] < 2) {
      throw std::invalid_argument(
          "--tip-windows: a half-width below 2 cannot support a parabola fit");
    }
  }
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
  cfg.fields.dir = opt.text("fields-dir", "");
  cfg.fields.every = opt.integer("fields-every", cfg.fields.every);
  cfg.fit_fraction = opt.real("fit-fraction", cfg.fit_fraction);
  cfg.tip_fit_halfwidth = opt.integer("tip-halfwidth", cfg.tip_fit_halfwidth);
  if (opt.has("tip-windows")) {
    parse_tip_windows(opt.text("tip-windows", ""), cfg.tip_windows);
  }
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

  const bool want_elastic = opt.flag("elastic", false);
#if ALLOY_DENDRITE_HAVE_ELASTICITY
  namespace mat = alloy_dendrite::material;
  cfg.elastic = want_elastic;
  auto &ep = cfg.elastic_params;
  const double soften = opt.real("el-soften", mat::kSofteningAtTm);
  ep.c_solid = mat::al_cu_solid_stiffness(soften);
  ep.c_solid.c11 = opt.real("el-c11", ep.c_solid.c11);
  ep.c_solid.c12 = opt.real("el-c12", ep.c_solid.c12);
  ep.c_solid.c44 = opt.real("el-c44", ep.c_solid.c44);
  ep.mu_liquid_fraction = opt.real("el-mu-liquid", ep.mu_liquid_fraction);
  ep.bulk_liquid_fraction = opt.real("el-bulk-liquid", ep.bulk_liquid_fraction);
  ep.eps_c = opt.real("eps-c", mat::kEpsC);
  ep.eps_T = opt.real("eps-T", mat::kEpsT);
  // The melt is unstrained: at t = 0 every cell is at U = -omega, so taking
  // U_ref = -omega makes the eigenstrain zero in the far field and puts the
  // whole elastic effect where the composition has actually changed. Any
  // other choice adds a uniform eigenstrain, which under the zero-mean-stress
  // macroscopic condition is elastically invisible but under the clamped one
  // is not -- another reason the two modes are both available.
  ep.U_ref = opt.real("u-ref", -cfg.omega);
  ep.theta_ref = opt.real("theta-ref", 0.0);
  {
    const std::string macro = opt.text("el-macro", "free");
    if (macro == "free") {
      ep.macro_strain = alloy_dendrite::MacroStrainMode::ZeroMeanStress;
    } else if (macro == "clamped") {
      ep.macro_strain = alloy_dendrite::MacroStrainMode::Clamped;
    } else {
      throw std::invalid_argument("--el-macro must be 'free' or 'clamped'");
    }
    const std::string scheme = opt.text("el-scheme", "em");
    if (scheme == "em") {
      ep.scheme = pfc::apps::MicroelasticityScheme::EyreMilton;
    } else if (scheme == "basic") {
      ep.scheme = pfc::apps::MicroelasticityScheme::Basic;
    } else {
      throw std::invalid_argument("--el-scheme must be 'em' or 'basic'");
    }
  }
  ep.tol_el = opt.real("el-tol", ep.tol_el);
  ep.n_el_iter = opt.integer("el-iter", ep.n_el_iter);
  ep.n_el_substep = opt.integer("n-el-substep", ep.n_el_substep);
  ep.warm_start = opt.flag("el-warm-start", ep.warm_start);
  // lambda_el follows lambda unless it is named: with the stiffnesses in
  // units of f_ref that is the calibrated coupling, and a value that differs
  // from lambda is then an explicit statement that the coupling is being
  // scaled, not an accident of a default.
  cfg.model.lambda_el =
      want_elastic ? opt.real("lambda-el", cfg.model.lambda) : 0.0;
#else
  if (want_elastic) {
    throw std::runtime_error(
        "--elastic=1 needs equations (5)-(7), which are a spectral solve and "
        "are only compiled in when OpenPFC is built with HeFFTe. Reconfigure "
        "with -DOpenPFC_ENABLE_HEFFTE=ON.");
  }
#endif

  cfg.csv_timeseries = opt.text("csv", cfg.csv_timeseries);
  cfg.csv_summary = opt.text("summary", cfg.csv_summary);
  cfg.run_id = opt.text("run-id", cfg.run_id);
  cfg.quiet = opt.flag("quiet", cfg.quiet);
  opt.require_all_consumed();

  if (rank == 0 && !cfg.quiet &&
      cfg.model.eps4 > alloy_dendrite::kEps4StiffnessLimit) {
    std::cerr << "alloy_dendrite_growth: WARNING eps4 = " << cfg.model.eps4
              << " exceeds 1/15, where the 2-D interfacial stiffness "
                 "1 - 15 eps4 cos 4theta changes sign. The equilibrium shape "
                 "then has missing orientations and the smooth-tip selection "
                 "theory sigma* is compared against does not apply.\n";
  }

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
            << "  Omega         " << cfg.omega << " (initial), "
            << res.omega_eff << " (effective, -min U at t_end)\n"
            << "  ------------------------------------------------------------\n"
            << "  tip position  " << res.x_tip << "\n"
            << "  tip velocity  " << res.v_tip << "  (W0/tau0)\n"
            << "  tip radius    " << res.rho_tip << "  (W0), fit half-width "
            << cfg.tip_fit_halfwidth << " cells\n"
            << "    windows     " << res.rho_window[0] << ", "
            << res.rho_window[1] << ", " << res.rho_window[2] << ", "
            << res.rho_window[3] << "  spread " << 100.0 * res.rho_window_spread
            << " %\n"
            << "  steadiness    dV/V = " << 100.0 * res.v_drift
            << " %, drho/rho = " << 100.0 * res.rho_drift
            << " % across the fit window\n"
            << "  selection     sigma* = 2 d0 D / (V rho^2) = " << res.sigma_star
            << "\n"
            << "  Ivantsov      V rho = " << res.v_rho << " vs 2 D P(Omega_eff) = "
            << res.v_rho_ivantsov << "\n";
  if (res.el_solves > 0) {
    std::cout << "  ------------------------------------------------------------\n"
              << "  elastic       " << res.el_solves << " solves, mean "
              << res.el_iter_mean << " iterations (max " << res.el_iter_max
              << "), " << res.el_nonconverged << " unconverged\n"
              << "                int f_el dV = " << res.el_energy
              << ", max |df_el/dphi| = " << res.el_max_dfel << "\n"
              << "                mean pressure " << res.el_mean_stress
              << " (zero only for a homogeneous modulus)\n";
  }
  std::cout << "  ------------------------------------------------------------\n"
            << "  solute drift  " << res.solute_drift_rel << " (relative)\n"
            << "  heat balance  " << res.heat_drift_rel << " (relative"
            << (cfg.model.evolve_theta ? "" : "; meaningless with --evolve-theta=0")
            << ")\n"
            << "  phi range     [" << res.phi_min << ", " << res.phi_max << "]\n"
            << "  samples       " << res.n_samples << ", of which "
            << res.n_samples_failed << " had no measurable tip\n";
  if (!res.state_finite) {
    // Loud, because the failure mode is a *plausible* number: samples whose
    // tip fit fails are skipped, so the trailing-window fit silently falls
    // back on the last healthy samples and reports a velocity for a field
    // that is now full of NaN. `dx = 1.0 W0` does exactly this.
    std::cout << "\n  *** DIVERGED: the final state is not a phase field "
                 "(phi outside [-1.1, 1.1] or U non-finite).\n"
                 "      Every number above was fitted to the samples taken "
                 "before it blew up and is NOT a\n"
                 "      result. Reduce dx (this model is unstable above about "
                 "0.8 W0) or dt_safety.\n";
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
            std::cerr << "alloy_dendrite_growth: " << e.what() << "\n";
            print_usage(std::cerr, app_argv[0]);
          }
          return EXIT_FAILURE;
        }
      });
}
