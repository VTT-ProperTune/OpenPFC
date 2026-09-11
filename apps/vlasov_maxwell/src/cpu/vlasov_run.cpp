// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vlasov_run.cpp
 * @brief One driver for the whole validation ladder of issue #84.
 *
 * @details
 * There is one binary and one stepper, and the benchmark is selected by
 * `--case=`. That is deliberate: a Vlasov-Poisson result and a
 * Vlasov-Maxwell result produced by two different programs prove nothing
 * about each other, while the same transport, the same deposition and the
 * same diagnostics under a runtime reduction let the electrostatic stage
 * verify machinery the electromagnetic stage then relies on.
 *
 * Cases, which are the rungs of the ladder:
 *
 *  - `wave`        stage 1, a vacuum electromagnetic wave, no plasma
 *  - `landau`      stage 2, linear Landau damping (electrostatic)
 *  - `twostream`   stage 2, the electrostatic two-stream instability
 *  - `gyro`        stage 3, gyro-motion in an imposed uniform `B_z`
 *  - `weibel`      stage 4/5, the electromagnetic anisotropy instability
 *  - `filament`    stage 4/5, cold counter-streaming filamentation
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc_apps/plasma_dispersion.hpp>
#include <vlasov_maxwell/cli.hpp>
#include <vlasov_maxwell/diagnostics.hpp>
#include <vlasov_maxwell/field_output.hpp>
#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/step.hpp>

#ifdef VLASOV_ENABLE_HIP
#include <vlasov_maxwell/device_step_hip.hpp>
#endif

namespace {

using vlasov::Ledger;
using vlasov::PhaseSpace;
using vlasov::SimParams;
using vlasov::Species;

void print_usage(std::ostream &os, const char *exe) {
  SimParams d;
  os << "usage: " << exe << " [--key=value ...]\n\n"
     << "1D2V electromagnetic Vlasov-Maxwell. One stepper; --case selects\n"
     << "the rung of the validation ladder.\n\n"
     << "Case\n"
     << "  --case=NAME         wave | landau | twostream | gyro | weibel |\n"
     << "                      filament                        (landau)\n\n"
     << "Phase-space grid\n"
     << "  --nx=N              cells in x, periodic          (" << d.nx << ")\n"
     << "  --nvx=N --nvy=N     cells in velocity             (" << d.nvx << ", "
     << d.nvy << ")\n"
     << "  --Lx=X              box length in skin depths     (case default)\n"
     << "  --vmax=X            velocity half-extent, in c    (case default)\n"
     << "  --vy-halo=N         halo on the distributed axis; 0 derives it (0)\n"
     << "  --vy-shift-budget=X largest velocity shift, in cells, the halo\n"
     << "                      must absorb; only used when --vy-halo=0  (4)\n"
     << "  --interp=N          Lagrange points/order, odd    (" << d.interp_order
     << ")\n\n"
     << "Time\n"
     << "  --dt=X              step; 0 = auto                (" << d.dt << ")\n"
     << "  --dt-safety=X       fraction of the limit         (" << d.dt_safety
     << ")\n"
     << "  --t-end=X           end time in 1/omega_pe        (case default)\n"
     << "  --samples=N         diagnostic samples            (" << d.n_sample
     << ")\n\n"
     << "Physics\n"
     << "  --vth=X             thermal velocity / c          (case default)\n"
     << "  --vthy=X            v_th along y (weibel)         (case default)\n"
     << "  --drift=X           beam drift / c                (case default)\n"
     << "  --amp=X             initial perturbation amplitude(case default)\n"
     << "  --mode=N            perturbed mode number         (1)\n"
     << "  --bext=X            imposed uniform B_z           (0)\n"
     << "  --mobile-ions=0|1   add a proton species          (0)\n\n"
     << "Numerics\n"
     << "  --gauss-correction=0|1  project E_x onto Gauss; the uncorrected\n"
     << "                      residual is reported either way      (0)\n\n"
     << "Output\n"
     << "  --csv=PATH          per-sample ledger (appended, never truncated)\n"
     << "  --summary=PATH      one row per run\n"
     << "  --fields-dir=DIR    raw-brick phase space + fields + manifest\n"
     << "  --fields-every=N    snapshot every N-th sample            (1)\n"
     << "  --run-id=NAME       identifier written into every row  (vlasov)\n"
     << "  --quiet=1           suppress the human-readable report\n\n"
     << "Device\n"
     << "  --device=host|hip   where the transport and moment reduction run.\n"
     << "                      hip requires a ROCm build; the host path stays\n"
     << "                      the reference and is what every oracle in the\n"
     << "                      validation ladder was measured against  (host)\n"
     << "  --device-x=0|1      run the spectral x-shift on the device too. 0\n"
     << "                      keeps it on the host, which costs four whole-\n"
     << "                      brick bus crossings per step               (1)\n";
}

/// Per-case defaults. Collected here so that "the benchmark" is one object
/// a reader can check against the chapter, not a scatter of literals.
struct Case {
  std::string name{"landau"};
  double Lx{0.0};
  double vmax{0.0};
  double vth{0.05};
  double vthy{0.0};
  double drift{0.0};
  double amp{0.01};
  double t_end{40.0};
  bool electrostatic{true};
  bool self_consistent{true};
};

[[nodiscard]] Case make_case(const std::string &n, int mode) {
  Case c;
  c.name = n;
  const double twopi = 2.0 * std::acos(-1.0);
  if (n == "landau") {
    // k lambda_D = 0.5 with lambda_D = vth (in skin depths, since
    // lambda_D = (vth/c) d_e and vth is in c). One mode in the box:
    // k = 2 pi m / Lx, so Lx = 2 pi m vth / 0.5.
    c.vth = 0.05;
    c.Lx = twopi * mode * c.vth / 0.5;
    c.vmax = 8.0 * c.vth;
    c.amp = 0.01;
    c.t_end = 40.0;
  } else if (n == "twostream") {
    c.vth = 0.02;
    c.drift = 0.1;
    c.Lx = twopi * mode / (0.5 / c.drift); // k v0 = 0.5 omega_pe
    c.vmax = c.drift + 8.0 * c.vth;
    c.amp = 1.0e-4;
    c.t_end = 120.0;
  } else if (n == "wave") {
    c.Lx = twopi;       // k = 1 for mode 1
    c.vth = 0.05;
    c.vmax = 8.0 * c.vth;
    c.amp = 1.0e-3;
    c.t_end = 20.0;
    c.electrostatic = false;
  } else if (n == "gyro") {
    c.vth = 0.05;
    c.drift = 0.2;
    c.Lx = twopi;
    c.vmax = c.drift + 8.0 * c.vth;
    c.amp = 0.0;
    c.t_end = 40.0;
    c.electrostatic = false;
    c.self_consistent = false;
  } else if (n == "weibel") {
    c.vth = 0.05;   // along x, the wave vector
    c.vthy = 0.15;  // across it: A = 9, comfortably unstable
    c.Lx = twopi / 0.5; // k = 0.5 / d_e for mode 1
    c.vmax = 8.0 * c.vthy;
    c.amp = 1.0e-5;
    c.t_end = 150.0;
    c.electrostatic = false;
  } else if (n == "filament") {
    c.vth = 0.01;
    c.drift = 0.2;
    c.Lx = twopi / 0.5;
    c.vmax = c.drift + 10.0 * c.vth;
    c.amp = 1.0e-5;
    c.t_end = 150.0;
    c.electrostatic = false;
  } else {
    throw std::invalid_argument(
        "unknown --case='" + n +
        "'; expected wave, landau, twostream, gyro, weibel or filament");
  }
  return c;
}

} // namespace

int run(int argc, char **argv, int rank, int nproc) {
  vlasov::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) print_usage(std::cout, argv[0]);
    return EXIT_SUCCESS;
  }

  const std::string case_name = opt.text("case", "landau");
  const int mode = opt.integer("mode", 1);
  Case c = make_case(case_name, mode);

  SimParams p;
  p.nx = opt.integer("nx", 64);
  p.nvx = opt.integer("nvx", 64);
  p.nvy = opt.integer("nvy", 64);
  p.Lx = opt.real("Lx", c.Lx);
  p.v_max = opt.real("vmax", c.vmax);
  p.interp_order = opt.integer("interp", p.interp_order);
  p.vy_halo = opt.integer("vy-halo", 0);
  p.dt = opt.real("dt", 0.0);
  p.dt_safety = opt.real("dt-safety", p.dt_safety);
  p.t_end = opt.real("t-end", c.t_end);
  p.n_sample = opt.integer("samples", 100);
  p.electrostatic = opt.flag("electrostatic", c.electrostatic);
  p.self_consistent = opt.flag("self-consistent", c.self_consistent);
  p.b_ext = opt.real("bext", c.name == "gyro" ? 0.5 : 0.0);
  p.gauss_correction = opt.flag("gauss-correction", false);
  const double opt_shift_budget = opt.real("vy-shift-budget", 4.0);

  const double vth = opt.real("vth", c.vth);
  const double vthy = opt.real("vthy", c.vthy > 0.0 ? c.vthy : vth);
  const double drift = opt.real("drift", c.drift);
  const double amp = opt.real("amp", c.amp);
  p.v_thermal = std::max(vth, vthy);

  p.species.clear();
  p.species.push_back(Species{"electron", -1.0, 1.0});
  if (opt.flag("mobile-ions", false)) {
    p.species.push_back(
        Species{"ion", +1.0, vlasov::kProtonElectronMassRatio});
  }

  const std::string csv = opt.text("csv", "");
  const std::string summary = opt.text("summary", "");
  vlasov::FieldOutputConfig fo;
  fo.dir = opt.text("fields-dir", "");
  fo.every = opt.integer("fields-every", 1);
  const std::string device = opt.text("device", "host");
  const bool device_x = opt.flag("device-x", true);
  if (device != "host" && device != "hip") {
    throw std::invalid_argument("--device must be 'host' or 'hip'");
  }
#ifndef VLASOV_ENABLE_HIP
  if (device == "hip") {
    throw std::invalid_argument(
        "--device=hip needs a ROCm build; configure with --with-rocm");
  }
#endif
  const std::string run_id = opt.text("run-id", "vlasov");
  const bool quiet = opt.flag("quiet", false);
  opt.require_all_consumed();
  p.validate();

  // The velocity box has to hold the distribution. This is checked rather
  // than assumed because a truncated Maxwellian has the wrong moments, so
  // the deposited charge is wrong, and the run still looks healthy while
  // damping at the wrong rate.
  if (c.name != "wave") {
    const double vth_min = std::min(vth, vthy);
    vlasov::ics::require_resolved_tail(p.v_max - std::fabs(drift),
                                       std::max(vth, vthy), 1.0e-10);
    // And the grid inside the box has to integrate what the box holds.
    // Checked on the *narrowest* thermal width, since that is the one the
    // spacing has to resolve, and on both axes independently because they
    // are sized independently and a defect on one is invisible to a scan
    // over the other.
    vlasov::ics::require_resolved_spacing(p.dvx(), vth_min, 1.0e-10);
    vlasov::ics::require_resolved_spacing(p.dvy(), vth_min, 1.0e-10);
  }

  const double k = p.k_skin(mode);
  // The halo has to absorb the largest velocity-space shift any step of the
  // run will take, and at start-up the fields that set it do not exist yet:
  // in an instability run they are seeded at 1e-5 and grow by four orders of
  // magnitude. So it is sized from an explicit *budget* in cells rather than
  // from the initial state, and the runtime guard in `advect_vy` is what
  // makes that safe -- it throws rather than wrapping, and the driver
  // reports the peak width actually used so the budget can be checked
  // against reality afterwards instead of trusted.
  const double shift_budget = opt_shift_budget;
  const int halo = p.vy_halo > 0
                       ? p.vy_halo
                       : vlasov::required_halo_width(shift_budget, p.interp_order);
  PhaseSpace ps(p, halo, MPI_COMM_WORLD);
  vlasov::Stepper st(p, ps);

  // ---- initial condition -------------------------------------------------
  for (std::size_t s = 0; s < p.species.size(); ++s) {
    const bool ion = p.species[s].sigma > 0.0;
    ps.initialise(s, [&](double x, double vx, double vy) {
      // Stage 1 is a *vacuum* wave: no plasma at all, so rho = J = 0 and
      // the only thing being tested is the field integrator. Seeding a
      // Maxwellian here instead would make it a wave in a responding
      // plasma, which is a different and much weaker test.
      if (c.name == "wave") return 0.0;
      const double n = ion ? 1.0 : vlasov::ics::density_perturbation(x, k, amp);
      if (c.name == "twostream") {
        return n * vlasov::ics::two_stream_x(vx, vy, vth, drift);
      }
      if (c.name == "filament") {
        return n * vlasov::ics::two_stream_y(vx, vy, vth, drift);
      }
      if (c.name == "weibel") {
        return n * vlasov::ics::bi_maxwellian(vx, vy, vth, vthy);
      }
      if (c.name == "gyro") {
        return vlasov::ics::maxwellian(vx, vy, vth, 1.0, drift, 0.0);
      }
      return n * vlasov::ics::maxwellian(vx, vy, vth);
    });
  }
  vlasov::check_adapter(ps, ps.f(0));

  st.deposit_all();
  // The initial E_x must satisfy Gauss in *both* paths. Ampere only
  // evolves E_x; it cannot create the field a non-uniform initial charge
  // density already implies. Starting the electromagnetic path from
  // E_x = 0 with a perturbed density leaves a Gauss residual of order one
  // for the whole run -- measured, before this was fixed -- and the
  // electrostatic force on the plasma is simply missing.
  {
    const auto sol = vlasov::solve_gauss(st.line, st.sources.rho,
                                         p.neutrality_tol);
    if (!sol.neutral && rank == 0 && !quiet) {
      std::cout << "  WARNING net charge " << sol.net_charge
                << " exceeds the neutrality tolerance; the k = 0 mode of "
                   "Gauss is not solvable and has been left at zero\n";
    }
    st.fields.Ex = sol.Ex;
  }
  if (c.name == "wave") {
    // A right-travelling transverse mode has B_z = E_y, which is the
    // analytic solution the run is compared against.
    for (int i = 0; i < p.nx; ++i) {
      st.fields.Ey[i] = amp * std::sin(k * p.x_of(i));
      st.fields.Bz[i] = amp * std::sin(k * p.x_of(i));
    }
  } else if (c.name == "weibel" || c.name == "filament") {
    // (c) The electromagnetic instabilities have to be seeded in the field
    // they grow. A density perturbation seeds an *electrostatic* mode; the
    // Weibel mode then has to grow out of round-off, and whichever mode
    // wins is the fastest-growing one rather than the one whose growth
    // rate the oracle was asked for. Seeding B_z at the chosen mode makes
    // the measurement and the prediction refer to the same wave.
    for (int i = 0; i < p.nx; ++i) {
      st.fields.Bz[i] = amp * std::cos(k * p.x_of(i));
    }
  }

  // ---- timestep ----------------------------------------------------------
  double emax = 0.0, bmax = std::fabs(p.b_ext);
  for (int i = 0; i < p.nx; ++i) {
    emax = std::fmax(emax, std::fabs(st.fields.Ex[i]));
    bmax = std::fmax(bmax, std::fabs(st.fields.Bz[i]));
  }
  emax = std::fmax(emax, 1.0e-3); // the instability will grow one
  const double lim = vlasov::step_limit(p, 1.0, emax, std::fmax(bmax, 0.1), halo);
  const double dt = (p.dt > 0.0) ? p.dt : p.dt_safety * lim;
  const int n_steps = std::max(1, static_cast<int>(std::llround(p.t_end / dt)));
  const int sample_every = std::max(1, n_steps / std::max(1, p.n_sample));

  if (rank == 0 && !quiet) {
    std::cout << "vlasov_run: case=" << c.name << "\n"
              << "  phase space   " << p.nx << " x " << p.nvx << " x " << p.nvy
              << " = " << p.cells() << " cells/species\n"
              << "  memory        "
              << p.cells() * 8.0 * static_cast<double>(p.species.size()) / 1e9
              << " GB/copy, " << p.species.size() << " species\n"
              << "  ranks         " << nproc << " (cap " << p.nvy / halo << ")"
              << ", halo " << halo << "\n"
              << "  box           Lx=" << p.Lx << " d_e, k=" << k
              << ", vmax=" << p.v_max << " c\n"
              << "  k lambda_D    " << k * vth << "\n"
              << "  dt            " << dt << " (limit " << lim << "), "
              << n_steps << " steps to t=" << p.t_end << "\n"
              << "  device        " << device
              << (device == "hip" && !device_x ? " (x-shift on host)" : "")
              << "\n"
              << "  model         "
              << (p.electrostatic ? "electrostatic" : "electromagnetic")
              << (p.self_consistent ? "" : ", fields frozen") << "\n";
  }

  // ---- output sinks ------------------------------------------------------
  vlasov::CsvAppender ts;
  if (!csv.empty()) ts = vlasov::CsvAppender(csv, vlasov::ledger_header(), rank);
  const auto &ob = ps.owned_box();
  vlasov::FieldSnapshotWriter snap(
      fo, run_id, {p.nx, p.nvx, p.nvy},
      {ob.size[0], ob.size[1], ob.size[2]},
      {ob.low[0], ob.low[1], ob.low[2]}, p.dx(), rank, ps.comm());
  std::vector<std::string> snap_fields{"f"};

  // ---- time loop ---------------------------------------------------------
  std::vector<double> t_s, e_ex, e_bz, e_em, m_ex, m_bz, p_x, p_y;
  Ledger ref = vlasov::make_ledger(p, st.line, st.moments, st.sources,
                                   st.fields, st.gauss, 0.0, 0, mode);
  Ledger now = ref;
  int n_seen = 0, n_snap = 0;
  auto record = [&](int step, double t) {
    now = vlasov::make_ledger(p, st.line, st.moments, st.sources, st.fields,
                              st.gauss, t, step, mode);
    vlasov::set_drifts(now, ref);
    if (ts.active()) ts.row(vlasov::ledger_row(run_id, now));
    t_s.push_back(t);
    e_ex.push_back(now.energy_ex);
    e_bz.push_back(now.energy_bz);
    e_em.push_back(now.energy_em);
    m_ex.push_back(now.mode_ex);
    m_bz.push_back(now.mode_bz);
    p_x.push_back(now.momentum_x);
    p_y.push_back(now.momentum_y);
    if (snap.due(n_seen)) {
      snap.note_time(t);
      snap.write("f", n_snap, ps.f(0));
      ++n_snap;
    }
    ++n_seen;
  };
  record(0, 0.0);

  double t = 0.0;
#ifdef VLASOV_ENABLE_HIP
  if (device == "hip") {
    vlasov::hip::DeviceStepper ds(st, ps, device_x);
    for (int step = 1; step <= n_steps; ++step) {
      ds.advance(dt);
      t = static_cast<double>(step) * dt;
      if (step % sample_every == 0 || step == n_steps) {
        // The ledger and the snapshots read the host brick, so it has to be
        // current. Downloading only on a sample rather than every step is
        // the whole reason the device path is worth having: the brick stays
        // on the GCD for `sample_every` steps at a time.
        ds.download_all();
        record(step, t);
      }
    }
    ds.download_all();
  } else
#endif
  {
    for (int step = 1; step <= n_steps; ++step) {
      st.advance(dt);
      t = static_cast<double>(step) * dt;
      if (step % sample_every == 0 || step == n_steps) record(step, t);
    }
  }
  snap.write_manifest(snap_fields);

  // ---- rate fits against the oracles -------------------------------------
  // The fit window is the trailing half for a growing mode and the leading
  // half for a damped one: a Landau wave reaches the round-off floor and
  // then stops damping, so fitting late measures the floor.
  namespace pd = pfc::apps::plasma;
  double gamma_fit = std::nan(""), omega_fit = std::nan("");
  double fit_t0 = std::nan(""), fit_t1 = std::nan("");
  double gamma_ref = std::nan(""), omega_ref = std::nan("");
  if (c.name == "landau") {
    // The envelope, not every sample: see fit_envelope_rate. The window
    // starts after one oscillation period so the ballistic transient that
    // precedes the asymptotic Landau regime is excluded, and ends before
    // the mode reaches the round-off floor where it stops damping.
    gamma_fit = vlasov::fit_envelope_rate(t_s, m_ex, 3.0, 0.7 * p.t_end);
    omega_fit = vlasov::frequency_from_minima(t_s, m_ex, 0.0, 0.6 * p.t_end);
    const auto r = pd::solve_langmuir_root(k * vth);
    gamma_ref = r.omega.imag();
    omega_ref = r.omega.real();
  } else if (c.name == "twostream") {
    // ceiling 0.05 rather than the default 0.2: the two-stream mode
    // saturates hard, and at 0.2 of its maximum the last fifth of the
    // window is already rolling over -- measured, -13.5% on the rate.
    const auto w = vlasov::auto_growth_window(t_s, m_ex, 5.0, 0.05);
    gamma_fit = vlasov::fit_exponential_rate(t_s, m_ex, w[0], w[1]);
    fit_t0 = w[0];
    fit_t1 = w[1];
    pd::TwoStreamMaxwellians ts2;
    ts2.v_drift = drift;
    ts2.v_th = vth;
    const auto r = pd::solve_two_stream_root(k, ts2);
    gamma_ref = r.omega.imag();
  } else if (c.name == "gyro") {
    // The magnetic force does no work, so the mean velocity rotates
    // rigidly at omega_c = sigma B_z / mu. Both the rate and the
    // conservation of |v| are the test; the rate is reported through the
    // frequency columns because that is what it is.
    bool aliased = false;
    omega_fit = vlasov::fit_rotation_rate(t_s, p_x, p_y, &aliased);
    // Sign: d/dt (v_x + i v_y) = (q/m) B_z (v_y - i v_x)
    //                          = -i (q/m) B_z (v_x + i v_y),
    // so the *phase* advances at -(q/m) B_z. An earlier revision compared
    // against +(q/m) B_z and reported a 200% error on a rotation whose
    // magnitude was right to seven digits -- the oracle was wrong, not the
    // code, which is the failure mode a sign convention always has.
    omega_ref = -p.species[0].qm() * p.b_ext;
    if (aliased && rank == 0 && !quiet) {
      std::cout << "  WARNING the gyro-phase advanced more than half a turn "
                   "between samples; the rotation rate is aliased. Increase "
                   "--samples.\n";
    }
  } else if (c.name == "weibel" || c.name == "filament") {
    const auto w = vlasov::auto_growth_window(t_s, m_bz);
    gamma_fit = vlasov::fit_exponential_rate(t_s, m_bz, w[0], w[1]);
    fit_t0 = w[0];
    fit_t1 = w[1];
    pd::BiMaxwellian bm;
    bm.v_th_x = vth;
    bm.v_th_y = (c.name == "weibel") ? vthy : std::sqrt(drift * drift + vth * vth);
    const auto r = pd::solve_weibel_root(k, bm);
    gamma_ref = r.omega.imag();
  }

  if (rank == 0 && !quiet) {
    std::cout << "  ------------------------------------------------------\n"
              << "  energy drift  " << now.d_energy << " (relative)\n"
              << "  number drift  " << now.d_number << "\n"
              << "  entropy drift " << now.d_entropy << "\n"
              << "  Gauss residual" << now.gauss_residual << "\n"
              << "  min f         " << now.f_min << "\n"
              << "  boundary      ratio " << now.boundary_ratio
              << ", fraction " << now.boundary_fraction << "\n"
              << "  halo used     " << st.peak_halo_used << " of " << halo
              << "\n";
    if (std::isfinite(gamma_fit)) {
      std::cout << "  growth rate   " << gamma_fit;
      if (std::isfinite(gamma_ref)) {
        std::cout << "  vs oracle " << gamma_ref << "  ("
                  << 100.0 * (gamma_fit - gamma_ref) / std::fabs(gamma_ref)
                  << " %)";
      }
      std::cout << "\n";
    }
    if (std::isfinite(omega_fit)) {
      std::cout << "  frequency     " << omega_fit;
      if (std::isfinite(omega_ref)) {
        std::cout << "  vs oracle " << omega_ref << "  ("
                  << 100.0 * (omega_fit - omega_ref) / std::fabs(omega_ref)
                  << " %)";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }

  if (rank == 0 && !summary.empty()) {
    vlasov::CsvAppender sum(
        summary,
        "run_id,case,nx,nvx,nvy,Lx,vmax,k,k_lambda_D,vth,vthy,drift,amp,"
        "dt,t_end,n_steps,ranks,halo,halo_used,interp,fit_t0,fit_t1,"
        "gamma_fit,gamma_ref,gamma_rel_err,omega_fit,omega_ref,omega_rel_err,"
        "d_energy,d_number,d_entropy,d_l1,d_l2,d_momentum_x,"
        "gauss_residual,f_min,boundary_ratio,boundary_fraction,"
        "energy_ex,energy_em,energy_bz,cells",
        rank);
    char buf[2048];
    std::snprintf(
        buf, sizeof(buf),
        "%s,%s,%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,"
        "%.10g,%.10g,%d,%d,%d,%d,%d,%.6g,%.6g,"
        "%.10g,%.10g,%.6g,%.10g,%.10g,%.6g,"
        "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
        "%.6e,%.6e,%.6e,%.6e,"
        "%.10g,%.10g,%.10g,%.0f",
        run_id.c_str(), c.name.c_str(), p.nx, p.nvx, p.nvy, p.Lx, p.v_max, k,
        k * vth, vth, vthy, drift, amp, dt, p.t_end, n_steps, nproc, halo,
        st.peak_halo_used, p.interp_order, fit_t0, fit_t1, gamma_fit, gamma_ref,
        (std::isfinite(gamma_ref) && gamma_ref != 0.0)
            ? (gamma_fit - gamma_ref) / std::fabs(gamma_ref)
            : std::nan(""),
        omega_fit, omega_ref,
        (std::isfinite(omega_ref) && omega_ref != 0.0)
            ? (omega_fit - omega_ref) / std::fabs(omega_ref)
            : std::nan(""),
        now.d_energy, now.d_number, now.d_entropy, now.d_l1, now.d_l2,
        now.d_momentum_x, now.gauss_residual, now.f_min, now.boundary_ratio,
        now.boundary_fraction, now.energy_ex, now.energy_em, now.energy_bz,
        p.cells());
    sum.row(std::string(buf));
  }
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    status = run(argc, argv, rank, nproc);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "vlasov_run: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
