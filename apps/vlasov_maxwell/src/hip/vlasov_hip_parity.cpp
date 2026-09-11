// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file vlasov_hip_parity.cpp
 * @brief Does the device compute the same 1D2V Vlasov step as the host?
 *        Operator by operator first, then integrated over many steps.
 *
 * @details
 * ## Two questions, not one
 *
 * "CPU/GPU parity on a small case" can mean two quite different things and
 * conflating them hides failures. This driver asks both separately.
 *
 *  1. **Operator parity.** Given *identical* input, does one call of the
 *     device gather / reduction return what the host call returns? This is
 *     a statement about the kernels and about nothing else.
 *  2. **Integrated parity.** After `n` Strang steps from the same initial
 *     condition, do the two paths still agree? This is a statement about
 *     error *growth*, and it can fail while (1) passes -- a Vlasov system
 *     is a transport problem with an instability in it, and a difference of
 *     one ulp in the deposited current is a difference in the field, which
 *     is a difference in the next shift.
 *
 * ## The tolerances, derived rather than fitted
 *
 * Let `u = 2^-53 = 1.11e-16` be the double-precision unit round-off.
 *
 * **The gathers are expected to be bitwise.** Both paths accumulate
 * `p` products in the same order with the same weights (the weights are
 * computed once on the host and uploaded -- `device_step_hip.hpp` explains
 * why), the gather does no reduction across threads, and contraction is
 * switched off in the kernels. So there is no arithmetic freedom left and
 * the test is `==`, not a tolerance. The driver nevertheless *reports* the
 * worst difference and the fraction of bitwise-equal cells, because "we
 * asserted bitwise and it held" and "we asserted a tolerance loose enough
 * to hide a one-ulp drift" look identical in a pass/fail line. If the host
 * compiler contracts the host loop into an FMA -- which it may, depending on
 * `-march` -- the difference becomes one rounding per stencil point, bounded
 * by `p u max|w f|`, and the driver says so instead of failing silently.
 *
 * **The reduction cannot be bitwise, and the bound says how far off it may
 * be.** Summing `N_v = N_vx N_vy` terms sequentially has a worst-case error
 * `(N_v - 1) u S` with `S = sum |f_i|`; the device's partial-then-combine
 * order has its own error of the same form. The difference of the two is
 * therefore bounded by
 *
 *     |dev - host| <= 2 (N_v - 1) u S ,
 *
 * and relative to the value itself, with the condition number
 * `kappa = S / |sum f_i|`,
 *
 *     eta := 2 N_v u kappa .
 *
 * `kappa` is measured from the actual initial state rather than assumed:
 * `f` is a distribution function and is non-negative to within the
 * interpolation's undershoot, so `kappa` is very close to one here, but a
 * run in which it is not is a run whose moments are ill-conditioned and the
 * reader should be told. The realistic error is `O(sqrt(N_v) u)`, two orders
 * of magnitude smaller than the bound at these sizes, and the driver prints
 * the measured value next to the bound so that the margin is visible.
 *
 * The entropy column carries one extra `u` per term, because `log` is
 * correctly rounded to within one ulp by both libms but not necessarily to
 * the *same* ulp; that is already inside the factor 2 above.
 *
 * **The fields inherit the moments' error.** `rho` and `J` enter Gauss,
 * Ampere and the ETD2 transverse update, all of which are `O(N_x log N_x)`
 * linear operations, so the field error is `eta` plus an FFT round-off of
 * `O(log N_x) u`, which is negligible beside it. The tolerance used is
 * `4 eta`, the factor absorbing the handful of linear operations between
 * the deposited current and the field.
 *
 * **The distribution's error after `n` steps.** A field difference `dE`
 * changes the semi-Lagrangian shift by `d(alpha) = (sigma/mu) dE dt / dv`,
 * and the distribution responds with `df ~ |d f / d alpha| d(alpha)`, which
 * for a grid-resolved `f` is at most `max|f|` per cell of shift. Errors add
 * linearly across steps in the worst case, so
 *
 *     tol_f = 8 n eta max(1, alpha_max) max|f| ,
 *
 * with `alpha_max` the largest shift in cells the run actually took. The
 * factor 8 is slack for the three shifts and two depositions inside one
 * Strang step; it is a constant, not a knob turned until the test passed,
 * and the driver prints the measured-to-tolerance ratio so a reader can see
 * how much of it was used.
 *
 * ## What a failure here means
 *
 * Operator parity failing is a kernel bug. Integrated parity failing while
 * operator parity holds is *not necessarily* a bug: it can be the physical
 * amplification of round-off by an instability, and the honest way to tell
 * them apart is to watch the growth with the number of steps, which is why
 * the driver prints the divergence at every sample rather than only at the
 * end.
 *
 * @see device_step_hip.hpp for what runs where
 * @see vlasov_hip_cost.cpp for whether it was worth running there
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "vlasov_hip_parity requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/cli.hpp>
#include <vlasov_maxwell/device_step_hip.hpp>
#include <vlasov_maxwell/diagnostics.hpp>
#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/step.hpp>

namespace {

using vlasov::Ledger;
using vlasov::PhaseSpace;
using vlasov::SimParams;
using vlasov::Species;
using vlasov::Stepper;
using vlasov::VelocityMoments;
using vlasov::hip::DeviceStepper;

/// IEEE double unit round-off, `2^-53`.
constexpr double kUnitRoundoff = 1.1102230246251565e-16;

/// A difference measured between the two paths.
struct Diff {
  double max_abs{0.0};
  double scale{0.0};     ///< the magnitude it should be judged against
  std::size_t n{0};      ///< values compared
  std::size_t bitwise{0}; ///< of those, how many agreed exactly

  [[nodiscard]] double relative() const {
    return max_abs / std::max(scale, vlasov::kTiny);
  }
  [[nodiscard]] double bitwise_fraction() const {
    return n > 0 ? static_cast<double>(bitwise) / static_cast<double>(n) : 1.0;
  }
};

/// Reduce a @ref Diff across the `v_y` ranks so the printed number is global.
[[nodiscard]] Diff reduce_diff(const Diff &d, MPI_Comm comm) {
  Diff out = d;
  double hi[2] = {d.max_abs, d.scale};
  MPI_Allreduce(MPI_IN_PLACE, hi, 2, MPI_DOUBLE, MPI_MAX, comm);
  out.max_abs = hi[0];
  out.scale = hi[1];
  long long counts[2] = {static_cast<long long>(d.n),
                         static_cast<long long>(d.bitwise)};
  MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_LONG_LONG, MPI_SUM, comm);
  out.n = static_cast<std::size_t>(counts[0]);
  out.bitwise = static_cast<std::size_t>(counts[1]);
  return out;
}

/// Compare the owned cells of two distributions living on the same geometry.
[[nodiscard]] Diff compare_fields(const PhaseSpace &ps,
                                  const vlasov::PhaseField &a,
                                  const vlasov::PhaseField &b) {
  Diff d;
  for (int k = 0; k < ps.nvy_local(); ++k) {
    for (int j = 0; j < ps.nvx(); ++j) {
      for (int i = 0; i < ps.nx(); ++i) {
        const double x = a(i, j, k);
        const double y = b(i, j, k);
        d.max_abs = std::max(d.max_abs, std::fabs(x - y));
        d.scale = std::max(d.scale, std::fabs(x));
        ++d.n;
        // Bit patterns, not `==`: this must count -0.0 against +0.0 and must
        // not call two NaNs equal.
        std::uint64_t bx = 0;
        std::uint64_t by = 0;
        std::memcpy(&bx, &x, sizeof bx);
        std::memcpy(&by, &y, sizeof by);
        if (bx == by) ++d.bitwise;
      }
    }
  }
  return reduce_diff(d, ps.comm());
}

/// Compare two replicated 1-D arrays (a field component, a moment profile).
[[nodiscard]] Diff compare_lines(const std::vector<double> &a,
                                 const std::vector<double> &b) {
  Diff d;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    d.max_abs = std::max(d.max_abs, std::fabs(a[i] - b[i]));
    d.scale = std::max(d.scale, std::fabs(a[i]));
    ++d.n;
    std::uint64_t bx = 0;
    std::uint64_t by = 0;
    std::memcpy(&bx, &a[i], sizeof bx);
    std::memcpy(&by, &b[i], sizeof by);
    if (bx == by) ++d.bitwise;
  }
  return d;
}

/// Accumulated pass/fail, so the process can exit non-zero on the first real
/// disagreement without stopping at it.
struct Verdict {
  int failures{0};
  int rank{0};

  void check(const char *what, double measured, double tol, const char *note = "") {
    const bool ok = std::isfinite(measured) && measured <= tol;
    if (!ok) ++failures;
    if (rank == 0) {
      std::printf("    %-34s %12.4e  tol %10.3e  %-4s %s\n", what, measured,
                  tol, ok ? "ok" : "FAIL", note);
    }
  }
  void report(const char *what, double value, const char *unit = "") {
    if (rank == 0) {
      std::printf("    %-34s %12.4e %s\n", what, value, unit);
    }
  }
};

/**
 * @brief A small but physically live phase space, built twice identically.
 *
 * Weibel-like: a bi-Maxwellian with an anisotropy, a seeded `B_z` and a
 * density perturbation. A parity case has to have non-zero fields, or the
 * velocity shifts are all exactly zero, the Lagrange weights come out as a
 * Kronecker delta, and every gather in the test is a memcpy that would agree
 * bitwise no matter how wrong the kernel was.
 */
struct Bench {
  SimParams p;
  std::unique_ptr<PhaseSpace> ps;
  std::unique_ptr<Stepper> st;
  double dt{0.0};
  double amp{1.0e-3};

  Bench(int nx, int nvx, int nvy, int interp, double dt_in) {
    const double twopi = 2.0 * std::acos(-1.0);
    const double vth = 0.05;
    const double vthy = 0.15;
    p.nx = nx;
    p.nvx = nvx;
    p.nvy = nvy;
    p.Lx = twopi / 0.5;
    p.v_max = 8.0 * vthy;
    p.v_thermal = vthy;
    p.interp_order = interp;
    p.electrostatic = false;
    p.species.clear();
    p.species.push_back(Species{"electron", -1.0, 1.0});
    p.validate();

    const int halo = vlasov::required_halo_width(4.0, interp);
    ps = std::make_unique<PhaseSpace>(p, halo, MPI_COMM_WORLD);
    st = std::make_unique<Stepper>(p, *ps);

    const double k = p.k_skin(1);
    ps->initialise(0, [&](double x, double vx, double vy) {
      return vlasov::ics::density_perturbation(x, k, amp) *
             vlasov::ics::bi_maxwellian(vx, vy, vth, vthy);
    });
    st->deposit_all();
    const auto sol =
        vlasov::solve_gauss(st->line, st->sources.rho, p.neutrality_tol);
    st->fields.Ex = sol.Ex;
    for (int i = 0; i < p.nx; ++i) {
      st->fields.Bz[i] = 1.0e-2 * std::cos(k * p.x_of(i));
      st->fields.Ey[i] = 1.0e-3 * std::sin(k * p.x_of(i));
    }
    dt = dt_in > 0.0
             ? dt_in
             : 0.2 * vlasov::step_limit(p, 1.0, 1.0e-2, 1.0e-2, halo);
  }
};

/// Snapshot of a padded brick, for restoring an operator's input.
[[nodiscard]] std::vector<double> snapshot(const vlasov::PhaseField &f) {
  return std::vector<double>(f.data(), f.data() + f.size());
}
void restore(vlasov::PhaseField &f, const std::vector<double> &s) {
  std::copy(s.begin(), s.end(), f.data());
  f.note_host_write();
}

/// `kappa = sum|f| / |sum f|`, the condition number of the moment sum. One
/// number, measured, because every tolerance below is proportional to it.
[[nodiscard]] double measure_kappa(const PhaseSpace &ps,
                                   const vlasov::PhaseField &f) {
  double s = 0.0;
  double a = 0.0;
  for (int k = 0; k < ps.nvy_local(); ++k) {
    for (int j = 0; j < ps.nvx(); ++j) {
      for (int i = 0; i < ps.nx(); ++i) {
        s += f(i, j, k);
        a += std::fabs(f(i, j, k));
      }
    }
  }
  double buf[2] = {s, a};
  MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, ps.comm());
  return buf[1] / std::max(std::fabs(buf[0]), vlasov::kTiny);
}

void print_usage(std::ostream &os, const char *exe) {
  os << "usage: " << exe << " [--key=value ...]\n\n"
     << "CPU-vs-HIP parity for the 1D2V Vlasov step.\n\n"
     << "  --nx=N --nvx=N --nvy=N   phase-space grid          (32, 32, 32)\n"
     << "  --interp=N               Lagrange points/order            (5)\n"
     << "  --steps=N                Strang steps in the integrated test (20)\n"
     << "  --samples=N              divergence reports during them     (5)\n"
     << "  --dt=X                   step; 0 derives it                 (0)\n"
     << "  --device-x=0|1           run phase A on the device too       (1)\n"
     << "  --science=landau         CPU/GPU Landau γ instead of the 20-step\n"
     << "                          operator test (off)\n"
     << "  --quiet=1                pass/fail only\n";
}

/// Landau IC, host Stepper vs DeviceStepper, same host envelope fit on
/// `mode_ex`. The oracle is not the dispersion root: this asks whether the
/// two paths measure the same rate, not whether either is right.
int run_science_landau(int nx, int nvx, int nvy, int interp, double dt_in,
                       bool device_x, int rank, int nproc) {
  const double vth = 0.05;
  const double twopi = 2.0 * std::acos(-1.0);
  const double t_end = 12.0;
  const double amp = 0.01;

  struct LandauRun {
    std::unique_ptr<PhaseSpace> ps;
    std::unique_ptr<Stepper> st;
    double dt{0.0};
  };
  auto make_landau = [&]() {
    SimParams p;
    p.nx = nx;
    p.nvx = nvx;
    p.nvy = nvy;
    p.Lx = twopi * vth / 0.5;
    p.v_max = 8.0 * vth;
    p.v_thermal = vth;
    p.t_end = t_end;
    p.interp_order = interp;
    p.electrostatic = true;
    p.self_consistent = true;
    p.validate();
    const int halo = vlasov::required_halo_width(4.0, interp);
    LandauRun r;
    r.ps = std::make_unique<PhaseSpace>(p, halo, MPI_COMM_WORLD);
    r.st = std::make_unique<Stepper>(r.ps->params(), *r.ps);
    const double k = r.ps->params().k_skin(1);
    r.ps->initialise(0, [&](double x, double vx, double vy) {
      return vlasov::ics::density_perturbation(x, k, amp) *
             vlasov::ics::maxwellian(vx, vy, vth);
    });
    r.st->deposit_all();
    const auto sol = vlasov::solve_gauss(r.st->line, r.st->sources.rho,
                                         r.ps->params().neutrality_tol);
    r.st->fields.Ex = sol.Ex;
    double emax = 0.0;
    for (double v : r.st->fields.Ex) emax = std::fmax(emax, std::fabs(v));
    r.dt = dt_in > 0.0
               ? dt_in
               : r.ps->params().dt_safety *
                     vlasov::step_limit(r.ps->params(), 1.0,
                                        std::fmax(emax, 1.0e-3), 0.1, halo);
    return r;
  };

  auto cpu = make_landau();
  auto gpu = make_landau();
  auto &p = cpu.ps->params();
  auto &ps_cpu = *cpu.ps;
  auto &st_cpu = *cpu.st;
  auto &ps_gpu = *gpu.ps;
  auto &st_gpu = *gpu.st;
  const double dt = cpu.dt;
  const int n_steps = std::max(1, static_cast<int>(std::llround(t_end / dt)));
  const double k = p.k_skin(1);

  DeviceStepper ds(st_gpu, ps_gpu, device_x);
  ds.upload_all();
  st_cpu.work.measure_mass = false;
  st_gpu.work.measure_mass = false;

  std::vector<double> ts, mex_cpu, mex_gpu;
  auto sample = [&](int step, double t) {
    ds.download_all();
    const Ledger lc = vlasov::make_ledger(p, st_cpu.line, st_cpu.moments,
                                          st_cpu.sources, st_cpu.fields,
                                          st_cpu.gauss, t, step, 1);
    const Ledger lg = vlasov::make_ledger(p, st_gpu.line, st_gpu.moments,
                                          st_gpu.sources, st_gpu.fields,
                                          st_gpu.gauss, t, step, 1);
    ts.push_back(t);
    mex_cpu.push_back(lc.mode_ex);
    mex_gpu.push_back(lg.mode_ex);
  };
  sample(0, 0.0);
  for (int step = 1; step <= n_steps; ++step) {
    st_cpu.advance(dt);
    ds.advance(dt);
    sample(step, static_cast<double>(step) * dt);
  }

  const double t0 = 3.0;
  const double t1 = t_end;
  vlasov::require_fit_before_recurrence(t1, k, p.dvx());
  const double g_cpu = vlasov::fit_envelope_rate(ts, mex_cpu, t0, t1);
  const double g_hip = vlasov::fit_envelope_rate(ts, mex_gpu, t0, t1);
  const double denom = std::max(std::fabs(g_cpu), 1.0e-16);
  const double rel = std::fabs(g_hip - g_cpu) / denom;
  if (rank == 0) {
    std::printf("vlasov_hip_parity science=landau: %d x %d x %d, %d ranks, "
                "%d steps, dt %.6g\n",
                nx, nvx, nvy, nproc, n_steps, dt);
    std::printf("  gamma_cpu = %.10g\n", g_cpu);
    std::printf("  gamma_hip = %.10g\n", g_hip);
    std::printf("  |g_hip-g_cpu|/max(|g_cpu|,eps) = %.3e  (tol 1e-6)\n", rel);
  }
  const bool ok = std::isfinite(g_cpu) && std::isfinite(g_hip) && rel < 1.0e-6;
  if (rank == 0) {
    std::printf("\n  VLASOV_HIP_SCIENCE_RATE %s\n", ok ? "PASS" : "FAIL");
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(int argc, char **argv, int rank, int nproc) {
  vlasov::Options opt(argc, argv);
  if (opt.help()) {
    if (rank == 0) print_usage(std::cout, argv[0]);
    return EXIT_SUCCESS;
  }
  const int nx = opt.integer("nx", 32);
  const int nvx = opt.integer("nvx", 32);
  const int nvy = opt.integer("nvy", 32);
  const int interp = opt.integer("interp", 5);
  const int steps = opt.integer("steps", 20);
  const int samples = opt.integer("samples", 5);
  const double dt_in = opt.real("dt", 0.0);
  const bool device_x = opt.flag("device-x", true);
  const bool quiet = opt.flag("quiet", false);
  const std::string science = opt.text("science", "");
  opt.require_all_consumed();

  if (science == "landau") {
    vlasov::hip::bind_local_device(rank);
    return run_science_landau(nx, nvx, nvy, interp, dt_in, device_x, rank,
                              nproc);
  }
  if (!science.empty()) {
    throw std::invalid_argument(
        "--science must be omitted or 'landau', got '" + science + "'");
  }

  vlasov::hip::bind_local_device(rank);

  Verdict v;
  v.rank = rank;
  if (rank == 0) {
    std::printf("vlasov_hip_parity: %d x %d x %d, interp %d, %d ranks, "
                "device %s\n",
                nx, nvx, nvy, interp, nproc, vlasov::hip::device_name());
    std::printf("  phase A (spectral x shift) runs on the %s\n",
                device_x ? "device (hipFFT)" : "host (FFTW)");
  }

  // ---- tolerances, from the arithmetic ---------------------------------
  Bench ref(nx, nvx, nvy, interp, dt_in);
  const double kappa = measure_kappa(*ref.ps, ref.ps->f(0));
  const double nv = static_cast<double>(nvx) * static_cast<double>(nvy);
  const double eta = 2.0 * nv * kUnitRoundoff * kappa;
  // Round-off of one spectral shift, relative to max|f|. Higham's bound for
  // a radix-2 FFT is ||e||_2 <= c log2(N) u ||f||_2 with c a small constant;
  // a shift is a forward transform, a multiply and an inverse, so twice
  // that, and the difference between two *implementations* of it is bounded
  // by the sum of their errors. Converting to the max norm with
  // ||f||_2 <= sqrt(N) ||f||_inf and taking c = 4 gives the constant below.
  // This term is zero unless phase A runs on the device, because otherwise
  // both paths call the identical FFTW plan on identical data.
  const double log2nx = std::log2(static_cast<double>(nx));
  const double eps_fft =
      device_x ? 8.0 * log2nx * kUnitRoundoff * std::sqrt(static_cast<double>(nx))
               : 0.0;
  if (rank == 0) {
    std::printf("  dt %.6g, %d steps\n", ref.dt, steps);
    std::printf("  N_v = %.0f, kappa = sum|f|/|sum f| = %.6f\n", nv, kappa);
    std::printf("  eta = 2 N_v u kappa = %.3e   (the moment tolerance)\n", eta);
    std::printf("  eps_fft = 8 log2(N_x) u sqrt(N_x) = %.3e   (one x shift)\n",
                eps_fft);
  }

  // =====================================================================
  // 1. Operator parity
  // =====================================================================
  if (rank == 0) std::printf("\n  [1] operator parity, one call on identical input\n");
  {
    auto &ps = *ref.ps;
    auto &st = *ref.st;
    const auto Bz = st.effective_bz();
    const double qm = ref.p.species[0].qm();
    const double h = 0.5 * ref.dt;
    const auto f0 = snapshot(ps.f(0));

    DeviceStepper ds(st, ps, device_x);

    // -- step A ---------------------------------------------------------
    // Only meaningful when the device has its own transform; otherwise both
    // paths are the same FFTW call and the comparison is a tautology.
    if (device_x) {
      restore(ps.f(0), f0);
      vlasov::advect_x(ps, ps.f(0), h, st.xplan, st.work);
      const auto host_x = snapshot(ps.f(0));

      restore(ps.f(0), f0);
      ds.upload_all();
      ds.device_advect_x(ds.brick(0), h);
      ds.download_all();
      vlasov::PhaseField tmp = ps.make_field();
      std::copy(ps.f(0).data(), ps.f(0).data() + ps.f(0).size(), tmp.data());
      tmp.note_host_write();
      restore(ps.f(0), host_x);
      const Diff d = compare_fields(ps, ps.f(0), tmp);
      v.check("A advect_x   max|dev-host|", d.max_abs, eps_fft * d.scale,
              "(rocFFT vs FFTW)");
      v.report("A advect_x   bitwise fraction", d.bitwise_fraction());
    }

    // -- step B ---------------------------------------------------------
    restore(ps.f(0), f0);
    vlasov::advect_vx(ps, ps.f(0), qm, h, st.fields.Ex, Bz, interp, st.work);
    const auto host_vx = snapshot(ps.f(0));

    restore(ps.f(0), f0);
    ds.upload_all();
    ds.device_advect_vx(ds.brick(0), qm, h, st.fields.Ex, Bz);
    ds.download_all();
    const auto dev_vx = snapshot(ps.f(0));
    {
      // Compare through the field's own accessor so only owned cells count.
      restore(ps.f(0), host_vx);
      vlasov::PhaseField tmp = ps.make_field();
      std::copy(dev_vx.begin(), dev_vx.end(), tmp.data());
      tmp.note_host_write();
      const Diff d = compare_fields(ps, ps.f(0), tmp);
      // p products, each of which the host may or may not have fused.
      const double tol = static_cast<double>(interp) * kUnitRoundoff * d.scale;
      v.check("B advect_vx  max|dev-host|", d.max_abs, tol,
              d.bitwise_fraction() >= 1.0 ? "(bitwise)" : "(not bitwise)");
      v.report("B advect_vx  bitwise fraction", d.bitwise_fraction());
    }

    // -- step C ---------------------------------------------------------
    restore(ps.f(0), f0);
    vlasov::advect_vy(ps, ps.f(0), qm, ref.dt, st.fields.Ey, Bz, interp, st.work);
    const auto host_vy = snapshot(ps.f(0));

    restore(ps.f(0), f0);
    ds.upload_all();
    ds.device_advect_vy(ds.brick(0), qm, ref.dt, st.fields.Ey, Bz);
    ds.download_all();
    const auto dev_vy = snapshot(ps.f(0));
    {
      restore(ps.f(0), host_vy);
      vlasov::PhaseField tmp = ps.make_field();
      std::copy(dev_vy.begin(), dev_vy.end(), tmp.data());
      tmp.note_host_write();
      const Diff d = compare_fields(ps, ps.f(0), tmp);
      const double tol = static_cast<double>(interp) * kUnitRoundoff * d.scale;
      v.check("C advect_vy  max|dev-host|", d.max_abs, tol,
              d.bitwise_fraction() >= 1.0 ? "(bitwise)" : "(not bitwise)");
      v.report("C advect_vy  bitwise fraction", d.bitwise_fraction());
    }

    // -- phase D --------------------------------------------------------
    restore(ps.f(0), f0);
    ds.upload_all();
    vlasov::ReductionOptions ropt;
    ropt.v_thermal = ref.p.v_thermal;
    ropt.comm = ps.comm();
    const VelocityMoments mh =
        vlasov::reduce_velocity(ref.p, vlasov::view_of(ps, ps.f(0)), ropt);
    const VelocityMoments md = ds.reduce_device(ds.brick(0), ropt);
    {
      const Diff dn = compare_lines(mh.n, md.n);
      const Diff dx = compare_lines(mh.flux_x, md.flux_x);
      const Diff dy = compare_lines(mh.flux_y, md.flux_y);
      const Diff de = compare_lines(mh.v2, md.v2);
      // A round-off bound is a bound on the *absolute* error of a sum, and
      // it is governed by the sum of the magnitudes of the terms, not by the
      // magnitude of the answer. For the density that distinction does not
      // arise -- `f` is non-negative, so the two are the same number. For
      // the fluxes it is the whole story: `sum v_x f` over a distribution
      // symmetric in `v_x` cancels to zero to fifteen digits, so judging it
      // against its own value asks the two paths to agree to 1e-30 and is a
      // test of nothing. The honest scale is `sum |v_x f| <= v_max sum f`,
      // i.e. `v_max` times the density profile, and that is what is used.
      const double n_scale = dn.scale;
      const double flux_scale = ref.p.v_max * n_scale;
      const double v2_scale = ref.p.v_max * ref.p.v_max * n_scale;
      v.check("D rho profile  max|dev-host|", dn.max_abs, eta * n_scale);
      v.check("D J_x profile  max|dev-host|", dx.max_abs, eta * flux_scale,
              "(scale v_max*max n)");
      v.check("D J_y profile  max|dev-host|", dy.max_abs, eta * flux_scale,
              "(scale v_max*max n)");
      v.check("D v2 profile   max|dev-host|", de.max_abs, eta * v2_scale);
      auto scalar = [&](const char *what, double a, double b) {
        v.check(what, std::fabs(a - b), eta * std::fabs(a));
      };
      scalar("D number", mh.number, md.number);
      scalar("D l1", mh.l1, md.l1);
      scalar("D l2", mh.l2, md.l2);
      scalar("D entropy", mh.entropy, md.entropy);
      // Extrema are order-independent: these must be bitwise.
      v.check("D f_min (must be exact)", std::fabs(mh.f_min - md.f_min), 0.0);
      v.check("D f_max (must be exact)", std::fabs(mh.f_max - md.f_max), 0.0);
      v.check("D f_face_max (must be exact)",
              std::fabs(mh.f_face_max - md.f_face_max), 0.0);
      v.check("D boundary_fraction", std::fabs(mh.boundary_fraction -
                                               md.boundary_fraction),
              eta * std::max(1.0, std::fabs(mh.boundary_fraction)));
      v.report("D measured/bound, rho profile",
               dn.max_abs / std::max(eta * dn.scale, vlasov::kTiny));
    }
    restore(ps.f(0), f0);
  }

  // =====================================================================
  // 2. Integrated parity
  // =====================================================================
  if (rank == 0) {
    std::printf("\n  [2] integrated parity, %d Strang steps from one state\n",
                steps);
  }
  {
    Bench a(nx, nvx, nvy, interp, dt_in);   // host reference
    Bench b(nx, nvx, nvy, interp, dt_in);   // device path
    a.st->work.measure_mass = false;
    b.st->work.measure_mass = false;
    DeviceStepper ds(*b.st, *b.ps, device_x);
    ds.upload_all();

    // The largest shift in cells the run takes, for the tolerance below.
    double alpha_max = 0.0;
    const int every = std::max(1, steps / std::max(1, samples));
    for (int s = 1; s <= steps; ++s) {
      a.st->advance(a.dt);
      ds.advance(b.dt);
      alpha_max = std::max(
          alpha_max,
          vlasov::max_vy_shift_cells(b.p, b.p.species[0].qm(), b.dt,
                                     b.st->fields.Ey, b.st->effective_bz()));
      if (s % every == 0 || s == steps) {
        ds.download_all();
        const Diff df = compare_fields(*a.ps, a.ps->f(0), b.ps->f(0));
        const Diff dex = compare_lines(a.st->fields.Ex, b.st->fields.Ex);
        const Diff dbz = compare_lines(a.st->fields.Bz, b.st->fields.Bz);
        if (rank == 0 && !quiet) {
          std::printf("    step %4d   f %.3e (rel %.3e)   E_x %.3e   "
                      "B_z %.3e   bitwise f %.4f\n",
                      s, df.max_abs, df.relative(), dex.max_abs, dbz.max_abs,
                      df.bitwise_fraction());
        }
      }
    }
    ds.download_all();

    // Two independent per-step sources, added:
    //  - the moments' round-off, which reaches `f` through the field and
    //    hence through the shift, with `8` covering the three shifts and two
    //    depositions a Strang step contains;
    //  - the two spectral x shifts, when they run on the device.
    // Both accumulate at worst linearly in the number of steps.
    const double n_steps = static_cast<double>(std::max(1, steps));
    const double tol_f_rel =
        n_steps * (8.0 * eta * std::max(1.0, alpha_max) + 2.0 * eps_fft);
    const double tol_field_rel = n_steps * (4.0 * eta + 2.0 * eps_fft);

    const Diff df = compare_fields(*a.ps, a.ps->f(0), b.ps->f(0));
    v.report("alpha_max (v_y cells)", alpha_max);
    v.check("f        max|dev-host| / max|f|", df.relative(), tol_f_rel);

    const Diff drho = compare_lines(a.st->sources.rho, b.st->sources.rho);
    const Diff djx = compare_lines(a.st->sources.Jx, b.st->sources.Jx);
    const Diff djy = compare_lines(a.st->sources.Jy, b.st->sources.Jy);
    v.check("rho      max|dev-host| / max|rho|", drho.relative(), tol_field_rel);
    v.check("J_x      max|dev-host| / max|J_x|", djx.relative(), tol_field_rel);
    v.check("J_y      max|dev-host| / max|J_y|", djy.relative(), tol_field_rel);

    const Diff dex = compare_lines(a.st->fields.Ex, b.st->fields.Ex);
    const Diff dey = compare_lines(a.st->fields.Ey, b.st->fields.Ey);
    const Diff dbz = compare_lines(a.st->fields.Bz, b.st->fields.Bz);
    v.check("E_x      max|dev-host| / max|E_x|", dex.relative(), tol_field_rel);
    v.check("E_y      max|dev-host| / max|E_y|", dey.relative(), tol_field_rel);
    v.check("B_z      max|dev-host| / max|B_z|", dbz.relative(), tol_field_rel);

    // ---- the ledger ----------------------------------------------------
    const Ledger la = vlasov::make_ledger(a.p, a.st->line, a.st->moments,
                                          a.st->sources, a.st->fields,
                                          a.st->gauss, 0.0, steps, 1);
    const Ledger lb = vlasov::make_ledger(b.p, b.st->line, b.st->moments,
                                          b.st->sources, b.st->fields,
                                          b.st->gauss, 0.0, steps, 1);
    auto ledger_check = [&](const char *what, double x, double y) {
      v.check(what, std::fabs(x - y),
              tol_field_rel * std::max(std::fabs(x), vlasov::kTiny));
    };
    ledger_check("ledger number", la.number, lb.number);
    ledger_check("ledger kinetic_energy", la.kinetic_energy, lb.kinetic_energy);
    // Same cancellation as the flux profiles: total momentum is zero in
    // every benchmark here by construction, so it is judged against
    // `mu v_max N`, the sum of the magnitudes its terms could have had.
    const double mom_scale =
        a.p.species[0].mu * a.p.v_max * std::fabs(la.number);
    v.check("ledger momentum_x", std::fabs(la.momentum_x - lb.momentum_x),
            tol_field_rel * mom_scale, "(scale mu*v_max*N)");
    v.check("ledger momentum_y", std::fabs(la.momentum_y - lb.momentum_y),
            tol_field_rel * mom_scale, "(scale mu*v_max*N)");
    ledger_check("ledger l1", la.l1, lb.l1);
    ledger_check("ledger l2", la.l2, lb.l2);
    ledger_check("ledger entropy", la.entropy, lb.entropy);
    ledger_check("ledger energy_ex", la.energy_ex, lb.energy_ex);
    ledger_check("ledger energy_em", la.energy_em, lb.energy_em);
    ledger_check("ledger total_energy", la.total_energy, lb.total_energy);
    // The Gauss residual is the sharpest diagnostic in the application and
    // is a ratio of two small numbers, so it is compared in absolute terms
    // against the field scale rather than relatively against itself.
    v.check("ledger gauss_residual (abs)",
            std::fabs(la.gauss_residual - lb.gauss_residual),
            std::max(tol_field_rel, 1.0e-12));
    v.report("ledger gauss_residual host", la.gauss_residual);
    v.report("ledger gauss_residual device", lb.gauss_residual);
    v.report("f divergence / tolerance", df.relative() /
                                             std::max(tol_f_rel, vlasov::kTiny));
    v.report("peak v_y halo used (host)",
             static_cast<double>(a.st->peak_halo_used));
    v.report("peak v_y halo used (device)",
             static_cast<double>(ds.peak_halo_used()));
  }

  int failures = v.failures;
  MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("\n  VLASOV_HIP_PARITY %s (%d failing check%s)\n",
                failures == 0 ? "PASS" : "FAIL", failures,
                failures == 1 ? "" : "s");
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  int status = 0;
  try {
    status = run(argc, argv, rank, nproc);
  } catch (const std::exception &e) {
    std::cerr << "vlasov_hip_parity[" << rank << "]: " << e.what() << "\n";
    status = 2;
  }
  MPI_Finalize();
  return status;
}
