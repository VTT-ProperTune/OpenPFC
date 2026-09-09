// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_kawahara.cpp
 * @brief Catch2 tests for the Kawahara spectral app.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <numbers>
#include <tuple>

#include <nlohmann/json.hpp>

#include <kawahara/capillary_gravity_mapping.hpp>
#include <kawahara/kawahara_physics.hpp>
#include <kawahara/kawahara_session.hpp>
#include <kawahara/wave_packet_diagnostics.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

double cosine_phase(const pfc::data::Field<double> &u, int nx) {
  const auto n = u.local_size();
  const auto sp = u.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double twopi = 2.0 * std::numbers::pi;
  double c = 0.0;
  double s = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = u.coords(i, j, k);
        const double th = twopi * static_cast<double>(nx) * x[0] / Lx;
        c += u(i, j, k) * std::cos(th);
        s += u(i, j, k) * std::sin(th);
      }
    }
  }
  return std::atan2(s, c);
}

double cosine_amplitude(const pfc::data::Field<double> &u, int nx) {
  const auto n = u.local_size();
  const auto sp = u.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double twopi = 2.0 * std::numbers::pi;
  double num = 0.0;
  double den = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = u.coords(i, j, k);
        const double w = std::cos(twopi * static_cast<double>(nx) * x[0] / Lx);
        num += u(i, j, k) * w;
        den += w * w;
      }
    }
  }
  return (den > 0.0) ? num / den : 0.0;
}

double mean_u(const pfc::data::Field<double> &u) {
  double sum = 0.0;
  std::size_t count = 0;
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        sum += u(i, j, k);
        ++count;
      }
    }
  }
  return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

} // namespace

TEST_CASE("Kawahara omega(k) = beta k^3 + gamma k^5 and L = -i omega",
          "[kawahara][physics][symbol]") {
  kawahara::KawaharaPhysics<> phys;
  REQUIRE_THAT(phys.params.alpha, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(phys.params.beta, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(phys.params.gamma, WithinAbs(-1.0, 1e-15));
  const double k = 0.5;
  const double w = kawahara::omega_k(k, phys.params);
  REQUIRE_THAT(w, WithinAbs(phys.params.beta * k * k * k +
                                phys.params.gamma * k * k * k * k * k,
                            1e-15));
  const auto L = phys.linear_symbol(k, 0.0, 0.0);
  REQUIRE_THAT(L.real(), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(L.imag(), WithinAbs(-w, 1e-15));
  const auto M = phys.nonlinear_symbol(k, 0.0, 0.0);
  REQUIRE_THAT(M.real(), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(M.imag(), WithinAbs(-0.5 * phys.params.alpha * k, 1e-15));
}

TEST_CASE("Kawahara linear cosine tracks omega(k) without damping",
          "[kawahara][spectral][phase]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 64;
  constexpr int nx = 16; // k = nx/32 = 0.5 when Lx = 64 pi, dx = pi
  constexpr double amp0 = 0.08;
  constexpr double dt = 0.02;
  constexpr int n_steps = 10;
  const double dx = std::numbers::pi;
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  // Independent manufactured solutions of u_t - beta*u_xxx + gamma*u_xxxxx=0.
  // For cos(k*x-w*t), the third and fifth derivatives are k^3*sin and
  // -k^5*sin. Do not obtain the expected phase from omega_k().
  double beta = 1.0, gamma = -1.0;
  SECTION("third derivative only") { gamma = 0.0; }
  SECTION("fifth derivative only") { beta = 0.0; }
  SECTION("competing derivatives") {}
  json params{{"alpha", 0.0}, {"beta", beta}, {"gamma", gamma}};
  auto phys = kawahara::KawaharaPhysics<>::from_json(params, domain,
                                                     stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &u = state.get_field<double>("u");
  const double Lx = static_cast<double>(N) * dx;
  const double k = 2.0 * std::numbers::pi * static_cast<double>(nx) / Lx;
  REQUIRE_THAT(k, WithinAbs(0.5, 1e-12));
  u.apply([&](double x, double, double) { return amp0 * std::cos(k * x); });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "u";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<kawahara::KawaharaPhysics<>> sys(phys, stack.fft(),
                                                               state, dt, opt);
  const double omega = beta * 0.125 + gamma * 0.03125; // k=1/2
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE_THAT(cosine_amplitude(u, nx), WithinRel(amp0, 1.0e-3));
  REQUIRE_THAT(cosine_phase(u, nx), WithinAbs(omega * t, 1.0e-5));
  u.for_each_owned([&](int i, int j, int z) {
    const auto x = u.coords(i, j, z);
    REQUIRE_THAT(u(i, j, z),
                 WithinAbs(amp0 * std::cos(k * x[0] - omega * t), 1e-12));
  });
  REQUIRE_THAT(mean_u(u), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Kawahara k^3 and k^5 reverse the phase velocity across |k|=1",
          "[kawahara][spectral][k3k5]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 128;
  constexpr double dx = 0.5 * std::numbers::pi;
  constexpr double dt = 0.01;
  constexpr int n_steps = 8;
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  json params{{"alpha", 0.0}, {"beta", 1.0}, {"gamma", -1.0}};
  auto phys = kawahara::KawaharaPhysics<>::from_json(params, domain,
                                                     stack.fft().get_inbox_bounds());
  const double Lx = static_cast<double>(N) * dx;
  auto run_mode = [&](int nx) {
    pfc::SimulationState state;
    phys.declare_fields(state);
    auto &u = state.get_field<double>("u");
    const double k = 2.0 * std::numbers::pi * static_cast<double>(nx) / Lx;
    u.apply([&](double x, double, double) { return 0.05 * std::cos(k * x); });
    pfc::sim::SpectralETDOptions opt;
    opt.psi_name = "u";
    opt.dealias = true;
    pfc::sim::SpectralETDSystem<kawahara::KawaharaPhysics<>> sys(phys, stack.fft(),
                                                                 state, dt, opt);
    double t = 0.0;
    for (int step = 0; step < n_steps; ++step) {
      t = sys.step(t);
    }
    const double omega = kawahara::omega_k(k, phys.params);
    return std::tuple<double, double, double, double>{k, omega, cosine_phase(u, nx),
                                                      t};
  };

  const auto [k_lo, w_lo, ph_lo, t_lo] = run_mode(16); // k = 0.5, c_p > 0
  const auto [k_hi, w_hi, ph_hi, t_hi] = run_mode(48); // k = 1.5, c_p < 0
  REQUIRE(w_lo / k_lo > 0.0);
  REQUIRE(w_hi / k_hi < 0.0);
  REQUIRE_THAT(ph_lo, WithinAbs(w_lo * t_lo, 1.0e-4));
  REQUIRE_THAT(ph_hi, WithinAbs(w_hi * t_hi, 1.0e-4));
}

TEST_CASE("Kawahara nonlinear pulse conserves mean u",
          "[kawahara][spectral][volume]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 64;
  constexpr double dx = std::numbers::pi;
  constexpr double dt = 0.02;
  constexpr int n_steps = 8;
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = kawahara::KawaharaPhysics<>::from_json(json::object(), domain,
                                                     stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &u = state.get_field<double>("u");
  const double Lx = static_cast<double>(N) * dx;
  const double x0 = 0.5 * Lx;
  const double sig = 4.0 * dx;
  u.apply([&](double x, double, double) {
    const double d = x - x0;
    return 0.3 * std::exp(-0.5 * d * d / (sig * sig));
  });
  const double mean0 = mean_u(u);
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "u";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<kawahara::KawaharaPhysics<>> sys(phys, stack.fft(),
                                                               state, dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE_THAT(mean_u(u), WithinAbs(mean0, 1e-10));
}

TEST_CASE("KawaharaSession runs a short JSON case", "[kawahara][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  json settings = {
      {"model",
       {{"name", "kawahara"},
        {"params", {{"alpha", 1.0}, {"beta", 1.0}, {"gamma", -1.0}}}}},
      {"domain",
       {{"Lx", 32},
        {"Ly", 1},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.05}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "u"},
         {"type", "cosine_mode"},
         {"u0", 0.0},
         {"amplitude", 0.05},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
  kawahara::register_catalog();
  kawahara::KawaharaSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.1).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Science-case tests (`#119`): documented capillary-gravity mapping, wave
// -packet group/phase-velocity vs the dispersion relation, and the fifth
// -order term's effect on a nonlinear pulse. These are additional to, and do
// not replace, the arbitrary-coefficient verification tests above.
// ---------------------------------------------------------------------------

namespace {

/// d(omega)/dk for omega(k) = beta*k^3 + gamma*k^5 (independent derivative of
/// kawahara::omega_k, not obtained from it).
double domega_dk(double k, double beta, double gamma) {
  return 3.0 * beta * k * k + 5.0 * gamma * k * k * k * k;
}

/// Wrap an angle into (-pi, pi].
double wrap_angle(double theta) { return std::atan2(std::sin(theta), std::cos(theta)); }

} // namespace

TEST_CASE("Kawahara capillary-gravity mapping: alpha/beta/gamma and crossover",
          "[kawahara][mapping]") {
  using kawahara::CapillaryGravityRegime;
  const CapillaryGravityRegime regime{1.0, 1.0, 0.30};
  REQUIRE_THAT(kawahara::c0_of(regime), WithinAbs(1.0, 1e-14));
  REQUIRE_THAT(kawahara::alpha_of(regime), WithinAbs(1.5, 1e-14));
  REQUIRE_THAT(kawahara::beta_of(regime), WithinAbs(-1.0 / 60.0, 1e-14));
  REQUIRE_THAT(kawahara::gamma_of(regime), WithinAbs(1.0 / 90.0, 1e-14));
  REQUIRE_THAT(kawahara::crossover_k(regime), WithinAbs(std::sqrt(1.5), 1e-12));

  // gamma > 0 always (this mapping); beta changes sign at the critical Bond
  // number tau=1/3, which is exactly the point this app's crossover formula
  // (k_c = sqrt(-beta/gamma)) requires.
  REQUIRE(kawahara::gamma_of(regime) > 0.0);
  REQUIRE(kawahara::beta_of(regime) < 0.0); // tau=0.30 < 1/3
  const CapillaryGravityRegime critical{1.0, 1.0, 1.0 / 3.0};
  REQUIRE_THAT(kawahara::beta_of(critical), WithinAbs(0.0, 1e-14));
  const CapillaryGravityRegime supercritical{1.0, 1.0, 0.40};
  REQUIRE(kawahara::beta_of(supercritical) > 0.0); // tau=0.40 > 1/3: no real crossover
}

TEST_CASE("Kawahara wave-packet diagnostics recover a known static envelope",
          "[kawahara][diagnostics]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral diagnostics");
  }
  constexpr int N = 512;
  constexpr double dx = 0.5;
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  pfc::data::Field<double> u(domain, stack.fft().get_inbox_bounds(), 0);
  const double amp = 0.2, sigma = 12.0, k0 = 0.8, x0 = 100.0;
  u.apply([&](double x, double, double) {
    const double d = x - x0;
    return amp * std::exp(-d * d / (2.0 * sigma * sigma)) * std::cos(k0 * d);
  });
  kawahara::WavePacketDiagnostics diag(domain, stack.fft(), MPI_COMM_WORLD, k0);
  const auto s = diag.sample(u);
  REQUIRE_THAT(s.centroid, WithinAbs(x0, 0.5));
  REQUIRE_THAT(std::sqrt(s.width2), WithinRel(sigma, 0.1));
  // mode_amplitude is the discrete projection onto exactly k0, i.e. (for a
  // narrowband envelope) the envelope's own DC Fourier content divided by Lx,
  // *not* the envelope's peak amplitude -- amp*sigma*sqrt(2*pi)/Lx here.
  const double Lx = static_cast<double>(N) * dx;
  const double expected_mode_amplitude = amp * sigma * std::sqrt(2.0 * std::numbers::pi) / Lx;
  REQUIRE_THAT(s.mode_amplitude, WithinRel(expected_mode_amplitude, 0.05));
  REQUIRE(s.edge_fraction < 1e-6);
}

TEST_CASE("Kawahara wave packet: group/phase velocity vs the dispersion "
          "relation across the crossover",
          "[kawahara][wavepacket][science]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  using kawahara::CapillaryGravityRegime;
  const CapillaryGravityRegime regime{1.0, 1.0, 0.30};
  const double beta = kawahara::beta_of(regime);
  const double gamma = kawahara::gamma_of(regime);
  const double kc = kawahara::crossover_k(regime);

  constexpr int N = 512;
  constexpr double dx = 0.5; // Lx = 256
  constexpr double dt = 0.5;
  constexpr int n_steps = 160; // T = 80
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  const double Lx = static_cast<double>(N) * dx;

  int nx = 0;
  double v_g_tol_rel = 0.0, v_g_tol_abs = 0.0;
  bool expect_cp_negative = false;
  SECTION("below crossover (k0 < kc)") {
    nx = 29;
    v_g_tol_abs = 5.0e-3;
    v_g_tol_rel = 1.0; // v_g is tiny here; the absolute tolerance governs
    expect_cp_negative = true;
  }
  SECTION("above crossover (k0 > kc)") {
    nx = 81;
    v_g_tol_abs = 0.0;
    v_g_tol_rel = 0.1;
    expect_cp_negative = false;
  }
  const double k0 = 2.0 * std::numbers::pi * static_cast<double>(nx) / Lx;
  if (nx == 29) {
    REQUIRE(k0 < kc);
  } else {
    REQUIRE(k0 > kc);
  }
  json params{{"alpha", 0.0}, {"beta", beta}, {"gamma", gamma}};
  auto phys = kawahara::KawaharaPhysics<>::from_json(params, domain,
                                                     stack.fft().get_inbox_bounds());
  const double omega = kawahara::omega_k(k0, phys.params);
  const double v_g_analytic = domega_dk(k0, beta, gamma);
  const double c_p_analytic = omega / k0;
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &u = state.get_field<double>("u");
  const double amp = 0.05, sigma = 10.0, x0 = 80.0;
  u.apply([&](double x, double, double) {
    const double d = x - x0;
    return amp * std::exp(-d * d / (2.0 * sigma * sigma)) * std::cos(k0 * d);
  });

  kawahara::WavePacketDiagnostics diag(domain, stack.fft(), MPI_COMM_WORLD, k0);
  const auto sample0 = diag.sample(u);
  REQUIRE(sample0.edge_fraction < 1e-2); // packet resolved away from the boundary

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "u";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<kawahara::KawaharaPhysics<>> sys(phys, stack.fft(), state,
                                                               dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  const auto sampleT = diag.sample(u);

  // Domain large enough that the packet has not reached the periodic
  // boundary over the reported interval: checked automatically here, not
  // just asserted in prose.
  REQUIRE(sampleT.edge_fraction < 1.0e-2);

  const double v_g_measured = (sampleT.centroid - sample0.centroid) / t;
  const double v_g_diff = std::abs(v_g_measured - v_g_analytic);
  const double v_g_ok = v_g_diff <= v_g_tol_abs ||
                        v_g_diff <= v_g_tol_rel * std::abs(v_g_analytic);
  INFO("k0=" << k0 << " v_g_measured=" << v_g_measured
             << " v_g_analytic=" << v_g_analytic << " diff=" << v_g_diff);
  REQUIRE(v_g_ok);

  // Phase velocity: the discrete Fourier coefficient at exactly k0 evolves
  // *exactly* as exp(-i*omega(k0)*t) under the linear (alpha=0) ETD step,
  // independent of the packet's envelope/bandwidth, so this comparison is
  // tight rather than an envelope-based approximation. mode_phase =
  // atan2(s,c) with s=sum(u sin(kx)), c=sum(u cos(kx)) is the *negative* of
  // the standard exp(ikx)-Fourier-coefficient argument (for u=A*cos(kx+phi),
  // atan2(s,c) = -phi), and phi(t) = phi(0) - omega*t, so mode_phase(t) =
  // mode_phase(0) + omega*t: the observed sign here (verified against the
  // measured run) is +omega*t, not -omega*t.
  const double expected_dtheta = wrap_angle(omega * t);
  const double measured_dtheta = wrap_angle(sampleT.mode_phase - sample0.mode_phase);
  REQUIRE_THAT(measured_dtheta, WithinAbs(expected_dtheta, 1.0e-6));

  // The crossover is where the phase velocity c_p = omega/k changes sign
  // (README's own definition); confirm the two chosen k0 land on the sides
  // this test claims.
  if (expect_cp_negative) {
    REQUIRE(c_p_analytic < 0.0);
  } else {
    REQUIRE(c_p_analytic > 0.0);
  }
}

TEST_CASE("Kawahara nonlinear pulse: fifth-order term changes the trailing "
          "radiation vs a third-order-only control",
          "[kawahara][pulse][science]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  using kawahara::CapillaryGravityRegime;
  const CapillaryGravityRegime regime{1.0, 1.0, 0.30};
  const double alpha = kawahara::alpha_of(regime);
  const double beta = kawahara::beta_of(regime);
  const double gamma = kawahara::gamma_of(regime);

  constexpr int N = 512;
  constexpr double dx = 0.25; // Lx = 128
  constexpr double dt = 0.005;
  // T=40 (n_steps=8000, as originally chosen here) sits too close to this
  // control's own wave-breaking time to be a reliable "control": for
  // u_t+alpha*u*u_x=0 (beta -> 0), a Gaussian bump breaks at
  // t_break = sigma / (0.6065 * alpha * amp) (the inviscid-Burgers
  // characteristic-crossing time, 0.6065=exp(-1/2) locating the steepest
  // slope of a Gaussian). With sigma=6, alpha=alpha_of(regime)=1.5,
  // amp=0.15 below, t_break ~= 44. beta here is weak (tau=0.30 is close to
  // the critical 1/3), so it barely delays that estimate -- measured on
  // this build, the gamma=0 (third-order-only) run is flat to 5 significant
  // digits out to t~=27 and then starts an accelerating, resolution-
  // independent (checked at both N=512 and N=1024) amplitude growth that is
  // the numerical approach to that same breaking singularity. Right at/after
  // a finite-time singularity, the exact step at which floating-point noise
  // tips the run into instability is platform-sensitive (different
  // compiler/libm/FFT rounding), which is why this test passed on LUMI/Cray
  // but produced a NaN mean_drift_third on ubuntu-24.04/gcc-13 CI. Running
  // only to T=20 (n_steps=4000) stays inside the flat, pre-breaking regime
  // with a >=1.35x margin below the observed t~=27 departure from flat and
  // a >=2x margin below the t_break~=44 estimate, on both tested platforms.
  constexpr int n_steps = 4000; // T = 20, safely below t_break (see above)
  const auto domain = pfc::domain::create(pfc::GridSize({N, 1, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  const double amp = 0.15, sigma = 6.0, x0 = 64.0;

  auto run = [&](double gamma_run) {
    json params{{"alpha", alpha}, {"beta", beta}, {"gamma", gamma_run}};
    auto phys = kawahara::KawaharaPhysics<>::from_json(params, domain,
                                                       stack.fft().get_inbox_bounds());
    pfc::SimulationState state;
    phys.declare_fields(state);
    auto &u = state.get_field<double>("u");
    u.apply([&](double x, double, double) {
      const double d = x - x0;
      return amp * std::exp(-d * d / (2.0 * sigma * sigma));
    });
    const double mean0 = mean_u(u);
    pfc::sim::SpectralETDOptions opt;
    opt.psi_name = "u";
    opt.dealias = true;
    pfc::sim::SpectralETDSystem<kawahara::KawaharaPhysics<>> sys(phys, stack.fft(), state,
                                                                 dt, opt);
    double t = 0.0;
    for (int step = 0; step < n_steps; ++step) {
      t = sys.step(t);
    }
    kawahara::PulseDiagnostics diag(MPI_COMM_WORLD, 4.0 * sigma);
    const auto sample = diag.sample(u);
    return std::pair<double, double>{mean_u(u) - mean0, sample.tail_rms};
  };

  const auto [mean_drift_third, tail_third] = run(0.0);
  const auto [mean_drift_full, tail_full] = run(gamma);
  INFO("tail_rms third-order-only=" << tail_third << " full=" << tail_full);
  REQUIRE_THAT(mean_drift_third, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(mean_drift_full, WithinAbs(0.0, 1e-9));
  // The fifth-order term must have a reproducible, nonzero effect on the
  // trailing dispersive radiation at this amplitude/duration. Measured
  // |tail_full-tail_third| at T=20 is ~3.5-4.0e-7 (repeatable on both
  // N=512 and N=1024, i.e. not a resolution/aliasing artifact), roughly
  // four orders of magnitude above the double-precision noise floor for
  // this quantity (O(1e-16) relative to an O(0.15)-magnitude field,
  // accumulated over a few FFTs/step across 4000 steps stays well under
  // 1e-12 in absolute terms), so 1e-7 leaves a >=3x margin below the
  // measured effect while remaining far above rounding noise.
  REQUIRE(std::abs(tail_full - tail_third) > 1.0e-7);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
