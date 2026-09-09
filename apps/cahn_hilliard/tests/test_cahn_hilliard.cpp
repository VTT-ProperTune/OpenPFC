// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_cahn_hilliard.cpp
 * @brief Catch2 tests for the Fe–Cr Cahn–Hilliard spectral app.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <mpi.h>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include <cahn_hilliard/cahn_hilliard_physics.hpp>
#include <cahn_hilliard/cahn_hilliard_session.hpp>
#include <cahn_hilliard/cosine_mode.hpp>
#include <cahn_hilliard/fe_cr_thermo.hpp>
#include <openpfc_apps/structure_factor.hpp>
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

cahn_hilliard::CahnHilliardPhysics<> default_physics(const pfc::Domain &domain,
                                                     const pfc::Box3i &box) {
  return cahn_hilliard::CahnHilliardPhysics<>::from_json(json::object(), domain,
                                                         box);
}

double cosine_amplitude(const pfc::data::Field<double> &c, double c0, int nx,
                        int ny) {
  const auto n = c.local_size();
  const auto sp = c.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double Ly = static_cast<double>(n[1]) * sp[1];
  const double twopi = 2.0 * std::numbers::pi;
  double num = 0.0;
  double den = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = c.coords(i, j, k);
        const double w = std::cos(twopi * (static_cast<double>(nx) * x[0] / Lx +
                                           static_cast<double>(ny) * x[1] / Ly));
        num += (c(i, j, k) - c0) * w;
        den += w * w;
      }
    }
  }
  return (den > 0.0) ? num / den : 0.0;
}

double mean_c(const pfc::data::Field<double> &c) {
  double sum = 0.0;
  const auto n = c.local_size();
  std::size_t count = 0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        sum += c(i, j, k);
        ++count;
      }
    }
  }
  return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

double variance_c(const pfc::data::Field<double> &c) {
  const double mu = mean_c(c);
  double acc = 0.0;
  std::size_t count = 0;
  const auto n = c.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const double d = c(i, j, k) - mu;
        acc += d * d;
        ++count;
      }
    }
  }
  return (count > 0) ? acc / static_cast<double>(count) : 0.0;
}

json mini_session_json() {
  return {
      {"model", {{"name", "cahn_hilliard"}, {"params", {{"c0", 0.32}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.05}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "c"},
         {"type", "cosine_mode"},
         {"c0", 0.32},
         {"amplitude", 0.01},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
}

} // namespace

TEST_CASE("CahnHilliard schema defaults put Fe-32Cr in the spinodal",
          "[cahn_hilliard][physics]") {
  cahn_hilliard::CahnHilliardPhysics<> phys;
  REQUIRE_THAT(phys.params.c0, WithinAbs(0.32, 1e-15));
  REQUIRE_THAT(phys.params.T, WithinAbs(748.15, 1e-12));
  REQUIRE(phys.params.omega_nd > 3.0);
  REQUIRE(phys.params.omega_nd < 3.5);
  REQUIRE(phys.in_spinodal(0.32));
  REQUIRE(phys.in_spinodal(0.50));
  REQUIRE_FALSE(phys.in_spinodal(0.05));
  REQUIRE_FALSE(phys.in_spinodal(0.95));
}

TEST_CASE("CahnHilliard schema round-trips JSON and rejects bad c0",
          "[cahn_hilliard][physics][schema]") {
  auto schema = cahn_hilliard::CahnHilliardPhysics<>::schema();
  const auto vals = schema.parse({{"c0", 0.45}, {"M", 2.0}});
  REQUIRE_THAT(vals.c0, WithinAbs(0.45, 1e-15));
  REQUIRE_THAT(vals.M, WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(vals.T, WithinAbs(748.15, 1e-12));
  REQUIRE_THROWS_AS(schema.parse({{"c0", 0.0}}), std::invalid_argument);
}

TEST_CASE("CahnHilliard L(k) is M (f''(c0) k_lap - kappa k_lap^2)",
          "[cahn_hilliard][physics][symbol]") {
  cahn_hilliard::CahnHilliardPhysics<> phys;
  const double k_lap = -4.0;
  const double expected =
      phys.params.M * (phys.params.fpp0 * k_lap - phys.params.kappa * k_lap * k_lap);
  REQUIRE_THAT(phys.linear_symbol(k_lap), WithinAbs(expected, 1e-14));
  REQUIRE_THAT(phys.linear_symbol(0.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.nonlinear_symbol(k_lap),
               WithinAbs(phys.params.M * k_lap, 1e-15));
  REQUIRE(phys.params.fpp0 < 0.0);
  REQUIRE(phys.linear_symbol(-0.25) > 0.0);
}

TEST_CASE("CahnHilliard n_nl vanishes at c0", "[cahn_hilliard][physics]") {
  const auto pw = cahn_hilliard::CahnHilliardPhysics<>{}.pointwise();
  REQUIRE_THAT(pw.n_nl(pw.c0), WithinAbs(0.0, 1e-14));
}

TEST_CASE("cosine_mode JSON sets a single periodic mode", "[cahn_hilliard][ic]") {
  json j = {{"type", "cosine_mode"},
            {"c0", 0.32},
            {"amplitude", 0.02},
            {"nx", 3},
            {"ny", 1},
            {"nz", 0}};
  cahn_hilliard::CosineMode ic;
  cahn_hilliard::from_json(j, ic);
  REQUIRE_THAT(ic.c0(), WithinAbs(0.32, 1e-15));
  REQUIRE_THAT(ic.amplitude(), WithinAbs(0.02, 1e-15));
  REQUIRE(ic.nx() == 3);
  REQUIRE(ic.ny() == 1);
  REQUIRE(ic.nz() == 0);
}

TEST_CASE("CahnHilliard ETD conserves mean c and matches linear growth",
          "[cahn_hilliard][spectral][mass][spinodal]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr int ny = 0;
  constexpr double amp0 = 1.0e-4;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = default_physics(domain, stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &c = state.get_field<double>("c");
  const double c0 = phys.params.c0;
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  c.apply([&](double x, double y, double) {
    return c0 + amp0 * std::cos(twopi * static_cast<double>(nx) * x / Lx +
                                twopi * static_cast<double>(ny) * y / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "c";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<cahn_hilliard::CahnHilliardPhysics<>> sys(
      phys, stack.fft(), state, dt, opt);

  const double k = twopi * static_cast<double>(nx) / Lx;
  const double k_lap = -(k * k);
  const double lambda = phys.linear_symbol(k_lap);
  REQUIRE(lambda > 0.0);

  const double mean0 = mean_c(c);
  REQUIRE_THAT(mean0, WithinAbs(c0, 1e-12));

  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE_THAT(mean_c(c), WithinAbs(mean0, 1e-12));
  const double amp = cosine_amplitude(c, c0, nx, ny);
  const double expected = amp0 * std::exp(lambda * t);
  REQUIRE_THAT(amp, WithinRel(expected, 0.05));
}

TEST_CASE("CahnHilliard spinodal mode grows and total free energy falls",
          "[cahn_hilliard][spectral][energy]") {
  if (world_size() != 1) {
    SKIP("single-rank free-energy comparison");
  }
  constexpr int N = 32;
  constexpr double dt = 0.05;
  constexpr int n_steps = 6;
  constexpr double amp0 = 0.03;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = default_physics(domain, stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &c = state.get_field<double>("c");
  const double c0 = phys.params.c0;
  const double twopi = 2.0 * std::numbers::pi;
  c.apply([&](double x, double y, double) {
    return c0 + amp0 * std::cos(twopi * 2.0 * x / static_cast<double>(N) +
                                twopi * 1.0 * y / static_cast<double>(N));
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "c";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<cahn_hilliard::CahnHilliardPhysics<>> sys(
      phys, stack.fft(), state, dt, opt);

  const double var0 = variance_c(c);
  cahn_hilliard::Diagnostics<pfc::HostSpace> diagnostics(domain, stack.fft(),
                                                         MPI_COMM_WORLD);
  auto previous = diagnostics.sample(c, phys.params);
  double t = 0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
    const auto current = diagnostics.sample(c, phys.params);
    REQUIRE(current.total_energy() <= previous.total_energy() + 1e-12);
    REQUIRE_THAT(current.mass, WithinAbs(previous.mass, 1e-10));
    previous = current;
  }
  REQUIRE(variance_c(c) > var0);
}

TEST_CASE("Diagnostics match the analytical gradient energy on distributed grids",
          "[cahn_hilliard][diagnostics]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto domain = pfc::domain::create(pfc::GridSize({32, 16, 1}),
                                          pfc::PhysicalOrigin({-2.0, 3.0, 0.0}),
                                          pfc::GridSpacing({0.5, 2.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, world_size(),
                                           MPI_COMM_WORLD);
  auto &c = stack.u();
  cahn_hilliard::CahnHilliardParams params;
  cahn_hilliard::Diagnostics<pfc::HostSpace> diagnostics(domain, stack.fft(),
                                                         MPI_COMM_WORLD);
  const double kx = 2 * std::numbers::pi / 16;
  const double ky = 4 * std::numbers::pi / 32;
  c.apply([&](double x, double y, double) {
    return 0.32 + 0.02 * std::cos(kx * x + ky * y);
  });
  const auto s = diagnostics.sample(c, params);
  REQUIRE_THAT(s.mean, WithinAbs(0.32, 1e-13));
  REQUIRE_THAT(s.mass, WithinAbs(0.32 * 512, 1e-10));
  REQUIRE_THAT(
      s.gradient_energy,
      WithinAbs(512 * params.kappa * 0.02 * 0.02 * (kx * kx + ky * ky) / 4, 1e-12));
  REQUIRE(s.invalid_cells == 0);
  c.apply([](double, double, double) { return 0.32; });
  const auto flat = diagnostics.sample(c, params);
  REQUIRE_THAT(flat.gradient_energy, WithinAbs(0, 1e-12));
  const auto pw = cahn_hilliard::CahnHilliardPointwise{.omega_nd = params.omega_nd};
  REQUIRE_THAT(flat.bulk_energy, WithinAbs(512 * pw.f_bulk(0.32), 1e-10));
  c.apply([](double, double, double) { return 1e-14; });
  const auto dilute = diagnostics.sample(c, params);
  const double dilute_energy = params.omega_nd * 1e-14 * (1 - 1e-14) +
                               1e-14 * std::log(1e-14) +
                               (1 - 1e-14) * std::log1p(-1e-14);
  REQUIRE_THAT(dilute.bulk_energy, WithinAbs(512 * dilute_energy, 1e-22));
  c.apply([](double, double, double) { return 1.1; });
  const auto bad = diagnostics.sample(c, params);
  REQUIRE(bad.invalid_cells == 512);
  REQUIRE(std::isnan(bad.total_energy()));
}

TEST_CASE("Seeded noise has the same mean and cells on every decomposition",
          "[cahn_hilliard][noise]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto domain =
      pfc::domain::create(pfc::GridSize({32, 16, 1}), pfc::PhysicalOrigin({0, 0, 0}),
                          pfc::GridSpacing({1, 1, 1}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, rank, world_size(),
                                           MPI_COMM_WORLD);
  cahn_hilliard::SeededNoise noise;
  cahn_hilliard::from_json(json{{"type", "seeded_noise"},
                                {"c0", 0.32},
                                {"amplitude", 0.02},
                                {"seed", 1234}},
                           noise);
  const pfc::SimulationContext context(MPI_COMM_WORLD);
  pfc::apply_field_modifier(noise, stack.u(), 0, &context);
  auto full = pfc::data::field_from_inbox<double>(
      domain, pfc::Box3i::from_bounds({0, 0, 0}, {31, 15, 0}));
  pfc::apply_field_modifier(noise, full, 0);
  auto &c = stack.u();
  c.for_each_owned([&](int i, int j, int k) {
    REQUIRE(c(i, j, k) ==
            full(i + c.box().low[0], j + c.box().low[1], k + c.box().low[2]));
  });
  REQUIRE_THAT(mean_c(full), WithinAbs(0.32, 1e-14));
  REQUIRE(variance_c(full) > 0);
  const auto before = full.vec();
  ++noise.seed;
  pfc::apply_field_modifier(noise, full, 0);
  REQUIRE(full.vec() != before);
  noise.amplitude = 0.5;
  REQUIRE_THROWS(pfc::apply_field_modifier(noise, full, 0));
  REQUIRE_THROWS(cahn_hilliard::from_json(
      json{
          {"type", "seeded_noise"}, {"c0", 0.32}, {"amplitude", 0.02}, {"seed", -1}},
      noise));
}

TEST_CASE("CahnHilliardSession runs a short JSON case", "[cahn_hilliard][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  cahn_hilliard::register_catalog();
  cahn_hilliard::CahnHilliardSession session(mini_session_json(), 0, 1,
                                             MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.1).margin(1e-12));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

// ---------------------------------------------------------------------------
// Fe-Cr thermodynamics (#113)
// ---------------------------------------------------------------------------

TEST_CASE("Redlich-Kister with L1 = 0 is exactly the regular solution",
          "[cahn_hilliard][thermo]") {
  // The reduced verifier must remain a special case of the assessed model,
  // not a parallel code path that can drift away from it.
  constexpr double omega = 3.23;
  cahn_hilliard::CahnHilliardPointwise regular{.omega_nd = omega, .l1_nd = 0.0};
  for (int i = 1; i < 100; ++i) {
    const double c = i / 100.0;
    const double u = c;
    REQUIRE_THAT(regular.f_bulk(c),
                 WithinRel(omega * u * (1 - u) + u * std::log(u) +
                               (1 - u) * std::log(1 - u),
                           1e-13));
    REQUIRE_THAT(regular.f_prime(c),
                 WithinRel(omega * (1 - 2 * u) + std::log(u / (1 - u)), 1e-13));
    REQUIRE_THAT(regular.f_double_prime(c),
                 WithinRel(-2 * omega + 1 / (u * (1 - u)), 1e-13));
  }
}

TEST_CASE("Redlich-Kister derivatives match finite differences",
          "[cahn_hilliard][thermo]") {
  cahn_hilliard::CahnHilliardPointwise pw{.omega_nd = 2.4, .l1_nd = 0.7};
  // Separate steps: the second difference divides by h^2, so its round-off
  // floor is eps/h^2. At h=1e-6 that is 1e-4 -- the same size as the check.
  const double h1 = 1e-6, h2 = 1e-4;
  for (const double c : {0.2, 0.35, 0.5, 0.65, 0.8}) {
    const double fd1 = (pw.f_bulk(c + h1) - pw.f_bulk(c - h1)) / (2 * h1);
    const double fd2 =
        (pw.f_bulk(c + h2) - 2 * pw.f_bulk(c) + pw.f_bulk(c - h2)) / (h2 * h2);
    REQUIRE_THAT(pw.f_prime(c), WithinRel(fd1, 1e-6));
    REQUIRE_THAT(pw.f_double_prime(c), WithinRel(fd2, 1e-5));
  }
}

TEST_CASE("Spinodal range agrees with the closed form when L1 = 0",
          "[cahn_hilliard][thermo]") {
  const double l0 = 2.5;
  const auto range = cahn_hilliard::spinodal_range(l0, 0.0);
  REQUIRE(range.exists);
  // f'' < 0 requires c(1-c) > 1/(2 l0); the roots are symmetric about 1/2.
  const double disc = std::sqrt(1.0 - 2.0 / l0);
  REQUIRE_THAT(range.lower, WithinAbs(0.5 * (1.0 - disc), 1e-9));
  REQUIRE_THAT(range.upper, WithinAbs(0.5 * (1.0 + disc), 1e-9));
  REQUIRE(range.contains(0.5));
  REQUIRE_FALSE(range.contains(0.02));
}

TEST_CASE("A shallow interaction leaves no miscibility gap",
          "[cahn_hilliard][thermo]") {
  // Below l0 = 2 the solution is stable at every composition.
  REQUIRE_FALSE(cahn_hilliard::spinodal_range(1.5, 0.0).exists);
  REQUIRE(cahn_hilliard::spinodal_range(2.5, 0.0).exists);
}

TEST_CASE("Andersson-Sundman puts Fe-32Cr outside the 475 C spinodal",
          "[cahn_hilliard][thermo]") {
  // Recorded because it drives the preset design: the assessed interaction is
  // shallower than the app's original representative Omega, so the classic
  // Fe-32Cr composition sits in the nucleation regime rather than the
  // spinodal, and a spinodal science preset must move to ~40-60% Cr.
  const cahn_hilliard::RedlichKister rk; // shipped defaults
  const double T = 748.15;
  const auto range = cahn_hilliard::spinodal_range(rk.l0_nd(T), rk.l1_nd(T));
  REQUIRE(range.exists);
  REQUIRE_FALSE(range.contains(0.32));
  REQUIRE(range.contains(0.50));
  REQUIRE_THAT(rk.l0_nd(T), WithinRel(2.1313, 1e-3));
}

TEST_CASE("Physical scales convert code units to nm and seconds",
          "[cahn_hilliard][thermo]") {
  const cahn_hilliard::PhysicalScales sc;
  const double T = 748.15;
  // l = sqrt(kappa Vm / (R T)); t = l^4 f0 / D.
  const double f0 = cahn_hilliard::kGasConstant * T / sc.Vm;
  REQUIRE_THAT(sc.f0(T), WithinRel(f0, 1e-12));
  REQUIRE_THAT(sc.length_m(T), WithinRel(std::sqrt(sc.kappa / f0), 1e-12));
  REQUIRE_THAT(sc.length_nm(T), WithinRel(1e9 * sc.length_m(T), 1e-12));
  const double l = sc.length_m(T);
  const double fpp = 0.2627; // |f''| at c0 = 0.5 for the shipped coefficients
  REQUIRE_THAT(sc.time_s(T, fpp), WithinRel(l * l * fpp / sc.D(T), 1e-12));
  REQUIRE(sc.length_nm(T) > 0.0);
  // Ageing at 475 C is an hours-to-weeks process: a code time unit must land
  // in minutes, not microseconds. This caught an incorrect first derivation.
  REQUIRE(sc.time_s(T, fpp) > 60.0);
  REQUIRE(sc.time_s(T, fpp) < 1.0e4);
  REQUIRE_THAT(sc.time_hours(T, fpp), WithinRel(sc.time_s(T, fpp) / 3600.0, 1e-12));
}

TEST_CASE("Redlich-Kister JSON drives the model and reports its spinodal",
          "[cahn_hilliard][thermo][schema]") {
  cahn_hilliard::CahnHilliardParams p;
  cahn_hilliard::apply_cahn_hilliard_json(
      json{{"c0", 0.5}, {"T", 748.15}, {"L0_a", 20500.0}, {"L0_b", -9.68}}, p);
  const cahn_hilliard::RedlichKister rk{20500.0, -9.68, 0.0, 0.0};
  REQUIRE_THAT(p.omega_nd, WithinRel(rk.l0_nd(748.15), 1e-12));
  REQUIRE(p.spinodal.exists);
  REQUIRE(p.spinodal.contains(0.5));
  REQUIRE(p.fpp0 < 0.0); // unstable at the linearization point

  // Omega still drives the model when no RK coefficients are given.
  cahn_hilliard::CahnHilliardParams q;
  cahn_hilliard::apply_cahn_hilliard_json(json{{"Omega", 20100.0}}, q);
  REQUIRE_THAT(q.omega_nd, WithinRel(20100.0 / (q.R * q.T), 1e-12));
  REQUIRE_THAT(q.l1_nd, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Structure factor (#113)
// ---------------------------------------------------------------------------

TEST_CASE("Structure factor finds the wave number of a single mode",
          "[cahn_hilliard][structure_factor]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral check");
  }
  constexpr int N = 64;
  constexpr int nx = 6;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &c = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double L = static_cast<double>(N);
  c.apply([&](double x, double, double) {
    return 0.32 + 0.01 * std::cos(twopi * nx * x / L);
  });

  pfc::data::Field<std::complex<double>> hat(domain,
                                             stack.fft().get_outbox_bounds(), 0);
  pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), c, hat);
  pfc::apps::StructureFactor sf;
  hat.with_host_view([&](std::complex<double> *h, std::size_t) {
    sf = pfc::apps::shell_average(stack.fft().get_outbox_bounds(), domain, h,
                                      MPI_COMM_WORLD, 64);
  });

  const double k_expected = twopi * nx / L;
  REQUIRE(sf.total_power > 0.0);
  // The peak shell must bracket the imposed wave number.
  const double bin = (std::numbers::pi / 1.0) / 64.0;
  REQUIRE_THAT(sf.k_peak, WithinAbs(k_expected, bin));
  REQUIRE_THAT(sf.dominant_wavelength(), WithinRel(L / nx, 0.1));
  // A single mode puts essentially all power in one shell, so the first
  // moment lands on it too.
  REQUIRE_THAT(sf.k1, WithinAbs(k_expected, 2 * bin));
  REQUIRE(sf.domain_length() > 0.0);
}

TEST_CASE("Coarsening exponent recovers a known power law",
          "[cahn_hilliard][structure_factor]") {
  std::vector<double> t, L;
  for (int i = 1; i <= 50; ++i) {
    const double ti = 0.5 * i;
    t.push_back(ti);
    L.push_back(3.7 * std::pow(ti, 1.0 / 3.0));
  }
  REQUIRE_THAT(pfc::apps::coarsening_exponent(t, L),
               WithinRel(1.0 / 3.0, 1e-9));
  // A t=0 sample is ignored rather than poisoning the log fit.
  t.insert(t.begin(), 0.0);
  L.insert(L.begin(), 0.0);
  REQUIRE_THAT(pfc::apps::coarsening_exponent(t, L),
               WithinRel(1.0 / 3.0, 1e-9));
  REQUIRE(pfc::apps::coarsening_exponent({1.0}, {2.0}) == 0.0);
}
