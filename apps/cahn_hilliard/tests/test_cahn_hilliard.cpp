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

TEST_CASE("CahnHilliard spinodal mode grows and bulk free energy falls",
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
  (void)sys.step(0.0);
  const double bulk0 = sys.last_free_energy();
  double t = dt;
  for (int step = 1; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE(variance_c(c) > var0);
  REQUIRE(sys.last_free_energy() <= bulk0 + 1.0e-12);
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
