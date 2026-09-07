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

#include <kawahara/kawahara_physics.hpp>
#include <kawahara/kawahara_session.hpp>
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

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
