// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_gradient_elasticity.cpp
 * @brief Catch2 tests for Helmholtz–Navier gradient elasticity.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <mpi.h>
#include <numbers>

#include <nlohmann/json.hpp>

#include <gradient_elasticity/gradient_elasticity_physics.hpp>
#include <gradient_elasticity/gradient_elasticity_session.hpp>
#include <gradient_elasticity/gradient_elasticity_solve.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
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

double mean_field(const pfc::data::Field<double> &f) {
  double sum = 0.0;
  std::size_t count = 0;
  f.for_each_owned([&](int i, int j, int k) {
    sum += f(i, j, k);
    ++count;
  });
  return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

double max_abs_err_sine(const pfc::data::Field<double> &ux, double amp, double kx,
                        double ky) {
  double m = 0.0;
  const auto n = ux.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = ux.coords(i, j, k);
        const double expect = amp * std::sin(kx * x[0] + ky * x[1]);
        m = std::max(m, std::abs(ux(i, j, k) - expect));
      }
    }
  }
  return m;
}

} // namespace

TEST_CASE("Helmholtz alpha is 1 + ell^2 k^2 at fourth order",
          "[gradient_elasticity][physics][alpha]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  REQUIRE_THAT(phys.params.ell, WithinAbs(1.0, 1e-15));
  REQUIRE(phys.params.order == 4);
  REQUIRE_THAT(phys.helmholtz_alpha(0.0), WithinAbs(1.0, 1e-15));
  const double k2 = 0.25;
  REQUIRE_THAT(phys.helmholtz_alpha(k2), WithinAbs(1.0 + k2, 1e-15));
}

TEST_CASE("Sixth-order alpha is (1 + ell^2 k^2)^2",
          "[gradient_elasticity][physics][alpha6]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  gradient_elasticity::apply_gradient_elasticity_json({{"order", 6}, {"ell", 2.0}},
                                                      phys.params);
  const double k2 = 0.25;
  const double a1 = 1.0 + 4.0 * k2;
  REQUIRE_THAT(phys.helmholtz_alpha(k2), WithinAbs(a1 * a1, 1e-15));
}

TEST_CASE("ell4>0 selects mixed sixth-order alpha",
          "[gradient_elasticity][physics][ell4]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  gradient_elasticity::apply_gradient_elasticity_json(
      {{"ell", 1.0}, {"ell4", 2.0}, {"order", 4}}, phys.params);
  const double k2 = 0.25;
  const double expect = 1.0 + k2 + 16.0 * k2 * k2;
  REQUIRE_THAT(phys.helmholtz_alpha(k2), WithinAbs(expect, 1e-15));
}

TEST_CASE("E and nu convert to Lamé moduli",
          "[gradient_elasticity][physics][lame]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  gradient_elasticity::apply_gradient_elasticity_json({{"E", 2.6}, {"nu", 0.3}},
                                                      phys.params);
  REQUIRE_THAT(phys.params.mu, WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(phys.params.lambda, WithinAbs(1.5, 1e-12));
}

TEST_CASE("k=0 mode is projected to zero displacement",
          "[gradient_elasticity][physics][nullspace]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  const auto u = phys.invert(0.0, 0.0, 0.0, {1.0, 2.0});
  REQUIRE_THAT(u.ux.real(), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(u.uy.real(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Longitudinal invert matches 1 / (-alpha (lambda+2 mu) k^2)",
          "[gradient_elasticity][physics][lt]") {
  gradient_elasticity::GradientElasticityPhysics<> phys;
  const double kx = 0.5;
  const double ky = 0.0;
  const double k2 = kx * kx;
  const gradient_elasticity::ModeForce f{{1.0, 0.0}, {0.0, 0.0}};
  const auto u = phys.invert(kx, ky, 0.0, f);
  const double den =
      -phys.helmholtz_alpha(k2) * (phys.params.lambda + 2.0 * phys.params.mu) * k2;
  REQUIRE_THAT(u.ux.real(), WithinAbs(1.0 / den, 1e-15));
  REQUIRE_THAT(u.uy.real(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Finite ell reduces high-k amplitude versus classical elasticity",
          "[gradient_elasticity][physics][highk]") {
  gradient_elasticity::GradientElasticityPhysics<> classical;
  gradient_elasticity::apply_gradient_elasticity_json({{"ell", 0.0}},
                                                      classical.params);
  gradient_elasticity::GradientElasticityPhysics<> graded;
  gradient_elasticity::apply_gradient_elasticity_json({{"ell", 2.0}}, graded.params);
  const double k_lo = 0.2;
  const double k_hi = 2.0;
  const gradient_elasticity::ModeForce f{{1.0, 0.0}, {0.0, 0.0}};
  const auto u_lo_c = classical.invert(k_lo, 0.0, 0.0, f);
  const auto u_hi_c = classical.invert(k_hi, 0.0, 0.0, f);
  const auto u_lo_g = graded.invert(k_lo, 0.0, 0.0, f);
  const auto u_hi_g = graded.invert(k_hi, 0.0, 0.0, f);
  const double r_lo = std::abs(u_lo_g.ux) / std::abs(u_lo_c.ux);
  const double r_hi = std::abs(u_hi_g.ux) / std::abs(u_hi_c.ux);
  REQUIRE_THAT(r_lo, WithinAbs(1.0 / graded.helmholtz_alpha(k_lo * k_lo), 1e-12));
  REQUIRE_THAT(r_hi, WithinAbs(1.0 / graded.helmholtz_alpha(k_hi * k_hi), 1e-12));
  REQUIRE(r_hi < r_lo);
  REQUIRE(r_hi < 0.1);
}

TEST_CASE("ell->0 recovers the classical Fourier solution",
          "[gradient_elasticity][spectral][classical]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double amp_g = 1.0;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      json{{"ell", 0.0}, {"eps0", 0.05}, {"mu", 1.0}, {"lambda", 1.0}}, domain,
      stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &g = state.get_field<double>("g");
  auto &ux = state.get_field<double>("ux");
  auto &uy = state.get_field<double>("uy");
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  const double kx = twopi * static_cast<double>(nx) / Lx;
  g.apply([&](double x, double, double) { return amp_g * std::cos(kx * x); });

  gradient_elasticity::solve_displacement(stack.fft(), phys, g, ux, uy);

  const double k2 = kx * kx;
  const double B = phys.cosine_displacement_prefactor(k2) * amp_g;
  REQUIRE_THAT(max_abs_err_sine(ux, B * kx, kx, 0.0), WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(mean_field(uy), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(mean_field(ux), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Cosine eigenstrain matches the analytical Fourier displacement",
          "[gradient_elasticity][spectral][fourier]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr int ny = 1;
  constexpr double amp_g = 1.0;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      json{{"ell", 2.0}, {"eps0", 0.04}, {"mu", 1.0}, {"lambda", 1.0}}, domain,
      stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &g = state.get_field<double>("g");
  auto &ux = state.get_field<double>("ux");
  auto &uy = state.get_field<double>("uy");
  const double twopi = 2.0 * std::numbers::pi;
  const double L = static_cast<double>(N);
  const double kx = twopi * static_cast<double>(nx) / L;
  const double ky = twopi * static_cast<double>(ny) / L;
  g.apply(
      [&](double x, double y, double) { return amp_g * std::cos(kx * x + ky * y); });

  gradient_elasticity::solve_displacement(stack.fft(), phys, g, ux, uy);

  const double k2 = kx * kx + ky * ky;
  const double B = phys.cosine_displacement_prefactor(k2) * amp_g;
  REQUIRE_THAT(max_abs_err_sine(ux, B * kx, kx, ky), WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(max_abs_err_sine(uy, B * ky, kx, ky), WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(mean_field(ux), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(mean_field(uy), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Finite ell damps a high cosine mode versus ell=0",
          "[gradient_elasticity][spectral][regularize]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 8;
  constexpr double amp_g = 1.0;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  const json j0{{"eps0", 0.05}, {"mu", 1.0}, {"lambda", 1.0}, {"ell", 0.0}};
  const json j1{{"eps0", 0.05}, {"mu", 1.0}, {"lambda", 1.0}, {"ell", 3.0}};
  auto phys0 = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      j0, domain, stack.fft().get_inbox_bounds());
  auto phys1 = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      j1, domain, stack.fft().get_inbox_bounds());

  pfc::SimulationState s0;
  pfc::SimulationState s1;
  phys0.declare_fields(s0);
  phys1.declare_fields(s1);
  const double twopi = 2.0 * std::numbers::pi;
  const double kx = twopi * static_cast<double>(nx) / static_cast<double>(N);
  auto fill = [&](pfc::data::Field<double> &g) {
    g.apply([&](double x, double, double) { return amp_g * std::cos(kx * x); });
  };
  fill(s0.get_field<double>("g"));
  fill(s1.get_field<double>("g"));
  gradient_elasticity::solve_displacement(
      stack.fft(), phys0, s0.get_field<double>("g"), s0.get_field<double>("ux"),
      s0.get_field<double>("uy"));
  gradient_elasticity::solve_displacement(
      stack.fft(), phys1, s1.get_field<double>("g"), s1.get_field<double>("ux"),
      s1.get_field<double>("uy"));
  const double k2 = kx * kx;
  const double B0 = phys0.cosine_displacement_prefactor(k2) * amp_g;
  const double B1 = phys1.cosine_displacement_prefactor(k2) * amp_g;
  REQUIRE(std::abs(B1) < std::abs(B0));
  REQUIRE_THAT(max_abs_err_sine(s0.get_field<double>("ux"), B0 * kx, kx, 0.0),
               WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(max_abs_err_sine(s1.get_field<double>("ux"), B1 * kx, kx, 0.0),
               WithinAbs(0.0, 1e-10));
}

TEST_CASE("GradientElasticitySession runs a short JSON case",
          "[gradient_elasticity][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  json settings = {
      {"model",
       {{"name", "gradient_elasticity"},
        {"params", {{"ell", 1.0}, {"eps0", 0.02}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 1.0}, {"dt", 1.0}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "g"},
         {"type", "cosine_mode"},
         {"g0", 0.0},
         {"amplitude", 1.0},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
  gradient_elasticity::register_catalog();
  gradient_elasticity::GradientElasticityCPUSession session(settings, 0, 1,
                                                            MPI_COMM_WORLD);
  session.run();
  REQUIRE_THAT(mean_field(session.ux()), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(mean_field(session.uy()), WithinAbs(0.0, 1e-12));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
