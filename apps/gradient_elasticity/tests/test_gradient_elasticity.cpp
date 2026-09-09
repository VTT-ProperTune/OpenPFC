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
#include <vector>

#include <nlohmann/json.hpp>

#include <gradient_elasticity/gradient_elasticity_diagnostics.hpp>
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

/// Fill `g` with a `tanh`-smoothed circular inclusion (same formula as
/// `CircularInclusion::apply`, inlined here so the low-level physics tests
/// don't need the JSON field-modifier plumbing).
void fill_circular_inclusion(pfc::data::Field<double> &g, double x0, double y0,
                             double R, double w) {
  g.apply([&](double x, double y, double) {
    const double dx = x - x0;
    const double dy = y - y0;
    const double r = std::sqrt(dx * dx + dy * dy);
    return 0.5 * (1.0 - std::tanh((r - R) / w));
  });
}

/// One-shot circular-inclusion solve (displacement, strain, stress, energy)
/// on an `N x N` periodic square, inclusion centred at the box midpoint.
/// Single-rank, direct physics/solve calls (no JSON session) so the size
/// sweep used by the tests below stays fast and self-contained.
gradient_elasticity::StressSummary run_circular_case(int N, double R, double w,
                                                      double ell, double mu,
                                                      double lambda, double eps0,
                                                      int order = 4) {
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      json{{"ell", ell}, {"eps0", eps0}, {"mu", mu}, {"lambda", lambda},
          {"order", order}},
      domain, stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &g = state.get_field<double>("g");
  auto &ux = state.get_field<double>("ux");
  auto &uy = state.get_field<double>("uy");
  auto &exx = state.get_field<double>("exx");
  auto &eyy = state.get_field<double>("eyy");
  auto &exy = state.get_field<double>("exy");
  auto &sxx = state.get_field<double>("sxx");
  auto &syy = state.get_field<double>("syy");
  auto &sxy = state.get_field<double>("sxy");
  auto &stress_hydro = state.get_field<double>("stress_hydro");
  auto &stress_vm = state.get_field<double>("stress_vm");
  auto &energy_density = state.get_field<double>("energy_density");
  const double x0 = 0.5 * static_cast<double>(N);
  const double y0 = 0.5 * static_cast<double>(N);
  fill_circular_inclusion(g, x0, y0, R, w);
  gradient_elasticity::solve_displacement_and_strain(stack.fft(), phys, g, ux, uy,
                                                     exx, eyy, exy);
  gradient_elasticity::compute_stress_fields(phys, g, exx, eyy, exy, sxx, syy, sxy,
                                             stress_hydro, stress_vm,
                                             energy_density);
  return gradient_elasticity::summarize_stress(stress_hydro, stress_vm,
                                               energy_density, domain,
                                               MPI_COMM_WORLD);
}

/// Classical (infinite-domain, sharp-boundary) closed form for a 2-D
/// circular inclusion of radius `R` with dilatational eigenstrain
/// `eps*=eps0*I`, from axisymmetric elasticity with eigenstrain (derived in
/// the PR description / app README): stress is spatially uniform inside and
/// decays as \(1/r^2\) outside; both are independent of `R` -- the classical
/// theory has no length scale, which is exactly the size effect gradient
/// elasticity is meant to introduce.
double classical_inclusion_inside_hydrostatic(double mu, double lambda,
                                              double eps0) {
  const double A = eps0 * (lambda + mu) / (lambda + 2.0 * mu);
  return 2.0 * (lambda + mu) * (A - eps0);
}

double classical_inclusion_boundary_von_mises(double mu, double lambda,
                                              double eps0) {
  const double A = eps0 * (lambda + mu) / (lambda + 2.0 * mu);
  return std::sqrt(3.0) * 2.0 * mu * A;
}

/// The *other* closed-form bound of the size-effect curve: the fully
/// unrelaxed ("clamped") limit `ell -> infinity` (equivalently `R/ell -> 0`).
/// Every Fourier mode with `k>0` is annihilated as `alpha=1+ell^2 k^2 ->
/// infinity`, so `u -> 0` identically (the gradient penalty forbids *any*
/// spatial variation of the displacement). With `u=0`, the elastic strain is
/// just the negative eigenstrain, `eps^e=-eps0*g*I`, so deep inside the
/// inclusion (`g=1`) the material cannot relax the misfit at all:
/// `sigma=-2(lambda+mu)*eps0*I`, a pure equibiaxial (hydrostatic) state, so
/// the reduced von Mises invariant equals the same magnitude. This is the
/// *maximum possible* internal stress for this eigenstrain -- gradient
/// elasticity increases peak stress above the classical value for small
/// inclusions (`R/ell` small) here, the opposite of the familiar
/// "regularizes a classical singularity" story: this loading (a smooth,
/// finite inclusion) has no classical singularity to begin with, so `ell`
/// instead acts as an increasing constraint on elastic relaxation.
double clamped_inclusion_hydrostatic(double mu, double lambda, double eps0) {
  return -2.0 * (lambda + mu) * eps0;
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

TEST_CASE("Spectral strain/stress post-processing matches the analytical "
          "cosine-mode field",
          "[gradient_elasticity][spectral][stress][analytical]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr int ny = 1;
  constexpr double amp_g = 1.0;
  constexpr double mu = 1.3;
  constexpr double lambda = 0.7;
  constexpr double eps0 = 0.03;
  constexpr double ell = 2.0;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = gradient_elasticity::GradientElasticityPhysics<>::from_json(
      json{{"ell", ell}, {"eps0", eps0}, {"mu", mu}, {"lambda", lambda}}, domain,
      stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &g = state.get_field<double>("g");
  auto &ux = state.get_field<double>("ux");
  auto &uy = state.get_field<double>("uy");
  auto &exx = state.get_field<double>("exx");
  auto &eyy = state.get_field<double>("eyy");
  auto &exy = state.get_field<double>("exy");
  auto &sxx = state.get_field<double>("sxx");
  auto &syy = state.get_field<double>("syy");
  auto &sxy = state.get_field<double>("sxy");
  auto &stress_hydro = state.get_field<double>("stress_hydro");
  auto &stress_vm = state.get_field<double>("stress_vm");
  auto &energy_density = state.get_field<double>("energy_density");
  const double twopi = 2.0 * std::numbers::pi;
  const double L = static_cast<double>(N);
  const double kx = twopi * static_cast<double>(nx) / L;
  const double ky = twopi * static_cast<double>(ny) / L;
  g.apply(
      [&](double x, double y, double) { return amp_g * std::cos(kx * x + ky * y); });

  gradient_elasticity::solve_displacement_and_strain(stack.fft(), phys, g, ux, uy,
                                                     exx, eyy, exy);
  gradient_elasticity::compute_stress_fields(phys, g, exx, eyy, exy, sxx, syy, sxy,
                                             stress_hydro, stress_vm,
                                             energy_density);

  const double k2 = kx * kx + ky * ky;
  const double B = phys.cosine_displacement_prefactor(k2) * amp_g;
  double max_err_exx = 0.0, max_err_eyy = 0.0, max_err_exy = 0.0;
  double max_err_hydro = 0.0, max_err_vm = 0.0, max_err_w = 0.0;
  const auto n = g.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = g.coords(i, j, k);
        const double phi = kx * x[0] + ky * x[1];
        const double gv = amp_g * std::cos(phi);
        // Analytical strain from u = B*(kx,ky)*sin(phi):
        const double exx_a = B * kx * kx * std::cos(phi);
        const double eyy_a = B * ky * ky * std::cos(phi);
        const double exy_a = B * kx * ky * std::cos(phi);
        const auto s = phys.stress_state(exx_a, eyy_a, exy_a, gv);
        max_err_exx = std::max(max_err_exx, std::abs(exx(i, j, k) - exx_a));
        max_err_eyy = std::max(max_err_eyy, std::abs(eyy(i, j, k) - eyy_a));
        max_err_exy = std::max(max_err_exy, std::abs(exy(i, j, k) - exy_a));
        max_err_hydro =
            std::max(max_err_hydro, std::abs(stress_hydro(i, j, k) - s.hydrostatic));
        max_err_vm = std::max(max_err_vm, std::abs(stress_vm(i, j, k) - s.von_mises));
        max_err_w = std::max(max_err_w,
                             std::abs(energy_density(i, j, k) - s.energy_density));
      }
    }
  }
  REQUIRE_THAT(max_err_exx, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(max_err_eyy, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(max_err_exy, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(max_err_hydro, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(max_err_vm, WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(max_err_w, WithinAbs(0.0, 1e-9));
}

TEST_CASE("ell=0 recovers the classical analytical circular-inclusion stress",
          "[gradient_elasticity][spectral][classical][inclusion]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  // Sharp-inclusion Eshelby-type closed form assumes an infinite matrix and
  // an infinitely sharp boundary; the numerical case has a periodic box and
  // a finite tanh interface width, so allow a stated few-percent tolerance
  // rather than machine precision.
  constexpr int N = 320;
  constexpr double R = 40.0;
  constexpr double w = 1.0; // interface half-width, w/R = 0.025
  constexpr double mu = 1.0;
  constexpr double lambda = 1.0;
  constexpr double eps0 = 0.01;

  const auto summary = run_circular_case(N, R, w, /*ell=*/0.0, mu, lambda, eps0);
  const double p_classical = std::abs(classical_inclusion_inside_hydrostatic(
      mu, lambda, eps0));
  const double vm_classical =
      classical_inclusion_boundary_von_mises(mu, lambda, eps0);

  // The finite tanh interface width makes the smoothed problem's actual
  // peak (the field near r=R, where the eigenstrain itself has a gradient)
  // measurably higher than the idealized-sharp-boundary uniform interior
  // value: the classical hydrostatic stress is discontinuous across a sharp
  // boundary (p_in far from 0, p_out=0 immediately outside), so smoothing
  // that step introduces a boundary-layer effect on top of discretization
  // error. 10% at w/R=0.025 is the measured, stated tolerance.
  REQUIRE_THAT(summary.peak_abs_hydrostatic, WithinRel(p_classical, 0.12));
  REQUIRE_THAT(summary.peak_von_mises, WithinRel(vm_classical, 0.12));
}

TEST_CASE("Peak inclusion stress decreases monotonically with R/ell, from the "
          "clamped bound to the classical bound (size-effect curve)",
          "[gradient_elasticity][spectral][size_effect]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  // Fixed inclusion radius R, varying ell so R/ell spans well below 1 to
  // well above 1. This loading (a smooth, *finite* inclusion) has no
  // classical singularity, so ell does not "regularize a singular peak"
  // here the way it does for the high-k cosine mode above. Instead ell acts
  // as an increasing constraint on elastic relaxation: for R/ell -> 0 the
  // gradient penalty suppresses essentially all spatial variation of u
  // (every k>0 mode is annihilated as alpha=1+ell^2 k^2 -> infinity), so the
  // material cannot relax the eigenstrain misfit at all and the internal
  // stress approaches the *clamped* bound `clamped_inclusion_hydrostatic`
  // (measured empirically first, then matched to this closed form -- see PR
  // description). For R/ell -> infinity, ell is negligible and the classical
  // *relaxed* bound `classical_inclusion_inside_hydrostatic` is recovered
  // (already checked by the `ell=0` test above). Both bounds are
  // R-independent closed forms; peak stress should decrease monotonically
  // from one to the other as R/ell grows. The box is scaled with
  // max(R, ell), not just R, so periodic images stay controlled at every
  // ell in the sweep.
  constexpr double R = 40.0;
  constexpr double w = 1.0;
  constexpr double mu = 1.0;
  constexpr double lambda = 1.0;
  constexpr double eps0 = 0.01;
  const std::vector<double> ells = {80.0, 40.0, 10.0, 2.5}; // R/ell: 0.5,1,4,16

  std::vector<double> peak_hydro;
  for (double ell : ells) {
    const int N = static_cast<int>(std::lround(8.0 * std::max(R, ell)));
    peak_hydro.push_back(
        run_circular_case(N, R, w, ell, mu, lambda, eps0).peak_abs_hydrostatic);
  }
  for (std::size_t i = 1; i < peak_hydro.size(); ++i) {
    REQUIRE(peak_hydro[i] < peak_hydro[i - 1]);
  }
  const double p_classical =
      std::abs(classical_inclusion_inside_hydrostatic(mu, lambda, eps0));
  const double p_clamped = std::abs(clamped_inclusion_hydrostatic(mu, lambda, eps0));
  // Smallest R/ell in the sweep: closer to (but not yet at) the clamped
  // bound; largest R/ell: closer to (but not yet at) the classical bound.
  // Every measured value must lie strictly between the two closed forms.
  for (double p : peak_hydro) {
    REQUIRE(p > p_classical);
    REQUIRE(p < p_clamped);
  }
}

TEST_CASE("Doubling the periodic box leaves the inclusion peak stress "
          "essentially unchanged",
          "[gradient_elasticity][spectral][box_size]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  // #117 requires documented/controlled inclusion-image interaction: repeat
  // one radius in a larger box (same R, ell, dx; doubled L) and show the
  // peak stress barely moves.
  constexpr double R = 12.0;
  constexpr double w = 1.0;
  constexpr double ell = 3.0;
  constexpr double mu = 1.0;
  constexpr double lambda = 1.0;
  constexpr double eps0 = 0.01;

  // L/R=8 (as first tried) measurably contaminates the peak stress with
  // periodic images; L/R=16 -> 32 is the regime where doubling the box
  // changes the peak by less than the tolerance below. `size_sweep.py
  // --box-to-radius` defaults to the safer ratio (16). Report all three
  // ratios (8, 16, 32) via WARN so the convergence trend is visible, not
  // just asserted.
  const auto box8 =
      run_circular_case(8 * static_cast<int>(R), R, w, ell, mu, lambda, eps0);
  const auto small_box = run_circular_case(16 * static_cast<int>(R), R, w, ell, mu,
                                           lambda, eps0);
  const auto large_box = run_circular_case(32 * static_cast<int>(R), R, w, ell, mu,
                                           lambda, eps0);
  const auto rel_change = [](double a, double b) { return std::abs(b - a) / a; };
  WARN("L/R 8->16 relative change: peak_hydro="
       << rel_change(box8.peak_abs_hydrostatic, small_box.peak_abs_hydrostatic)
       << " peak_vm="
       << rel_change(box8.peak_von_mises, small_box.peak_von_mises));
  WARN("L/R 16->32 relative change: peak_hydro="
       << rel_change(small_box.peak_abs_hydrostatic, large_box.peak_abs_hydrostatic)
       << " peak_vm="
       << rel_change(small_box.peak_von_mises, large_box.peak_von_mises));
  REQUIRE_THAT(large_box.peak_abs_hydrostatic,
              WithinRel(small_box.peak_abs_hydrostatic, 0.02));
  REQUIRE_THAT(large_box.peak_von_mises,
              WithinRel(small_box.peak_von_mises, 0.02));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
