// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_surface_diffusion.cpp
 * @brief Catch2 tests for Mullins surface-diffusion spectral app.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <numbers>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <surface_diffusion/surface_diffusion_physics.hpp>
#include <surface_diffusion/surface_diffusion_session.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

double cosine_amplitude(const pfc::data::Field<double> &h, int nx, int ny) {
  const auto n = h.local_size();
  const auto sp = h.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double Ly = static_cast<double>(n[1]) * sp[1];
  const double twopi = 2.0 * std::numbers::pi;
  double num = 0.0;
  double den = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = h.coords(i, j, k);
        const double w = std::cos(twopi * (static_cast<double>(nx) * x[0] / Lx +
                                           static_cast<double>(ny) * x[1] / Ly));
        num += h(i, j, k) * w;
        den += w * w;
      }
    }
  }
  return (den > 0.0) ? num / den : 0.0;
}

double mean_h(const pfc::data::Field<double> &h) {
  double sum = 0.0;
  std::size_t count = 0;
  const auto n = h.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        sum += h(i, j, k);
        ++count;
      }
    }
  }
  return (count > 0) ? sum / static_cast<double>(count) : 0.0;
}

} // namespace

TEST_CASE("SurfaceDiffusion L(k) is -B k_lap^2",
          "[surface_diffusion][physics][symbol]") {
  surface_diffusion::SurfaceDiffusionPhysics<> phys;
  REQUIRE_THAT(phys.params.B, WithinAbs(1.0, 1e-15));
  const double k_lap = -0.25;
  REQUIRE_THAT(phys.linear_symbol(k_lap),
               WithinAbs(-phys.params.B * k_lap * k_lap, 1e-15));
  REQUIRE_THAT(phys.linear_symbol(0.0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("SurfaceDiffusion ETD matches exp(-B k^4 t) and conserves mean h",
          "[surface_diffusion][spectral][volume]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double amp0 = 0.1;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = surface_diffusion::SurfaceDiffusionPhysics<>::from_json(
      json::object(), domain, stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &h = state.get_field<double>("h");
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  h.apply([&](double x, double, double) {
    return amp0 * std::cos(twopi * static_cast<double>(nx) * x / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  pfc::sim::SpectralETDSystem<surface_diffusion::SurfaceDiffusionPhysics<>> sys(
      phys, stack.fft(), state, dt, opt);

  const double k = twopi * static_cast<double>(nx) / Lx;
  const double lambda = -phys.params.B * k * k * k * k;
  REQUIRE_THAT(phys.linear_symbol(-(k * k)), WithinAbs(lambda, 1e-14));

  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE_THAT(mean_h(h), WithinAbs(0.0, 1e-12));
  const double amp = cosine_amplitude(h, nx, 0);
  REQUIRE_THAT(amp, WithinRel(amp0 * std::exp(lambda * t), 1.0e-8));
}

TEST_CASE("SurfaceDiffusion two-mode decay rates scale as k^4",
          "[surface_diffusion][spectral][k4]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr double a1 = 0.08;
  constexpr double a2 = 0.05;
  constexpr double dt = 0.02;
  constexpr int n_steps = 6;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = surface_diffusion::SurfaceDiffusionPhysics<>::from_json(
      json::object(), domain, stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &h = state.get_field<double>("h");
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  h.apply([&](double x, double, double) {
    return a1 * std::cos(twopi * 1.0 * x / Lx) + a2 * std::cos(twopi * 2.0 * x / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  pfc::sim::SpectralETDSystem<surface_diffusion::SurfaceDiffusionPhysics<>> sys(
      phys, stack.fft(), state, dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  const double r1 = cosine_amplitude(h, 1, 0) / a1;
  const double r2 = cosine_amplitude(h, 2, 0) / a2;
  REQUIRE(r1 > 0.0);
  REQUIRE(r2 > 0.0);
  const double rate1 = -std::log(r1) / t;
  const double rate2 = -std::log(r2) / t;
  REQUIRE_THAT(rate2 / rate1, WithinAbs(16.0, 1.0e-6));
}

TEST_CASE("SurfaceDiffusionSession runs a short JSON case",
          "[surface_diffusion][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  json settings = {
      {"model", {{"name", "surface_diffusion"}, {"params", {{"B", 1.0}}}}},
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
       {{{"target", "h"},
         {"type", "cosine_mode"},
         {"h0", 0.0},
         {"amplitude", 0.05},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
  surface_diffusion::register_catalog();
  surface_diffusion::SurfaceDiffusionSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.1).margin(1e-12));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
