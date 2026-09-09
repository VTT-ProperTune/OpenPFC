// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_surface_diffusion.cpp
 * @brief Catch2 tests for Mullins surface-diffusion spectral app.
 *
 * The isotropic exact-decay tests below (`L(k) is -B k_lap^2`, `ETD matches
 * exp(-B k^4 t)`, `two-mode decay rates scale as k^4`, `Session runs a short
 * JSON case`) are the `#115` numerical oracle and are unchanged by the
 * anisotropic science case added lower in this file (`AnisotropicSurface*`
 * test cases): the roadmap rule is "keep the verifier, add the science case
 * on top of it."
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
#include <surface_diffusion/anisotropic_flux.hpp>
#include <surface_diffusion/anisotropy.hpp>
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

// ---------------------------------------------------------------------------
// Anisotropic surface-diffusion science case (#115). The isotropic tests
// above are untouched and remain the numerical oracle.
// ---------------------------------------------------------------------------

TEST_CASE("SurfaceStiffness reduces exactly to B0 when eps_a = 0",
          "[surface_diffusion][anisotropy]") {
  const surface_diffusion::SurfaceStiffness B{/*B0=*/1.7, /*eps_a=*/0.0, /*m=*/4};
  for (double theta : {-std::numbers::pi, -1.3, -0.5, 0.0, 0.7, 1.9,
                       std::numbers::pi}) {
    REQUIRE_THAT(B(theta), WithinAbs(1.7, 1e-15));
  }
}

TEST_CASE("SurfaceStiffness has the symmetry it claims",
          "[surface_diffusion][anisotropy]") {
  for (int m : {4, 6}) {
    const surface_diffusion::SurfaceStiffness B{/*B0=*/1.0, /*eps_a=*/0.6, m};
    const double period = 2.0 * std::numbers::pi / static_cast<double>(m);
    for (double theta : {-2.7, -0.9, 0.0, 0.4, 1.1, 2.3}) {
      REQUIRE_THAT(B(theta), WithinAbs(B(theta + period), 1e-13));
    }
    // m=4 additionally makes theta=0 and theta=pi/2 equivalent; m=6 does not
    // -- this is exactly why the science preset below uses m=6 for the
    // crossed x/y corrugation (see surface_diffusion_anisotropic.cpp).
    if (m == 4) {
      REQUIRE_THAT(B(0.0), WithinAbs(B(std::numbers::pi / 2.0), 1e-13));
    } else {
      REQUIRE(std::abs(B(0.0) - B(std::numbers::pi / 2.0)) > 0.1);
    }
  }
}

namespace {

/// One `AnisotropicSurfaceDiffusionETD` run of a single cosine mode; returns
/// the measured decay rate `-log(amp(t)/amp(0))/t`.
double anisotropic_single_mode_rate(int nx, int ny,
                                    const surface_diffusion::SurfaceStiffness &B,
                                    int n_steps, double dt) {
  constexpr int N = 32;
  constexpr double amp0 = 0.05;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  h.apply([&](double x, double y, double) {
    return amp0 * std::cos(twopi * (static_cast<double>(nx) * x / Lx +
                                    static_cast<double>(ny) * y / Lx));
  });
  surface_diffusion::AnisotropicSurfaceDiffusionETD stepper(domain, stack.fft(), dt,
                                                            B);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) t = stepper.step(t, h);
  const double amp = cosine_amplitude(h, nx, ny);
  REQUIRE(amp > 0.0);
  return -std::log(amp / amp0) / t;
}

} // namespace

TEST_CASE("AnisotropicSurfaceDiffusionETD reduces to isotropic k^4 decay at "
          "eps_a = 0",
          "[surface_diffusion][anisotropy][spectral]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr double B0 = 1.0;
  const surface_diffusion::SurfaceStiffness B{B0, /*eps_a=*/0.0, /*m=*/6};
  const double rate = anisotropic_single_mode_rate(2, 0, B, 8, 0.02);
  const double twopi = 2.0 * std::numbers::pi;
  const double k = twopi * 2.0 / 32.0;
  const double expected = B0 * k * k * k * k;
  REQUIRE_THAT(rate, WithinRel(expected, 1.0e-6));
}

TEST_CASE("AnisotropicSurfaceDiffusionETD: single-orientation decay matches "
          "B(theta)",
          "[surface_diffusion][anisotropy][spectral]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr double B0 = 1.0;
  constexpr double eps_a = 0.5;
  constexpr int m = 6;
  const surface_diffusion::SurfaceStiffness B{B0, eps_a, m};

  // x-only ridges: grad h is purely in x, theta = 0 (mod pi) a.e., so
  // B(theta) = B0*(1+eps_a) is the constant effective stiffness -- the
  // orientation-dependent remainder handed to the stepper's *explicit* part
  // is then spatially uniform, but that remainder is still only integrated
  // to first order in dt (only the eps_a=0 part of L(k) is exponentiated
  // exactly), so the measured rate matches B(theta)*k^4 to O(dt), not to
  // machine precision; WithinRel below is loose enough to accommodate that
  // and still catch a wrong-sign or wrong-magnitude anisotropy.
  const double rate_x = anisotropic_single_mode_rate(2, 0, B, 8, 0.02);
  // y-only ridges: theta = +-pi/2 a.e., cos(6*pi/2) = -1, so
  // B(theta) = B0*(1-eps_a).
  const double rate_y = anisotropic_single_mode_rate(0, 2, B, 8, 0.02);

  const double twopi = 2.0 * std::numbers::pi;
  const double k = twopi * 2.0 / 32.0;
  const double k4 = k * k * k * k;
  REQUIRE_THAT(rate_x, WithinRel(B0 * (1.0 + eps_a) * k4, 5.0e-3));
  REQUIRE_THAT(rate_y, WithinRel(B0 * (1.0 - eps_a) * k4, 5.0e-3));

  // The faceting/orientation-selection claim of #115, made quantitative: the
  // stiffer (x) orientation relaxes measurably faster than the softer (y)
  // orientation under the same sixfold anisotropy.
  REQUIRE(rate_x > 1.4 * rate_y);
}

TEST_CASE("AnisotropicSurfaceDiffusionETD conserves mean height",
          "[surface_diffusion][anisotropy][volume]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  const double h0 = 0.37;
  h.apply([&](double x, double y, double) {
    return h0 + 0.05 * std::cos(twopi * 2.0 * x / Lx) +
          0.05 * std::cos(twopi * 3.0 * y / Lx);
  });
  REQUIRE_THAT(mean_h(h), WithinAbs(h0, 1e-10));

  const surface_diffusion::SurfaceStiffness B{1.0, 0.6, 6};
  surface_diffusion::AnisotropicSurfaceDiffusionETD stepper(domain, stack.fft(), 0.02,
                                                            B);
  double t = 0.0;
  for (int step = 0; step < 20; ++step) t = stepper.step(t, h);
  REQUIRE_THAT(mean_h(h), WithinAbs(h0, 1e-9));
}

TEST_CASE("AnisotropicSurfaceDiffusionETD selects orientation on a crossed "
          "corrugation",
          "[surface_diffusion][anisotropy][faceting]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int n = 3;
  constexpr double amp0 = 0.04;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  auto crossed_ic = [&](double x, double y, double) {
    return amp0 * std::cos(twopi * n * x / Lx) + amp0 * std::cos(twopi * n * y / Lx);
  };

  auto run = [&](const surface_diffusion::SurfaceStiffness &B) {
    pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
    auto &h = stack.u();
    h.apply(crossed_ic);
    surface_diffusion::AnisotropicSurfaceDiffusionETD stepper(domain, stack.fft(),
                                                              0.02, B);
    double t = 0.0;
    for (int step = 0; step < 60; ++step) t = stepper.step(t, h);
    return cosine_amplitude(h, n, 0) / cosine_amplitude(h, 0, n);
  };

  const double ratio_isotropic = run({1.0, 0.0, 6});
  const double ratio_anisotropic = run({1.0, 0.5, 6});

  // Isotropic dynamics is linear (no real-space coupling between the two
  // orthogonal ridge sets), so this ratio holds exactly. The anisotropic run
  // is a genuinely nonlinear, mode-coupled evolution: even though the
  // isolated-orientation rates above differ by a factor ~3, the two ridge
  // sets are present *simultaneously* here and each one's local B(theta)
  // depends on the combined gradient, not on that mode alone, which dilutes
  // (but does not remove) the orientation-selection signal relative to the
  // naive single-mode estimate. The threshold below is set from the
  // measured, reproducible effect at this resolution/time, not derived
  // analytically -- unlike the single-mode test above.
  REQUIRE_THAT(ratio_isotropic, WithinAbs(1.0, 1.0e-6));
  REQUIRE(std::abs(ratio_anisotropic - 1.0) > 0.01);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
