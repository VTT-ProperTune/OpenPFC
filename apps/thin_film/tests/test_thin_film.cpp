// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_thin_film.cpp
 * @brief Catch2 tests for the lubrication thin-film spectral app.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <numbers>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <thin_film/cosine_mode.hpp>
#include <openpfc_apps/spectral_flux.hpp>
#include <openpfc_apps/structure_factor.hpp>
#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_physics.hpp>
#include <thin_film/thin_film_session.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

double cosine_amplitude(const pfc::data::Field<double> &h, double h0, int nx,
                        int ny) {
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
        num += (h(i, j, k) - h0) * w;
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

void run_mode(thin_film::ThinFilmPhysics<> phys, int nx, double amp0, double dt,
              int n_steps, double &amp_end, double &mean_end, double &lambda) {
  constexpr int N = 32;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  phys.domain = domain;
  phys.box = stack.fft().get_inbox_bounds();
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &h = state.get_field<double>("h");
  const double h0 = phys.params.h0;
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  h.apply([&](double x, double, double) {
    return h0 + amp0 * std::cos(twopi * static_cast<double>(nx) * x / Lx);
  });
  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<thin_film::ThinFilmPhysics<>> sys(phys, stack.fft(),
                                                                state, dt, opt);
  const double k = twopi * static_cast<double>(nx) / Lx;
  lambda = phys.linear_symbol(-(k * k));
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  amp_end = cosine_amplitude(h, h0, nx, 0);
  mean_end = mean_h(h);
}

} // namespace

TEST_CASE("ThinFilm schema defaults are an unstable coating",
          "[thin_film][physics]") {
  thin_film::ThinFilmPhysics<> phys;
  REQUIRE_THAT(phys.params.h0, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(phys.params.A, WithinAbs(0.05, 1e-15));
  REQUIRE(phys.params.Pip0 > 0.0);
  REQUIRE(phys.k_peak_sq() > 0.0);
  REQUIRE_THAT(phys.params.Pip0,
               WithinAbs(6.0 * phys.params.A / phys.params.h0, 1e-12));
}

TEST_CASE("ThinFilm A=0 is leveling (k_peak vanishes)", "[thin_film][physics]") {
  thin_film::ThinFilmPhysics<> phys;
  thin_film::apply_thin_film_json({{"A", 0.0}}, phys.params);
  REQUIRE_THAT(phys.params.A, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.params.Pip0, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.k_peak_sq(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("ThinFilm L(k) matches -M0 gamma k_lap^2 - M0 Pi' k_lap",
          "[thin_film][physics][symbol]") {
  thin_film::ThinFilmPhysics<> phys;
  const double k_lap = -0.25;
  const double expected = -phys.params.M0 * phys.params.gamma * k_lap * k_lap -
                          phys.params.M0 * phys.params.Pip0 * k_lap;
  REQUIRE_THAT(phys.linear_symbol(k_lap), WithinAbs(expected, 1e-14));
  REQUIRE_THAT(phys.linear_symbol(0.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.nonlinear_symbol(k_lap),
               WithinAbs(-phys.params.M0 * k_lap, 1e-15));
}

TEST_CASE("ThinFilm n_nl vanishes at h0", "[thin_film][physics]") {
  const auto pw = thin_film::ThinFilmPhysics<>{}.pointwise();
  REQUIRE_THAT(pw.n_nl(pw.h0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ThinFilm ETD conserves volume and matches linear growth/decay",
          "[thin_film][spectral][volume][dispersion]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr double amp0 = 1.0e-4;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;
  thin_film::ThinFilmPhysics<> phys;

  double amp_g = 0.0;
  double mean_g = 0.0;
  double lam_g = 0.0;
  run_mode(phys, /*nx=*/2, amp0, dt, n_steps, amp_g, mean_g, lam_g);
  REQUIRE(lam_g > 0.0);
  REQUIRE_THAT(mean_g, WithinAbs(phys.params.h0, 1e-12));
  REQUIRE_THAT(amp_g, WithinRel(amp0 * std::exp(lam_g * dt * n_steps), 0.05));

  double amp_d = 0.0;
  double mean_d = 0.0;
  double lam_d = 0.0;
  run_mode(phys, /*nx=*/4, amp0, dt, n_steps, amp_d, mean_d, lam_d);
  REQUIRE(lam_d < 0.0);
  REQUIRE_THAT(mean_d, WithinAbs(phys.params.h0, 1e-12));
  REQUIRE_THAT(amp_d, WithinRel(amp0 * std::exp(lam_d * dt * n_steps), 0.05));
}

TEST_CASE("ThinFilm A=0 levels a cosine (capillary k^4 decay)",
          "[thin_film][spectral][leveling]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  thin_film::ThinFilmPhysics<> phys;
  thin_film::apply_thin_film_json({{"A", 0.0}}, phys.params);
  constexpr double amp0 = 0.02;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;
  double amp = 0.0;
  double mean = 0.0;
  double lam = 0.0;
  run_mode(phys, /*nx=*/2, amp0, dt, n_steps, amp, mean, lam);
  REQUIRE(lam < 0.0);
  REQUIRE_THAT(mean, WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(amp, WithinRel(amp0 * std::exp(lam * dt * n_steps), 0.05));
  REQUIRE(amp < amp0);
}

TEST_CASE("ThinFilmSession runs a short JSON dewetting case",
          "[thin_film][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  json settings = {
      {"model", {{"name", "thin_film"}, {"params", {{"h0", 1.0}, {"A", 0.05}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.2}, {"dt", 0.1}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "h"},
         {"type", "cosine_mode"},
         {"h0", 1.0},
         {"amplitude", 0.01},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
  thin_film::register_catalog();
  thin_film::ThinFilmSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.2).margin(1e-12));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}

// ---------------------------------------------------------------------------
// Nonlinear lubrication (#114)
// ---------------------------------------------------------------------------

TEST_CASE("Cubic mobility reduces to M0 at the reference thickness",
          "[thin_film][nonlinear]") {
  const thin_film::CubicMobility m{2.5, 1.3};
  REQUIRE_THAT(m(1.3), WithinRel(2.5, 1e-14));
  REQUIRE_THAT(m(2.6), WithinRel(2.5 * 8.0, 1e-14)); // (2h0)^3 = 8 M0
  REQUIRE_THAT(m(0.65), WithinRel(2.5 / 8.0, 1e-14));
  // A ruptured cell must still give a finite, non-negative mobility.
  REQUIRE(m(0.0) > 0.0);
  REQUIRE(m(-1.0) > 0.0);
}

TEST_CASE("Flux stepper reproduces the analytical k^4 decay",
          "[thin_film][nonlinear][flux]") {
  if (world_size() != 1) {
    SKIP("single-rank analytical comparison");
  }
  // The whole point of keeping the linear verifier: with a constant mobility
  // and a small perturbation the nonlinear flux solver must reproduce
  // exp(-M0 gamma k^4 t) exactly, or the flux path is wrong.
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double h0 = 1.0, gamma = 1.0, M0 = 1.0;
  constexpr double amp = 1.0e-6, dt = 0.01;
  constexpr int steps = 20;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  const double L = static_cast<double>(N);
  h.apply([&](double x, double, double) {
    return h0 + amp * std::cos(twopi * nx * x / L);
  });

  std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
  pfc::fft::kspace::for_each_kpoint(
      stack.fft().get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        k_lap[i] = -(kx * kx + ky * ky + kz * kz);
      });

  // A = 0, so Pi == 0 and p = -gamma lap h. Constant mobility M0.
  pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double kl) {
    return -M0 * gamma * kl * kl;
  });
  auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                       pfc::data::Field<double> &, 
                       pfc::data::Field<std::complex<double>> &out) {
    h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
      out.with_host_view([&](std::complex<double> *o, std::size_t) {
        for (std::size_t i = 0; i < m; ++i) o[i] = -gamma * k_lap[i] * hv[i];
      });
    });
  };
  auto constant_mobility = [=](double) { return M0; };

  double t = 0.0;
  for (int s = 0; s < steps; ++s)
    t = stepper.step(t, h, potential, constant_mobility);

  const double k = twopi * nx / L;
  const double expected = amp * std::exp(-M0 * gamma * k * k * k * k * t);
  // Project onto the mode.
  double num = 0.0, den = 0.0;
  const auto n = h.local_size();
  for (int j = 0; j < n[1]; ++j)
    for (int i = 0; i < n[0]; ++i) {
      const auto x = h.coords(i, j, 0);
      const double w = std::cos(twopi * nx * x[0] / L);
      num += (h(i, j, 0) - h0) * w;
      den += w * w;
    }
  REQUIRE_THAT(num / den, WithinRel(expected, 1e-8));
}

TEST_CASE("Cubic mobility conserves liquid volume", "[thin_film][nonlinear]") {
  if (world_size() != 1) {
    SKIP("single-rank conservation check");
  }
  constexpr int N = 32;
  constexpr double h0 = 1.0, dt = 0.005;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h = stack.u();
  const double twopi = 2.0 * std::numbers::pi;
  h.apply([&](double x, double y, double) {
    return h0 * (1.0 + 0.2 * std::cos(twopi * 2 * x / N) *
                           std::cos(twopi * 2 * y / N));
  });

  std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
  pfc::fft::kspace::for_each_kpoint(
      stack.fft().get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        k_lap[i] = -(kx * kx + ky * ky + kz * kz);
      });

  pfc::apps::FluxETD stepper(domain, stack.fft(), dt,
                             [=](double kl) { return -kl * kl; });
  auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                       pfc::data::Field<double> &,
                       pfc::data::Field<std::complex<double>> &out) {
    h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
      out.with_host_view([&](std::complex<double> *o, std::size_t) {
        for (std::size_t i = 0; i < m; ++i) o[i] = -k_lap[i] * hv[i];
      });
    });
  };
  const thin_film::CubicMobility mobility{1.0, h0};

  const auto v0 = thin_film::sample_film(h, domain, h0, 0.05, MPI_COMM_WORLD);
  double t = 0.0;
  for (int s = 0; s < 40; ++s) t = stepper.step(t, h, potential, mobility);
  const auto v1 = thin_film::sample_film(h, domain, h0, 0.05, MPI_COMM_WORLD);

  // A divergence form conserves the integral to round-off, whatever M does.
  REQUIRE_THAT(v1.volume, WithinRel(v0.volume, 1e-11));
  REQUIRE(v1.max_h != v0.max_h); // and the profile did evolve
}

TEST_CASE("Gaussian defect is a localized depression", "[thin_film][nonlinear]") {
  const auto domain = pfc::domain::create(pfc::GridSize({64, 64, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto d = thin_film::GaussianDefect::centred(domain, 0.2, 4.0);
  REQUIRE_THAT(d(32.0, 32.0), WithinRel(-0.2, 1e-12)); // deepest at the centre
  REQUIRE(std::abs(d(0.0, 0.0)) < 1e-6);               // and local
  const thin_film::GaussianDefect none{};
  REQUIRE(none(1.0, 2.0) == 0.0);
}
