// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_ehd_film.cpp
 * @brief Catch2 tests for the EHD flexible-plate spectral app.
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

#include <ehd_film/ehd_film_physics.hpp>
#include <ehd_film/ehd_film_session.hpp>
#include <ehd_film/nonlinear.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <openpfc_apps/spectral_flux.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

double cosine_amplitude(const pfc::data::Field<double> &h, double h0, int nx) {
  const auto n = h.local_size();
  const auto sp = h.spacing();
  const double Lx = static_cast<double>(n[0]) * sp[0];
  const double twopi = 2.0 * std::numbers::pi;
  double num = 0.0;
  double den = 0.0;
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto x = h.coords(i, j, k);
        const double w = std::cos(twopi * static_cast<double>(nx) * x[0] / Lx);
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

} // namespace

TEST_CASE("EhdFilm L(k) is M0 B k_lap^3 minus lower-order terms",
          "[ehd_film][physics][symbol]") {
  ehd_film::EhdFilmPhysics<> phys;
  REQUIRE_THAT(phys.params.B, WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(phys.params.gamma, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.params.A, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.params.Pip0, WithinAbs(0.0, 1e-15));
  const double k_lap = -0.25;
  const double k2 = k_lap * k_lap;
  REQUIRE_THAT(phys.linear_symbol(k_lap),
               WithinAbs(phys.params.M0 * phys.params.B * k_lap * k2, 1e-15));
  REQUIRE_THAT(phys.linear_symbol(0.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(phys.k_peak_sq(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("EhdFilm ETD matches exp(-M0 B k^6 t) and conserves mean h",
          "[ehd_film][spectral][volume]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double amp0 = 0.05;
  constexpr double dt = 0.05;
  constexpr int n_steps = 8;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = ehd_film::EhdFilmPhysics<>::from_json(json::object(), domain,
                                                    stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &h = state.get_field<double>("h");
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  const double h0 = phys.params.h0;
  h.apply([&](double x, double, double) {
    return h0 + amp0 * std::cos(twopi * static_cast<double>(nx) * x / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<ehd_film::EhdFilmPhysics<>> sys(phys, stack.fft(),
                                                              state, dt, opt);
  const double k = twopi * static_cast<double>(nx) / Lx;
  const double k2 = k * k;
  const double lambda = -phys.params.M0 * phys.params.B * k2 * k2 * k2;
  REQUIRE_THAT(phys.linear_symbol(-k2), WithinAbs(lambda, 1e-14));

  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  REQUIRE_THAT(mean_h(h), WithinAbs(h0, 1e-12));
  REQUIRE_THAT(cosine_amplitude(h, h0, nx),
               WithinRel(amp0 * std::exp(lambda * t), 1.0e-8));
}

TEST_CASE("EhdFilm two-mode decay rates scale as k^6", "[ehd_film][spectral][k6]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  constexpr int N = 32;
  constexpr double a1 = 0.04;
  constexpr double a2 = 0.03;
  constexpr double dt = 0.02;
  constexpr int n_steps = 6;

  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto phys = ehd_film::EhdFilmPhysics<>::from_json(json::object(), domain,
                                                    stack.fft().get_inbox_bounds());
  pfc::SimulationState state;
  phys.declare_fields(state);
  auto &h = state.get_field<double>("h");
  const double twopi = 2.0 * std::numbers::pi;
  const double Lx = static_cast<double>(N);
  const double h0 = phys.params.h0;
  h.apply([&](double x, double, double) {
    return h0 + a1 * std::cos(twopi * 2.0 * x / Lx) +
           a2 * std::cos(twopi * 4.0 * x / Lx);
  });

  pfc::sim::SpectralETDOptions opt;
  opt.psi_name = "h";
  opt.dealias = true;
  pfc::sim::SpectralETDSystem<ehd_film::EhdFilmPhysics<>> sys(phys, stack.fft(),
                                                              state, dt, opt);
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    t = sys.step(t);
  }
  const double r1 = cosine_amplitude(h, h0, 2) / a1;
  const double r2 = cosine_amplitude(h, h0, 4) / a2;
  REQUIRE(r1 > 0.0);
  REQUIRE(r2 > 0.0);
  const double rate1 = -std::log(r1) / t;
  const double rate2 = -std::log(r2) / t;
  REQUIRE_THAT(rate2 / rate1, WithinAbs(64.0, 1.0e-4));
}

TEST_CASE("EhdFilm A>0 is unstable at low k and damped at high k",
          "[ehd_film][spectral][spinodal]") {
  if (world_size() != 1) {
    SKIP("single-rank spectral comparison");
  }
  ehd_film::EhdFilmPhysics<> phys;
  ehd_film::apply_ehd_film_json({{"A", 0.05}}, phys.params);
  REQUIRE(phys.params.Pip0 > 0.0);
  REQUIRE(phys.k_peak_sq() > 0.0);
  const double k_lo = 0.2;
  const double k_hi = 2.0;
  REQUIRE(phys.linear_symbol(-(k_lo * k_lo)) > 0.0);
  REQUIRE(phys.linear_symbol(-(k_hi * k_hi)) < 0.0);
}

TEST_CASE("EhdFilmSession runs a short JSON case", "[ehd_film][session]") {
  if (world_size() != 1) {
    SKIP("single-rank session smoke");
  }
  json settings = {
      {"model", {{"name", "ehd_film"}, {"params", {{"B", 1.0}, {"A", 0.0}}}}},
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
         {"h0", 1.0},
         {"amplitude", 0.05},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
  ehd_film::register_catalog();
  ehd_film::EhdFilmSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(pfc::time::current(session.time()) == Catch::Approx(0.1).margin(1e-12));
}

// ---------------------------------------------------------------------------
// Nonlinear compliant lubrication (#116)
// ---------------------------------------------------------------------------

TEST_CASE("EHD cubic mobility reduces to M0 at the reference thickness",
          "[ehd_film][nonlinear]") {
  const ehd_film::CubicMobility m{2.5, 1.3};
  REQUIRE_THAT(m(1.3), WithinRel(2.5, 1e-14));
  REQUIRE_THAT(m(2.6), WithinRel(2.5 * 8.0, 1e-14)); // (2h0)^3 = 8 M0
  REQUIRE_THAT(m(0.65), WithinRel(2.5 / 8.0, 1e-14));
  // A plate pinned near zero gap must still give a finite, positive mobility.
  REQUIRE(m(0.0) > 0.0);
  REQUIRE(m(-1.0) > 0.0);
}

TEST_CASE("EHD flux stepper reproduces the analytical k^6 decay",
          "[ehd_film][nonlinear][flux]") {
  if (world_size() != 1) {
    SKIP("single-rank analytical comparison");
  }
  // The point of keeping the linear k^6 verifier: with a constant mobility,
  // pure bending (gamma=A=0) and a small perturbation, the nonlinear flux
  // solver must reproduce exp(-M0 B k^6 t) exactly, or the flux path -- and
  // not just the linear ETD symbol -- is wrong.
  constexpr int N = 32;
  constexpr int nx = 2;
  constexpr double h0 = 1.0, B = 1.0, M0 = 1.0;
  constexpr double amp = 1.0e-6, dt = 0.02;
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

  // gamma = A = 0, so p = B lap^2 h. Constant mobility M0.
  pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double kl) {
    return M0 * B * kl * kl * kl;
  });
  auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                       pfc::data::Field<double> &,
                       pfc::data::Field<std::complex<double>> &out) {
    h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
      out.with_host_view([&](std::complex<double> *o, std::size_t) {
        for (std::size_t i = 0; i < m; ++i) o[i] = B * k_lap[i] * k_lap[i] * hv[i];
      });
    });
  };
  auto constant_mobility = [=](double) { return M0; };

  double t = 0.0;
  for (int s = 0; s < steps; ++s)
    t = stepper.step(t, h, potential, constant_mobility);

  const double k = twopi * nx / L;
  const double k6 = k * k * k * k * k * k;
  const double expected = amp * std::exp(-M0 * B * k6 * t);
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

TEST_CASE("EHD cubic mobility conserves liquid volume under bending, tension "
          "and adhesion",
          "[ehd_film][nonlinear]") {
  if (world_size() != 1) {
    SKIP("single-rank conservation check");
  }
  // Exercises Pi(h) through the flux path, not just B and gamma: the
  // perturbation is bounded well away from h=0 (h in [0.8h0, 1.2h0]) so the
  // default (no h_star) disjoining form -- known stiff near h -> 0 -- stays
  // smooth here, per the thin_film #114 experience with this potential.
  constexpr int N = 32;
  constexpr double h0 = 1.0, B = 1.0, gamma = 0.1, A = 0.05, dt = 0.01;
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

  const ehd_film::EhdFilmPointwise pw{.A = A, .h0 = h0};
  const double Pip0 = pw.Pi_prime(h0);

  std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
  pfc::fft::kspace::for_each_kpoint(
      stack.fft().get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        k_lap[i] = -(kx * kx + ky * ky + kz * kz);
      });

  pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double kl) {
    return B * kl * kl * kl - gamma * kl * kl - Pip0 * kl;
  });
  pfc::data::Field<double> pi_real(domain, stack.fft().get_inbox_bounds(), 0);
  pfc::data::Field<std::complex<double>> pi_hat(
      domain, stack.fft().get_outbox_bounds(), 0);
  auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                       pfc::data::Field<double> &hh,
                       pfc::data::Field<std::complex<double>> &out) {
    hh.with_host_view([&](double *hv, std::size_t cnt) {
      pi_real.with_host_view([&](double *pv, std::size_t) {
        for (std::size_t i = 0; i < cnt; ++i) pv[i] = pw.Pi(hv[i]);
      });
    });
    pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), pi_real, pi_hat);
    h_hat.with_host_view([&](std::complex<double> *hv, std::size_t m) {
      pi_hat.with_host_view([&](std::complex<double> *piv, std::size_t) {
        out.with_host_view([&](std::complex<double> *o, std::size_t) {
          for (std::size_t i = 0; i < m; ++i)
            o[i] = B * k_lap[i] * k_lap[i] * hv[i] - gamma * k_lap[i] * hv[i] -
                   piv[i];
        });
      });
    });
  };
  const ehd_film::CubicMobility mobility{1.0, h0};

  pfc::data::Field<double> p_dummy(domain, stack.fft().get_inbox_bounds(), 0);
  const auto v0 = ehd_film::sample_ehd_film(h, p_dummy, domain, h0, MPI_COMM_WORLD);
  double t = 0.0;
  for (int s = 0; s < 40; ++s) t = stepper.step(t, h, potential, mobility);
  const auto v1 = ehd_film::sample_ehd_film(h, p_dummy, domain, h0, MPI_COMM_WORLD);

  // A divergence form conserves the integral to round-off, whatever M does.
  REQUIRE_THAT(v1.volume, WithinRel(v0.volume, 1e-11));
  REQUIRE(v1.h_center != v0.h_center); // and the profile did evolve
}

TEST_CASE("EHD localized load thins the gap at its centre and conserves volume",
          "[ehd_film][nonlinear][load]") {
  if (world_size() != 1) {
    SKIP("single-rank sign-convention check");
  }
  // Documents the sign in ehd_film/nonlinear.hpp: p_ext > 0 is a load
  // pressing the plate down, so a positive Gaussian load must thin the gap
  // at its centre (deflection > 0) and thicken it in the surrounding
  // annulus, while the divergence-form flux still conserves total volume --
  // this holds regardless of B, gamma, A because p_ext only ever enters
  // inside grad(p).
  constexpr int N = 64;
  constexpr double h0 = 1.0, M0 = 1.0, dt = 0.02;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h = stack.u();
  h.apply([&](double, double, double) { return h0; });

  // No bending, tension or adhesion here: isolate the load's own sign.
  pfc::apps::FluxETD stepper(domain, stack.fft(), dt,
                             [](double) { return 0.0; });
  const auto load = ehd_film::GaussianLoad::centred(domain, /*p0=*/0.5,
                                                     /*a=*/6.0, /*t_load=*/1.0e9);
  pfc::data::Field<double> pext_real(domain, stack.fft().get_inbox_bounds(), 0);
  auto potential = [&](pfc::data::Field<std::complex<double>> &,
                       pfc::data::Field<double> &,
                       pfc::data::Field<std::complex<double>> &out) {
    const auto n = pext_real.local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) {
          const auto x = pext_real.coords(i, j, k);
          pext_real(i, j, k) = load(x[0], x[1], 0.0);
        }
    pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), pext_real, out);
  };
  const ehd_film::CubicMobility mobility{M0, h0};

  pfc::data::Field<double> p_dummy(domain, stack.fft().get_inbox_bounds(), 0);
  const auto before = ehd_film::sample_ehd_film(h, p_dummy, domain, h0, MPI_COMM_WORLD);
  double t = 0.0;
  for (int s = 0; s < 10; ++s) t = stepper.step(t, h, potential, mobility);
  const auto after = ehd_film::sample_ehd_film(h, p_dummy, domain, h0, MPI_COMM_WORLD);

  REQUIRE(after.h_center < before.h_center); // gap thins under the load
  REQUIRE_THAT(after.volume, WithinRel(before.volume, 1e-10));
}

TEST_CASE("GaussianLoad turns off at t_load", "[ehd_film][nonlinear]") {
  const auto domain = pfc::domain::create(pfc::GridSize({32, 32, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto load = ehd_film::GaussianLoad::centred(domain, 0.5, 4.0, 2.0);
  REQUIRE_THAT(load(16.0, 16.0, 0.0), WithinRel(0.5, 1e-12)); // on, peak
  REQUIRE(load(16.0, 16.0, 2.0) == 0.0);                      // off at t_load
  REQUIRE(load(16.0, 16.0, 5.0) == 0.0);                      // stays off
  const ehd_film::GaussianLoad none{};
  REQUIRE(none(1.0, 2.0, 0.0) == 0.0);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
