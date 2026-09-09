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

#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/kernel/simulation/time.hpp>
#include <thin_film/cosine_mode.hpp>
#include <openpfc_apps/spectral_flux.hpp>
#include <openpfc_apps/structure_factor.hpp>
#include <thin_film/fd_flux.hpp>
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

// ---------------------------------------------------------------------------
// Conservative face-flux FD solver (#124): the two-method flagship.
// ---------------------------------------------------------------------------

namespace {

/// Common single-rank FD fixture: `N x N`, `dx = 1`, decomposition of one.
struct FDFixture {
  static constexpr int N = 32;
  pfc::Domain domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                           pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                           pfc::GridSpacing({1.0, 1.0, 1.0}));
  pfc::decomposition::Decomposition decomp = pfc::decomposition::create(domain, 1);
  pfc::Box3i box = pfc::decomposition::local_box(decomp, 0);
  int nx = box.size[0];
  int ny = box.size[1];
};

std::vector<double> cosine_field(const FDFixture &fx, double h0, double amp, int mx,
                                 int my) {
  std::vector<double> h(static_cast<std::size_t>(fx.nx) *
                        static_cast<std::size_t>(fx.ny));
  const double twopi = 2.0 * std::numbers::pi;
  for (int iy = 0; iy < fx.ny; ++iy) {
    for (int ix = 0; ix < fx.nx; ++ix) {
      const double w = std::cos(twopi * mx * ix / fx.N) * std::cos(twopi * my * iy / fx.N);
      h[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * fx.nx] =
          h0 * (1.0 + amp * w);
    }
  }
  return h;
}

double sum_vec(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}

} // namespace

TEST_CASE("FD flux divergence sums to zero to round-off, before any timestep",
          "[thin_film][fd][mass]") {
  if (world_size() != 1) {
    SKIP("single-rank FD fixture");
  }
  FDFixture fx;
  auto h = cosine_field(fx, /*h0=*/1.0, /*amp=*/0.2, /*mx=*/3, /*my=*/2);

  thin_film::ThinFilmParams p; // defaults: h0=1, gamma=1, M0=1, A=0.05
  const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0, .h_star = p.h_star,
                                        .Pi0 = p.Pi0, .Pip0 = p.Pip0};
  const thin_film::CubicMobility mobility{p.M0, p.h0};

  thin_film::FDFluxSolver solver(fx.domain, fx.decomp, 0, MPI_COMM_WORLD, h,
                                 /*order=*/2);
  std::vector<double> dhdt(h.size());
  solver.compute_rhs(h, p.gamma, pw, mobility, thin_film::FaceMobility::Harmonic,
                     dhdt);

  // The divergence form telescopes exactly: every interior face flux is
  // added once and subtracted once, so the sum over all cells is zero to
  // round-off *before any timestep is taken* -- this is a statement about
  // the spatial discretization, not about dt.
  REQUIRE_THAT(sum_vec(dhdt), WithinAbs(0.0, 1e-10));

  // And volume stays put over many explicit steps, whatever dt is (subject
  // to the scheme's own stability limit).
  const auto v0 = thin_film::sample_film_fd(h, fx.domain, p.h0, 0.05, MPI_COMM_WORLD);
  for (int s = 0; s < 50; ++s) {
    solver.step(h, /*dt=*/1.0e-3, p.gamma, pw, mobility,
               thin_film::FaceMobility::Harmonic);
  }
  const auto v1 = thin_film::sample_film_fd(h, fx.domain, p.h0, 0.05, MPI_COMM_WORLD);
  REQUIRE_THAT(v1.volume, WithinRel(v0.volume, 1.0e-9));
  REQUIRE(v1.max_h != v0.max_h); // and the profile did evolve
}

TEST_CASE("FD reproduces the analytical k^4 decay at constant mobility",
          "[thin_film][fd][dispersion]") {
  if (world_size() != 1) {
    SKIP("single-rank analytical comparison");
  }
  // Bypass CubicMobility with a constant-mobility lambda, exactly as the
  // spectral test does -- this isolates the FD curvature/flux operators
  // from the h^3 nonlinearity, so any mismatch against exp(-M0 gamma k^4 t)
  // is discretization error in those operators, not the nonlinearity.
  FDFixture fx;
  constexpr double h0 = 1.0, gamma = 1.0, M0 = 1.0;
  constexpr double amp = 1.0e-4, dt = 1.0e-3;
  constexpr int steps = 200;
  constexpr int mode = 2;
  auto h = cosine_field(fx, h0, amp, mode, 0);

  const thin_film::ThinFilmPointwise pw{.A = 0.0, .h0 = h0}; // A=0: Pi == 0
  const auto constant_mobility = [](double) { return M0; };

  thin_film::FDFluxSolver solver(fx.domain, fx.decomp, 0, MPI_COMM_WORLD, h,
                                 /*order=*/2);
  for (int s = 0; s < steps; ++s) {
    solver.step(h, dt, gamma, pw, constant_mobility, thin_film::FaceMobility::Harmonic);
  }

  const double twopi = 2.0 * std::numbers::pi;
  const double k = twopi * mode / fx.N;
  const double expected = amp * std::exp(-M0 * gamma * k * k * k * k * dt * steps);
  double num = 0.0, den = 0.0;
  for (int iy = 0; iy < fx.ny; ++iy) {
    for (int ix = 0; ix < fx.nx; ++ix) {
      const double w = std::cos(twopi * mode * ix / fx.N);
      num += (h[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * fx.nx] -
             h0) *
             w;
      den += w * w;
    }
  }
  // Order-2 central differences on a coarse grid (32 points, wavelength 16
  // points) carry an O((k dx)^2) discretization error the spectral method
  // does not have; 5% captures that while still catching a wrong sign,
  // missing factor, or wrong exponent in the FD operators.
  REQUIRE_THAT(num / den, WithinRel(expected, 0.05));
}

TEST_CASE("FD reproduces the analytical unstable growth rate with the "
          "disjoining pressure linearized in",
          "[thin_film][fd][dispersion]") {
  if (world_size() != 1) {
    SKIP("single-rank analytical comparison");
  }
  // Mirrors the spectral suite's "ETD conserves volume and matches linear
  // growth/decay": constant mobility, but now A != 0 so the destabilizing
  // Pi'(h0) term is exercised too, not just the stabilizing curvature term
  // the k^4-decay test above isolates. lambda(k) = M0 k^2 (Pip0 - gamma k^2)
  // is the same formula `ThinFilmPhysics::linear_symbol` encodes for the
  // spectral path; here it is the FD flux/curvature operators being
  // checked against it instead.
  FDFixture fx;
  constexpr double h0 = 1.0, gamma = 1.0, M0 = 1.0, A = 0.05;
  constexpr double amp = 1.0e-4, dt = 1.0e-3;
  constexpr int steps = 200;
  constexpr int mode = 2;
  auto h = cosine_field(fx, h0, amp, mode, 0);

  thin_film::ThinFilmParams p;
  thin_film::apply_thin_film_json({{"h0", h0}, {"gamma", gamma}, {"M0", M0}, {"A", A}},
                                  p);
  const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0, .h_star = p.h_star,
                                        .Pi0 = p.Pi0, .Pip0 = p.Pip0};
  const auto constant_mobility = [](double) { return M0; };

  thin_film::FDFluxSolver solver(fx.domain, fx.decomp, 0, MPI_COMM_WORLD, h,
                                 /*order=*/2);
  for (int s = 0; s < steps; ++s) {
    solver.step(h, dt, gamma, pw, constant_mobility, thin_film::FaceMobility::Harmonic);
  }

  const double twopi = 2.0 * std::numbers::pi;
  const double k = twopi * mode / fx.N;
  const double lambda = M0 * k * k * (p.Pip0 - gamma * k * k);
  REQUIRE(lambda > 0.0); // this mode must be unstable, or the test proves nothing
  const double expected = amp * std::exp(lambda * dt * steps);
  double num = 0.0, den = 0.0;
  for (int iy = 0; iy < fx.ny; ++iy) {
    for (int ix = 0; ix < fx.nx; ++ix) {
      const double w = std::cos(twopi * mode * ix / fx.N);
      num += (h[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * fx.nx] -
             h0) *
             w;
      den += w * w;
    }
  }
  REQUIRE_THAT(num / den, WithinRel(expected, 0.05));
}

TEST_CASE("FD face mobility: harmonic mean stays non-negative through rupture, "
          "arithmetic mean does not",
          "[thin_film][fd][positivity]") {
  if (world_size() != 1) {
    SKIP("single-rank rupture drive");
  }
  // A deep, localized precursor-form defect (same shape as the science
  // case's defect-triggered dewetting, `thin_film_defect.json`, just deeper
  // and on a much smaller grid) driven forward long enough to cross into
  // the precursor. This is the actual claim the application makes about
  // the face-mobility choice -- checked here, not just asserted in the
  // README -- and it was calibrated against exactly this behaviour with
  // `apps/thin_film/inputs_json/thin_film_fd_probe2*.json` on a real run:
  // harmonic settles just above `h_star`; arithmetic overflows once the
  // dip gets close to it (there, at t ~ 24 of 30; here, well inside the
  // step budget below).
  constexpr int N = 32;
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({0.5, 0.5, 0.5}));
  const auto decomp = pfc::decomposition::create(domain, 1);
  const auto box = pfc::decomposition::local_box(decomp, 0);
  const int nx = box.size[0], ny = box.size[1];

  thin_film::ThinFilmParams p;
  thin_film::apply_thin_film_json({{"A", 8.6022}, {"h_star", 0.15}}, p);
  const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0, .h_star = p.h_star,
                                        .Pi0 = p.Pi0, .Pip0 = p.Pip0};
  const thin_film::CubicMobility mobility{p.M0, p.h0};
  const auto defect = thin_film::GaussianDefect::centred(domain, /*amplitude=*/0.87,
                                                         /*sigma=*/3.0);
  constexpr double dt = 5.0e-4;
  constexpr int steps = 50000;

  auto run = [&](thin_film::FaceMobility kind) {
    std::vector<double> h(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const double x = ix * 0.5, y = iy * 0.5;
        h[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx] =
            p.h0 * (1.0 + defect(x, y));
      }
    }
    thin_film::FDFluxSolver solver(domain, decomp, 0, MPI_COMM_WORLD, h, /*order=*/2);
    double min_h = *std::min_element(h.begin(), h.end());
    for (int s = 0; s < steps; ++s) {
      solver.step(h, dt, p.gamma, pw, mobility, kind);
      min_h = std::min(min_h, *std::min_element(h.begin(), h.end()));
      if (!std::isfinite(min_h)) break;
    }
    return min_h;
  };

  const double min_harmonic = run(thin_film::FaceMobility::Harmonic);
  const double min_arithmetic = run(thin_film::FaceMobility::Arithmetic);

  REQUIRE(std::isfinite(min_harmonic));
  REQUIRE(min_harmonic > -1.0e-9); // stays non-negative to round-off
  REQUIRE(min_harmonic < 0.5 * p.h0); // and it actually got close to rupture
  // The arithmetic mean only halves the flux out of a near-empty cell; it
  // does not have to stop that cell from being driven below zero, and in
  // this case it does not: the run overflows before the loop ends.
  REQUIRE_FALSE(std::isfinite(min_arithmetic));
}

TEST_CASE("FD and spectral agree in the smooth pre-rupture regime",
          "[thin_film][fd][spectral][agreement]") {
  if (world_size() != 1) {
    SKIP("single-rank cross-validation");
  }
  // Same equation, same IC, same domain, both methods still small-amplitude
  // (no precursor engaged): this is the cross-validation the two-method
  // story depends on. Growth is exponential here, so the two solutions
  // should differ only by discretization error, not by O(1) physics.
  constexpr int N = 64;
  constexpr double h0 = 1.0, gamma = 1.0, M0 = 1.0, A = 0.05;
  constexpr double amp = 1.0e-3, dt = 2.0e-3;
  constexpr int steps = 100;
  constexpr int mode = 2;

  thin_film::ThinFilmParams p;
  thin_film::apply_thin_film_json({{"h0", h0}, {"gamma", gamma}, {"M0", M0}, {"A", A}},
                                  p);
  const thin_film::ThinFilmPointwise pw{.A = p.A, .h0 = p.h0, .h_star = p.h_star,
                                        .Pi0 = p.Pi0, .Pip0 = p.Pip0};
  const thin_film::CubicMobility mobility{p.M0, p.h0};
  const double twopi = 2.0 * std::numbers::pi;

  // FD.
  const auto domain = pfc::domain::create(pfc::GridSize({N, N, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto decomp = pfc::decomposition::create(domain, 1);
  const auto box = pfc::decomposition::local_box(decomp, 0);
  const int nx = box.size[0], ny = box.size[1];
  std::vector<double> h_fd(static_cast<std::size_t>(nx) *
                           static_cast<std::size_t>(ny));
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix)
      h_fd[static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx] =
          h0 * (1.0 + amp * std::cos(twopi * mode * ix / N));
  thin_film::FDFluxSolver fd_solver(domain, decomp, 0, MPI_COMM_WORLD, h_fd,
                                    /*order=*/2);
  for (int s = 0; s < steps; ++s) {
    fd_solver.step(h_fd, dt, p.gamma, pw, mobility, thin_film::FaceMobility::Harmonic);
  }
  const auto fd_sample =
      thin_film::sample_film_fd(h_fd, domain, h0, 0.05, MPI_COMM_WORLD);

  // Spectral.
  pfc::sim::stacks::SpectralCPUStack stack(domain, 0, 1, MPI_COMM_WORLD);
  auto &h_spec = stack.u();
  h_spec.apply([&](double x, double, double) {
    return h0 * (1.0 + amp * std::cos(twopi * mode * x / N));
  });
  const double k_lap_scale = -(twopi * mode / N) * (twopi * mode / N);
  (void)k_lap_scale;
  std::vector<double> k_lap(stack.fft().size_outbox(), 0.0);
  pfc::fft::kspace::for_each_kpoint(
      stack.fft().get_outbox_bounds(), domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        k_lap[i] = -(kx * kx + ky * ky + kz * kz);
      });
  pfc::apps::FluxETD stepper(domain, stack.fft(), dt, [=](double kl) {
    return -M0 * gamma * kl * kl - M0 * p.Pip0 * kl;
  });
  auto potential = [&](pfc::data::Field<std::complex<double>> &h_hat,
                       pfc::data::Field<double> &hh,
                       pfc::data::Field<std::complex<double>> &out) {
    std::vector<double> pi_real(hh.local_size()[0] * hh.local_size()[1]);
    hh.with_host_view([&](double *hv, std::size_t cnt) {
      for (std::size_t i = 0; i < cnt; ++i) pi_real[i] = pw.Pi(hv[i]);
    });
    pfc::data::Field<double> pi_field(domain, stack.fft().get_inbox_bounds(), 0);
    pi_field.with_host_view([&](double *pv, std::size_t cnt) {
      for (std::size_t i = 0; i < cnt; ++i) pv[i] = pi_real[i];
    });
    pfc::data::Field<std::complex<double>> pi_hat(domain, stack.fft().get_outbox_bounds(),
                                                  0);
    pfc::sim::SpectralETDOps<pfc::HostSpace>::forward(stack.fft(), pi_field, pi_hat);
    h_hat.with_host_view([&](std::complex<double> *hvv, std::size_t m) {
      pi_hat.with_host_view([&](std::complex<double> *pvv, std::size_t) {
        out.with_host_view([&](std::complex<double> *o, std::size_t) {
          for (std::size_t i = 0; i < m; ++i) o[i] = -gamma * k_lap[i] * hvv[i] - pvv[i];
        });
      });
    });
  };
  double t = 0.0;
  for (int s = 0; s < steps; ++s) t = stepper.step(t, h_spec, potential, mobility);
  const auto spec_sample = thin_film::sample_film(h_spec, domain, h0, 0.05, MPI_COMM_WORLD);

  // Both are unstable-mode growth from the same IC; require them to agree to
  // a couple of percent in the two headline observables this application
  // reports -- min thickness and total volume -- while amplitude is still
  // small (no precursor engaged on either side).
  REQUIRE_THAT(fd_sample.min_h, WithinRel(spec_sample.min_h, 0.02));
  REQUIRE_THAT(fd_sample.volume, WithinRel(spec_sample.volume, 1.0e-6));
}
