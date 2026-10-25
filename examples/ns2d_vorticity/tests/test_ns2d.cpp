// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_ns2d.cpp
 * @brief Catch2 tests for the 2-D vorticity–streamfunction prototype.
 *
 * Taylor–Green is spectrally exact and cannot test nonlinear stability
 * or temporal order of the Jacobian. Those use the two-mode IC.
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/band_error.hpp>
#include <ns2d/cases.hpp>
#include <ns2d/flux_balance.hpp>
#include <ns2d/temporal_order.hpp>
#include <ns2d/vorticity_stream.hpp>

using Catch::Matchers::WithinAbs;

namespace {

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

struct StackSolver {
  pfc::sim::stacks::SpectralCPUStack stack;
  ns2d::VorticityStreamCPU solver;
  StackSolver(int n, ns2d::Params p, double length = 2.0 * pfc::pi)
      : stack(ns2d::make_slab(n, length), world_rank(), world_size(),
              MPI_COMM_WORLD),
        solver(stack, p) {}
};

double reduce_max(double local) {
  double g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

} // namespace

TEST_CASE("Poisson invert recovers Taylor-Green streamfunction velocity",
          "[ns2d][poisson]") {
  StackSolver ss(32, ns2d::Params{0.1, 0.01});
  ss.solver.initialize_omega([](double x, double y, double) {
    return ns2d::taylor_green_omega(x, y, 0.1, 0.0);
  });

  double u_err = 0.0, v_err = 0.0;
  ss.stack.u().for_each_owned([&](int i, int j, int k) {
    const auto c = ss.stack.u().coords(i, j, k);
    const auto inbox = ss.stack.fft().get_inbox_bounds();
    const std::size_t idx =
        static_cast<std::size_t>(i) +
        static_cast<std::size_t>(j) * static_cast<std::size_t>(inbox.size[0]) +
        static_cast<std::size_t>(k) * static_cast<std::size_t>(inbox.size[0]) *
            static_cast<std::size_t>(inbox.size[1]);
    u_err = std::max(u_err, std::abs(ss.solver.u()[idx] -
                                     ns2d::taylor_green_u(c[0], c[1], 0.1, 0.0)));
    v_err = std::max(v_err, std::abs(ss.solver.v()[idx] -
                                     ns2d::taylor_green_v(c[0], c[1], 0.1, 0.0)));
  });
  REQUIRE_THAT(reduce_max(u_err), WithinAbs(0.0, 1.0e-12));
  REQUIRE_THAT(reduce_max(v_err), WithinAbs(0.0, 1.0e-12));
}

TEST_CASE("zero mode of the streamfunction is exactly zero", "[ns2d][zero-mode]") {
  StackSolver ss(16, ns2d::Params{0.05, 0.02});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  const auto &inv = ss.solver.psi_inv_k2();
  pfc::fft::kspace::for_each_kpoint(
      ss.stack.fft().get_outbox_bounds(), ss.stack.u().global_size(),
      ss.stack.u().spacing(),
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        if (i == 0 && j == 0 && k == 0) {
          REQUIRE_THAT(inv[idx], WithinAbs(0.0, 0.0));
        }
      });
}

TEST_CASE("2/3 dealias mask zeros the high-k third", "[ns2d][dealias]") {
  StackSolver ss(24, ns2d::Params{0.1, 0.01});
  const auto &mask = ss.solver.dealias_mask();
  const auto spacing = ss.stack.u().spacing();
  int kept = 0, killed = 0;
  pfc::fft::kspace::for_each_kpoint(
      ss.stack.fft().get_outbox_bounds(), ss.stack.u().global_size(), spacing,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        const bool keep = pfc::fft::kspace::two_thirds_keep(kx, ky, kz, spacing);
        if (keep) {
          REQUIRE_THAT(mask[idx], WithinAbs(1.0, 0.0));
          ++kept;
        } else {
          REQUIRE_THAT(mask[idx], WithinAbs(0.0, 0.0));
          ++killed;
        }
      });
  REQUIRE(kept > 0);
  REQUIRE(killed > 0);
}

TEST_CASE("unprojected high-k product aliases into k=1; 2/3 projection does not",
          "[ns2d][aliasing]") {
  constexpr int N = 32;
  auto k1_probe = [](const pfc::data::Field<double> &w) {
    double s = 0.0;
    w.for_each_owned(
        [&](double x, double, double, double val) { s += val * std::cos(x); });
    double g = 0.0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::abs(g);
  };

  StackSolver unproj(N, ns2d::Params{0.0, 0.01});
  unproj.stack.u().apply([](double x, double, double) {
    return std::sin(12.0 * x) * std::sin(13.0 * x);
  });
  const double aliased = k1_probe(unproj.stack.u());

  StackSolver proj(N, ns2d::Params{0.0, 0.01});
  proj.stack.u().apply([](double x, double, double) { return std::sin(12.0 * x); });
  proj.solver.project_current_state();
  auto f = proj.stack.u().vec();
  proj.stack.u().apply([](double x, double, double) { return std::sin(13.0 * x); });
  proj.solver.project_current_state();
  auto g = proj.stack.u().vec();
  for (std::size_t i = 0; i < proj.stack.u().vec().size(); ++i) {
    proj.stack.u().vec()[i] = f[i] * g[i];
  }
  const double cleaned = k1_probe(proj.stack.u());

  REQUIRE(aliased > 10.0 * (cleaned + 1.0e-14));
  REQUIRE(cleaned < 1.0e-10 * (aliased + 1.0));
}

TEST_CASE("Taylor-Green decays to spectral accuracy under IFRK4",
          "[ns2d][taylor-green]") {
  const double nu = 0.1;
  const double dt = 0.05;
  const int steps = 8;
  StackSolver ss(32, ns2d::Params{nu, dt});
  ss.solver.initialize_omega([nu](double x, double y, double) {
    return ns2d::taylor_green_omega(x, y, nu, 0.0);
  });
  for (int i = 0; i < steps; ++i) ss.solver.step();
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  const double T = static_cast<double>(steps) * dt;
  const double linf =
      ss.solver.linf_omega_error(MPI_COMM_WORLD, [nu, T](double x, double y) {
        return ns2d::taylor_green_omega(x, y, nu, T);
      });
  REQUIRE_THAT(linf, WithinAbs(0.0, 1.0e-11));
  REQUIRE_THAT(d.ke, WithinAbs(ns2d::taylor_green_ke(nu, T), 1.0e-12));
  REQUIRE_THAT(d.enstrophy, WithinAbs(ns2d::taylor_green_enstrophy(nu, T), 1.0e-12));
  REQUIRE_THAT(d.div_linf, WithinAbs(0.0, 1.0e-11));
  REQUIRE_THAT(d.mean_omega, WithinAbs(0.0, 1.0e-14));
}

TEST_CASE("Taylor-Green error is already at roundoff on N=16 and N=32",
          "[ns2d][spatial]") {
  const double nu = 0.05;
  const double dt = 0.02;
  const int steps = 5;
  const double T = steps * dt;
  auto run = [&](int n) {
    StackSolver ss(n, ns2d::Params{nu, dt});
    ss.solver.initialize_omega([nu](double x, double y, double) {
      return ns2d::taylor_green_omega(x, y, nu, 0.0);
    });
    for (int i = 0; i < steps; ++i) ss.solver.step();
    return ss.solver.linf_omega_error(MPI_COMM_WORLD, [nu, T](double x, double y) {
      return ns2d::taylor_green_omega(x, y, nu, T);
    });
  };
  REQUIRE(run(16) < 1.0e-11);
  REQUIRE(run(32) < 1.0e-11);
}

TEST_CASE("mean vorticity is conserved on a nonlinear two-mode field",
          "[ns2d][invariants]") {
  StackSolver ss(32, ns2d::Params{0.01, 0.005});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  const double mean0 = ss.solver.diagnostics(MPI_COMM_WORLD).mean_omega;
  for (int i = 0; i < 40; ++i) ss.solver.step();
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.mean_omega, WithinAbs(mean0, 1.0e-12));
  REQUIRE(d.div_linf < 1.0e-10);
  REQUIRE(d.ke > 0.0);
}

TEST_CASE("IFRK4 is higher than first order on a two-mode Jacobian",
          "[ns2d][timestep]") {
  const double nu = 0.05;
  const double T = 0.2;
  auto run = [&](double dt) {
    const int steps = static_cast<int>(std::llround(T / dt));
    StackSolver ss(32, ns2d::Params{nu, dt});
    ss.solver.initialize_omega(
        [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
    for (int i = 0; i < steps; ++i) ss.solver.step();
    std::vector<double> local(static_cast<std::size_t>(ss.stack.u().size()), 0.0);
    ss.stack.u().for_each_owned([&](int i, int j, int k) {
      local[ss.stack.u().idx(i, j, k)] = ss.stack.u()(i, j, k);
    });
    return local;
  };
  const auto fine = run(0.0025);
  auto err = [&](double dt) {
    auto w = run(dt);
    double local = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
      local = std::max(local, std::abs(w[i] - fine[i]));
    }
    return reduce_max(local);
  };
  const double e_coarse = err(0.02);
  const double e_mid = err(0.01);
  REQUIRE(e_coarse > 0.0);
  REQUIRE(e_mid > 0.0);
  // RK4 / IFRK4: halving dt should drop the error by ~16. Allow a wide
  // window so a first-order method (ratio ~2) fails this test.
  REQUIRE(e_coarse / e_mid > 8.0);
}

TEST_CASE("inviscid two-mode field stays finite and nearly conserves energy",
          "[ns2d][stability]") {
  StackSolver ss(32, ns2d::Params{0.0, 0.02});
  ss.solver.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });
  const auto d0 = ss.solver.diagnostics(MPI_COMM_WORLD);
  for (int i = 0; i < 80; ++i) ss.solver.step();
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(std::isfinite(d.ke));
  REQUIRE(std::isfinite(d.max_abs_omega));
  REQUIRE(d.div_linf < 1.0e-10);
  REQUIRE(std::abs(d.ke - d0.ke) / d0.ke < 1.0e-3);
  REQUIRE_THAT(d.mean_omega, WithinAbs(d0.mean_omega, 1.0e-12));
}

TEST_CASE("Minion-Brown unit-square shear has two opposite sheets at t=0",
          "[ns2d][shear]") {
  constexpr double rho = 30.0;
  constexpr double eps = 0.05;
  StackSolver ss(64, ns2d::Params{1.0e-4, 0.005}, 1.0);
  ss.solver.initialize_omega([](double x, double y, double) {
    return ns2d::double_shear_omega(x, y, rho, eps);
  });
  const auto d = ss.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.max_abs_omega > 20.0);
  // N=64 places y=1/2 on a node; the two tanh branches then have 33 vs 31
  // cells, so the sampled mean is O(1e-6), not machine zero.
  REQUIRE_THAT(d.mean_omega, WithinAbs(0.0, 1.0e-5));
  REQUIRE(d.div_linf < 1.0e-6);
  REQUIRE(ns2d::shear_cells_per_thickness(64, rho) ==
          Catch::Approx(64.0 / 30.0).margin(1.0e-12));
}

TEST_CASE("observed order is log2 of successive errors above the floor",
          "[ns2d][order]") {
  const auto fourth = ns2d::observed_order(16.0, 1.0, 1.0e-11);
  REQUIRE(fourth.has_value());
  REQUIRE_THAT(*fourth, WithinAbs(4.0, 1.0e-12));
  REQUIRE_FALSE(ns2d::observed_order(1.0e-12, 1.0e-13, 1.0e-11).has_value());
  REQUIRE_FALSE(ns2d::observed_order(1.0e-4, 0.0, 1.0e-11).has_value());
}

TEST_CASE("spectral face flux matches the integral of a cosine", "[ns2d][flux]") {
  StackSolver ss(32, ns2d::Params{0.0, 0.01}, 1.0);
  ss.stack.u().apply(
      [](double x, double, double) { return std::cos(2.0 * pfc::pi * x); });
  std::vector<ns2d::SpectralPlane::Complex> u_hat(ss.solver.plane().out_n());
  std::vector<ns2d::SpectralPlane::Complex> v_hat(ss.solver.plane().out_n());
  ss.solver.plane().fft().forward(ss.stack.u().vec(), u_hat);
  const ns2d::AxisBox box{0.25, 0.5, 0.1, 0.7};
  const auto flux = ns2d::face_flux(ss.solver.plane(), u_hat, v_hat, box);
  const double height = box.y1 - box.y0;
  const double exact =
      height * (std::cos(2.0 * pfc::pi * box.x1) - std::cos(2.0 * pfc::pi * box.x0));
  REQUIRE_THAT(flux.net, WithinAbs(exact, 1.0e-12));
  REQUIRE_THAT(flux.imag, WithinAbs(0.0, 1.0e-12));
  const ns2d::AxisBox period{0.0, 1.0, 0.0, 1.0};
  const auto closed = ns2d::face_flux(ss.solver.plane(), u_hat, v_hat, period);
  REQUIRE_THAT(closed.net, WithinAbs(0.0, 1.0e-12));
  REQUIRE(ns2d::modal_div_amplitude(ss.solver.plane(), u_hat, v_hat) > 0.1);
}

TEST_CASE("streamfunction velocity has roundoff closed-box flux", "[ns2d][flux]") {
  StackSolver ss(32, ns2d::Params{0.1, 0.01});
  ss.solver.initialize_omega([](double x, double y, double) {
    return ns2d::taylor_green_omega(x, y, 0.1, 0.0);
  });
  std::vector<ns2d::SpectralPlane::Complex> u_hat(ss.solver.plane().out_n());
  std::vector<ns2d::SpectralPlane::Complex> v_hat(ss.solver.plane().out_n());
  ss.solver.plane().fft().forward(ss.solver.u(), u_hat);
  ss.solver.plane().fft().forward(ss.solver.v(), v_hat);
  const double length = 2.0 * pfc::pi;
  const double dx = length / 32.0;
  const ns2d::AxisBox boxes[] = {
      {0.0, length, 0.0, length},
      {0.0, dx, 0.0, dx},
      {2.0 * dx, 7.0 * dx, dx, 4.0 * dx},
      {0.2 * length, 0.7 * length, 0.15 * length, 0.45 * length},
  };
  for (const auto &box : boxes) {
    const auto flux = ns2d::face_flux(ss.solver.plane(), u_hat, v_hat, box);
    REQUIRE_THAT(flux.net, WithinAbs(0.0, 1.0e-9));
    REQUIRE_THAT(flux.imag, WithinAbs(0.0, 1.0e-9));
  }
  REQUIRE(ns2d::modal_div_amplitude(ss.solver.plane(), u_hat, v_hat) < 1.0e-12);

  // Mode (1,1) cancels under this trapezoid. Mode (3,1) is still a curl
  // inside the 2/3 mask, and the same nodes do not.
  StackSolver unequal(32, ns2d::Params{0.1, 0.01});
  unequal.solver.initialize_omega(
      [](double x, double y, double) { return std::sin(3.0 * x) * std::sin(y); });
  std::vector<ns2d::SpectralPlane::Complex> uu(unequal.solver.plane().out_n());
  std::vector<ns2d::SpectralPlane::Complex> vv(unequal.solver.plane().out_n());
  unequal.solver.plane().fft().forward(unequal.solver.u(), uu);
  unequal.solver.plane().fft().forward(unequal.solver.v(), vv);
  const ns2d::AxisBox cell{2.0 * dx, 7.0 * dx, dx, 4.0 * dx};
  const auto spectral = ns2d::face_flux(unequal.solver.plane(), uu, vv, cell);
  const auto trap = ns2d::trapezoid_flux(unequal.solver.plane(), unequal.solver.u(),
                                         unequal.solver.v(), cell);
  REQUIRE(trap.has_value());
  REQUIRE_THAT(spectral.net, WithinAbs(0.0, 1.0e-9));
  REQUIRE(std::abs(*trap) > 1.0e-4);
  REQUIRE(std::abs(*trap) > 100.0 * std::abs(spectral.net));
}

TEST_CASE("common-band error vanishes for a shared Fourier mode", "[ns2d][band]") {
  StackSolver fine(32, ns2d::Params{0.0, 0.01}, 1.0);
  StackSolver coarse(16, ns2d::Params{0.0, 0.01}, 1.0);
  const auto ic = [](double x, double y, double) {
    return std::sin(2.0 * pfc::pi * x) * std::sin(2.0 * pfc::pi * y);
  };
  fine.solver.initialize_omega(ic);
  coarse.solver.initialize_omega(ic);
  std::vector<ns2d::SpectralPlane::Complex> hat;
  fine.solver.copy_omega_hat(hat);
  const auto err = ns2d::common_band_error_from_hat(fine.solver.plane(), hat,
                                                    coarse.solver, MPI_COMM_WORLD);
  REQUIRE_THAT(err.omega_l2, WithinAbs(0.0, 1.0e-10));
  REQUIRE_THAT(err.omega_linf, WithinAbs(0.0, 1.0e-10));
  REQUIRE_THAT(err.velocity_l2, WithinAbs(0.0, 1.0e-10));
  REQUIRE_THAT(err.omega_l2_rel, WithinAbs(0.0, 1.0e-10));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
