// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_mhd.cpp
 * @brief Catch2 verification hierarchy for 2-D incompressible MHD (#23).
 */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

#include <ns2d/cases.hpp>
#include <ns2d/mhd.hpp>
#include <ns2d/mhd_cases.hpp>
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

double reduce_max(double local) {
  double g = 0.0;
  MPI_Allreduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

struct MHDStack {
  pfc::sim::stacks::SpectralCPUStack stack;
  ns2d::MHDSolver solver;
  MHDStack(int n, ns2d::MHDParams p)
      : stack(ns2d::make_twopi_slab(n), world_rank(), world_size(), MPI_COMM_WORLD),
        solver(stack, p) {}
};

} // namespace

TEST_CASE("a=0 MHD reproduces the NS solver on a two-mode field",
          "[mhd][hydro-reduction]") {
  const double nu = 0.05;
  const double dt = 0.01;
  const int steps = 20;
  pfc::sim::stacks::SpectralCPUStack ns_stack(
      ns2d::make_twopi_slab(32), world_rank(), world_size(), MPI_COMM_WORLD);
  ns2d::VorticityStreamCPU ns(ns_stack, ns2d::Params{nu, dt});
  ns.initialize_omega(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); });

  MHDStack mhd(32, ns2d::MHDParams{nu, nu, dt, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::two_mode_omega(x, y); },
      [](double, double, double) { return 0.0; });

  for (int i = 0; i < steps; ++i) {
    ns.step();
    mhd.solver.step();
  }
  double local = 0.0;
  ns.omega().for_each_owned([&](int i, int j, int k) {
    local =
        std::max(local, std::abs(ns.omega()(i, j, k) - mhd.solver.omega()(i, j, k)));
  });
  REQUIRE(reduce_max(local) < 1.0e-11);
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.me < 1.0e-20);
  REQUIRE(d.max_b < 1.0e-10);
}

TEST_CASE("force-free Fourier mode decays as exp(-eta k^2 t) with u=0",
          "[mhd][force-free]") {
  const double eta = 0.1;
  const double dt = 0.02;
  const int steps = 10;
  MHDStack mhd(32, ns2d::MHDParams{0.05, eta, dt, +1.0});
  mhd.solver.initialize(
      [](double, double, double) { return 0.0; },
      [](double x, double y, double) { return ns2d::force_free_a(x, y); });
  for (int i = 0; i < steps; ++i) mhd.solver.step();
  const double T = steps * dt;
  const double aerr =
      mhd.solver.linf_a_error(MPI_COMM_WORLD, [eta, T](double x, double y) {
        return ns2d::force_free_a_exact(x, y, eta, T);
      });
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(aerr, WithinAbs(0.0, 1.0e-11));
  REQUIRE(d.max_speed < 1.0e-12);
  REQUIRE(d.max_abs_omega < 1.0e-12);
  REQUIRE_THAT(d.me, WithinAbs(0.25 * std::exp(-4.0 * eta * T), 1.0e-12));
}

TEST_CASE("correct Lorentz sign cancels Alfvénic u=B; the wrong sign does not",
          "[mhd][lorentz-sign]") {
  auto nlinf = [](double sign) {
    MHDStack mhd(32, ns2d::MHDParams{0.0, 0.0, 0.01, sign});
    mhd.solver.initialize(
        [](double x, double y, double) { return ns2d::alfven_omega(x, y); },
        [](double x, double y, double) { return ns2d::alfven_phi(x, y); });
    mhd.solver.nonlinear_omega_from_current();
    return mhd.solver.n_omega_linf(MPI_COMM_WORLD);
  };
  const double good = nlinf(+1.0);
  const double bad = nlinf(-1.0);
  REQUIRE(good < 1.0e-10);
  REQUIRE(bad > 1.0);
  REQUIRE(bad > 1.0e6 * (good + 1.0e-18));
}

TEST_CASE("Alfvénic u=B with nu=eta decays as a viscous eigenmode",
          "[mhd][alfven]") {
  const double nu = 0.05;
  const double dt = 0.01;
  const int steps = 15;
  MHDStack mhd(32, ns2d::MHDParams{nu, nu, dt, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::alfven_omega(x, y); },
      [](double x, double y, double) { return ns2d::alfven_phi(x, y); });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d0.ke - d0.me, WithinAbs(0.0, 1.0e-12));
  for (int i = 0; i < steps; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.ke - d.me, WithinAbs(0.0, 1.0e-11));
  REQUIRE(d.div_u_linf < 1.0e-11);
  REQUIRE(d.div_b_linf < 1.0e-11);
}

TEST_CASE("ideal two-mode MHD conserves E, Hc and A2 over a short window",
          "[mhd][invariants]") {
  MHDStack mhd(32, ns2d::MHDParams{0.0, 0.0, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  for (int i = 0; i < 40; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(std::isfinite(d.energy));
  REQUIRE(std::abs(d.energy - d0.energy) / d0.energy < 2.0e-3);
  REQUIRE(std::abs(d.a2 - d0.a2) / d0.a2 < 2.0e-3);
  REQUIRE(std::abs(d.cross_helicity - d0.cross_helicity) < 2.0e-3);
  REQUIRE(d.div_u_linf < 1.0e-10);
  REQUIRE(d.div_b_linf < 1.0e-10);
}

TEST_CASE("dissipative energy budget matches nu <w^2> + eta <j^2>",
          "[mhd][budget]") {
  MHDStack mhd(32, ns2d::MHDParams{0.05, 0.05, 0.005, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  (void)mhd.solver.diagnostics(MPI_COMM_WORLD);
  mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(std::abs(d.energy_budget_residual) < 5.0e-3);
  REQUIRE(d.dissipation > 0.0);
}

TEST_CASE("Elsasser speeds are max components and magnitudes of u±B",
          "[mhd][elsasser]") {
  {
    std::vector<double> u{1.0}, v{0.0}, bx{1.0}, by{0.0};
    const auto s = ns2d::elsasser_speeds(u, v, bx, by);
    REQUIRE_THAT(s.max_inf, WithinAbs(2.0, 1.0e-15));
    REQUIRE_THAT(s.max_mag, WithinAbs(2.0, 1.0e-15));
    REQUIRE_THAT(s.max_l1, WithinAbs(2.0, 1.0e-15));
  }
  {
    std::vector<double> u{1.0}, v{0.0}, bx{0.0}, by{1.0};
    const auto s = ns2d::elsasser_speeds(u, v, bx, by);
    REQUIRE_THAT(s.max_inf, WithinAbs(1.0, 1.0e-15));
    REQUIRE_THAT(s.max_mag, WithinAbs(std::sqrt(2.0), 1.0e-15));
    REQUIRE_THAT(s.max_l1, WithinAbs(2.0, 1.0e-15));
  }
  MHDStack mhd(32, ns2d::MHDParams{0.02, 0.02, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.max_z_inf, WithinAbs(2.0, 1.0e-12));
  REQUIRE_THAT(d.cfl_ub, WithinAbs(d.cfl_nominal, 1.0e-12));
  REQUIRE_THAT(d.cfl_elsasser, WithinAbs(2.0 * d.cfl_nominal, 1.0e-12));
  REQUIRE(d.cfl_elsasser > d.cfl_ub);
  REQUIRE(d.cfl_elsasser_sum >= d.cfl_elsasser);
}

TEST_CASE("diagonal Elsasser propagation makes the 2-D sum CFL larger",
          "[mhd][elsasser-sum]") {
  // z+ = (1, 1), z- = (1, 1): component-max is 1, L1 sum is 2.
  std::vector<double> u{1.0}, v{1.0}, bx{0.0}, by{0.0};
  const double dx = 0.5, dy = 0.5;
  const auto s = ns2d::elsasser_speeds(u, v, bx, by, dx, dy);
  REQUIRE_THAT(s.max_inf, WithinAbs(1.0, 1.0e-15));
  REQUIRE_THAT(s.max_l1, WithinAbs(2.0, 1.0e-15));
  REQUIRE(s.max_l1 > s.max_inf);
  REQUIRE_THAT(s.max_sum_inv, WithinAbs(4.0, 1.0e-15));
  const double dt = 0.1;
  const double cfl_comp = dt * s.max_inf / dx;
  const double cfl_sum = dt * s.max_sum_inv;
  REQUIRE(cfl_sum > cfl_comp);
  REQUIRE_THAT(cfl_sum, WithinAbs(2.0 * cfl_comp, 1.0e-15));
}

TEST_CASE("A2 uses the zero-mean gauge and a_hat(0) is removed", "[mhd][a2-gauge]") {
  MHDStack mhd(32, ns2d::MHDParams{0.02, 0.02, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y) + 3.0; });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d0.mean_a, WithinAbs(0.0, 1.0e-12));
  REQUIRE_THAT(d0.a2, WithinAbs(0.3125, 1.0e-12));
  for (int i = 0; i < 10; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.mean_a, WithinAbs(0.0, 1.0e-12));
  REQUIRE(d.a2 > 0.0);
}

TEST_CASE("ideal OT invariant drift decreases faster than first order in dt",
          "[mhd][invariants-refine]") {
  const double T = 0.2;
  auto drifts = [&](double dt) {
    const int steps = static_cast<int>(std::llround(T / dt));
    MHDStack mhd(32, ns2d::MHDParams{0.0, 0.0, dt, +1.0});
    mhd.solver.initialize(
        [](double x, double y, double) { return ns2d::ot_omega(x, y); },
        [](double x, double y, double) { return ns2d::ot_a(x, y); });
    const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
    for (int i = 0; i < steps; ++i) mhd.solver.step();
    const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
    std::array<double, 3> e{};
    e[0] = std::abs(d.energy - d0.energy) / d0.energy;
    e[1] = std::abs(d.cross_helicity - d0.cross_helicity) /
           (std::abs(d0.cross_helicity) + 1.0e-16);
    e[2] = std::abs(d.a2 - d0.a2) / d0.a2;
    return e;
  };
  const auto e_dt = drifts(0.02);
  const auto e_h = drifts(0.01);
  const auto e_q = drifts(0.005);
  REQUIRE(e_dt[0] > e_h[0]);
  REQUIRE(e_h[0] > e_q[0]);
  REQUIRE(e_dt[2] > e_h[2]);
  REQUIRE(e_h[2] > e_q[2]);
  REQUIRE(e_dt[1] > e_h[1]);
  // IFRK4 with L=0 is RK4. Require clearly higher than first order (ratio 2)
  // before claiming a formal order. Spatial/roundoff error may cap the last
  // halving.
  REQUIRE(e_dt[0] / e_h[0] > 4.0);
  REQUIRE(e_dt[2] / e_h[2] > 4.0);
  REQUIRE(e_dt[1] / e_h[1] > 4.0);
}

TEST_CASE("trapezoidal energy-budget residual drops as the diag interval shrinks",
          "[mhd][budget-refine]") {
  auto max_residual = [](int stride, int steps) {
    MHDStack mhd(32, ns2d::MHDParams{0.05, 0.05, 0.005, +1.0});
    mhd.solver.initialize(
        [](double x, double y, double) { return ns2d::ot_omega(x, y); },
        [](double x, double y, double) { return ns2d::ot_a(x, y); });
    (void)mhd.solver.diagnostics(MPI_COMM_WORLD);
    double maxr = 0.0;
    for (int i = 1; i <= steps; ++i) {
      mhd.solver.step();
      if (i % stride == 0) {
        const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
        maxr = std::max(maxr, std::abs(d.energy_budget_residual));
      }
    }
    return maxr;
  };
  const double r1 = max_residual(1, 16);
  const double r4 = max_residual(4, 16);
  const double r8 = max_residual(8, 16);
  REQUIRE(r1 < 5.0e-3);
  REQUIRE(r1 < r4);
  REQUIRE(r4 < r8);
}

TEST_CASE("restrict_hat_by_k maps a shared trigonometric polynomial",
          "[mhd][restrict]") {
  pfc::sim::stacks::SpectralCPUStack fine_stack(
      ns2d::make_twopi_slab(32), world_rank(), world_size(), MPI_COMM_WORLD);
  pfc::sim::stacks::SpectralCPUStack coarse_stack(
      ns2d::make_twopi_slab(16), world_rank(), world_size(), MPI_COMM_WORLD);
  ns2d::SpectralPlane fine(fine_stack.fft(), fine_stack.u());
  ns2d::SpectralPlane coarse(coarse_stack.fft(), coarse_stack.u());
  auto trig = [](double x, double y, double) {
    return std::cos(2.0 * x) + std::cos(y);
  };
  fine_stack.u().apply(trig);
  coarse_stack.u().apply(trig);
  std::vector<ns2d::SpectralPlane::Complex> fhat(fine.out_n()), chat(coarse.out_n()),
      rhat;
  fine.fft().forward(fine_stack.u().vec(), fhat);
  coarse.fft().forward(coarse_stack.u().vec(), chat);
  ns2d::restrict_hat_by_k(fine, fhat, coarse, rhat);
  double err = 0.0, nrm = 0.0;
  for (std::size_t i = 0; i < chat.size(); ++i) {
    const auto d = rhat[i] - chat[i];
    err += d.real() * d.real() + d.imag() * d.imag();
    nrm += chat[i].real() * chat[i].real() + chat[i].imag() * chat[i].imag();
  }
  REQUIRE(std::sqrt(err) / (std::sqrt(nrm) + 1.0e-30) < 1.0e-10);
  std::vector<double> recovered(coarse.in_n(), 0.0);
  coarse.fft().backward(rhat, recovered);
  double rerr = 0.0;
  coarse_stack.u().for_each_owned([&](int i, int j, int k) {
    const std::size_t c = coarse.real_idx(i, j, k);
    rerr = std::max(rerr, std::abs(recovered[c] - coarse_stack.u()(i, j, k)));
  });
  REQUIRE(reduce_max(rerr) < 1.0e-10);
}

TEST_CASE("u and B remain solenoidal to transform roundoff", "[mhd][divergence]") {
  MHDStack mhd(32, ns2d::MHDParams{0.02, 0.02, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::ot_omega(x, y); },
      [](double x, double y, double) { return ns2d::ot_a(x, y); });
  for (int i = 0; i < 10; ++i) mhd.solver.step();
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d.div_u_linf < 1.0e-11);
  REQUIRE(d.div_b_linf < 1.0e-11);
}

TEST_CASE("coalescence perturbation is frozen, divergence-free, and diagonal",
          "[mhd][coalescence-ic]") {
  REQUIRE_THAT(ns2d::coalescence_eps, WithinAbs(0.01, 0.0));
  REQUIRE_THAT(ns2d::coalescence_abar, WithinAbs(0.4, 0.0));
  const double pi = std::acos(-1.0);
  const auto u_omax = ns2d::coalescence_u(0.5 * pi, 0.5 * pi);
  REQUIRE_THAT(u_omax[0], WithinAbs(ns2d::coalescence_eps, 1.0e-15));
  REQUIRE_THAT(u_omax[1], WithinAbs(ns2d::coalescence_eps, 1.0e-15));
  const auto u_omax2 = ns2d::coalescence_u(1.5 * pi, 1.5 * pi);
  REQUIRE_THAT(u_omax2[0], WithinAbs(-ns2d::coalescence_eps, 1.0e-15));
  REQUIRE_THAT(u_omax2[1], WithinAbs(-ns2d::coalescence_eps, 1.0e-15));
  const auto u_omin = ns2d::coalescence_u(0.5 * pi, 1.5 * pi);
  REQUIRE_THAT(u_omin[0], WithinAbs(-ns2d::coalescence_eps, 1.0e-15));
  REQUIRE_THAT(u_omin[1], WithinAbs(ns2d::coalescence_eps, 1.0e-15));
  const auto u_omin2 = ns2d::coalescence_u(1.5 * pi, 0.5 * pi);
  REQUIRE_THAT(u_omin2[0], WithinAbs(ns2d::coalescence_eps, 1.0e-15));
  REQUIRE_THAT(u_omin2[1], WithinAbs(-ns2d::coalescence_eps, 1.0e-15));
  for (double x : {0.0, pi}) {
    for (double y : {0.0, pi}) {
      const auto u_x = ns2d::coalescence_u(x, y);
      REQUIRE_THAT(u_x[0], WithinAbs(0.0, 1.0e-15));
      REQUIRE_THAT(u_x[1], WithinAbs(0.0, 1.0e-15));
    }
  }

  MHDStack mhd(32, ns2d::MHDParams{0.01, 0.01, 0.01, +1.0});
  mhd.solver.initialize(
      [](double x, double y, double) { return ns2d::coalescence_omega(x, y); },
      [](double x, double y, double) { return ns2d::coalescence_a(x, y); });
  const double aerr = mhd.solver.linf_a_error(
      MPI_COMM_WORLD, [](double x, double y) { return ns2d::coalescence_a(x, y); });
  REQUIRE_THAT(aerr, WithinAbs(0.0, 1.0e-12));
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(d.mean_sq_j, WithinAbs(8.0 * d.a2, 1.0e-10));
  REQUIRE(d.div_u_linf < 1.0e-11);
  REQUIRE(d.div_b_linf < 1.0e-11);
  REQUIRE(d.max_speed > 0.5 * ns2d::coalescence_eps);
  REQUIRE(d.max_speed < 1.5 * ns2d::coalescence_eps);
}

TEST_CASE("unperturbed coalescence flux is a static decaying eigenmode",
          "[mhd][coalescence-equilibrium]") {
  const double eta = 0.1;
  const double dt = 0.02;
  const int steps = 10;
  MHDStack mhd(32, ns2d::MHDParams{0.05, eta, dt, +1.0});
  mhd.solver.initialize(
      [](double, double, double) { return 0.0; },
      [](double x, double y, double) { return ns2d::coalescence_a(x, y); });
  const auto d0 = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE(d0.max_speed < 1.0e-12);
  REQUIRE(d0.max_abs_omega < 1.0e-12);
  mhd.solver.nonlinear_omega_from_current();
  REQUIRE(mhd.solver.n_omega_linf(MPI_COMM_WORLD) < 1.0e-10);
  for (int i = 0; i < steps; ++i) mhd.solver.step();
  const double T = steps * dt;
  const double aerr =
      mhd.solver.linf_a_error(MPI_COMM_WORLD, [eta, T](double x, double y) {
        return ns2d::coalescence_a_exact(x, y, eta, T);
      });
  const auto d = mhd.solver.diagnostics(MPI_COMM_WORLD);
  REQUIRE_THAT(aerr, WithinAbs(0.0, 1.0e-11));
  REQUIRE(d.max_speed < 1.0e-12);
  REQUIRE(d.max_abs_omega < 1.0e-12);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
