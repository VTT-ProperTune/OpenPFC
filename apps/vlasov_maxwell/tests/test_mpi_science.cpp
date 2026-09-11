// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_mpi_science.cpp
 * @brief 1-rank vs N-rank agreement of a science observable (Landau γ).
 *
 * @details
 * `test_transport.cpp` already asserts that `advect_vy` is bitwise
 * independent of the rank count. That is a statement about the gather.
 * This file asks the question the gather test cannot: does the *fitted
 * damping rate* of a Landau run agree between a rank-local brick
 * (`MPI_COMM_SELF`) and a `v_y` split (`MPI_COMM_WORLD`)? The oracle is
 * the 1-rank γ, not the dispersion root — a parallel bug that shifted
 * every rank the same way would still match the kinetic theory.
 */

#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <vlasov_maxwell/diagnostics.hpp>
#include <vlasov_maxwell/ics.hpp>
#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>
#include <vlasov_maxwell/step.hpp>

using Catch::Approx;

namespace {

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

vlasov::SimParams landau_params() {
  const double vth = 0.05;
  const double twopi = 2.0 * std::acos(-1.0);
  vlasov::SimParams p;
  p.nx = 32;
  p.nvx = 32;
  p.nvy = 32;
  p.Lx = twopi * vth / 0.5;
  p.v_max = 8.0 * vth;
  p.v_thermal = vth;
  p.t_end = 12.0;
  p.interp_order = 5;
  p.electrostatic = true;
  p.self_consistent = true;
  p.validate();
  return p;
}

/// Landau damping rate of `mode_ex` on communicator @p comm.
double landau_gamma(const vlasov::SimParams &p, MPI_Comm comm, int halo) {
  const double k = p.k_skin(1);
  const double amp = 0.01;
  vlasov::PhaseSpace ps(p, halo, comm);
  ps.initialise(0, [&](double x, double vx, double vy) {
    return vlasov::ics::density_perturbation(x, k, amp) *
           vlasov::ics::maxwellian(vx, vy, p.v_thermal);
  });
  vlasov::Stepper st(p, ps);
  st.deposit_all();
  {
    const auto sol = vlasov::solve_gauss(st.line, st.sources.rho, p.neutrality_tol);
    st.fields.Ex = sol.Ex;
  }

  double emax = 0.0;
  for (double v : st.fields.Ex) emax = std::fmax(emax, std::fabs(v));
  const double dt =
      p.dt_safety * vlasov::step_limit(p, 1.0, std::fmax(emax, 1.0e-3), 0.1, halo);
  const int n_steps = std::max(1, static_cast<int>(std::llround(p.t_end / dt)));

  std::vector<double> ts, mex;
  auto sample = [&](double t) {
    const auto L = vlasov::make_ledger(p, st.line, st.moments, st.sources,
                                       st.fields, st.gauss, t, 0, 1);
    ts.push_back(t);
    mex.push_back(L.mode_ex);
  };
  sample(0.0);
  for (int step = 1; step <= n_steps; ++step) {
    st.advance(dt);
    sample(static_cast<double>(step) * dt);
  }
  const double t0 = 3.0;
  const double t1 = p.t_end;
  vlasov::require_fit_before_recurrence(t1, k, p.dvx());
  return vlasov::fit_envelope_rate(ts, mex, t0, t1);
}

} // namespace

TEST_CASE("Landau gamma agrees between 1 rank and N ranks", "[mpi][science]") {
  const auto p = landau_params();
  const int halo = vlasov::required_halo_width(4.0, p.interp_order);
  REQUIRE(p.nvy > halo * world_size());
  if (world_size() > vlasov::PhaseSpace::max_ranks(p, halo)) {
    SKIP("more ranks than the v_y cap for this Landau grid");
  }

  const double gamma_1 = landau_gamma(p, MPI_COMM_SELF, halo);
  REQUIRE(std::isfinite(gamma_1));

  const double gamma_n = landau_gamma(p, MPI_COMM_WORLD, halo);
  INFO("ranks = " << world_size() << "  gamma_1 = " << gamma_1
                  << "  gamma_N = " << gamma_n);
  REQUIRE(std::isfinite(gamma_n));
  REQUIRE(gamma_n == Approx(gamma_1).epsilon(1e-8));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
