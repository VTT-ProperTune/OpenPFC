// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_heat3d_fd_convergence.cpp
 * @brief Regression guard: the compact FD Laplacian's *observed* order of
 *        accuracy matches its *design* order for two representative even
 *        stencil orders.
 *
 * @details
 * `heat3d_fd_convergence_study` (see `apps/heat3d/README.md` for the full
 * measured table across orders 2..12) is a manual sweep, not a test. This
 * file pins down two points of that sweep as an automatic check that a
 * future change to `pfc::gradient::FDGradient<G>`'s stencil coefficients
 * cannot silently regress the design order without a test failing.
 *
 * fd_order 4 and fd_order 8 are used (not fd_order 2, which needs no
 * higher-order machinery to get right, and not the highest orders 10/12,
 * which — per the measured table in the README — approach the round-off
 * floor by N=64, making a tight order-tolerance assertion there partly a
 * test of floating-point noise rather than of the stencil). Orders 4 and
 * 8 stay well above that floor across the whole N in {16..64} range (see
 * the README table), so this guard uses all five grid sizes.
 *
 * See `heat3d/convergence_study.hpp` for how this has zero
 * time-discretization error by construction (single Fourier mode IC,
 * evolved by its own exact eigenvalue rather than a stepper), so what is
 * measured here is purely the stencil's spatial truncation error.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <mpi.h>

#include <heat3d/convergence_study.hpp>

using heat3d::convergence::ConvergenceCase;
using heat3d::convergence::observed_order;
using heat3d::convergence::run_case;

namespace {

/// Average observed order across consecutive N in {16, 24, 32, 48, 64}, at
/// one fd_order.
double average_observed_order(int fd_order) {
  const std::vector<int> grid_sizes = {16, 24, 32, 48, 64};
  std::vector<ConvergenceCase> cases;
  cases.reserve(grid_sizes.size());
  for (int N : grid_sizes) cases.push_back(run_case(fd_order, N));

  double sum = 0.0;
  for (std::size_t i = 1; i < cases.size(); ++i) {
    const double p = observed_order(cases[i - 1], cases[i]);
    INFO("fd_order=" << fd_order << " N=" << cases[i - 1].N << "->" << cases[i].N
                     << " l2=" << cases[i - 1].l2_error << "->" << cases[i].l2_error
                     << " observed_order=" << p);
    sum += p;
  }
  return sum / static_cast<double>(cases.size() - 1);
}

} // namespace

TEST_CASE("heat3d FD Laplacian: fd_order=4 observed order matches design order",
          "[heat3d][convergence][fd_order]") {
  // Measured (see apps/heat3d/README.md): 3.84, 3.92, 3.96, 3.98 -> avg ~3.92.
  const double p = average_observed_order(4);
  INFO("average observed order (fd_order=4) = " << p);
  REQUIRE(p > 3.6);
  REQUIRE(p < 4.3);
}

TEST_CASE("heat3d FD Laplacian: fd_order=8 observed order matches design order",
          "[heat3d][convergence][fd_order]") {
  // Measured (see apps/heat3d/README.md): 7.57, 7.79, 7.89, 7.95 -> avg ~7.8.
  const double p = average_observed_order(8);
  INFO("average observed order (fd_order=8) = " << p);
  REQUIRE(p > 7.3);
  REQUIRE(p < 8.4);
}

int main(int argc, char *argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "test_heat3d_fd_convergence: MPI_Init failed\n";
    return 1;
  }
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
