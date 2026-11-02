// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file bounded_axis_cost.hpp
 * @brief Collective Chebyshev Poisson solves and their cost against differences.
 *
 * Each solve gathers the Lobatto line and calls the existing bounded
 * operator: Dirichlet, Neumann, or Robin. The cost record times the
 * Dirichlet solve, the order-4 and order-8 uniform Dirichlet solves, a
 * one-rank `ComplexLineFFT` of the even extension, and
 * `DistributedComplexLineFFT` of the same extension. Production
 * applications are not switched here.
 */

#include <mpi.h>
#include <span>
#include <vector>

namespace pfc::fft {

/** @brief Owned slice of a gathered Dirichlet Chebyshev Poisson solve.
 * `solution` follows the Lobatto grid from +1 down to -1.
 */
[[nodiscard]] std::vector<double>
distributed_chebyshev_dirichlet_poisson(std::span<const double> local_forcing,
                                        int degree, double value_at_plus,
                                        double value_at_minus, MPI_Comm comm);

/** @brief Owned slice of a gathered Neumann Chebyshev Poisson solve.
 * The additive constant is the zero-integral gauge of
 * `chebyshev_neumann_poisson`. Incompatible slopes are rejected on every
 * rank.
 */
[[nodiscard]] std::vector<double>
distributed_chebyshev_neumann_poisson(std::span<const double> local_forcing,
                                      int degree, double slope_at_plus,
                                      double slope_at_minus, MPI_Comm comm);

/** @brief Owned slice of a gathered Robin Chebyshev Poisson solve.
 * Each end imposes `weight_value * u + weight_slope * u' = data`, through
 * `chebyshev_robin_poisson`.
 */
[[nodiscard]] std::vector<double> distributed_chebyshev_robin_poisson(
    std::span<const double> local_forcing, int degree, double value_weight_plus,
    double slope_weight_plus, double data_plus, double value_weight_minus,
    double slope_weight_minus, double data_minus, MPI_Comm comm);

/** @brief Neutral bounded-axis comparison at one polynomial degree.
 * `points` is `degree + 1` on both the Lobatto grid and the uniform grid.
 * Times are the maximum across ranks. The line times start after the plan
 * exists. The Poisson time includes the plans that solve builds on each
 * call. `prefer_chebyshev` is true when the Lobatto error is strictly
 * below both uniform errors. That flag does not edit a production
 * application.
 */
struct BoundedAxisComparison {
  int ranks = 0;
  int points = 0;
  double chebyshev_error = 0;
  double fd4_error = 0;
  double fd8_error = 0;
  double chebyshev_seconds = 0;
  double distributed_poisson_seconds = 0;
  double local_line_seconds = 0;
  double distributed_line_seconds = 0;
  double fd4_seconds = 0;
  double fd8_seconds = 0;
  std::size_t distributed_workspace_bytes = 0;
  std::size_t distributed_comm_bytes = 0;
  std::size_t fd_matrix_bytes = 0;
  double distributed_roundtrip_peak = 0;
  bool prefer_chebyshev = false;
};

[[nodiscard]] BoundedAxisComparison compare_bounded_axis(int degree, MPI_Comm comm);

} // namespace pfc::fft
