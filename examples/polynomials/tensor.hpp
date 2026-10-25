// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "collocation.hpp"
#include <iostream>

namespace tensor {
struct Solution {
  polynomial::Vector nodes, values;
  double residual;
};

template <class Forcing, class Boundary>
inline Solution solve(int n, double k2, Forcing forcing, Boundary boundary) {
  auto x = polynomial::nodes(n);
  const auto d2 = polynomial::square(polynomial::differentiation(x));
  const int m = n - 1;
  polynomial::Vector full((n + 1) * (n + 1)), b(m * m), u(m * m), diagonal(m * m);
  auto at = [&](int i, int j) -> double & { return full[i + (n + 1) * j]; };
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
      if (i == 0 || i == n || j == 0 || j == n) at(i, j) = boundary(x[i], x[j]);
  for (int j = 1; j < n; ++j)
    for (int i = 1; i < n; ++i) {
      const int row = i - 1 + m * (j - 1);
      b[row] = forcing(x[i], x[j]);
      b[row] -= d2[i][0] * at(0, j) + d2[i][n] * at(n, j) + d2[j][0] * at(i, 0) +
                d2[j][n] * at(i, n);
      diagonal[row] = d2[i][i] + d2[j][j] + k2;
    }
  // start snippet tensor
  auto action = [&](const auto &v, auto &out) {
    std::fill(out.begin(), out.end(), 0);
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        const int row = i + m * j;
        out[row] = k2 * v[row];
        for (int k = 0; k < m; ++k)
          out[row] +=
              d2[i + 1][k + 1] * v[k + m * j] + d2[j + 1][k + 1] * v[i + m * k];
      }
  };
  // end snippet tensor
  pfc::solvers::VectorOperations<> ops;
  pfc::solvers::KrylovControls controls;
  controls.relative_tolerance = 1e-11;
  controls.absolute_tolerance = 1e-12;
  controls.restart = m * m;
  controls.maximum_iterations = 2 * m * m;
  const auto result = pfc::solvers::gmres(
      ops, action, b, u, controls, [&](const auto &r, auto &z, int) {
        for (int i = 0; i < m * m; ++i)
          z[i] = r[i] / (std::abs(diagonal[i]) > 1e-12 ? diagonal[i] : 1.0);
      });
  if (!result.converged()) throw std::runtime_error("Tensor GMRES did not converge");
  for (int j = 1; j < n; ++j)
    for (int i = 1; i < n; ++i) at(i, j) = u[i - 1 + m * (j - 1)];
  return {std::move(x), std::move(full), result.residual};
}

template <class Forcing, class Boundary>
inline int run(int n, double k2, Forcing forcing, Boundary boundary) {
  if (n < 2 || n > 32) throw std::invalid_argument("Use n in [2,32]");
  const auto solution = solve(n, k2, forcing, boundary);
  auto output = polynomial::output("solution.csv");
  output << "i,j,x,y,u\n";
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
      output << i << ',' << j << ',' << solution.nodes[i] << ',' << solution.nodes[j]
             << ',' << solution.values[i + (n + 1) * j] << '\n';
  std::cout << "residual=" << solution.residual << '\n';
  return 0;
}
} // namespace tensor
