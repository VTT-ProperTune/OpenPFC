// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/solvers/multigrid.hpp>
namespace pfc::solvers {
/** Second-order finite-difference auxiliary operator in a unit cube.
 * Scalar: -mu*Laplacian. Vector: -mu*Laplacian(u)-(lambda+mu)*grad(div(u)),
 * using centered mixed derivatives and compact diagonal second derivatives
 * for the Laplacian. With mu>0, lambda+mu>=0 it is SPD modulo periodic
 * translations. Multiplication by cell volume makes transfers use P^T.
 * Components are coupled through mixed derivatives; no crystal update occurs.
 */
inline SparseOperator elastic_auxiliary(const StructuredGrid &g, double lambda = 1,
                                        double mu = .5) {
  g.validate();
  if ((g.components != 1 && g.components != 3) || !(mu > 0) || !(lambda + mu >= 0) ||
      !std::isfinite(lambda) || !std::isfinite(mu))
    throw std::invalid_argument("invalid elastic auxiliary operator");
  double h[3], volume = 1;
  for (int d = 0; d < 3; ++d) {
    h[d] = 1.0 / (g.shape[d] - (g.boundary == Boundary::Dirichlet));
    volume *= h[d];
  }
  std::vector<std::map<int, double>> rows(g.size());
  for (int z = 0; z < g.shape[2]; ++z)
    for (int y = 0; y < g.shape[1]; ++y)
      for (int x = 0; x < g.shape[0]; ++x) {
        int pos[3] = {x, y, z}, node = g.node(x, y, z);
        for (int c = 0; c < g.components; ++c) {
          int row = g.components * node + c;
          if (g.constrained(x, y, z)) {
            rows[row][row] = 1;
            continue;
          }
          auto add = [&](std::array<int, 3> offset, int component, double v) {
            int q[3];
            for (int d = 0; d < 3; ++d) {
              q[d] = pos[d] + offset[d];
              if (g.boundary == Boundary::Periodic)
                q[d] = (q[d] % g.shape[d] + g.shape[d]) % g.shape[d];
              else if (q[d] < 0 || q[d] >= g.shape[d])
                return;
            }
            if (g.constrained(q[0], q[1], q[2])) return;
            rows[row][g.components * g.node(q[0], q[1], q[2]) + component] +=
                volume * v;
          };
          for (int d = 0; d < 3; ++d) {
            add({0, 0, 0}, c, 2 * mu / (h[d] * h[d]));
            for (int sign : {-1, 1}) {
              std::array<int, 3> o{};
              o[d] = sign;
              add(o, c, -mu / (h[d] * h[d]));
            }
          }
          if (g.components == 3)
            for (int d = 0; d < 3; ++d) {
              // -D_c D_d; diagonal d=c uses compact second derivative so the
              // Dirichlet stencil stays inside the one-layer boundary contract.
              if (c == d) {
                add({0, 0, 0}, c, 2 * (lambda + mu) / (h[c] * h[c]));
                for (int sign : {-1, 1}) {
                  std::array<int, 3> o{};
                  o[c] = sign;
                  add(o, c, -(lambda + mu) / (h[c] * h[c]));
                }
              } else
                for (int sc : {-1, 1})
                  for (int sd : {-1, 1}) {
                    std::array<int, 3> o{};
                    o[c] = sc;
                    o[d] = sd;
                    add(o, d, -(lambda + mu) * sc * sd / (4 * h[c] * h[d]));
                  }
            }
        }
      }
  return SparseOperator::from_rows(rows);
}
inline SparseOperator poisson_auxiliary(const StructuredGrid &g) {
  if (g.components != 1) throw std::invalid_argument("Poisson needs scalar grid");
  return elastic_auxiliary(g, 0, 1);
}
} // namespace pfc::solvers
