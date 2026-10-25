// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "collocation.hpp"
#include <openpfc/kernel/fft/chebyshev.hpp>

namespace clamped {
struct Operators {
  polynomial::Vector x;
  polynomial::Matrix laplacian, biharmonic;
};

inline Operators operators(int n) {
  if (n < 4 || n > 100)
    throw std::invalid_argument("Clamped reference degree must be in [4,100]");
  // start snippet native
  const auto x = pfc::fft::chebyshev_lobatto(n);
  polynomial::Matrix d(n + 1, polynomial::Vector(n + 1));
  polynomial::Vector basis(n + 1);
  for (int j = 0; j <= n; ++j) {
    basis.at(j) = 1;
    const auto column = pfc::fft::chebyshev_derivative(basis);
    for (int i = 0; i <= n; ++i) d[i][j] = column[i];
    basis[j] = 0;
  }
  // end snippet native
  const auto d2 = polynomial::square(d), d4 = polynomial::square(d2);
  polynomial::Matrix d3(n + 1, polynomial::Vector(n + 1));
  for (int i = 0; i <= n; ++i)
    for (int k = 0; k <= n; ++k)
      for (int j = 0; j <= n; ++j) d3[i][j] += d2[i][k] * d[k][j];
  Operators result{x, polynomial::Matrix(n - 1, polynomial::Vector(n - 1)),
                   polynomial::Matrix(n - 1, polynomial::Vector(n - 1))};
  // start snippet clamped
  for (int i = 1; i < n; ++i)
    for (int j = 1; j < n; ++j) {
      result.laplacian[i - 1][j - 1] = d2[i][j];
      result.biharmonic[i - 1][j - 1] =
          ((1 - x[i] * x[i]) * d4[i][j] - 8 * x[i] * d3[i][j] - 12 * d2[i][j]) /
          (1 - x[j] * x[j]);
    }
  // end snippet clamped
  return result;
}
} // namespace clamped
