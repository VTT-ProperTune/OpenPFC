// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../polynomials/pencil.hpp"
#include "../polynomials/singular_values.hpp"
#include <iostream>

int main() {
  try {
    using namespace eigen_reference;
    const std::vector<Matrix> matrices{
        {{1., 1000., 0.}, {0., 0., -1.}, {0., 1., 0.}},
        {{Complex(1, 1), 3., 4.}, {0., Complex(2, -3), 5.}, {0., 0., 4.}},
        {{0., 0.}, {0., 0.}}};
    const std::vector<std::vector<Complex>> expected{
        {1., Complex(0, 1), Complex(0, -1)},
        {Complex(1, 1), Complex(2, -3), 4.},
        {0., 0.}};
    for (std::size_t example = 0; example < matrices.size(); ++example) {
      const auto result = solve(matrices[example]);
      auto remaining = expected[example];
      for (const auto &pair : result) {
        const auto nearest = std::min_element(
            remaining.begin(), remaining.end(), [&](auto a, auto b) {
              return std::abs(a - pair.value) < std::abs(b - pair.value);
            });
        if (nearest == remaining.end() || std::abs(*nearest - pair.value) > 1e-9)
          throw std::runtime_error("Wrong analytic dense reference eigenvalue");
        remaining.erase(nearest);
      }
      if (!remaining.empty()) throw std::runtime_error("Missing eigenvalue");
    }
    const polynomial::Matrix mass{{0, 2, 1}, {3, 1, -1}, {1, 0, 4}};
    const Matrix exact{
        {Complex(1, 2), 0., 3.}, {Complex(-2, 1), 4., 5.}, {7., Complex(0, -3), 2.}};
    Matrix product(3, std::vector<Complex>(3));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) product[i][j] += mass[i][k] * exact[k][j];
    const auto recovered = pencil::mass_solve(mass, product);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (std::abs(recovered[i][j] - exact[i][j]) > 1e-12)
          throw std::runtime_error(
              "Pivoted mass solve failed its exact complex system");
    for (double delta : {1.0, 1e-6, 1e-12}) {
      const Matrix triangular{{1., Complex(0, 1)}, {0., delta}};
      const double frobenius2 = 2 + delta * delta;
      const double largest =
          std::sqrt((frobenius2 + std::hypot(2.0, delta * delta)) / 2);
      const double expected_smallest = delta / largest;
      const double actual = singular_values::smallest(triangular);
      if (std::abs(actual / expected_smallest - 1) > 1e-12)
        throw std::runtime_error(
            "Lost a small singular value of a complex triangular matrix");
    }
    if (singular_values::smallest({{0., 0.}, {0., 0.}}) != 0)
      throw std::runtime_error("Zero matrix has a nonzero singular value");
    std::cout
        << "Complex pairs, split Hessenberg blocks and zero matrix verified.\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
