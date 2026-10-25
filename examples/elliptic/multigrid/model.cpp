// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/structured_operators.hpp>

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input");
      file >> input;
    }
    const auto sizes = input.value("sizes", std::vector<int>{9, 17, 33});
    if (sizes.empty() || sizes.size() > 8)
      throw std::invalid_argument("Use1..8 sizes");
    std::filesystem::create_directories("results");
    std::ofstream convergence("results/convergence.csv");
    if (!convergence) throw std::runtime_error("Cannot open convergence CSV");
    convergence << std::setprecision(17)
                << "n,h,relative_error,iterations,residual\n";
    for (int n : sizes) {
      if (n < 5 || n > 33 || ((n - 1) & (n - 2)))
        throw std::invalid_argument("Use n=2^k+1 in [5,33]");
      const double h = 1.0 / (n - 1), pi = std::numbers::pi;
      pfc::solvers::StructuredGrid grid{
          {n, n, n}, 1, pfc::solvers::Boundary::Dirichlet};
      std::vector<double> exact(grid.size()), b(grid.size()), u(grid.size());
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < n; ++i) {
            const int row = grid.node(i, j, k);
            const double low =
                std::sin(pi * i * h) * std::sin(pi * j * h) * std::sin(pi * k * h);
            const double high = .2 * std::sin(2 * pi * i * h) *
                                std::sin(3 * pi * j * h) * std::sin(pi * k * h);
            exact[row] = low + high;
            b[row] = h * h * h * pi * pi * (3 * low + 14 * high);
          }
      grid.project(exact);
      grid.project(b);
      // start snippet vcycle
      const auto action = pfc::solvers::poisson_auxiliary(grid);
      pfc::solvers::Multigrid hierarchy(grid, pfc::solvers::poisson_auxiliary);
      pfc::solvers::VectorOperations<> ops;
      pfc::solvers::KrylovControls controls;
      controls.relative_tolerance = 1e-11;
      const auto result = pfc::solvers::gmres(
          ops, action, b, u, controls, [&](const auto &r, auto &z, int iteration) {
            hierarchy(r, z, iteration);
          });
      if (!result.converged())
        throw std::runtime_error("Multigrid-GMRES did not converge");
      // end snippet vcycle
      double error2 = 0, norm2 = 0;
      for (std::size_t i = 0; i < u.size(); ++i) {
        error2 += std::pow(u[i] - exact[i], 2);
        norm2 += exact[i] * exact[i];
      }
      convergence << n << ',' << h << ',' << std::sqrt(error2 / norm2) << ','
                  << result.iterations << ',' << result.residual << '\n';
      std::ofstream field("results/field_" + std::to_string(n) + ".csv");
      if (!field) throw std::runtime_error("Cannot open field CSV");
      field << std::setprecision(17) << "i,j,k,u,exact\n";
      for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < n; ++i) {
            const int row = grid.node(i, j, k);
            field << i << ',' << j << ',' << k << ',' << u[row] << ',' << exact[row]
                  << '\n';
          }
      if (!field) throw std::runtime_error("Cannot write field CSV");
    }
    if (!convergence) throw std::runtime_error("Cannot write convergence CSV");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
