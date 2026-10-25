// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../matrix.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input");
      file >> input;
    }
    const auto sizes = input.value("sizes", std::vector<int>{6, 12, 18, 24, 30, 36});
    if (sizes.empty() || sizes.size() > 32)
      throw std::invalid_argument("Use 1..32 sizes");
    std::filesystem::create_directories("results");
    std::ofstream output("results/energies.csv");
    if (!output) throw std::runtime_error("Cannot open energy CSV");
    output << std::setprecision(17) << "n,mode,eigenvalue,exact\n";
    // start snippet oscillator
    for (int n : sizes) {
      if (n < 6 || n > 128 || n % 2)
        throw std::invalid_argument("Use even n in [6,128]");
      constexpr double length = 16;
      const double h = length / n;
      const auto domain = pfc::domain::create(pfc::GridSize{{n, 1, 1}},
                                              pfc::PhysicalOrigin{{-8 + h, 0, 0}},
                                              pfc::GridSpacing{{h, 1, 1}});
      pfc::data::Field<double> potential(domain, pfc::domain::index_box(domain), 0);
      potential.apply([](double x, double, double) { return x * x; });
      auto hamiltonian = fourier_matrix::negative_laplacian(n, length);
      for (int j = 0; j < n; ++j) hamiltonian[j][j] += potential(j, 0, 0);
      const auto energies = fourier_matrix::eigenvalues(hamiltonian);
      for (int mode = 0; mode < 4; ++mode)
        output << n << ',' << mode + 1 << ',' << energies[mode] << ','
               << 2 * mode + 1 << '\n';
    }
    // end snippet oscillator
    if (!output) throw std::runtime_error("Cannot write energy CSV");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
