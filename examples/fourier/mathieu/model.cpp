// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../matrix.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input");
      file >> input;
    }
    const int n = input.value("n", 42), intervals = input.value("intervals", 75);
    const double maximum = input.value("q_max", 15.0);
    if (n < 12 || n > 128 || n % 2 || intervals < 1 || intervals > 150 ||
        !std::isfinite(maximum) || maximum < 0 || maximum > 20)
      throw std::invalid_argument(
          "Use even n in [12,128], intervals1..150 and q_max0..20");
    std::filesystem::create_directories("results");
    std::ofstream output("results/mathieu.csv");
    if (!output) throw std::runtime_error("Cannot open Mathieu CSV");
    output << std::setprecision(17) << "q,mode,eigenvalue\n";
    // start snippet potential
    const auto kinetic = fourier_matrix::negative_laplacian(n, 2 * std::numbers::pi);
    for (int step = 0; step <= intervals; ++step) {
      const double q = maximum * step / intervals;
      auto a = kinetic;
      for (int j = 0; j < n; ++j) {
        const double x = 2 * std::numbers::pi * (j + 1) / n;
        a[j][j] += 2 * q * std::cos(2 * x);
      }
      const auto spectrum = fourier_matrix::eigenvalues(a);
      for (int mode = 0; mode < 11; ++mode)
        output << q << ',' << mode + 1 << ',' << spectrum[mode] << '\n';
    }
    // end snippet potential
    if (!output) throw std::runtime_error("Cannot write Mathieu CSV");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
