// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numbers>

using Complex = std::complex<double>;
// start snippet hankel
Complex reciprocal_gamma(Complex z, int samples) {
  constexpr double center = -11, radius = 16;
  Complex sum = 0;
  for (int j = 0; j < samples; ++j) {
    const double theta =
        -std::numbers::pi + 2 * std::numbers::pi * (j + .5) / samples;
    const Complex t = center + radius * std::exp(Complex(0, theta));
    sum += std::exp(t - z * std::log(t)) * (t - center);
  }
  return sum / static_cast<double>(samples);
}
// end snippet hankel

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input");
      file >> input;
    }
    const int samples = input.value("samples", 70);
    if (samples < 8 || samples > 256)
      throw std::invalid_argument("Use samples in [8,256]");
    std::filesystem::create_directories("results");
    std::ofstream output("results/gamma.csv"),
        convergence("results/convergence.csv");
    if (!output || !convergence) throw std::runtime_error("Cannot open gamma CSV");
    output << std::setprecision(17)
           << "real,imag,reciprocal_real,reciprocal_imag,gamma_abs,pole\n";
    for (int j = 0; j <= 50; ++j)
      for (int i = 0; i <= 75; ++i) {
        const double x = -3.5 + .1 * i, y = -2.5 + .1 * j;
        const auto value = reciprocal_gamma({x, y}, samples);
        const bool pole =
            std::abs(y) < 1e-14 && x <= 0 && std::abs(x - std::round(x)) < 1e-12;
        const double magnitude =
            pole ? std::numeric_limits<double>::quiet_NaN() : 1 / std::abs(value);
        output << x << ',' << y << ',' << value.real() << ',' << value.imag() << ','
               << magnitude << ',' << pole << '\n';
      }
    convergence << std::setprecision(17) << "samples,z,reciprocal,exact,error\n";
    for (int n : {20, 40, 70, 100})
      for (double z : {.5, 1.0, 1.5, 2.0, 3.0, 4.0}) {
        const double value = reciprocal_gamma(z, n).real(),
                     exact = 1 / std::tgamma(z);
        convergence << n << ',' << z << ',' << value << ',' << exact << ','
                    << std::abs(value - exact) << '\n';
      }
    if (!output || !convergence) throw std::runtime_error("Cannot write gamma CSV");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
