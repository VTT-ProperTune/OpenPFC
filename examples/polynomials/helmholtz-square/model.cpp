// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../tensor.hpp"

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 24);
    const bool manufactured = input.value("manufactured", false);
    const double wave = input.value("wave", 9.0);
    if (!std::isfinite(wave) || wave < 0 || wave > 20)
      throw std::invalid_argument("Use wave in [0,20]");
    // start snippet source
    auto source = [=](double x, double y) {
      return manufactured
                 ? -4 + 2 * x * x + 2 * y * y +
                       wave * wave * (1 - x * x) * (1 - y * y)
                 : std::exp(-10 * ((y - 1) * (y - 1) + (x - .5) * (x - .5)));
    };
    auto boundary = [](double, double) { return 0.; };
    return tensor::run(n, wave * wave, source, boundary);
    // end snippet source
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
