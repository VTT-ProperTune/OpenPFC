// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../tensor.hpp"

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 24);
    const bool manufactured = input.value("manufactured", false);
    // start snippet source
    auto source = [](double, double) { return 0.; };
    auto boundary = [=](double x, double y) {
      if (manufactured) return x * x - y * y;
      return (y == 1 && x < 0 ? std::pow(std::sin(std::numbers::pi * x), 4) : 0) +
             (x == 1 ? .2 * std::sin(3 * std::numbers::pi * y) : 0);
    };
    return tensor::run(n, 0, source, boundary);
    // end snippet source
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
