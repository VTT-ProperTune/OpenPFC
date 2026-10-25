// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../tensor.hpp"

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 24);
    const bool manufactured = input.value("manufactured", false);
    // start snippet source
    auto source = [=](double x, double y) {
      return manufactured ? -4 + 2 * x * x + 2 * y * y
                          : 10 * std::sin(8 * x * (y - 1));
    };
    auto boundary = [](double, double) { return 0.; };
    return tensor::run(n, 0, source, boundary);
    // end snippet source
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
