// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../allen_cahn.hpp"

int main(int argc, char **argv) {
  // start snippet model
  const auto right_boundary = [](double t) {
    return 1 + std::pow(std::sin(t / 5), 2);
  };
  return allen_cahn::run(argc, argv, false, right_boundary);
  // end snippet model
}
