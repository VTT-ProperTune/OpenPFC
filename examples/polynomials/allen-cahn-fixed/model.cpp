// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../allen_cahn.hpp"

int main(int argc, char **argv) {
  // start snippet model
  const auto right_boundary = [](double) { return 1.0; };
  return allen_cahn::run(argc, argv, true, right_boundary);
  // end snippet model
}
