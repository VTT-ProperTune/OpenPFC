// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_helpers_example.cpp
 * @brief Example: Domain construction helpers
 */

#include <iomanip>
#include <iostream>

#include <openpfc/kernel/data/domain.hpp>

using namespace pfc;

int main() {
  std::cout << "=== Domain Construction Helpers Example ===\n\n";

  auto d1 = domain::create({64, 64, 64});
  std::cout << "create({64,64,64}): " << d1 << "\n";

  auto d2 = domain::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0, 0.0}),
                           GridSpacing({0.5, 0.5, 0.5}));
  std::cout << "custom spacing: " << d2 << "\n";

  auto d3 = domain::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
  std::cout << "from_bounds periodic: " << d3 << "\n";

  auto d4 = domain::from_bounds({100, 100, 100}, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
                                {false, true, true});
  std::cout << "from_bounds non-periodic x: " << d4 << "\n";

  auto d5 = domain::with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});
  std::cout << "with_spacing: " << d5 << "\n";
  return 0;
}
