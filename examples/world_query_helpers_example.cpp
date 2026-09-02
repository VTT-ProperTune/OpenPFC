// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_query_helpers_example.cpp
 * @brief Domain query helpers (volume, dimensionality, bounds)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>

using namespace pfc;

void print_domain_info(const Domain &domain, const std::string &name) {
  std::cout << "\n" << name << ":\n";
  std::cout << std::string(60, '=') << "\n";

  auto size = domain::get_size(domain);
  std::cout << "  Grid size:        " << size[0] << " × " << size[1] << " × "
            << size[2] << "\n";
  std::cout << "  Total points:     " << domain::get_total_size(domain) << "\n";

  auto spacing = domain::get_spacing(domain);
  auto origin = domain::get_origin(domain);
  std::cout << "  Spacing:          (" << spacing[0] << ", " << spacing[1] << ", "
            << spacing[2] << ")\n";
  std::cout << "  Origin:           (" << origin[0] << ", " << origin[1] << ", "
            << origin[2] << ")\n";

  std::cout << "\n  Physical volume:  " << std::fixed << std::setprecision(6)
            << domain::physical_volume(domain) << "\n";

  std::cout << "  Dimensionality:   " << domain::dimensionality(domain) << "D\n";
  std::cout << "    is_1d():        " << (domain::is_1d(domain) ? "true" : "false")
            << "\n";
  std::cout << "    is_2d():        " << (domain::is_2d(domain) ? "true" : "false")
            << "\n";
  std::cout << "    is_3d():        " << (domain::is_3d(domain) ? "true" : "false")
            << "\n";

  auto lower = domain::get_lower_bounds(domain);
  auto upper = domain::get_upper_bounds(domain);
  std::cout << "\n  Lower bounds:     (" << lower[0] << ", " << lower[1] << ", "
            << lower[2] << ")\n";
  std::cout << "  Upper bounds:     (" << upper[0] << ", " << upper[1] << ", "
            << upper[2] << ")\n";

  std::cout << "  Physical extent:  (" << upper[0] - lower[0] << " × "
            << upper[1] - lower[1] << " × " << upper[2] - lower[2] << ")\n";
}

int main() {
  std::cout << "=============================================================\n";
  std::cout << "OpenPFC Example: Domain Query Convenience Functions\n";
  std::cout << "=============================================================\n";

  std::cout << "\nExample 1: Standard 3D simulation domain\n";
  auto d3 = domain::create(GridSize({64, 64, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                           GridSpacing({0.1, 0.1, 0.1}));
  print_domain_info(d3, "3D Cubic Domain (64³, dx=0.1)");

  std::cout << "\n\nExample 2: 2D simulation (thin film)\n";
  auto d2 = domain::create(GridSize({128, 128, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                           GridSpacing({0.01, 0.01, 1.0}));
  print_domain_info(d2, "2D Domain (128² × 1, dx=0.01)");

  std::cout << "\n\nExample 3: 1D simulation (line)\n";
  auto d1 = domain::create(GridSize({256, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                           GridSpacing({0.05, 1.0, 1.0}));
  print_domain_info(d1, "1D Domain (256 × 1 × 1, dx=0.05)");

  std::cout << "\n\nExample 4: Non-cubic domain with custom origin\n";
  auto offset =
      domain::create(GridSize({100, 100, 50}), PhysicalOrigin({-5.0, -5.0, 0.0}),
                     GridSpacing({0.1, 0.1, 0.2}));
  print_domain_info(offset, "Offset Domain (100×100×50, origin at (-5,-5,0))");

  std::cout << "\n\nExample 5: Using ADL (no namespace prefix needed)\n";
  std::cout << std::string(60, '=') << "\n";
  {
    using namespace pfc::domain;

    auto d = create(GridSize({32, 32, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({1.0, 1.0, 1.0}));

    std::cout << "  Volume:           " << physical_volume(d) << "\n";
    std::cout << "  Is 3D:            " << (is_3d(d) ? "yes" : "no") << "\n";
    std::cout << "  Dimensionality:   " << dimensionality(d) << "D\n";

    auto lower = get_lower_bounds(d);
    auto upper = get_upper_bounds(d);
    std::cout << "  Bounds:           [" << lower[0] << ", " << upper[0] << "] × "
              << "[" << lower[1] << ", " << upper[1] << "] × "
              << "[" << lower[2] << ", " << upper[2] << "]\n";
  }

  std::cout << "\n\nExample 6: Manual vs. convenience function\n";
  std::cout << std::string(60, '=') << "\n";
  {
    auto d = domain::create(GridSize({50, 50, 50}), PhysicalOrigin({0.0, 0.0, 0.0}),
                            GridSpacing({0.2, 0.2, 0.2}));

    auto spacing = domain::get_spacing(d);
    auto size = domain::get_size(d);
    double manual_vol =
        spacing[0] * spacing[1] * spacing[2] * size[0] * size[1] * size[2];
    double conv_vol = domain::physical_volume(d);

    std::cout << "  Manual calculation:   " << std::fixed << std::setprecision(6)
              << manual_vol << "\n";
    std::cout << "  Convenience function: " << conv_vol << "\n";
    std::cout << "  Match:                "
              << (std::abs(manual_vol - conv_vol) < 1e-10 ? "✓" : "✗") << "\n";
  }

  std::cout << "\n\n=============================================================\n";
  std::cout << "Summary:\n";
  std::cout << "=============================================================\n";
  std::cout << "Convenience functions in pfc::domain:\n\n";
  std::cout << "  physical_volume(domain)     - Calculate domain volume\n";
  std::cout << "  is_1d(domain)               - Check if 1D (nx>1, ny=1, nz=1)\n";
  std::cout << "  is_2d(domain)               - Check if 2D (nx>1, ny>1, nz=1)\n";
  std::cout << "  is_3d(domain)               - Check if 3D (all > 1)\n";
  std::cout << "  dimensionality(domain)      - Get 1, 2, or 3\n";
  std::cout << "  get_lower_bounds(domain)    - Physical coords at (0,0,0)\n";
  std::cout << "  get_upper_bounds(domain)    - Physical coords at max indices\n";
  std::cout << "=============================================================\n\n";

  return 0;
}
