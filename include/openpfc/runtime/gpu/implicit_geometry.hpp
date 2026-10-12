// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file implicit_geometry.hpp
 * @brief Device classification of Cartesian cells against one implicit body.
 *
 * Cells keep the `CellLayout` index. The result is a mask, not a mesh.
 * Cut-cell measures are a separate entry point once quadrature is available.
 */

#include <openpfc/kernel/geometry/classify.hpp>

#include <cstdint>
#include <vector>

namespace pfc::geometry {

struct DeviceProfile {
  double classify_seconds = 0;
  double measure_seconds = 0;
  double copy_to_device_seconds = 0;
  double copy_to_host_seconds = 0;
  std::size_t device_bytes = 0;
  std::size_t bytes_to_device = 0;
  std::size_t bytes_to_host = 0;
  int cells = 0;
  int cut_cells = 0;
  int interior = 0;
  int exterior = 0;
};

/// `mask[linear]`: 0 interior, 1 exterior, 2 cut, in x-fastest order.
void classify_on_device(const CellLayout &layout, const Implicit &shape,
                        const Policy &policy, std::vector<std::uint8_t> &mask,
                        DeviceProfile &profile);

/// Classify, then measure each cut cell. Volumes follow the ascending cut
/// indices. Requires the cut-cell integrator.
void evaluate_on_device(const CellLayout &layout, const Implicit &shape,
                        const Policy &policy, std::vector<std::uint8_t> &mask,
                        std::vector<double> &inside_volume,
                        std::vector<double> &outside_volume,
                        std::vector<double> &interface_area, DeviceProfile &profile);

} // namespace pfc::geometry
