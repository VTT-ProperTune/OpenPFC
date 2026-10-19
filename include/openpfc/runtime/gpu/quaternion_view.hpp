// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file quaternion_view.hpp
 * @brief Borrowed, device-resident four-field orientation view for FCC batches.
 *
 * Four component arrays are x-fastest and share the same padded/owned brick.
 * Data pointers are read-only and owned by the producing solver. No pack,
 * MPI operation or host/device transfer is performed by this view. The
 * source remains alive for every batch evaluation through the borrower's
 * separately owned lifetime token (see FccPoints::borrow_quaternion_fields).
 */

#include <cstddef>
#include <limits>
#include <openpfc/kernel/data/host_device.hpp>

namespace pfc::finite_strain {

struct FccQuaternionFieldView {
  const double *q[4]{};               // component-major pointers, one padded field each
  std::size_t owned[3]{};             // local real-space box size (x, y, z)
  std::size_t padded[3]{};            // allocated strides incl. halo (x-fastest)
  std::size_t storage_halo = 0;       // same width on every axis

  /// Host-side structural validation. This cannot check device pointer lifetime
  /// or physical origin: the owner-level adapter verifies Domain and Box3i.
  [[nodiscard]] bool valid_for(std::size_t count) const noexcept {
    if (count == 0) return false;
    for (const auto *component : q) if (!component) return false;
    std::size_t product = 1;
    for (int d = 0; d < 3; ++d) {
      if (owned[d] == 0 || storage_halo > (std::numeric_limits<std::size_t>::max() - owned[d]) / 2)
        return false;
      if (padded[d] != owned[d] + 2 * storage_halo) return false;
      if (product > std::numeric_limits<std::size_t>::max() / owned[d]) return false;
      product *= owned[d];
    }
    if (product != count) return false;
    std::size_t padded_product = 1;
    for (int d = 0; d < 3; ++d) {
      if (padded_product > std::numeric_limits<std::size_t>::max() / padded[d])
        return false;
      padded_product *= padded[d];
    }
    return true;
  }

  /// Map FCC's compact real-inbox point p to a padded source-field offset.
  /// Called on the GPU only after valid_for(count) and exact owner-box match.
  [[nodiscard]] OPENPFC_INLINE_HD std::size_t at(std::size_t p) const noexcept {
    const std::size_t x = p % owned[0];
    const std::size_t y = (p / owned[0]) % owned[1];
    const std::size_t z = p / (owned[0] * owned[1]);
    return (x + storage_halo) + padded[0] *
             ((y + storage_halo) + padded[1] * (z + storage_halo));
  }
};

} // namespace pfc::finite_strain
