// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file domain_geometry.hpp
 * @brief Shared geometric checks for MPI subarray (global/local/offset) domains.
 *
 * Used by BinaryWriter and BinaryReader before marking a domain valid, so
 * invalid extents fail closed with std::invalid_argument instead of reaching
 * MPI_Type_create_subarray. Kernel-only: no frontend includes.
 */

#include <array>
#include <stdexcept>
#include <string>

namespace pfc::mpi {

/**
 * @brief Reject non-positive dims, negative offsets, or pieces outside the
 *        global box before MPI subarray construction.
 *
 * Geometric half of the VTKWriter domain contract (positive global/local,
 * non-negative offsets, offset+local <= global per axis). Does not check
 * VTK origin/spacing or INT_MAX extent endpoints.
 *
 * @param global_size Global grid extents (must be > 0 on each axis).
 * @param local_size  Local brick extents (must be > 0 on each axis).
 * @param offset      Local brick origin in global indices (must be >= 0).
 * @param context_label Prefix for exception messages (e.g.
 *                      "BinaryWriter::set_domain").
 * @throws std::invalid_argument when any check fails.
 */
inline void validate_subarray_domain(const std::array<int, 3> &global_size,
                                     const std::array<int, 3> &local_size,
                                     const std::array<int, 3> &offset,
                                     const char *context_label) {
  const std::string prefix =
      std::string(context_label != nullptr ? context_label : "validate_subarray_domain") +
      ": ";
  for (int i = 0; i < 3; ++i) {
    if (global_size[i] <= 0 || local_size[i] <= 0) {
      throw std::invalid_argument(prefix +
                                  "global/local dimensions must be positive");
    }
    if (offset[i] < 0) {
      throw std::invalid_argument(prefix + "offsets must be non-negative");
    }
    const auto end = static_cast<long long>(offset[i]) +
                     static_cast<long long>(local_size[i]);
    if (end > static_cast<long long>(global_size[i])) {
      throw std::invalid_argument(
          prefix +
          "piece does not lie inside global domain (check offset + "
          "local_size vs global)");
    }
  }
}

} // namespace pfc::mpi
