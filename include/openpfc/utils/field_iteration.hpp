// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_iteration.hpp
 * @brief Helper functions for iterating over field subdomains
 *
 * @details
 * This file provides utility functions for common field iteration patterns,
 * particularly for iterating over MPI subdomain inboxes. These helpers reduce
 * code duplication and potential bugs in field modifier implementations.
 *
 * @example
 * @code
 * #include <openpfc/utils/field_iteration.hpp>
 *
 * void apply(pfc::Model& model, double time) override {
 *   auto& field = model.get_real_field(get_field_name());
 *   const auto& fft = model.get_fft();
 *   auto inbox = pfc::fft::get_inbox(fft);
 *
 *   // Use helper instead of nested loops
 *   pfc::utils::iterate_inbox(inbox, [&](const pfc::Int3& idx, int linear_idx) {
 *     auto pos = pfc::world::to_coords(model.get_world(), idx);
 *     field[linear_idx] = compute_value(pos, time);
 *   });
 * }
 * @endcode
 *
 * @see field_modifier.hpp for usage in modifiers
 * @see core/box3d.hpp for Box3D definition
 */

#ifndef PFC_UTILS_FIELD_ITERATION_HPP
#define PFC_UTILS_FIELD_ITERATION_HPP

#include "core/box3d.hpp"
#include "multi_index.hpp"
#include <functional>

namespace pfc {
namespace utils {

/**
 * @brief Iterate over all points in a 3D box (typically an MPI inbox)
 *
 * Calls the provided function for each point in the box, providing both
 * the 3D index coordinates and the linear index. This eliminates the need
 * for manual nested loops and index tracking.
 *
 * @tparam Func Callable type that accepts (const Int3&, int)
 * @param inbox The 3D box to iterate over (typically from fft::get_inbox())
 * @param func Function to call for each point: func(Int3{i,j,k}, linear_idx)
 *
 * @note The function is called with indices in order: k (outer), j (middle), i
 * (inner)
 * @note Linear index starts at 0 and increments for each point
 * @note This matches the storage order used in OpenPFC fields
 *
 * @example
 * @code
 * auto inbox = pfc::fft::get_inbox(fft);
 * pfc::utils::iterate_inbox(inbox, [&](const pfc::Int3& idx, int linear_idx) {
 *   // idx = {i, j, k} - 3D grid coordinates
 *   // linear_idx - linear array index
 *   field[linear_idx] = compute_value(idx);
 * });
 * @endcode
 */
template <typename Func> void iterate_inbox(const Box3D &inbox, Func &&func) {
  int linear_idx = 0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
        func(Int3{i, j, k}, linear_idx++);
      }
    }
  }
}

} // namespace utils
} // namespace pfc

#endif // PFC_UTILS_FIELD_ITERATION_HPP
