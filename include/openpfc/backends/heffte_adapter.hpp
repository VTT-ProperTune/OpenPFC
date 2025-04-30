// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_HEFFTE_ADAPTER_HPP
#define PFC_HEFFTE_ADAPTER_HPP

#include "openpfc/core/world.hpp"
#include <heffte.h>

namespace pfc {

/**
 * @brief Converts a World object to heffte::box3d<int>.
 *
 * This function allows explicit conversion of a World object to
 * heffte::box3d<int>. The resulting box represents the entire world domain.
 *
 * @param world The World object to convert.
 * @return A heffte::box3d<int> representing the world domain.
 */
inline heffte::box3d<int> to_heffte_box(const World &world) {
  const auto &size = get_size(world);
  return heffte::box3d<int>({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
}

} // namespace pfc

#endif // PFC_HEFFTE_ADAPTER_HPP
