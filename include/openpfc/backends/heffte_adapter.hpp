// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heffte_adapter.hpp
 * @brief Adapter functions for HeFFTe library integration
 *
 * @details
 * This header provides conversion functions between OpenPFC types and
 * HeFFTe library types, enabling seamless integration with the HeFFTe
 * distributed FFT backend.
 *
 * Key adapters:
 * - to_heffte_box(): Convert World to heffte::box3d<int>
 *
 * HeFFTe is used as the FFT backend for spectral methods in OpenPFC,
 * providing efficient distributed-memory parallel FFT operations.
 *
 * @code
 * #include <openpfc/backends/heffte_adapter.hpp>
 * #include <openpfc/core/world.hpp>
 *
 * pfc::World world({64, 64, 64}, {1.0, 1.0, 1.0});
 * auto heffte_box = pfc::to_heffte_box(world);
 * @endcode
 *
 * @see fft.hpp for FFT interface
 * @see core/world.hpp for World definition
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

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
