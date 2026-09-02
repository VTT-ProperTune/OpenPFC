// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heffte_adapter.hpp
 * @brief Adapter functions for HeFFTe library integration
 */

#ifndef PFC_HEFFTE_ADAPTER_HPP
#define PFC_HEFFTE_ADAPTER_HPP

#include <heffte.h>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>

namespace pfc {

inline heffte::box3d<int> to_heffte_box(const Domain &domain) {
  const auto &size = domain.size;
  return heffte::box3d<int>({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
}

inline heffte::box3d<int> to_heffte_box(const Box3i &box) {
  return heffte::box3d<int>({box.low[0], box.low[1], box.low[2]},
                            {box.high[0], box.high[1], box.high[2]});
}

} // namespace pfc

#endif // PFC_HEFFTE_ADAPTER_HPP
