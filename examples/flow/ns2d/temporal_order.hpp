// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file temporal_order.hpp
 * @brief Observed step order from two errors against the same reference.
 *
 * `p = log2(e(dt) / e(dt/2))`. The value is withheld when either error is
 * at or below `floor`, so a ratio is not reported from roundoff or from a
 * comparison that has already reached the reference.
 */

#include <cmath>
#include <optional>

namespace ns2d {

[[nodiscard]] inline std::optional<double>
observed_order(double e_coarse, double e_fine, double floor) {
  if (!(e_coarse > floor && e_fine > floor)) return std::nullopt;
  return std::log2(e_coarse / e_fine);
}

} // namespace ns2d
