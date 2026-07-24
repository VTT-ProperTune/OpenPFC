// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file box3i.hpp
 * @brief `pfc::Box3i` — the canonical inclusive integer index box (M1).
 *
 * @details
 * A 3D axis-aligned index box with **inclusive** low/high corners and per-axis
 * `size`. This is the single index-box type for OpenPFC 0.2, unifying the two
 * former integer index boxes (the removed dense-box type and `pfc::fft::Box3i`,
 * which is now an alias of this type, see `kernel/fft/box3i.hpp`). It carries no
 * HeFFTe-specific fields, so public headers stay HeFFTe-free.
 *
 * It is an aggregate (no user-declared constructors) so existing brace
 * initialization `{low, high, size}` and member access `.low/.high/.size`
 * continue to work. Use `from_bounds(low, high)` to construct with a validated,
 * computed `size`.
 */

#pragma once

#include <array>
#include <ostream>

namespace pfc {

/**
 * @brief Inclusive integer index box `[low, high]` with per-axis `size`.
 *
 * Invariant (when built via `from_bounds`): `size[d] == high[d] - low[d] + 1`.
 */
struct Box3i {
  std::array<int, 3> low{};
  std::array<int, 3> high{};
  std::array<int, 3> size{};

  /// Construct from inclusive corners, computing `size` (>= 1 per axis).
  static constexpr Box3i from_bounds(const std::array<int, 3> &lo,
                                     const std::array<int, 3> &hi) {
    return Box3i{lo, hi, {hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1}};
  }

  /// True iff `size` is consistent with `high - low + 1` and positive.
  [[nodiscard]] constexpr bool is_consistent() const {
    for (int d = 0; d < 3; ++d) {
      if (size[d] != high[d] - low[d] + 1 || size[d] < 1) return false;
    }
    return true;
  }

  /// Total number of index points.
  [[nodiscard]] constexpr long long count() const {
    return static_cast<long long>(size[0]) * size[1] * size[2];
  }

  /// True iff index `idx` lies within `[low, high]` on every axis.
  [[nodiscard]] constexpr bool contains(const std::array<int, 3> &idx) const {
    for (int d = 0; d < 3; ++d) {
      if (idx[d] < low[d] || idx[d] > high[d]) return false;
    }
    return true;
  }

  friend constexpr bool operator==(const Box3i &a, const Box3i &b) {
    return a.low == b.low && a.high == b.high && a.size == b.size;
  }
  friend constexpr bool operator!=(const Box3i &a, const Box3i &b) {
    return !(a == b);
  }
};

/// Stream operator for diagnostics output.
inline std::ostream &operator<<(std::ostream &os, const Box3i &b) {
  os << "Box3i(low={" << b.low[0] << "," << b.low[1] << "," << b.low[2]
     << "}, high={" << b.high[0] << "," << b.high[1] << "," << b.high[2]
     << "}, size={" << b.size[0] << "," << b.size[1] << "," << b.size[2] << "})";
  return os;
}

} // namespace pfc
