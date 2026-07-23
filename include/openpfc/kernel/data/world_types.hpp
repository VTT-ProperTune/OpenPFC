// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_types.hpp
 * @brief Core type definitions for World parameters
 *
 * @details
 * This header defines the fundamental types used to construct World objects:
 * - Int3, Real3, Bool3: Array aliases for 3D data
 * - Size3: Grid dimensions
 * - Periodic3: Periodicity flags per dimension
 * - LowerBounds3, UpperBounds3: Physical domain bounds
 * - Spacing3: Grid spacing per dimension
 *
 * These types provide strong typing to prevent confusion between different
 * 3-element arrays (size vs spacing vs bounds), improving code clarity
 * and catching errors at compile time.
 *
 * @code
 * #include <openpfc/kernel/data/world_types.hpp>
 *
 * pfc::types::Size3 size{64, 64, 64};
 * pfc::types::Spacing3 spacing{1.0, 1.0, 1.0};
 * pfc::types::Periodic3 periodic{true, true, false};  // Periodic in x, y only
 * @endcode
 *
 * @see world.hpp for usage in World construction
 * @see csys.hpp for coordinate system integration
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <array>
#include <stdexcept>

namespace pfc {

namespace types {

/// Type aliases for clarity
using Int3 = std::array<int, 3>;
using Real3 = std::array<double, 3>;
using Bool3 = std::array<bool, 3>;

/**
 * @brief Represents the size of the 3d simulation domain.
 */
struct Size3 {

  const std::array<int, 3> value;

  explicit Size3(const std::array<int, 3> &v) : value(v) {
    for (int dim : v) {
      if (dim <= 0) {
        throw std::invalid_argument("Size values must be positive.");
      }
    }
  };
};

/**
 * @brief Represents the periodicity of the 3d simulation domain.
 */
struct Periodic3 {

  const std::array<bool, 3> value;

  explicit Periodic3(const std::array<bool, 3> &v) : value(v) {}
};

/**
 * @brief Represents the lower bounds of the 3d simulation domain.
 */
struct LowerBounds3 {

  const std::array<double, 3> value;

  explicit LowerBounds3(const std::array<double, 3> &v) : value(v) {}
};

/**
 * @brief Represents the upper bounds of the 3d simulation domain.
 */
struct UpperBounds3 {

  const std::array<double, 3> value;

  explicit UpperBounds3(const std::array<double, 3> &v) : value(v) {}
};

/**
 * @brief Represents the spacing of the 3d simulation grid.
 */
struct Spacing3 {

  const std::array<double, 3> value;

  explicit Spacing3(const std::array<double, 3> &v) : value(v) {
    for (double dim : v) {
      if (dim <= 0.0) {
        throw std::invalid_argument("Spacing values must be positive.");
      }
    }
  };
};

} // namespace types

using Int3 = types::Int3;
using Real3 = types::Real3;
using Bool3 = types::Bool3;

} // namespace pfc
