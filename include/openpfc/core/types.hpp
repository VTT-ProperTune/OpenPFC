// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <stdexcept>

namespace pfc {

namespace types {

/// Type aliases for clarity
using Int3 = std::array<int, 3>;
using Real3 = std::array<double, 3>;
using Bool3 = std::array<bool, 3>;

// Forward declarations
struct Size3;
struct Periodic3;
struct LowerBounds3;
struct UpperBounds3;
struct Spacing3;

namespace utils {
/**
 * @brief Computes the upper bounds based on size, lower bounds, and spacing.
 * @param size The size of the simulation domain.
 * @param lower The lower bounds of the simulation domain.
 * @param spacing The spacing of the simulation grid.
 * @param periodic The periodicity of the simulation domain.
 * @return The computed upper bounds.
 */
const Real3 compute_upper_bounds(const Size3 &size, const LowerBounds3 &lower,
                                 const Spacing3 &spacing, const Periodic3 &periodic);

/**
 * @brief Computes the spacing based on size, lower bounds, and upper bounds.
 * @param size The size of the simulation domain.
 * @param lower The lower bounds of the simulation domain.
 * @param upper The upper bounds of the simulation domain.
 * @param periodic The periodicity of the simulation domain.
 * @return The computed spacing.
 */
Real3 compute_spacing(const Size3 &size, const LowerBounds3 &lower,
                      const UpperBounds3 &upper, const Periodic3 &periodic);

} // namespace utils

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

  /**
   * @brief Constructs an UpperBounds3 object from a Size3 and LowerBounds3.
   * @param size The size of the simulation domain.
   * @param lower_bounds The lower bounds of the simulation domain.
   */
  UpperBounds3(const Size3 &size, const LowerBounds3 &lower, const Spacing3 &spacing,
               const Periodic3 &periodic)
      : UpperBounds3(utils::compute_upper_bounds(size, lower, spacing, periodic)) {}
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

  /**
   * @brief Constructs a Spacing3 object from a Size3, LowerBounds3, UpperBounds3,
   * and Periodic3.
   * @param size The size of the simulation domain.
   * @param lower_bounds The lower bounds of the simulation domain.
   * @param upper_bounds The upper bounds of the simulation domain.
   * @param periodic The periodicity of the simulation domain.
   * @return The computed spacing.
   */
  Spacing3(const Size3 &size, const LowerBounds3 &lower, const UpperBounds3 &upper,
           const Periodic3 &periodic)
      : Spacing3(utils::compute_spacing(size, lower, upper, periodic)) {}
};

} // namespace types

using Int3 = types::Int3;
using Real3 = types::Real3;
using Bool3 = types::Bool3;

} // namespace pfc
