// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file strong_types.hpp
 * @brief Strong type aliases for geometric quantities
 *
 * @details
 * This header provides lightweight strong type wrappers for geometric quantities
 * used throughout OpenPFC. These types improve code clarity and type safety by
 * distinguishing between different kinds of 3D arrays (size vs spacing vs offset).
 *
 * ## Design Philosophy
 *
 * OpenPFC uses strong types to make code **self-documenting** and **type-safe**:
 *
 * **Before (primitive obsession):**
 * @code
 * Int3 size = {64, 64, 64};
 * Int3 offset = {0, 0, 0};
 * Real3 spacing = {1.0, 1.0, 1.0};
 *
 * auto world = create(size, offset, spacing);  // Which is which?
 * auto bad = create(offset, size, spacing);    // ❌ Compiles but wrong!
 * @endcode
 *
 * **After (strong types):**
 * @code
 * GridSize size({64, 64, 64});
 * GridSpacing spacing({1.0, 1.0, 1.0});
 * PhysicalOrigin origin({0.0, 0.0, 0.0});
 *
 * auto world = create(size, spacing, origin);  // ✅ Clear intent
 * auto bad = create(spacing, size, origin);    // ❌ Won't compile!
 * @endcode
 *
 * ## Zero-Cost Abstraction
 *
 * All strong types are **zero-cost** - they compile away completely:
 *
 * @code
 * static_assert(sizeof(GridSize) == sizeof(Int3));
 * static_assert(std::is_trivially_copyable_v<GridSize>);
 * @endcode
 *
 * Assembly output is identical to using raw `Int3` or `Real3` types.
 *
 * ## Explicit Conversions
 *
 * Strong types provide **explicit conversions** only to ensure type safety:
 *
 * @code
 * // Construction requires explicit factory method or explicit constructor
 * Int3 raw_size = {64, 64, 64};
 * GridSize size = GridSize::from_vector3(raw_size);  // ✅ Explicit factory
 * GridSize size2{raw_size};                          // ✅ Explicit construction
 *
 * // No implicit conversions from raw types
 * // GridSize size3 = raw_size;                      // ❌ Won't compile
 *
 * // Extract back to raw type (explicit only)
 * Int3 extracted = size.to_vector3();               // ✅ Explicit conversion
 * // Int3 extracted2 = size;                         // ❌ Won't compile
 * @endcode
 *
 * This explicit-only approach maximizes type safety by preventing accidental
 * conversions that could mask type mistakes.
 *
 * ## Available Types
 *
 * **Discrete (index) space:**
 * - `GridSize` - Grid dimensions (number of points per dimension)
 *
 * **Physical (coordinate) space:**
 * - `GridSpacing` - Physical spacing between grid points
 * - `PhysicalOrigin` - Physical origin of coordinate system
 *
 * ## Usage Examples
 *
 * ### Basic Construction
 *
 * @code
 * // From raw arrays (explicit factory method)
 * Int3 raw_size = {64, 64, 64};
 * GridSize size = GridSize::from_vector3(raw_size);
 *
 * // Direct brace initialization
 * GridSize size2({128, 128, 128});
 *
 * // Explicit conversion back to raw type
 * Int3 extracted = size.to_vector3();
 * @endcode
 *
 * ### Function Parameters
 *
 * @code
 * // Self-documenting function signatures
 * void setup(GridSize size, GridSpacing spacing, PhysicalOrigin origin);
 *
 * // Compiler catches argument order mistakes
 * setup(size, spacing, origin);      // ✅ Correct
 * // setup(spacing, size, origin);   // ❌ Won't compile!
 * @endcode
 *
 * ## When to Use
 *
 * **Use strong types for:**
 * - Function parameters (improves clarity)
 * - Public APIs (self-documenting)
 * - Struct members (semantic meaning)
 *
 * **Raw types are fine for:**
 * - Local variables in implementation
 * - Tight loops (no conversion overhead anyway)
 * - Internal helper functions
 *
 * ## Performance Notes
 *
 * Strong types have **zero runtime overhead**:
 * - No heap allocation
 * - No virtual functions
 * - Same size as underlying types
 * - Trivially copyable
 * - Standard layout
 * - Optimizes away completely
 *
 * @see core/types.hpp for raw type definitions (Int3, Real3)
 * @see core/world.hpp for usage in World construction
 *
 * @author OpenPFC Development Team
 * @date 2025-11-24
 */

#pragma once

#include <array>
// nvcc (__CUDACC__) and HIP clang (__HIPCC__/__HIP__) both reject
// <compare> / defaulted operator<=> in device TUs. ROCm clang also
// cannot find <compare> when compiling -x hip.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define OPENPFC_GPU_DEVICE_TU 1
#endif
#ifndef OPENPFC_GPU_DEVICE_TU
#include <compare>
#endif
#include <openpfc/kernel/data/types.hpp>
#include <type_traits>

namespace pfc {

// ============================================================================
// Strong Types for Discrete (Index) Space
// ============================================================================

/**
 * @brief Grid dimensions (number of grid points per dimension)
 *
 * Represents the size of the computational grid in each dimension.
 * Use this instead of raw `Int3` for function parameters to make
 * intent clear and catch argument order mistakes.
 *
 * @note Zero-cost: `sizeof(GridSize) == sizeof(Int3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * GridSize size({64, 64, 64});  // 64³ grid
 *
 * // Explicit conversion methods (preferred over implicit conversion)
 * Int3 raw = size.to_vector3();
 * GridSize size2 = GridSize::from_vector3(raw);
 * @endcode
 */
struct GridSize {
  Int3 value; ///< Underlying array value

  /**
   * @brief Construct from Int3 (explicit construction)
   * @param v Grid dimensions
   */
  explicit GridSize(const Int3 &v) : value(v) {}

  /**
   * @brief Create from Int3 (explicit factory method)
   * @param v Grid dimensions
   * @return GridSize instance
   */
  static GridSize from_vector3(const Int3 &v) noexcept { return GridSize(v); }

  /**
   * @brief Get underlying value
   * @return Reference to underlying Int3
   */
  const Int3 &get() const noexcept { return value; }

  /**
   * @brief Explicit conversion to Int3
   * @return Copy of underlying Int3
   */
  Int3 to_vector3() const noexcept { return value; }

  /** @brief Lexicographic comparison of underlying grid dimensions */
#ifndef OPENPFC_GPU_DEVICE_TU
  auto operator<=>(const GridSize &other) const noexcept = default;
#else
  /** @brief Equality comparison for CUDA (element-by-element) */
  __host__ __device__ constexpr bool
  operator==(const GridSize &other) const noexcept {
    return value[0] == other.value[0] && value[1] == other.value[1] &&
           value[2] == other.value[2];
  }

  /** @brief Inequality comparison for CUDA */
  __host__ __device__ constexpr bool
  operator!=(const GridSize &other) const noexcept {
    return !(*this == other);
  }
#endif
};

// ============================================================================
// Strong Types for Physical (Coordinate) Space
// ============================================================================

/**
 * @brief Physical spacing between grid points
 *
 * Represents the physical distance between adjacent grid points in each dimension.
 * Defines the resolution of the computational grid in physical units.
 *
 * @note Zero-cost: `sizeof(GridSpacing) == sizeof(Real3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * GridSpacing spacing({1.0, 1.0, 1.0});  // 1 unit spacing
 *
 * // Explicit conversion methods (preferred over implicit conversion)
 * Real3 raw = spacing.to_vector3();
 * GridSpacing spacing2 = GridSpacing::from_vector3(raw);
 * @endcode
 */
struct GridSpacing {
  Real3 value; ///< Underlying array value

  /**
   * @brief Construct from Real3 (explicit construction)
   * @param v Spacing in each dimension
   */
  explicit GridSpacing(const Real3 &v) : value(v) {}

  /**
   * @brief Create from Real3 (explicit factory method)
   * @param v Spacing in each dimension
   * @return GridSpacing instance
   */
  static GridSpacing from_vector3(const Real3 &v) noexcept { return GridSpacing(v); }

  /**
   * @brief Get underlying value
   * @return Reference to underlying Real3
   */
  const Real3 &get() const noexcept { return value; }

  /**
   * @brief Explicit conversion to Real3
   * @return Copy of underlying Real3
   */
  Real3 to_vector3() const noexcept { return value; }

  /** @brief Lexicographic comparison of underlying spacing */
#ifndef OPENPFC_GPU_DEVICE_TU
  auto operator<=>(const GridSpacing &other) const noexcept = default;
#else
  /** @brief Equality comparison for CUDA (element-by-element) */
  __host__ __device__ constexpr bool
  operator==(const GridSpacing &other) const noexcept {
    return value[0] == other.value[0] && value[1] == other.value[1] &&
           value[2] == other.value[2];
  }

  /** @brief Inequality comparison for CUDA */
  __host__ __device__ constexpr bool
  operator!=(const GridSpacing &other) const noexcept {
    return !(*this == other);
  }
#endif
};

/**
 * @brief Physical origin of coordinate system
 *
 * Represents the physical location of the coordinate system origin.
 * Defines where (0,0,0) in index space maps to in physical space.
 *
 * @note Zero-cost: `sizeof(PhysicalOrigin) == sizeof(Real3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * PhysicalOrigin origin({-10.0, -10.0, -10.0});  // Centered domain
 *
 * // Explicit conversion methods (preferred over implicit conversion)
 * Real3 raw = origin.to_vector3();
 * PhysicalOrigin origin2 = PhysicalOrigin::from_vector3(raw);
 * @endcode
 */
struct PhysicalOrigin {
  Real3 value; ///< Underlying array value

  /**
   * @brief Construct from Real3 (explicit construction)
   * @param v Origin coordinates
   */
  explicit PhysicalOrigin(const Real3 &v) : value(v) {}

  /**
   * @brief Create from Real3 (explicit factory method)
   * @param v Origin coordinates
   * @return PhysicalOrigin instance
   */
  static PhysicalOrigin from_vector3(const Real3 &v) noexcept {
    return PhysicalOrigin(v);
  }

  /**
   * @brief Get underlying value
   * @return Reference to underlying Real3
   */
  const Real3 &get() const noexcept { return value; }

  /**
   * @brief Explicit conversion to Real3
   * @return Copy of underlying Real3
   */
  Real3 to_vector3() const noexcept { return value; }

  /** @brief Lexicographic comparison of underlying coordinates */
#ifndef OPENPFC_GPU_DEVICE_TU
  auto operator<=>(const PhysicalOrigin &other) const noexcept = default;
#else
  /** @brief Equality comparison for CUDA (element-by-element) */
  __host__ __device__ constexpr bool
  operator==(const PhysicalOrigin &other) const noexcept {
    return value[0] == other.value[0] && value[1] == other.value[1] &&
           value[2] == other.value[2];
  }

  /** @brief Inequality comparison for CUDA */
  __host__ __device__ constexpr bool
  operator!=(const PhysicalOrigin &other) const noexcept {
    return !(*this == other);
  }
#endif
};

// ============================================================================
// Compile-Time Assertions (Zero-Cost Verification)
// ============================================================================

// Verify zero-cost: same size as underlying types
static_assert(sizeof(GridSize) == sizeof(Int3),
              "GridSize must be same size as Int3 (zero-cost)");
static_assert(sizeof(GridSpacing) == sizeof(Real3),
              "GridSpacing must be same size as Real3 (zero-cost)");
static_assert(sizeof(PhysicalOrigin) == sizeof(Real3),
              "PhysicalOrigin must be same size as Real3 (zero-cost)");

// Verify trivial copyability (required for performance)

#if __cplusplus >= 201703L && !defined(OPENPFC_GPU_DEVICE_TU)
static_assert(std::is_trivially_copyable_v<GridSize>,
              "GridSize must be trivially copyable");
static_assert(std::is_trivially_copyable_v<GridSpacing>,
              "GridSpacing must be trivially copyable");
static_assert(std::is_trivially_copyable_v<PhysicalOrigin>,
              "PhysicalOrigin must be trivially copyable");
#endif

// Verify standard layout (required for interop)

#if __cplusplus >= 201703L && !defined(OPENPFC_GPU_DEVICE_TU)
static_assert(std::is_standard_layout_v<GridSize>,
              "GridSize must have standard layout");
static_assert(std::is_standard_layout_v<GridSpacing>,
              "GridSpacing must have standard layout");
static_assert(std::is_standard_layout_v<PhysicalOrigin>,
              "PhysicalOrigin must have standard layout");
#endif

} // namespace pfc
