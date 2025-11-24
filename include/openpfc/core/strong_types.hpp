// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 * LocalOffset offset({0, 0, 0});
 * GridSpacing spacing({1.0, 1.0, 1.0});
 *
 * auto world = create(size, offset, spacing);  // ✅ Clear intent
 * auto bad = create(offset, size, spacing);    // ❌ Won't compile!
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
 * ## Backward Compatibility
 *
 * Strong types use **implicit conversions** for seamless backward compatibility:
 *
 * @code
 * // Old code still works
 * Int3 size = {64, 64, 64};
 * auto world = create(size, ...);  // ✅ Works
 *
 * // New code uses strong types
 * GridSize size({64, 64, 64});
 * auto world = create(size, ...);  // ✅ Also works
 *
 * // Can mix and match
 * auto world2 = create(GridSize({32, 32, 32}), {0, 0, 0}, ...);  // ✅ Works
 * @endcode
 *
 * ## Available Types
 *
 * **Discrete (index) space:**
 * - `GridSize` - Grid dimensions (number of points per dimension)
 * - `LocalOffset` - Subdomain offset in local coordinate system
 * - `GlobalOffset` - Subdomain offset in global coordinate system
 * - `IndexBounds` - Min/max indices for a region
 *
 * **Physical (coordinate) space:**
 * - `GridSpacing` - Physical spacing between grid points
 * - `PhysicalOrigin` - Physical origin of coordinate system
 * - `PhysicalCoords` - Physical position in space
 * - `PhysicalBounds` - Physical min/max coordinates for a region
 *
 * ## Usage Examples
 *
 * ### Basic Construction
 *
 * @code
 * // From raw arrays
 * Int3 raw_size = {64, 64, 64};
 * GridSize size(raw_size);
 *
 * // Direct brace initialization
 * GridSize size2({128, 128, 128});
 *
 * // Implicit conversion back to raw type
 * Int3 extracted = size;
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
 * ### Bounds Types
 *
 * @code
 * // Index space bounds
 * IndexBounds idx_bounds({0, 0, 0}, {63, 63, 63});
 * Int3 lower = idx_bounds.lower;
 * Int3 upper = idx_bounds.upper;
 *
 * // Physical space bounds
 * PhysicalBounds phys_bounds({-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
 * Real3 lower_phys = phys_bounds.lower;
 * Real3 upper_phys = phys_bounds.upper;
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
#include <openpfc/core/types.hpp>
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
 * Int3 raw = size;              // Implicit conversion
 * @endcode
 */
struct GridSize {
  Int3 value; ///< Underlying array value

  /**
   * @brief Construct from Int3 (implicit for backward compatibility)
   * @param v Grid dimensions
   */
  GridSize(const Int3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Int3
   */
  const Int3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Int3
   * @return Reference to underlying Int3
   */
  operator const Int3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other GridSize to compare
   * @return true if values are equal
   */
  bool operator==(const GridSize &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other GridSize to compare
   * @return true if values are not equal
   */
  bool operator!=(const GridSize &other) const noexcept {
    return value != other.value;
  }
};

/**
 * @brief Local subdomain offset in local coordinate system
 *
 * Represents the offset of a subdomain within a local coordinate frame.
 * Used in domain decomposition to specify where a subdomain starts.
 *
 * @note Zero-cost: `sizeof(LocalOffset) == sizeof(Int3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * LocalOffset offset({0, 0, 0});  // Starts at origin
 * Int3 raw = offset;              // Implicit conversion
 * @endcode
 */
struct LocalOffset {
  Int3 value; ///< Underlying array value

  /**
   * @brief Construct from Int3 (implicit for backward compatibility)
   * @param v Offset in each dimension
   */
  LocalOffset(const Int3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Int3
   */
  const Int3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Int3
   * @return Reference to underlying Int3
   */
  operator const Int3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other LocalOffset to compare
   * @return true if values are equal
   */
  bool operator==(const LocalOffset &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other LocalOffset to compare
   * @return true if values are not equal
   */
  bool operator!=(const LocalOffset &other) const noexcept {
    return value != other.value;
  }
};

/**
 * @brief Global subdomain offset in global coordinate system
 *
 * Represents the offset of a subdomain within the global computational domain.
 * Used in distributed-memory (MPI) parallelism to specify subdomain position.
 *
 * @note Zero-cost: `sizeof(GlobalOffset) == sizeof(Int3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * GlobalOffset offset({64, 0, 0});  // Second subdomain in x-direction
 * Int3 raw = offset;                // Implicit conversion
 * @endcode
 */
struct GlobalOffset {
  Int3 value; ///< Underlying array value

  /**
   * @brief Construct from Int3 (implicit for backward compatibility)
   * @param v Offset in each dimension
   */
  GlobalOffset(const Int3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Int3
   */
  const Int3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Int3
   * @return Reference to underlying Int3
   */
  operator const Int3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other GlobalOffset to compare
   * @return true if values are equal
   */
  bool operator==(const GlobalOffset &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other GlobalOffset to compare
   * @return true if values are not equal
   */
  bool operator!=(const GlobalOffset &other) const noexcept {
    return value != other.value;
  }
};

/**
 * @brief Index space bounds (min and max indices)
 *
 * Represents a rectangular region in index space with lower and upper bounds.
 * Commonly used to specify iteration ranges or subdomain boundaries.
 *
 * @note Zero-cost: Struct of two Int3 arrays, no overhead
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * IndexBounds bounds({0, 0, 0}, {63, 63, 63});
 * for (int k = bounds.lower[2]; k <= bounds.upper[2]; ++k) {
 *     // Iterate over region
 * }
 * @endcode
 */
struct IndexBounds {
  Int3 lower; ///< Lower bounds (inclusive)
  Int3 upper; ///< Upper bounds (inclusive)

  /**
   * @brief Construct from lower and upper bounds
   * @param lo Lower bounds (inclusive)
   * @param hi Upper bounds (inclusive)
   */
  IndexBounds(const Int3 &lo, const Int3 &hi) : lower(lo), upper(hi) {}
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
 * Real3 raw = spacing;                    // Implicit conversion
 * @endcode
 */
struct GridSpacing {
  Real3 value; ///< Underlying array value

  /**
   * @brief Construct from Real3 (implicit for backward compatibility)
   * @param v Spacing in each dimension
   */
  GridSpacing(const Real3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Real3
   */
  const Real3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Real3
   * @return Reference to underlying Real3
   */
  operator const Real3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other GridSpacing to compare
   * @return true if values are equal
   */
  bool operator==(const GridSpacing &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other GridSpacing to compare
   * @return true if values are not equal
   */
  bool operator!=(const GridSpacing &other) const noexcept {
    return value != other.value;
  }
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
 * Real3 raw = origin;                             // Implicit conversion
 * @endcode
 */
struct PhysicalOrigin {
  Real3 value; ///< Underlying array value

  /**
   * @brief Construct from Real3 (implicit for backward compatibility)
   * @param v Origin coordinates
   */
  PhysicalOrigin(const Real3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Real3
   */
  const Real3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Real3
   * @return Reference to underlying Real3
   */
  operator const Real3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other PhysicalOrigin to compare
   * @return true if values are equal
   */
  bool operator==(const PhysicalOrigin &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other PhysicalOrigin to compare
   * @return true if values are not equal
   */
  bool operator!=(const PhysicalOrigin &other) const noexcept {
    return value != other.value;
  }
};

/**
 * @brief Physical coordinates in space
 *
 * Represents a position in physical space (as opposed to index space).
 * Used for specifying locations, evaluating functions at positions, etc.
 *
 * @note Zero-cost: `sizeof(PhysicalCoords) == sizeof(Real3)`
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * PhysicalCoords pos({1.5, 2.5, 3.5});  // Point in space
 * Real3 raw = pos;                       // Implicit conversion
 * @endcode
 */
struct PhysicalCoords {
  Real3 value; ///< Underlying array value

  /**
   * @brief Construct from Real3 (implicit for backward compatibility)
   * @param v Physical coordinates
   */
  PhysicalCoords(const Real3 &v) : value(v) {}

  /**
   * @brief Get underlying value
   * @return Reference to underlying Real3
   */
  const Real3 &get() const noexcept { return value; }

  /**
   * @brief Implicit conversion to Real3
   * @return Reference to underlying Real3
   */
  operator const Real3 &() const noexcept { return value; }

  /**
   * @brief Equality comparison
   * @param other PhysicalCoords to compare
   * @return true if values are equal
   */
  bool operator==(const PhysicalCoords &other) const noexcept {
    return value == other.value;
  }

  /**
   * @brief Inequality comparison
   * @param other PhysicalCoords to compare
   * @return true if values are not equal
   */
  bool operator!=(const PhysicalCoords &other) const noexcept {
    return value != other.value;
  }
};

/**
 * @brief Physical space bounds (min and max coordinates)
 *
 * Represents a rectangular region in physical space with lower and upper bounds.
 * Used to specify the physical extent of the simulation domain.
 *
 * @note Zero-cost: Struct of two Real3 arrays, no overhead
 * @note Trivially copyable: No heap allocation or deep copy
 *
 * @code
 * PhysicalBounds bounds({-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
 * double volume = (bounds.upper[0] - bounds.lower[0]) *
 *                 (bounds.upper[1] - bounds.lower[1]) *
 *                 (bounds.upper[2] - bounds.lower[2]);
 * @endcode
 */
struct PhysicalBounds {
  Real3 lower; ///< Lower bounds
  Real3 upper; ///< Upper bounds

  /**
   * @brief Construct from lower and upper bounds
   * @param lo Lower bounds
   * @param hi Upper bounds
   */
  PhysicalBounds(const Real3 &lo, const Real3 &hi) : lower(lo), upper(hi) {}
};

// ============================================================================
// Compile-Time Assertions (Zero-Cost Verification)
// ============================================================================

// Verify zero-cost: same size as underlying types
static_assert(sizeof(GridSize) == sizeof(Int3),
              "GridSize must be same size as Int3 (zero-cost)");
static_assert(sizeof(LocalOffset) == sizeof(Int3),
              "LocalOffset must be same size as Int3 (zero-cost)");
static_assert(sizeof(GlobalOffset) == sizeof(Int3),
              "GlobalOffset must be same size as Int3 (zero-cost)");
static_assert(sizeof(GridSpacing) == sizeof(Real3),
              "GridSpacing must be same size as Real3 (zero-cost)");
static_assert(sizeof(PhysicalOrigin) == sizeof(Real3),
              "PhysicalOrigin must be same size as Real3 (zero-cost)");
static_assert(sizeof(PhysicalCoords) == sizeof(Real3),
              "PhysicalCoords must be same size as Real3 (zero-cost)");

// Verify trivial copyability (required for performance)
static_assert(std::is_trivially_copyable_v<GridSize>,
              "GridSize must be trivially copyable");
static_assert(std::is_trivially_copyable_v<LocalOffset>,
              "LocalOffset must be trivially copyable");
static_assert(std::is_trivially_copyable_v<GlobalOffset>,
              "GlobalOffset must be trivially copyable");
static_assert(std::is_trivially_copyable_v<IndexBounds>,
              "IndexBounds must be trivially copyable");
static_assert(std::is_trivially_copyable_v<GridSpacing>,
              "GridSpacing must be trivially copyable");
static_assert(std::is_trivially_copyable_v<PhysicalOrigin>,
              "PhysicalOrigin must be trivially copyable");
static_assert(std::is_trivially_copyable_v<PhysicalCoords>,
              "PhysicalCoords must be trivially copyable");
static_assert(std::is_trivially_copyable_v<PhysicalBounds>,
              "PhysicalBounds must be trivially copyable");

// Verify standard layout (required for interop)
static_assert(std::is_standard_layout_v<GridSize>,
              "GridSize must have standard layout");
static_assert(std::is_standard_layout_v<LocalOffset>,
              "LocalOffset must have standard layout");
static_assert(std::is_standard_layout_v<GlobalOffset>,
              "GlobalOffset must have standard layout");
static_assert(std::is_standard_layout_v<IndexBounds>,
              "IndexBounds must have standard layout");
static_assert(std::is_standard_layout_v<GridSpacing>,
              "GridSpacing must have standard layout");
static_assert(std::is_standard_layout_v<PhysicalOrigin>,
              "PhysicalOrigin must have standard layout");
static_assert(std::is_standard_layout_v<PhysicalCoords>,
              "PhysicalCoords must have standard layout");
static_assert(std::is_standard_layout_v<PhysicalBounds>,
              "PhysicalBounds must have standard layout");

} // namespace pfc
