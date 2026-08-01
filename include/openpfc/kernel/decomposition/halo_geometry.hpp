// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_geometry.hpp
 * @brief Shared halo geometry utilities for communication layer consolidation
 *
 * @details
 * Provides unified geometry abstractions for halo exchange operations,
 * consolidating duplicated implementations across CPU/GPU exchange variants.
 * Single source of truth for:
 * - Face/edge/corner direction classification
 * - Opposite slot calculation
 * - Deterministic tag allocation
 * - Slab geometry specifications
 *
 * Replaces 4-6 current re-implementations of these concepts in:
 * - halo_exchange.hpp
 * - padded_halo_exchange.hpp  
 * - full_padded_halo_exchange.hpp
 * - halo_persistent.hpp
 * - runtime/cuda/padded_device_halo_exchange.hpp
 * - runtime/cuda/full_padded_device_halo.hpp
 * - runtime/hip/padded_device_halo_exchange.hpp
 * - runtime/hip/full_padded_device_halo.hpp
 *
 * @note This header is part of M4 communication layer consolidation and
 *       provides CPU-only geometry abstractions. Device implementations
 *       can compile this header when it's included from device code,
 *       but the primary consumers are host-side exchange orchestration.
 *
 * @see kernel/decomposition/halo_directions.hpp for direction set abstractions
 * @see kernel/decomposition/decomposition_neighbors.hpp for neighbor finding
 * @author OpenPFC Development Team
 * @date 2026
 */

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/data/types.hpp>

namespace pfc::halo::geometry {

/**
 * @brief Classification of a neighbor direction in 3D
 */
enum class DirectionType {
  Face,   ///< Axis-aligned face (±X, ±Y, ±Z) - 6 total
  Edge,   ///< Edge between two faces (12 total)
  Corner  ///< Corner where three faces meet (8 total)
};

/**
 * @brief Classify a direction vector as Face, Edge, or Corner
 *
 * @param d Direction vector with components in {-1, 0, 1}, not {0,0,0}
 * @return DirectionType classification
 *
 * @throws std::invalid_argument if d is {0,0,0} or has out-of-range components
 */
[[nodiscard]] inline DirectionType classify_direction(const Int3 &d) {
  // Validate components
  for (int c : d) {
    if (c < -1 || c > 1) {
      throw std::invalid_argument(
          "classify_direction: components must be in {-1,0,1}, got (" +
          std::to_string(d[0]) + "," + std::to_string(d[1]) + "," +
          std::to_string(d[2]) + ")");
    }
  }
  
  if (d[0] == 0 && d[1] == 0 && d[2] == 0) {
    throw std::invalid_argument("classify_direction: {0,0,0} is not a valid direction");
  }
  
  // Count non-zero components
  int non_zero_count = 0;
  for (int c : d) {
    if (c != 0) non_zero_count++;
  }
  
  switch (non_zero_count) {
    case 1: return DirectionType::Face;
    case 2: return DirectionType::Edge;
    case 3: return DirectionType::Corner;
    default: 
      // This should never happen due to validation above
      throw std::invalid_argument("classify_direction: invalid direction");
  }
}

/**
 * @brief Canonical face slot enumeration (order must match legacy implementations)
 *
 * This order is the de facto standard across all exchange implementations:
 * +X, -X, +Y, -Y, +Z, -Z (slot numbers 0-5)
 */
enum class FaceSlot {
  PositiveX = 0,
  NegativeX = 1,
  PositiveY = 2,
  NegativeY = 3,
  PositiveZ = 4,
  NegativeZ = 5
};

/**
 * @brief Total number of face directions in 3D
 */
constexpr int kNumFaceDirections = 6;

/**
 * @brief Total number of edge directions in 3D
 */
constexpr int kNumEdgeDirections = 12;

/**
 * @brief Total number of corner directions in 3D
 */
constexpr int kNumCornerDirections = 8;

/**
 * @brief Total number of neighbor directions in 3D (excluding self)
 */
constexpr int kNumTotalDirections = 26;

/**
 * @brief Map a face direction vector to its canonical slot index
 *
 * @param d Face direction vector (must be axis-aligned)
 * @return FaceSlot enum value (0-5)
 *
 * @throws std::invalid_argument if d is not a valid face direction
 */
[[nodiscard]] inline FaceSlot direction_to_face_slot(const Int3 &d) {
  static constexpr std::array<Int3, 6> kFaceDirs = {
      Int3{1, 0, 0},   Int3{-1, 0, 0},
      Int3{0, 1, 0},   Int3{0, -1, 0},
      Int3{0, 0, 1},   Int3{0, 0, -1}
  };
  
  for (int i = 0; i < 6; ++i) {
    if (kFaceDirs[i] == d) {
      return static_cast<FaceSlot>(i);
    }
  }
  
  throw std::invalid_argument(
      "direction_to_face_slot: not a face direction (" +
      std::to_string(d[0]) + "," + std::to_string(d[1]) + "," +
      std::to_string(d[2]) + ")");
}

/**
 * @brief Map a face slot index to its direction vector
 *
 * @param slot Face slot (0-5)
 * @return Direction vector for that slot
 *
 * @throws std::out_of_range if slot is not in [0, 6)
 */
[[nodiscard]] inline Int3 face_slot_to_direction(int slot) {
  static constexpr std::array<Int3, 6> kFaceDirs = {
      Int3{1, 0, 0},   Int3{-1, 0, 0},
      Int3{0, 1, 0},   Int3{0, -1, 0},
      Int3{0, 0, 1},   Int3{0, 0, -1}
  };
  
  if (slot < 0 || slot >= 6) {
    throw std::out_of_range(
        "face_slot_to_direction: slot must be in [0,6), got " +
        std::to_string(slot));
  }
  
  return kFaceDirs[static_cast<std::size_t>(slot)];
}

/**
 * @brief Calculate the opposite face slot (single source of truth)
 *
 * This function consolidates the duplicated `opposite_slot` implementations
 * across multiple exchange classes. The canonical mapping is:
 * - +X (0) ↔ -X (1)
 * - +Y (2) ↔ -Y (3)  
 * - +Z (4) ↔ -Z (5)
 *
 * @param slot Face slot (0-5)
 * @return Opposite face slot
 *
 * @throws std::out_of_range if slot is not in [0, 6)
 *
 * @note For edge/corner directions, the opposite is simply `-d`,
 *       but this function specifically handles the face slot mapping
 *       used by most exchange implementations.
 */
[[nodiscard]] inline int opposite_slot(int slot) {
  static constexpr std::array<int, 6> kOppositeSlots = {
      1,  // +X (0) → -X (1)
      0,  // -X (1) → +X (0)
      3,  // +Y (2) → -Y (3)
      2,  // -Y (3) → +Y (2)
      5,  // +Z (4) → -Z (5)
      4   // -Z (5) → +Z (4)
  };
  
  if (slot < 0 || slot >= 6) {
    throw std::out_of_range(
        "opposite_slot: slot must be in [0,6), got " +
        std::to_string(slot));
  }
  
  return kOppositeSlots[static_cast<std::size_t>(slot)];
}

/**
 * @brief Calculate the opposite direction vector (works for all 26 directions)
 *
 * For any direction vector d, returns -d.
 * This is the general case of opposite_slot that works for edges and corners too.
 *
 * @param d Any valid direction vector (not {0,0,0})
 * @return Opposite direction vector
 */
[[nodiscard]] inline Int3 opposite_direction(const Int3 &d) {
  return Int3{-d[0], -d[1], -d[2]};
}

/**
 * @brief Slab geometry specification for face/edge/corner exchanges
 *
 * Defines the shape and size of halo regions that need to be exchanged
 * for different direction types. Used by exchange implementations to
 * calculate buffer sizes and packing strategies.
 */
struct SlabGeometry {
  DirectionType type;      ///< Face, Edge, or Corner
  Int3 dimensions;         ///< Size of the slab (in local grid units)
  Int3 offset;             ///< Offset from the local domain origin
  Int3 stride;             ///< Memory stride for packed/unpacked access
  
  /**
   * @brief Create face slab geometry
   * 
   * @param axis Axis normal to the face (0=X, 1=Y, 2=Z)
   * @param direction +1 for positive face, -1 for negative face
   * @param local_size Local domain size
   * @param halo_width Halo width
   * @return SlabGeometry for the face
   */
  static SlabGeometry create_face_slab(int axis, int direction,
                                       const Int3 &local_size, int halo_width) {
    SlabGeometry slab;
    slab.type = DirectionType::Face;
    
    // For face slabs, two dimensions span the full face, one dimension is halo_width
    slab.dimensions = local_size;
    slab.dimensions[axis] = halo_width;
    
    // Offset depends on direction
    slab.offset = Int3{0, 0, 0};
    if (direction < 0) {
      // Negative face: halo starts at 0
      slab.offset[axis] = 0;
    } else {
      // Positive face: halo starts at local_size[axis] - halo_width
      slab.offset[axis] = local_size[axis] - halo_width;
    }
    
    // Stride for packed access (row-major: Z, Y, X)
    slab.stride = Int3{1, local_size[0], local_size[0] * local_size[1]};
    
    return slab;
  }
  
  /**
   * @brief Create edge slab geometry
   * 
   * @param axes Two axes defining the edge (e.g., {0,1} for X-Y edge)
   * @param directions Direction along each axis ({±1, ±1})
   * @param local_size Local domain size
   * @param halo_width Halo width
   * @return SlabGeometry for the edge
   */
  static SlabGeometry create_edge_slab(const Int3 &axes, const Int3 &directions,
                                       const Int3 &local_size, int halo_width) {
    SlabGeometry slab;
    slab.type = DirectionType::Edge;
    
    // Find the axis normal to the edge (the one not in axes)
    int normal_axis = 3 - axes[0] - axes[1]; // 0+1+2=3, so 3-(0+1)=2, etc.
    
    // Edge spans halo_width along two axes, full size along the normal axis
    slab.dimensions = Int3{halo_width, halo_width, halo_width};
    slab.dimensions[normal_axis] = local_size[normal_axis];
    
    // Set correct dimensions for the edge axes
    slab.dimensions[axes[0]] = halo_width;
    slab.dimensions[axes[1]] = halo_width;
    
    // Calculate offset based on directions
    slab.offset = Int3{0, 0, 0};
    for (int i = 0; i < 3; ++i) {
      if (directions[i] < 0) {
        slab.offset[i] = 0;
      } else {
        slab.offset[i] = local_size[i] - slab.dimensions[i];
      }
    }
    
    // Stride for packed access
    slab.stride = Int3{1, local_size[0], local_size[0] * local_size[1]};
    
    return slab;
  }
  
  /**
   * @brief Create corner slab geometry
   * 
   * @param directions Direction along each axis ({±1, ±1, ±1})
   * @param local_size Local domain size
   * @param halo_width Halo width
   * @return SlabGeometry for the corner
   */
  static SlabGeometry create_corner_slab(const Int3 &directions,
                                         const Int3 &local_size, int halo_width) {
    SlabGeometry slab;
    slab.type = DirectionType::Corner;
    
    // Corner spans halo_width along all three axes
    slab.dimensions = Int3{halo_width, halo_width, halo_width};
    
    // Calculate offset based on directions
    slab.offset = Int3{0, 0, 0};
    for (int i = 0; i < 3; ++i) {
      if (directions[i] < 0) {
        slab.offset[i] = 0;
      } else {
        slab.offset[i] = local_size[i] - halo_width;
      }
    }
    
    // Stride for packed access
    slab.stride = Int3{1, local_size[0], local_size[0] * local_size[1]};
    
    return slab;
  }
};

/**
 * @brief Tag allocation scheme for deterministic MPI tag generation
 *
 * Provides collision-free tags for multi-field, multi-direction exchanges
 * without requiring hand-spaced application tags. The scheme is:
 * 
 * ```
 * tag = base_tag + field_offset * kTagFieldStride + direction_offset
 * ```
 *
 * Where:
 * - `base_tag` is the application-provided base (field-specific通过)
 * - `field_offset` is the field index (0 for first field, 1 for second, etc.)
 * - `kTagFieldStride` (34) ensures no overlap between fields
 * - `direction_offset` is in [0, 34) from `direction_to_canonical_tag()`
 *
 * This guarantees that for any number of fields and directions, tags never collide.
 */
class TagAllocator {
public:
  /// Stride between field tag blocks (must be ≥ max direction offset + 1)
  static constexpr int kTagFieldStride = 34;
  
  /// Maximum direction offset (33 for all 26 directions + buffer)
  static constexpr int kMaxDirectionOffset = 33;
  
  /// Validate that kTagFieldStride > kMaxDirectionOffset
  static_assert(kTagFieldStride > kMaxDirectionOffset,
                "Tag field stride must exceed max direction offset");
  
  /**
   * @brief Calculate deterministic tag offset for any 26-direction vector
   *
   * For faces (6 directions): returns canonical slot [0,5]
   * For edges (12 directions) + corners (8 directions): returns [6,33]
   *
   * The encoding depends only on the direction vector, ensuring peer ranks
   * calculate the same tag for the same exchange direction.
   *
   * @param d Direction vector
   * @return Tag offset in [0, 34)
   */
  [[nodiscard]] static int direction_to_canonical_tag(const Int3 &d) noexcept {
    try {
      const FaceSlot slot = direction_to_face_slot(d);
      return static_cast<int>(slot);
    } catch (const std::invalid_argument &) {
      // Not a face direction - calculate offset for edge/corner
      // Encoding: 6 + (dx+1) + 3*(dy+1) + 9*(dz+1) gives range [6, 33]
      return 6 + (d[0] + 1) + 3 * (d[1] + 1) + 9 * (d[2] + 1);
    }
  }
  
  /**
   * @brief Calculate tag for a specific field and direction
   *
   * @param base_tag Application-provided base tag (field-specific)
   * @param field_index Field index (0 for first field, 1 for second, etc.)
   * @param direction Exchange direction vector
   * @return Complete MPI tag
   */
  [[nodiscard]] static int calculate_tag(int base_tag, int field_index,
                                         const Int3 &direction) noexcept {
    const int direction_offset = direction_to_canonical_tag(direction);
    return base_tag + field_index * kTagFieldStride + direction_offset;
  }
  
  /**
   * @brief Calculate tag for multi-field batched exchange
   *
   * @param base_tag Application-provided base tag (exchange-specific)
   * @param field_index Field index within the batch
   * @param direction_slot Face slot for batched face exchanges [0,6)
   * @return Complete MPI tag
   *
   * @note This overload is optimized for batched face exchanges where
   *       direction_slot is pre-calculated.
   */
  [[nodiscard]] static int calculate_batched_tag(int base_tag, int field_index,
                                                 int direction_slot) noexcept {
    return base_tag + field_index * kTagFieldStride + direction_slot;
  }
  
  /**
   * @brief Calculate maximum safe base tag to avoid MPI tag overflow
   *
   * MPI tags are 32-bit signed integers. This function calculates the
   * maximum base tag that won't cause overflow for a given number of fields.
   *
   * @param num_fields Number of fields in the exchange
   * @return Maximum safe base tag
   */
  [[nodiscard]] static constexpr int max_safe_base_tag(int num_fields) noexcept {
    // Use int32_t max explicitly for MPI tag limits
    constexpr int kMaxMpitag = 2147483647; // INT32_MAX
    if (num_fields <= 0) return kMaxMpitag;
    return kMaxMpitag - (num_fields - 1) * kTagFieldStride - kMaxDirectionOffset;
  }
};

/**
 * @brief Validate that a slab geometry fits within local domain bounds
 *
 * @param slab Slab geometry to validate
 * @param local_size Local domain size
 * @return true if slab fits, false otherwise
 */
[[nodiscard]] inline bool slab_fits_in_domain(const SlabGeometry &slab,
                                              const Int3 &local_size) {
  // Calculate slab_end manually since Int3 (std::array) doesn't have operator+
  Int3 slab_end{
      slab.offset[0] + slab.dimensions[0],
      slab.offset[1] + slab.dimensions[1],
      slab.offset[2] + slab.dimensions[2]
  };
  
  return slab.offset[0] >= 0 && slab.offset[1] >= 0 && slab.offset[2] >= 0 &&
         slab_end[0] <= local_size[0] && 
         slab_end[1] <= local_size[1] && 
         slab_end[2] <= local_size[2];
}

} // namespace pfc::halo::geometry