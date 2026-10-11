// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/kernel/grain/topology.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>

namespace pfc::grain::tracking {

enum class Status {
  Success,
  Unseeded,
  IterationLimit,
  CapacityOverflow,
  UnknownIdentity
};
struct PropagationResult {
  Status status = Status::Success;
  std::size_t sweeps = 0;
};
struct AdjacencyResult {
  Status status = Status::Success;
  std::size_t edge_count = 0;
  std::vector<Id> active;
  /// Compact unique-contact table bytes; never a dense G×G matrix.
  std::size_t workspace_bytes = 0;
};

/// All spans except identities denote device storage; layouts are slot-major.
/// Occupancy is caller-defined (zero inactive, nonzero active), not a threshold.
/// Surviving prior seeds remain fixed. Newly covered cells use deterministic
/// synchronous breadth-first ownership, equal-distance ties choosing smaller Id.
/// No new identities are allocated. Unseeded active regions are an error.
/// Success publishes labels; expected failures leave output unchanged.
/// This synchronous API stages work internally and completes before returning.
#if defined(OpenPFC_ENABLE_CUDA)
PropagationResult propagate(backend::CUDATag, Grid2D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t max_sweeps);
AdjacencyResult adjacency(backend::CUDATag, Grid2D grid, Slot slots,
                          std::span<const Id> labels, std::span<const Id> identities,
                          std::span<Contact> output, std::size_t contact_radius = 1);
namespace detail {
PropagationResult propagate(backend::CUDATag, Grid3D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t max_sweeps);
AdjacencyResult adjacency(backend::CUDATag, Grid3D grid, Slot slots,
                          std::span<const Id> labels, std::span<const Id> identities,
                          std::span<Contact> output, std::size_t contact_radius = 1);
} // namespace detail
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline PropagationResult propagate(backend::CUDATag tag, Grid grid, Slot slots,
                                   std::span<const std::uint8_t> occupancy,
                                   std::span<const Id> seeds, std::span<Id> output,
                                   std::size_t max_sweeps) {
  return detail::propagate(tag, grid, slots, occupancy, seeds, output, max_sweeps);
}
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline AdjacencyResult
adjacency(backend::CUDATag tag, Grid grid, Slot slots, std::span<const Id> labels,
          std::span<const Id> identities, std::span<Contact> output,
          std::size_t contact_radius = 1) {
  return detail::adjacency(tag, grid, slots, labels, identities, output,
                           contact_radius);
}

#endif
#if defined(OpenPFC_ENABLE_HIP)
PropagationResult propagate(backend::HIPTag, Grid2D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t max_sweeps);
AdjacencyResult adjacency(backend::HIPTag, Grid2D grid, Slot slots,
                          std::span<const Id> labels, std::span<const Id> identities,
                          std::span<Contact> output, std::size_t contact_radius = 1);
namespace detail {
PropagationResult propagate(backend::HIPTag, Grid3D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t max_sweeps);
AdjacencyResult adjacency(backend::HIPTag, Grid3D grid, Slot slots,
                          std::span<const Id> labels, std::span<const Id> identities,
                          std::span<Contact> output, std::size_t contact_radius = 1);
} // namespace detail
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline PropagationResult propagate(backend::HIPTag tag, Grid grid, Slot slots,
                                   std::span<const std::uint8_t> occupancy,
                                   std::span<const Id> seeds, std::span<Id> output,
                                   std::size_t max_sweeps) {
  return detail::propagate(tag, grid, slots, occupancy, seeds, output, max_sweeps);
}
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline AdjacencyResult
adjacency(backend::HIPTag tag, Grid grid, Slot slots, std::span<const Id> labels,
          std::span<const Id> identities, std::span<Contact> output,
          std::size_t contact_radius = 1) {
  return detail::adjacency(tag, grid, slots, labels, identities, output,
                           contact_radius);
}

#endif
/// Adjacency checks same-cell cross-plane overlaps and all stencil neighbors
/// across all slots; contact_radius is explicit (Four/Six Manhattan /
/// Eight/TwentySix Chebyshev). Identities is a sorted unique nonzero HOST registry.
/// Output is a canonical DEVICE contact list; active is compact HOST metadata.
/// edge_count reports required capacity, including on CapacityOverflow. Unknown
/// labels and insufficient edge capacity leave output unchanged. Unique contacts
/// use an open-addressed table of O(edge capacity) slots, not a dense G×G
/// matrix. `workspace_bytes` is that table. Size overflow is rejected, and
/// allocation/runtime failures throw.

} // namespace pfc::grain::tracking
