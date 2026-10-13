// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/grain/diagnostics.hpp>
#include <openpfc/kernel/grain/topology.hpp>

#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace pfc::grain {

/// One persistent grain changes slot; source must describe the original state.
/// Duplicate identities are invalid. A validated self-move is a no-op.
struct Transfer {
  Id id = background;
  Slot source = unassigned;
  Slot destination = unassigned;
};

/// Failure leaves every input untouched and returns no staged state.
enum class TransferStatus : unsigned {
  Success = 0,
  InvalidLayout,
  InvalidRegistry,
  InvalidTransfer,
  StaleSource,
  MissingSupport,
  InvalidSupport,
  OccupiedDestination
};

/// All members are published together by the caller after Success.
template <typename Values = std::vector<double>, typename Labels = std::vector<Id>>
struct TransferResult {
  TransferStatus status = TransferStatus::Success;
  Values values;
  Labels labels;
  std::vector<Grain> grains;
};

/// Source/destination storage slots for one active physical UID. Trivially
/// copyable; validated plans may upload the table to device memory.
struct Assignment {
  Id id;
  Slot source;
  Slot destination;
};

/// Host-owned plan. A public preparation failure returns no payload.
/// Publish grains with values/labels only after every cell stage succeeds.
struct TransferPlan {
  TransferStatus status = TransferStatus::Success;
  std::vector<Assignment> assignments;
  std::vector<Grain> grains;
};

namespace detail {
// Preserve existing internal/source callers while making type ownership public.
using Assignment = pfc::grain::Assignment;
using PreparedTransfer = pfc::grain::TransferPlan;

template <class Grid>
inline TransferStatus transfer_layout(const Grid &grid, Slot slots,
                                      std::size_t values, std::size_t labels,
                                      double background_value) {
  if (!std::isfinite(background_value)) return TransferStatus::InvalidLayout;
  std::size_t cells;
  try {
    cells = cell_count(grid);
  } catch (const std::exception &) {
    return TransferStatus::InvalidLayout;
  }
  if (slots > std::numeric_limits<std::size_t>::max() / cells)
    return TransferStatus::InvalidLayout;
  const auto count = cells * slots;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(Id) ||
      values != count || labels != count)
    return TransferStatus::InvalidLayout;
  return TransferStatus::Success;
}

template <bool Observe = false>
OPENPFC_INLINE_HD std::size_t
find_assignment(const Assignment *assignments, std::size_t count, Id id,
                diagnostics::Accesses *counts = nullptr) {
  std::size_t first = 0, last = count;
  while (first < last) {
    const auto middle = first + (last - first) / 2;
    if (diagnostics::load<Observe>(assignments, middle, counts,
                                   diagnostics::Field::Assignments)
            .id < id)
      first = middle + 1;
    else
      last = middle;
  }
  return first < count && diagnostics::load<Observe>(assignments, first, counts,
                                                     diagnostics::Field::Assignments)
                                  .id == id
             ? first
             : count;
}

inline PreparedTransfer prepare_transfer(Slot slots, std::span<const Grain> grains,
                                         std::span<const Transfer> moves) {
  PreparedTransfer prepared;
  Id previous = background;
  for (const auto &grain : grains) {
    if (grain.id == background || grain.id <= previous ||
        (grain.active && (grain.slot == unassigned || grain.slot >= slots)) ||
        (!grain.active && grain.slot != unassigned)) {
      prepared.status = TransferStatus::InvalidRegistry;
      return prepared;
    }
    previous = grain.id;
    if (grain.active)
      prepared.assignments.push_back({grain.id, grain.slot, grain.slot});
  }
  prepared.grains.assign(grains.begin(), grains.end());
  std::vector<Id> moved;
  for (const auto &move : moves) {
    const auto index = find_assignment(prepared.assignments.data(),
                                       prepared.assignments.size(), move.id);
    if (move.id == background || move.destination >= slots ||
        index == prepared.assignments.size() ||
        std::find(moved.begin(), moved.end(), move.id) != moved.end()) {
      prepared.status = TransferStatus::InvalidTransfer;
      return prepared;
    }
    auto &assignment = prepared.assignments[index];
    if (move.source != assignment.source) {
      prepared.status = TransferStatus::StaleSource;
      return prepared;
    }
    moved.push_back(move.id);
    assignment.destination = move.destination;
  }
  for (auto &grain : prepared.grains) {
    if (!grain.active) continue;
    const auto index = find_assignment(prepared.assignments.data(),
                                       prepared.assignments.size(), grain.id);
    grain.slot = prepared.assignments[index].destination;
  }
  return prepared;
}

OPENPFC_INLINE_HD TransferStatus worst(TransferStatus first, TransferStatus second) {
  return static_cast<unsigned>(first) > static_cast<unsigned>(second) ? first
                                                                      : second;
}

/// One thread owns all slots at a cell. Proposals always read original buffers.
/// Returns a deterministic failure priority shared by host and device paths.
template <bool Observe = false>
OPENPFC_INLINE_HD TransferStatus stage_cell(
    std::size_t cell, std::size_t cells, Slot slots, const double *values,
    const Id *labels, const Assignment *assignments, std::size_t assignment_count,
    double background_value, double *staged_values, Id *staged_labels,
    diagnostics::Accesses *counts = nullptr) {
  auto status = TransferStatus::Success;
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    diagnostics::store<Observe>(
        staged_values, i,
        diagnostics::load<Observe>(values, i, counts, diagnostics::Field::Values),
        counts, diagnostics::Field::StagedValues);
    diagnostics::store<Observe>(
        staged_labels, i,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts, diagnostics::Field::StagedLabels);
    // Comparisons reject NaN and both infinities without vendor math intrinsics.
    if (!(diagnostics::load<Observe>(
              values, i, counts, diagnostics::Field::Values) >= background_value &&
          diagnostics::load<Observe>(
              values, i, counts, diagnostics::Field::Values) <= largest_finite) ||
        ((diagnostics::load<Observe>(labels, i, counts,
                                     diagnostics::Field::Labels) == background) !=
         (diagnostics::load<Observe>(
              values, i, counts, diagnostics::Field::Values) == background_value)))
      status = worst(status, TransferStatus::InvalidSupport);
    if (diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels) ==
        background)
      continue;
    const auto assignment = find_assignment<Observe>(
        assignments, assignment_count,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts);
    if (assignment == assignment_count ||
        diagnostics::load<Observe>(assignments, assignment, counts,
                                   diagnostics::Field::Assignments)
                .source != slot) {
      status = worst(status, TransferStatus::InvalidSupport);
      continue;
    }
    // Two original grains must never publish to the same slot at this cell.
    for (Slot other = 0; other < slot; ++other) {
      const auto label = diagnostics::load<Observe>(
          labels, cells * other + cell, counts, diagnostics::Field::Labels);
      if (label == background) continue;
      const auto neighbor =
          find_assignment<Observe>(assignments, assignment_count, label, counts);
      if (neighbor != assignment_count &&
          diagnostics::load<Observe>(assignments, neighbor, counts,
                                     diagnostics::Field::Assignments)
                  .source == other &&
          diagnostics::load<Observe>(assignments, neighbor, counts,
                                     diagnostics::Field::Assignments)
                  .destination ==
              diagnostics::load<Observe>(assignments, assignment, counts,
                                         diagnostics::Field::Assignments)
                  .destination)
        status = worst(status, TransferStatus::OccupiedDestination);
    }
  }
  if (status != TransferStatus::Success) return status;
  // Clear all vacating sources first; then populate from immutable originals.
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    if (diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels) ==
        background)
      continue;
    const auto assignment = find_assignment<Observe>(
        assignments, assignment_count,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts);
    if (diagnostics::load<Observe>(assignments, assignment, counts,
                                   diagnostics::Field::Assignments)
            .destination == slot)
      continue;
    diagnostics::store<Observe>(staged_values, i, background_value, counts,
                                diagnostics::Field::StagedValues);
    diagnostics::store<Observe>(staged_labels, i, Id{background}, counts,
                                diagnostics::Field::StagedLabels);
  }
  for (Slot slot = 0; slot < slots; ++slot) {
    const auto i = cells * slot + cell;
    if (diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels) ==
        background)
      continue;
    const auto assignment = find_assignment<Observe>(
        assignments, assignment_count,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts);
    const auto destination =
        diagnostics::load<Observe>(assignments, assignment, counts,
                                   diagnostics::Field::Assignments)
            .destination;
    if (destination == slot) continue;
    diagnostics::store<Observe>(
        staged_values, cells * destination + cell,
        diagnostics::load<Observe>(values, i, counts, diagnostics::Field::Values),
        counts, diagnostics::Field::StagedValues);
    diagnostics::store<Observe>(
        staged_labels, cells * destination + cell,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts, diagnostics::Field::StagedLabels);
  }
  return status;
}

} // namespace detail

[[nodiscard]] inline TransferPlan prepare_transfer(Slot slots,
                                                   std::span<const Grain> grains,
                                                   std::span<const Transfer> moves) {
  auto plan = detail::prepare_transfer(slots, grains, moves);
  if (plan.status != TransferStatus::Success) return {plan.status, {}, {}};
  return plan;
}

/// Binary lookup in a sorted table produced by prepare_transfer. Missing UID
/// returns count. Host/device callers own the table's lifetime and residency.
OPENPFC_INLINE_HD std::size_t find_assignment(const Assignment *assignments,
                                              std::size_t count, Id id) {
  return detail::find_assignment(assignments, count, id);
}

/**
 * Stage a local 2D transaction in slot-major arrays: slot*(nx*ny)+x+nx*y.
 *
 * Values must be finite and >= background_value. Every nonbackground value
 * has a nonzero persistent label, every background value has label zero, and
 * each active grain has support in its recorded source slot. Labels identify
 * the entire support, including diffuse tails; this API discards none.
 *
 * All moves read the original state. Valid cycles and noncontiguous supports
 * are supported. A destination may contain a simultaneously vacating grain,
 * but two supports cannot publish into the same cell/slot. Unrelated values
 * and labels are copied exactly. The multiset of nonbackground values per
 * cell is preserved, hence summed occupancy (value-background_value) is
 * conserved algebraically without tail loss. Floating summation order may
 * differ after a permutation; there is no transfer error or tolerance knob.
 *
 * No graph, tracker, halo, MPI, or snapshot epoch is modified. Publication is
 * caller-owned and must replace values, labels, and grain slots together.
 */
template <bool Observe, class Grid>
inline TransferResult<>
transfer_impl(const Grid &grid, Slot slots, std::span<const double> values,
              std::span<const Id> labels, std::span<const Grain> grains,
              std::span<const Transfer> moves, double background_value = 0.0) {
  const auto layout = detail::transfer_layout(grid, slots, values.size(),
                                              labels.size(), background_value);
  if (layout != TransferStatus::Success) return {layout, {}, {}, {}};
  auto prepared = detail::prepare_transfer(slots, grains, moves);
  if (prepared.status != TransferStatus::Success)
    return {prepared.status, {}, {}, {}};
  const auto cells = cell_count(grid);
  auto *counts = diagnostics::host_accesses();
  TransferResult<> result;
  result.values.resize(values.size());
  result.labels.resize(labels.size());
  std::vector<unsigned> present(prepared.assignments.size(), 0);
  if constexpr (Observe) {
    diagnostics::scan(counts, diagnostics::Phase::Transfer, slots);
    for (std::size_t i = 0; i < values.size(); ++i) {
      diagnostics::write(counts, diagnostics::Field::StagedValues, sizeof(double));
      diagnostics::write(counts, diagnostics::Field::StagedLabels, sizeof(Id));
    }
  }
  for (std::size_t cell = 0; cell < cells; ++cell) {
    result.status = detail::worst(
        result.status,
        detail::stage_cell<Observe>(cell, cells, slots, values.data(), labels.data(),
                                    prepared.assignments.data(), present.size(),
                                    background_value, result.values.data(),
                                    result.labels.data(), counts));
    for (Slot slot = 0; slot < slots; ++slot) {
      const auto index = detail::find_assignment<Observe>(
          prepared.assignments.data(), present.size(),
          diagnostics::load<Observe>(labels.data(), cells * slot + cell, counts,
                                     diagnostics::Field::Labels),
          counts);
      if (index != present.size()) {
        diagnostics::store<Observe>(present.data(), index, 1u, counts,
                                    diagnostics::Field::Counts);
      }
    }
  }
  if (std::find(present.begin(), present.end(), 0) != present.end())
    result.status = detail::worst(result.status, TransferStatus::MissingSupport);
  if (result.status != TransferStatus::Success) return {result.status, {}, {}, {}};
  result.grains = std::move(prepared.grains);
  return result;
}

template <class Grid>
inline TransferResult<>
transfer_dispatch(const Grid &grid, Slot slots, std::span<const double> values,
                  std::span<const Id> labels, std::span<const Grain> grains,
                  std::span<const Transfer> moves, double background_value = 0.0) {
  if (diagnostics::current) {
    try {
      diagnostics::current->covered();
      return transfer_impl<true>(grid, slots, values, labels, grains, moves,
                                 background_value);
    } catch (...) {
      diagnostics::current->fail();
      throw;
    }
  }
  return transfer_impl<false>(grid, slots, values, labels, grains, moves,
                              background_value);
}

inline TransferResult<>
transfer(const Grid2D &grid, Slot slots, std::span<const double> values,
         std::span<const Id> labels, std::span<const Grain> grains,
         std::span<const Transfer> moves, double background_value = 0.0) {
  return transfer_dispatch(grid, slots, values, labels, grains, moves,
                           background_value);
}
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline TransferResult<>
transfer(const Grid &grid, Slot slots, std::span<const double> values,
         std::span<const Id> labels, std::span<const Grain> grains,
         std::span<const Transfer> moves, double background_value = 0.0) {
  return transfer_dispatch(grid, slots, values, labels, grains, moves,
                           background_value);
}

} // namespace pfc::grain
