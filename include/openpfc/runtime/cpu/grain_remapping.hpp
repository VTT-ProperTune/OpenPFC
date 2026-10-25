// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/runtime/common/grain_remapping.hpp>
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>

namespace pfc::grain::remapping {

/** Stage a complete 2D observation and remap, leaving all inputs untouched.
 * q is finite/nonnegative with exact zero background. Seeds may cover only
 * part of each active component; surviving seeds must match its recorded slot.
 * Same-slot touching UID components are conservatively UnsafeCadence even
 * with complete labels: this operation cannot certify that prior evolution
 * did not mix amplitudes. Use buffered contacts before evolution.
 * This CPU path uses independent small-grid tracking/contact reference APIs;
 * it is not a scalable contact detector. Publish owning results together.
 */
template <class Grid>
inline Result<> remap_dispatch(Grid grid, Slot slots, std::span<const double> values,
                               std::span<const Id> seeds,
                               std::span<const Grain> grains,
                               const Options &options = {}) {
  const auto operation = [&]<bool Observe> {
    auto *counters = diagnostics::host_accesses();
    Result<> result;
    detail::StageTiming timing{result.statistics};
    const auto finish = [&](Status status) {
      result.status = timing.finish(status);
      return std::move(result);
    };
    if (!options.check_now) return finish(Status::Deferred);
    if (!detail::layout(grid, slots, values.size(), seeds.size(), options) ||
        (values.size() && (!values.data() || !seeds.data())))
      return finish(Status::InvalidInput);
    auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
    if (prepared.status != TransferStatus::Success) {
      result.transfer_status = prepared.status;
      return finish(Status::InvalidInput);
    }
    const auto detection = detail::Clock::now();
    timing.begin(result.statistics.detection_seconds, detection);
    const auto cells = cell_count(grid);
    std::vector<std::uint8_t> occupied(values.size());
    if constexpr (Observe) {
      diagnostics::scan(counters, diagnostics::Phase::Preflight, slots);
      for (std::size_t i = 0; i < values.size(); ++i)
        diagnostics::write(counters, diagnostics::Field::Occupancy,
                           sizeof(std::uint8_t));
    }
    auto bad = Status::Success;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!std::isfinite(diagnostics::load<Observe>(values.data(), i, counters,
                                                    diagnostics::Field::Values)) ||
          diagnostics::load<Observe>(values.data(), i, counters,
                                     diagnostics::Field::Values) < 0) {
        bad = static_cast<Status>(
            std::max(static_cast<unsigned>(bad),
                     static_cast<unsigned>(Status::InvalidInput)));
        continue;
      }
      diagnostics::store<Observe>(
          occupied.data(), i,
          diagnostics::load<Observe>(values.data(), i, counters,
                                     diagnostics::Field::Values) > 0,
          counters, diagnostics::Field::Occupancy);
      if (!diagnostics::load<Observe>(occupied.data(), i, counters,
                                      diagnostics::Field::Occupancy) ||
          !diagnostics::load<Observe>(seeds.data(), i, counters,
                                      diagnostics::Field::Labels))
        continue;
      const auto j = pfc::grain::detail::find_assignment<Observe>(
          prepared.assignments.data(), prepared.assignments.size(),
          diagnostics::load<Observe>(seeds.data(), i, counters,
                                     diagnostics::Field::Labels),
          counters);
      if (j == prepared.assignments.size())
        bad = Status::UnknownIdentity;
      else if (prepared.assignments[j].source != i / cells)
        bad = static_cast<Status>(
            std::max(static_cast<unsigned>(bad),
                     static_cast<unsigned>(Status::InvalidInput)));
    }
    if (bad != Status::Success) {
      if (bad == Status::InvalidInput)
        result.transfer_status = TransferStatus::InvalidSupport;
      return finish(bad);
    }
    auto propagated = tracking::reference::propagate(grid, slots, occupied, seeds);
    const auto horizon =
        detail::horizon<Observe>(grid, slots, occupied, seeds, propagated.complete);
    result.statistics.propagation_sweeps = std::min(horizon, options.max_sweeps);
    if (horizon > options.max_sweeps) return finish(Status::IterationLimit);
    if (!propagated.complete) return finish(Status::Unseeded);
    if (detail::unsafe<Observe>(grid, propagated.labels))
      return finish(Status::UnsafeCadence);
    const auto ids = detail::identities(prepared.assignments);
    std::vector<std::uint64_t> counts(ids.size(), 0);
    if constexpr (Observe)
      diagnostics::scan(counters, diagnostics::Phase::Inspection, slots);
    for (std::size_t i = 0; i < propagated.labels.size(); ++i) {
      const auto id = diagnostics::load<Observe>(
          propagated.labels.data(), i, counters, diagnostics::Field::Labels);
      if (id) ++counts[std::lower_bound(ids.begin(), ids.end(), id) - ids.begin()];
    }
    if (std::find(counts.begin(), counts.end(), 0) != counts.end()) {
      result.transfer_status = TransferStatus::MissingSupport;
      return finish(Status::MissingSupport);
    }
    result.statistics.detection_seconds = detail::seconds(detection);
    const std::vector<double> weights(counts.begin(), counts.end());
    const auto adjacency = detail::Clock::now();
    timing.begin(result.statistics.adjacency_seconds, adjacency);
    auto graph = tracking::reference::contact_graph(grid, slots, propagated.labels,
                                                    options.contact_radius);
    result.statistics.edges = graph.edges.size();
    result.statistics.adjacency_seconds = detail::seconds(adjacency);
    if (graph.edges.size() > options.contact_capacity)
      return finish(Status::CapacityOverflow);
    const auto solving = detail::Clock::now();
    timing.begin(result.statistics.decision_seconds, solving);
    auto decision = detail::decide(graph, grains, slots, weights, options);
    result.statistics.attempts = decision.attempts;
    result.statistics.conflict = decision.conflict;
    result.statistics.decision_seconds = detail::seconds(solving);
    if (decision.status != Status::Success) return finish(decision.status);
    detail::payload(result.statistics, ids, counts, decision.moves);
    const auto moving = detail::Clock::now();
    timing.begin(result.statistics.transfer_seconds, moving);
    result.statistics.staged_storage_bytes =
        values.size() * (sizeof(double) + sizeof(Id));
    auto transaction = pfc::grain::transfer(grid, slots, values, propagated.labels,
                                            grains, decision.moves);
    result.transfer_status = transaction.status;
    result.statistics.transfer_seconds = detail::seconds(moving);
    timing.end();
    if (transaction.status != TransferStatus::Success)
      return finish(Status::TransferFailure);
    result.values = std::move(transaction.values);
    result.labels = std::move(transaction.labels);
    result.snapshot = {options.epoch, std::move(transaction.grains),
                       std::move(graph)};
    pfc::grain::validate(result.snapshot, slots);
    result.statistics.published_storage_bytes =
        result.values.size() * sizeof(double) + result.labels.size() * sizeof(Id);
    result.statistics.wrapper_storage_bytes =
        occupied.size() + propagated.labels.size() * sizeof(Id);
    return finish(Status::Success);
  };
  return detail::observe_operation(options, operation);
}
inline Result<> remap(Grid2D grid, Slot slots, std::span<const double> values,
                      std::span<const Id> seeds, std::span<const Grain> grains,
                      const Options &options = {}) {
  return remap_dispatch(grid, slots, values, seeds, grains, options);
}
template <class Grid>
  requires std::same_as<Grid, Grid3D>
inline Result<> remap(Grid grid, Slot slots, std::span<const double> values,
                      std::span<const Id> seeds, std::span<const Grain> grains,
                      const Options &options = {}) {
  return remap_dispatch(grid, slots, values, seeds, grains, options);
}
} // namespace pfc::grain::remapping
