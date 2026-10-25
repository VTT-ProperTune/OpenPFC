// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <openpfc/kernel/grain/diagnostics.hpp>
#include <openpfc/kernel/grain/transfer.hpp>
#include <openpfc/runtime/cpu/detail/grain_coloring.hpp>

#include <chrono>
#include <queue>

namespace pfc::grain::remapping {

/// Expected failures never publish partial values, labels, or slot assignments.
enum class Status {
  Success,
  Deferred,
  InvalidInput,
  UnknownIdentity,
  Unseeded,
  IterationLimit,
  UnsafeCadence,
  CapacityOverflow,
  MissingSupport,
  SearchLimit,
  Infeasible,
  TransferFailure
};
enum class Method {
  Incremental,
  GlobalSaturation,
  GlobalLargestFirst,
  CompleteOracle
};

struct Options {
  Diagnostics *diagnostics =
      nullptr; ///< Null explicitly disables nested observation.
  bool check_now = true;
  std::uint64_t epoch = 0; ///< Caller observation epoch, not numerical time.
  std::size_t contact_radius = 3;
  std::size_t contact_capacity = 4096;
  std::size_t max_sweeps = 128;
  Method method = Method::Incremental;
  reference::Limits limits{};
  bool componentwise =
      true; ///< Practical global policies preserve proper components.
};

/// Synchronized wall-clock stage durations, not performance evidence.
/// Byte counts describe owned storage/logical payload, not HBM transactions.
struct Statistics {
  double detection_seconds = 0;
  double adjacency_seconds = 0;
  double decision_seconds = 0;
  double transfer_seconds = 0;
  double synchronization_seconds = 0;
  double total_seconds = 0;
  std::size_t propagation_sweeps = 0;
  std::size_t edges = 0;
  std::size_t attempts = 0;
  std::size_t changed_grains = 0;
  std::uint64_t moved_samples = 0;
  std::uint64_t moved_value_bytes = 0;
  std::uint64_t moved_label_bytes = 0;
  std::size_t published_storage_bytes = 0;
  std::size_t staged_storage_bytes =
      0; ///< Whole-array transfer staging, even for no-op plans.
  std::size_t wrapper_storage_bytes = 0; ///< Excludes internals of constituent APIs.
  std::size_t graph_device_to_host_bytes = 0;
  bool conflict = false;
};

/// On success publish all three owning members together. Snapshot topology is
/// rebuilt at every checked call, including no-op calls; it is never stale.
template <typename Values = std::vector<double>, typename Labels = std::vector<Id>>
struct Result {
  Status status = Status::InvalidInput;
  TransferStatus transfer_status = TransferStatus::Success;
  Values values;
  Labels labels;
  Snapshot snapshot;
  Statistics statistics;
};

namespace detail {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Shared dispatch bookkeeping; timing observes the same semantic stages on
// both backends and includes private scratch destruction in total_seconds.
struct StageTiming {
  Statistics &statistics;
  Clock::time_point start = Clock::now();
  Clock::time_point stage_start = start;
  double *stage = nullptr;
  void begin(double &duration, Clock::time_point now) {
    stage_start = now;
    stage = &duration;
  }
  void end() { stage = nullptr; }
  Status finish(Status status) {
    if (stage) *stage = seconds(stage_start);
    statistics.total_seconds = seconds(start);
    return status;
  }
};

template <class Operation>
auto observe_operation(const Options &options, Operation &&operation) {
  diagnostics::Scope scope(options.diagnostics);
  if (options.diagnostics) options.diagnostics->covered();
  const auto start = Clock::now();
  auto completed = [&] {
    try {
      return options.diagnostics ? operation.template operator()<true>()
                                 : operation.template operator()<false>();
    } catch (...) {
      if (options.diagnostics) options.diagnostics->fail();
      throw;
    }
  }(); // All per-operation scratch has been destroyed here.
  completed.statistics.total_seconds = seconds(start);
  return completed;
}

template <class Grid>
OPENPFC_INLINE_HD bool neighbor(Grid grid, std::size_t cell, int dx, int dy, int dz,
                                std::size_t &result) {
  return pfc::grain::detail::offset(grid, cell, dx, dy, dz, result);
}

template <class Grid>
inline bool layout(Grid grid, Slot slots, std::size_t values, std::size_t labels,
                   const Options &options) {
  return slots != 0 && slots != unassigned &&
         values <= std::numeric_limits<std::size_t>::max() / 64 &&
         pfc::grain::detail::transfer_layout(grid, slots, values, labels, 0) ==
             TransferStatus::Success &&
         grid.nx <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         grid.ny <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         pfc::grain::detail::depth(grid) <=
             static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         pfc::grain::detail::connectivity_valid(grid) &&
         options.contact_radius >= 1 &&
         options.contact_radius <
             static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         options.contact_capacity <=
             (std::numeric_limits<std::size_t>::max() - 64 * values) /
                 sizeof(Contact) &&
         (options.method == Method::Incremental ||
          options.method == Method::GlobalSaturation ||
          options.method == Method::GlobalLargestFirst ||
          options.method == Method::CompleteOracle);
}

inline std::vector<Id>
identities(std::span<const pfc::grain::detail::Assignment> assignments) {
  std::vector<Id> ids;
  for (const auto &assignment : assignments) ids.push_back(assignment.id);
  return ids;
}

// Independent distance BFS gives the exact synchronous propagation horizon;
// the ownership values themselves come from the prerequisite priority-queue oracle.
template <bool Observe = false, class Grid>
inline std::size_t horizon(Grid grid, Slot slots,
                           std::span<const std::uint8_t> occupied,
                           std::span<const Id> seeds, bool complete) {
  auto *counts = diagnostics::host_accesses();
  if constexpr (Observe)
    diagnostics::scan(counts, diagnostics::Phase::Inspection, slots);
  const auto cells = cell_count(grid),
             absent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> distance(occupied.size(), absent);
  std::queue<std::size_t> queue;
  for (std::size_t i = 0; i < occupied.size(); ++i)
    if (diagnostics::load<Observe>(occupied.data(), i, counts,
                                   diagnostics::Field::Occupancy) &&
        diagnostics::load<Observe>(seeds.data(), i, counts,
                                   diagnostics::Field::Labels)) {
      distance[i] = 0;
      queue.push(i);
    }
  std::size_t maximum = 0;
  while (!queue.empty()) {
    auto i = queue.front();
    queue.pop();
    maximum = std::max(maximum, distance[i]);
    const int zr = pfc::grain::detail::depth_radius(grid, 1);
    for (int dz = -zr; dz <= zr; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if ((!dx && !dy && !dz) ||
              !pfc::grain::detail::stencil(grid, dx, dy, dz, 1))
            continue;
          std::size_t cell;
          if (!neighbor(grid, i % cells, dx, dy, dz, cell)) continue;
          const auto j = (i / cells) * cells + cell;
          if (diagnostics::load<Observe>(occupied.data(), j, counts,
                                         diagnostics::Field::Occupancy) &&
              distance[j] == absent) {
            distance[j] = distance[i] + 1;
            queue.push(j);
          }
        }
  }
  (void)slots;
  return complete ? std::max<std::size_t>(1, maximum) : maximum + 1;
}

template <bool Observe = false, class Grid>
inline bool unsafe(Grid grid, std::span<const Id> labels) {
  auto *counts = diagnostics::host_accesses();
  const auto cells = cell_count(grid);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!diagnostics::load<Observe>(labels.data(), i, counts,
                                    diagnostics::Field::Labels))
      continue;
    const int zr = pfc::grain::detail::depth_radius(grid, 1);
    for (int dz = -zr; dz <= zr; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if ((!dx && !dy && !dz) ||
              !pfc::grain::detail::stencil(grid, dx, dy, dz, 1))
            continue;
          std::size_t cell;
          if (!neighbor(grid, i % cells, dx, dy, dz, cell)) continue;
          const auto other =
              diagnostics::load<Observe>(labels.data(), (i / cells) * cells + cell,
                                         counts, diagnostics::Field::Labels);
          if (other &&
              other != diagnostics::load<Observe>(labels.data(), i, counts,
                                                  diagnostics::Field::Labels))
            return true;
        }
  }
  return false;
}

struct Decision {
  Status status = Status::Success;
  std::vector<Transfer> moves;
  std::size_t attempts = 0;
  bool conflict = false;
};

inline Decision decide(const ContactGraph &contacts, std::span<const Grain> grains,
                       Slot slots, std::span<const double> weights,
                       const Options &options) {
  reference::Graph graph(contacts.vertices.size());
  std::vector<reference::Color> before;
  for (auto id : contacts.vertices) {
    auto grain =
        std::lower_bound(grains.begin(), grains.end(), id,
                         [](const Grain &g, Id value) { return g.id < value; });
    if (grain == grains.end() || grain->id != id || !grain->active)
      return {Status::UnknownIdentity, {}, 0, false};
    before.push_back(grain->slot);
  }
  for (auto edge : contacts.edges) {
    auto a = std::lower_bound(contacts.vertices.begin(), contacts.vertices.end(),
                              edge.first) -
             contacts.vertices.begin();
    auto b = std::lower_bound(contacts.vertices.begin(), contacts.vertices.end(),
                              edge.second) -
             contacts.vertices.begin();
    graph[a].push_back(b);
    graph[b].push_back(a);
  }
  Decision decision;
  if (reference::proper_coloring(graph, before, slots)) return decision;
  decision.conflict = true;
  reference::Result solved{reference::Status::success, before, 0};
  if (options.method == Method::Incremental) {
    solved = reference::incremental_coloring(graph, before, slots, options.limits);
  } else if (options.method == Method::CompleteOracle) {
    solved = reference::global_coloring(graph, slots, options.limits);
    if (solved.status == reference::Status::success)
      solved.colors = reference::align_colors(before, solved.colors, slots, weights);
  } else {
    std::vector<std::vector<std::size_t>> components;
    std::vector<bool> visited(graph.size(), false);
    if (!options.componentwise) {
      components.emplace_back();
      for (std::size_t i = 0; i < graph.size(); ++i) components.back().push_back(i);
    } else {
      for (std::size_t i = 0; i < graph.size(); ++i) {
        if (visited[i]) continue;
        components.push_back({i});
        visited[i] = true;
        for (std::size_t j = 0; j < components.back().size(); ++j)
          for (auto next : graph[components.back()[j]])
            if (!visited[next]) {
              visited[next] = true;
              components.back().push_back(next);
            }
        std::sort(components.back().begin(), components.back().end());
      }
    }
    for (const auto &component : components) {
      reference::Graph part(component.size());
      std::vector<reference::Color> prior;
      std::vector<double> weight;
      for (std::size_t v = 0; v < component.size(); ++v) {
        prior.push_back(before[component[v]]);
        weight.push_back(weights[component[v]]);
        for (auto u : graph[component[v]])
          part[v].push_back(std::lower_bound(component.begin(), component.end(), u) -
                            component.begin());
      }
      if (reference::proper_coloring(part, prior, slots)) continue;
      auto remaining = options.limits;
      remaining.attempts -= solved.attempts; // One budget across the invocation.
      auto colored = reference::global_greedy_coloring(
          part, slots,
          options.method == Method::GlobalSaturation
              ? reference::GlobalOrder::saturation
              : reference::GlobalOrder::largest_first,
          remaining);
      solved.attempts += colored.attempts;
      if (colored.status != reference::Status::success) {
        solved.status = colored.status;
        break;
      }
      auto aligned = reference::align_colors(prior, colored.colors, slots, weight);
      for (std::size_t i = 0; i < component.size(); ++i)
        solved.colors[component[i]] = aligned[i];
    }
  }
  decision.attempts = solved.attempts;
  if (solved.status != reference::Status::success) {
    decision.status = solved.status == reference::Status::infeasible
                          ? Status::Infeasible
                          : Status::SearchLimit;
    return decision;
  }
  if (!reference::proper_coloring(graph, solved.colors, slots))
    throw std::logic_error("invalid coloring result");
  for (std::size_t i = 0; i < before.size(); ++i)
    if (before[i] != solved.colors[i])
      decision.moves.push_back({contacts.vertices[i], static_cast<Slot>(before[i]),
                                static_cast<Slot>(solved.colors[i])});
  return decision;
}

inline void payload(Statistics &statistics, std::span<const Id> ids,
                    std::span<const std::uint64_t> counts,
                    std::span<const Transfer> moves) {
  statistics.changed_grains = moves.size();
  for (auto move : moves) {
    auto i = std::lower_bound(ids.begin(), ids.end(), move.id) - ids.begin();
    statistics.moved_samples += counts[i];
  }
  statistics.moved_value_bytes = statistics.moved_samples * sizeof(double);
  statistics.moved_label_bytes = statistics.moved_samples * sizeof(Id);
}
} // namespace detail

/// Graph-only decision: no tracking, support cutting or field transfer.
using Decision = detail::Decision;

/** Plan slot moves for a canonical contact graph and sorted grain registry.
 * Contacts may describe a subset of the active registry. Weights are finite,
 * nonnegative and match vertices, including for methods that ignore weights.
 * Malformed public inputs return InvalidInput; absent/inactive graph identities
 * return UnknownIdentity. Inputs remain unchanged; callers stage transfers and
 * publish their accepted state separately. The existing coloring policy is used.
 */
[[nodiscard]] inline Decision decide(const ContactGraph &contacts,
                                     std::span<const Grain> grains, Slot slots,
                                     std::span<const double> weights,
                                     const Options &options = {}) {
  if (options.method != Method::Incremental &&
      options.method != Method::GlobalSaturation &&
      options.method != Method::GlobalLargestFirst &&
      options.method != Method::CompleteOracle)
    return {Status::InvalidInput, {}, 0, false};
  try {
    pfc::grain::validate(contacts);
  } catch (const std::invalid_argument &) {
    return {Status::InvalidInput, {}, 0, false};
  }
  if (weights.size() != contacts.vertices.size())
    return {Status::InvalidInput, {}, 0, false};
  for (double w : weights)
    if (!std::isfinite(w) || w < 0) return {Status::InvalidInput, {}, 0, false};
  Id previous = background;
  for (const auto &grain : grains) {
    if (!grain.id || grain.id <= previous ||
        (grain.active && (grain.slot == unassigned || grain.slot >= slots)) ||
        (!grain.active && grain.slot != unassigned))
      return {Status::InvalidInput, {}, 0, false};
    previous = grain.id;
  }
  return detail::decide(contacts, grains, slots, weights, options);
}
} // namespace pfc::grain::remapping
