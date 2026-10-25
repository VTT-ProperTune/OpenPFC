// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Instantiate in a CUDA/HIP translation unit; field arrays stay device-owned.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include <openpfc/runtime/common/grain_remapping.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>

namespace pfc::grain::remapping {
namespace detail {
template <bool Observe>
__device__ void error_max(unsigned *error, Status status,
                          diagnostics::Accesses *counts) {
  if constexpr (Observe) {
    diagnostics::read(counts, diagnostics::Field::Flags, sizeof(unsigned));
    diagnostics::write(counts, diagnostics::Field::Flags, sizeof(unsigned));
  }
  atomicMax(error, static_cast<unsigned>(status));
}
template <bool Observe>
__global__ void
preflight(std::size_t cells, Slot slots, const double *values, const Id *seeds,
          const pfc::grain::detail::Assignment *assignments, std::size_t grains,
          std::uint8_t *occupied, unsigned *error, diagnostics::Accesses *totals) {
  std::conditional_t<Observe, diagnostics::Accesses, diagnostics::NoAccesses>
      local{};
  diagnostics::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < cells * slots; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto q =
        diagnostics::load<Observe>(values, i, counts, diagnostics::Field::Values);
    diagnostics::store<Observe>(occupied, i, std::uint8_t(q > 0), counts,
                                diagnostics::Field::Occupancy);
    if (!(q >= 0 && q <= pfc::grain::largest_finite)) {
      error_max<Observe>(error, Status::InvalidInput, counts);
      continue;
    }
    if (!diagnostics::load<Observe>(occupied, i, counts,
                                    diagnostics::Field::Occupancy) ||
        !diagnostics::load<Observe>(seeds, i, counts, diagnostics::Field::Labels))
      continue;
    const auto j = pfc::grain::detail::find_assignment<Observe>(
        assignments, grains,
        diagnostics::load<Observe>(seeds, i, counts, diagnostics::Field::Labels),
        counts);
    if (j == grains)
      error_max<Observe>(error, Status::UnknownIdentity, counts);
    else if (diagnostics::load<Observe>(assignments, j, counts,
                                        diagnostics::Field::Assignments)
                 .source != i / cells)
      error_max<Observe>(error, Status::InvalidInput, counts);
  }
  if constexpr (Observe) diagnostics::merge(totals, local);
}
template <bool Observe, class Grid>
__global__ void inspect(Grid grid, Slot slots, const Id *labels,
                        const pfc::grain::detail::Assignment *assignments,
                        std::size_t grains, unsigned long long *support,
                        unsigned *error, diagnostics::Accesses *totals) {
  std::conditional_t<Observe, diagnostics::Accesses, diagnostics::NoAccesses>
      local{};
  diagnostics::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  const auto cells = grid.nx * grid.ny * pfc::grain::detail::depth(grid);
  for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < cells * slots; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (!diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels))
      continue;
    const auto index = pfc::grain::detail::find_assignment<Observe>(
        assignments, grains,
        diagnostics::load<Observe>(labels, i, counts, diagnostics::Field::Labels),
        counts);
    if (index == grains) {
      error_max<Observe>(error, Status::UnknownIdentity, counts);
      continue;
    }
    if constexpr (Observe) {
      diagnostics::read(counts, diagnostics::Field::Counts,
                        sizeof(unsigned long long));
      diagnostics::write(counts, diagnostics::Field::Counts,
                         sizeof(unsigned long long));
    }
    atomicAdd(support + index, 1ULL);
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
              diagnostics::load<Observe>(labels, (i / cells) * cells + cell, counts,
                                         diagnostics::Field::Labels);
          if (other && other != diagnostics::load<Observe>(
                                    labels, i, counts, diagnostics::Field::Labels))
            error_max<Observe>(error, Status::UnsafeCadence, counts);
        }
  }
  if constexpr (Observe) diagnostics::merge(totals, local);
}
} // namespace detail

/** Same owning transaction and cadence contract as the CPU remap overload.
 * Synchronizes prior work on stream; all constituent staging completes before
 * returning. Only compact counts/graph/registry/status return to the host.
 * Fatal allocation/runtime exceptions preserve const originals where hardware
 * remains usable; this is not recovery from a poisoned device/runtime.
 */
template <typename Backend, class Grid>
Result<pfc::core::DataBuffer<Backend, double>, pfc::core::DataBuffer<Backend, Id>>
remap_dispatch(Grid grid, Slot slots,
               const pfc::core::DataBuffer<Backend, double> &values,
               const pfc::core::DataBuffer<Backend, Id> &seeds,
               std::span<const Grain> grains, const Options &options = {},
               pfc::gpuStream_t stream = nullptr) {
  const auto operation = [&]<bool Observe> {
    using Values = pfc::core::DataBuffer<Backend, double>;
    using Labels = pfc::core::DataBuffer<Backend, Id>;
    Result<Values, Labels> result;
    detail::StageTiming timing{result.statistics};
    const auto finish = [&](Status status) {
      result.status = timing.finish(status);
      return std::move(result);
    };
    if (!options.check_now) return finish(Status::Deferred);
    if (!detail::layout(grid, slots, values.size(), seeds.size(), options))
      return finish(Status::InvalidInput);
    auto prepared = pfc::grain::detail::prepare_transfer(slots, grains, {});
    if (prepared.status != TransferStatus::Success) {
      result.transfer_status = prepared.status;
      return finish(Status::InvalidInput);
    }
    const auto synchronize = detail::Clock::now();
    GPU_CHECK(pfc::gpuStreamSynchronize(stream));
    result.statistics.synchronization_seconds = detail::seconds(synchronize);
    const auto detection = detail::Clock::now();
    timing.begin(result.statistics.detection_seconds, detection);
    const auto cells = cell_count(grid), n = values.size();
    diagnostics::Observation<Backend> observation;
    diagnostics::Interval preflight_interval(diagnostics::Phase::Preflight);
    pfc::core::DataBuffer<Backend, pfc::grain::detail::Assignment> assignments(
        prepared.assignments.size());
    assignments.copy_from_host(prepared.assignments,
                               diagnostics::Field::Assignments);
    pfc::core::DataBuffer<Backend, std::uint8_t> occupied(n);
    Labels propagated(n);
    pfc::core::DataBuffer<Backend, unsigned> error(1);
    error.copy_from_host(std::vector<unsigned>{0}, diagnostics::Field::Flags);
    const auto blocks =
        static_cast<unsigned>(std::min<std::size_t>((n - 1) / 256 + 1, 65535));
    GPU_LAUNCH_KERNEL(detail::preflight<Observe>, blocks, 256,
                      (cells, slots, values.data(), seeds.data(), assignments.data(),
                       assignments.size(), occupied.data(), error.data(),
                       observation.data()),
                      0);
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    auto bad = static_cast<Status>(error.to_host(diagnostics::Field::Flags).front());
    diagnostics::scan(diagnostics::host_accesses(), diagnostics::Phase::Preflight,
                      slots);
    preflight_interval.finish();
    if (bad != Status::Success) {
      if (bad == Status::InvalidInput)
        result.transfer_status = TransferStatus::InvalidSupport;
      return finish(bad);
    }
    auto ownership = tracking::propagate(
        Backend{}, grid, slots, std::span<const std::uint8_t>(occupied.data(), n),
        std::span<const Id>(seeds.data(), n), std::span<Id>(propagated.data(), n),
        options.max_sweeps);
    result.statistics.propagation_sweeps = ownership.sweeps;
    if (ownership.status != tracking::Status::Success)
      return finish(ownership.status == tracking::Status::Unseeded
                        ? Status::Unseeded
                        : Status::IterationLimit);
    diagnostics::Interval inspection_interval(diagnostics::Phase::Inspection);
    pfc::core::DataBuffer<Backend, unsigned long long> device_counts(
        assignments.size());
    device_counts.copy_from_host(
        std::vector<unsigned long long>(assignments.size(), 0),
        diagnostics::Field::Counts);
    GPU_LAUNCH_KERNEL((detail::inspect<Observe, Grid>), blocks, 256,
                      (grid, slots, propagated.data(), assignments.data(),
                       assignments.size(), device_counts.data(), error.data(),
                       observation.data()),
                      0);
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    bad = static_cast<Status>(error.to_host(diagnostics::Field::Flags).front());
    diagnostics::scan(diagnostics::host_accesses(), diagnostics::Phase::Inspection,
                      slots);
    if (bad != Status::Success) return finish(bad);
    const auto counted = device_counts.to_host(diagnostics::Field::Counts);
    if (std::find(counted.begin(), counted.end(), 0) != counted.end()) {
      result.transfer_status = TransferStatus::MissingSupport;
      return finish(Status::MissingSupport);
    }
    const std::vector<std::uint64_t> counts(counted.begin(), counted.end());
    const std::vector<double> weights(counted.begin(), counted.end());
    const auto ids = detail::identities(prepared.assignments);
    result.statistics.detection_seconds = detail::seconds(detection);
    inspection_interval.finish();
    const auto adjacency = detail::Clock::now();
    timing.begin(result.statistics.adjacency_seconds, adjacency);
    pfc::core::DataBuffer<Backend, Contact> device_edges(options.contact_capacity);
    auto topology = tracking::adjacency(
        Backend{}, grid, slots, std::span<const Id>(propagated.data(), n), ids,
        std::span<Contact>(device_edges.data(), device_edges.size()),
        options.contact_radius);
    result.statistics.edges = topology.edge_count;
    if (topology.status != tracking::Status::Success) {
      result.statistics.adjacency_seconds = detail::seconds(adjacency);
      return finish(topology.status == tracking::Status::CapacityOverflow
                        ? Status::CapacityOverflow
                        : Status::UnknownIdentity);
    }
    diagnostics::Interval graph_publication_interval(diagnostics::Phase::Adjacency);
    std::vector<Contact> contacts(topology.edge_count);
    if (!contacts.empty()) {
      GPU_CHECK(pfc::gpuMemcpy(contacts.data(), device_edges.data(),
                               contacts.size() * sizeof(Contact),
                               pfc::gpuMemcpyDeviceToHost));
      diagnostics::copy(diagnostics::Direction::DeviceToHost,
                        diagnostics::Field::Contacts,
                        contacts.size() * sizeof(Contact));
    }
    auto graph = make_contact_graph(std::move(topology.active), std::move(contacts));
    if (graph.vertices != ids) return finish(Status::MissingSupport);
    result.statistics.graph_device_to_host_bytes =
        topology.edge_count * sizeof(Contact);
    result.statistics.adjacency_seconds = detail::seconds(adjacency);
    graph_publication_interval.finish();
    const auto solving = detail::Clock::now();
    diagnostics::Interval decision_interval(diagnostics::Phase::Decision);
    timing.begin(result.statistics.decision_seconds, solving);
    auto decision = detail::decide(graph, grains, slots, weights, options);
    result.statistics.attempts = decision.attempts;
    result.statistics.conflict = decision.conflict;
    result.statistics.decision_seconds = detail::seconds(solving);
    decision_interval.finish();
    if (decision.status != Status::Success) return finish(decision.status);
    detail::payload(result.statistics, ids, counts, decision.moves);
    const auto moving = detail::Clock::now();
    timing.begin(result.statistics.transfer_seconds, moving);
    result.statistics.staged_storage_bytes = n * (sizeof(double) + sizeof(Id));
    auto transaction = pfc::grain::transfer(grid, slots, values, propagated, grains,
                                            decision.moves);
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
    result.statistics.published_storage_bytes = n * (sizeof(double) + sizeof(Id));
    result.statistics.wrapper_storage_bytes =
        n * (sizeof(Id) + sizeof(std::uint8_t)) +
        device_edges.size() * sizeof(Contact) +
        assignments.size() * sizeof(pfc::grain::detail::Assignment) +
        device_counts.size() * sizeof(unsigned long long) + sizeof(unsigned);
    const auto completed = detail::Clock::now();
    GPU_CHECK(pfc::gpuDeviceSynchronize());
    result.statistics.synchronization_seconds += detail::seconds(completed);
    return finish(Status::Success);
  };
  return detail::observe_operation(options, operation);
}
template <typename Backend>
Result<pfc::core::DataBuffer<Backend, double>, pfc::core::DataBuffer<Backend, Id>>
remap(Grid2D grid, Slot slots, const pfc::core::DataBuffer<Backend, double> &values,
      const pfc::core::DataBuffer<Backend, Id> &seeds, std::span<const Grain> grains,
      const Options &options = {}, pfc::gpuStream_t stream = nullptr) {
  return remap_dispatch(grid, slots, values, seeds, grains, options, stream);
}
template <typename Backend, class Grid>
  requires std::same_as<Grid, Grid3D>
Result<pfc::core::DataBuffer<Backend, double>, pfc::core::DataBuffer<Backend, Id>>
remap(Grid grid, Slot slots, const pfc::core::DataBuffer<Backend, double> &values,
      const pfc::core::DataBuffer<Backend, Id> &seeds, std::span<const Grain> grains,
      const Options &options = {}, pfc::gpuStream_t stream = nullptr) {
  return remap_dispatch(grid, slots, values, seeds, grains, options, stream);
}
} // namespace pfc::grain::remapping
#endif
