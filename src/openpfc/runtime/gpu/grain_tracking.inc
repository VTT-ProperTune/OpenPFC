// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/kernel/grain/contact_hash.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/grain_diagnostics.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace pfc::grain::tracking {
#if defined(OPENPFC_GPU_API_HIP)
using Tag = backend::HIPTag;
#else
using Tag = backend::CUDATag;
#endif
namespace {
namespace dg = diagnostics;
using dg::Direction;
using dg::Field;
template <class T> struct Buffer {
  T *data = nullptr;
  std::optional<dg::AllocationToken> allocation;
  explicit Buffer(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("grain allocation byte count overflow");
    if (n) {
      GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)));
      allocation.emplace(
          dg::successful_allocation(dg::Space::Device, n * sizeof(T)));
    }
  }
  ~Buffer() {
    if (data) {
      const auto status = gpuFree(data);
      if (allocation) {
        if (status == gpuSuccess)
          allocation->release();
        else
          allocation->failed_release();
      }
    }
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};
template <class Grid> std::size_t storage_size(Grid grid, Slot slots) {
  const auto n = cell_count(grid);
  if (grid.nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      grid.ny > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      pfc::grain::detail::depth(grid) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "grain tracking axis exceeds signed coordinate limit");
  if (slots == 0 || slots == unassigned)
    throw std::invalid_argument("grain tracking requires a positive slot count");
  if (n > std::numeric_limits<std::size_t>::max() / slots)
    throw std::overflow_error("grain slot storage size overflow");
  if (!pfc::grain::detail::connectivity_valid(grid))
    throw std::invalid_argument("unknown grain connectivity");
  return n * slots;
}
template <class T> void check_span(std::span<T> data, std::size_t n) {
  if (data.size() != n || (n && !data.data()))
    throw std::invalid_argument("grain device span does not match slot grid");
}
template <class Grid>
__device__ bool neighbor(Grid grid, std::size_t cell, int dx, int dy, int dz,
                         std::size_t &result) {
  return pfc::grain::detail::offset(grid, cell, dx, dy, dz, result);
}
template <bool Observe>
__global__ void initialize(const std::uint8_t *occupancy, const Id *seeds,
                           Id *labels, std::size_t n, dg::Accesses *totals) {
  std::conditional_t<Observe, dg::Accesses, dg::NoAccesses> local{};
  dg::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    dg::store<Observe>(labels, i,
                       dg::load<Observe>(occupancy, i, counts, Field::Occupancy)
                           ? dg::load<Observe>(seeds, i, counts, Field::Labels)
                           : background,
                       counts, Field::StagedLabels);
  if constexpr (Observe) dg::merge(totals, local);
}
template <bool Observe, class Grid>
__global__ void sweep(Grid grid, std::size_t n, const std::uint8_t *occupancy,
                      const Id *before, Id *after, unsigned *flags,
                      dg::Accesses *totals) {
  std::conditional_t<Observe, dg::Accesses, dg::NoAccesses> local{};
  dg::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  const auto cells = grid.nx * grid.ny * pfc::grain::detail::depth(grid);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    Id chosen = dg::load<Observe>(before, i, counts, Field::StagedLabels);
    if (dg::load<Observe>(occupancy, i, counts, Field::Occupancy) &&
        chosen == background) {
      const auto plane = (i / cells) * cells;
      const int zr = pfc::grain::detail::depth_radius(grid, 1);
      for (int dz = -zr; dz <= zr; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if ((!dx && !dy && !dz) ||
                !pfc::grain::detail::stencil(grid, dx, dy, dz, 1))
              continue;
            std::size_t adjacent;
            if (neighbor(grid, i % cells, dx, dy, dz, adjacent)) {
              const Id candidate = dg::load<Observe>(before, plane + adjacent,
                                                     counts, Field::StagedLabels);
              if (candidate != background &&
                  (chosen == background || candidate < chosen))
                chosen = candidate;
            }
          }
    }
    dg::store<Observe>(after, i, chosen, counts, Field::StagedLabels);
    if (chosen != dg::load<Observe>(before, i, counts, Field::StagedLabels)) {
      if constexpr (Observe) dg::read(counts, Field::Flags, sizeof(unsigned));
      if constexpr (Observe) dg::write(counts, Field::Flags, sizeof(unsigned));
      atomicOr(flags, 1u);
    }
    if (dg::load<Observe>(occupancy, i, counts, Field::Occupancy) &&
        chosen == background) {
      if constexpr (Observe) dg::read(counts, Field::Flags, sizeof(unsigned));
      if constexpr (Observe) dg::write(counts, Field::Flags, sizeof(unsigned));
      atomicOr(flags, 2u);
    }
  }
  if constexpr (Observe) dg::merge(totals, local);
}
template <bool Observe, class T = unsigned>
__global__ void zero(T *data, std::size_t n, dg::Accesses *totals, Field field) {
  std::conditional_t<Observe, dg::Accesses, dg::NoAccesses> local{};
  dg::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    dg::store<Observe>(data, i, T{}, counts, field);
  if constexpr (Observe) dg::merge(totals, local);
}
template <bool Observe>
__device__ std::size_t find_id(const Id *ids, std::size_t n, Id id,
                               dg::Accesses *counts) {
  std::size_t first = 0, last = n;
  while (first < last) {
    const auto mid = first + (last - first) / 2;
    if (dg::load<Observe>(ids, mid, counts, Field::Registry) < id)
      first = mid + 1;
    else
      last = mid;
  }
  return first < n && dg::load<Observe>(ids, first, counts, Field::Registry) == id
             ? first
             : n;
}
template <bool Observe, class Grid>
__global__ void contacts(Grid grid, Slot slots, const Id *labels, const Id *ids,
                         std::size_t grains, unsigned long long *keys,
                         std::size_t table_slots, unsigned *active, unsigned *error,
                         int radius, dg::Accesses *totals) {
  std::conditional_t<Observe, dg::Accesses, dg::NoAccesses> local{};
  dg::Accesses *counts = nullptr;
  if constexpr (Observe) counts = &local;
  const auto cells = grid.nx * grid.ny * pfc::grain::detail::depth(grid),
             n = cells * slots;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const Id id = dg::load<Observe>(labels, i, counts, Field::Labels);
    if (id == background) continue;
    const auto a = find_id<Observe>(ids, grains, id, counts);
    if (a == grains) {
      {
        if constexpr (Observe) dg::read(counts, Field::Flags, sizeof(unsigned));
        if constexpr (Observe) dg::write(counts, Field::Flags, sizeof(unsigned));
        atomicOr(error, 1u);
      }
      continue;
    }
    if constexpr (Observe) dg::read(counts, Field::Counts, sizeof(unsigned));
    if constexpr (Observe) dg::write(counts, Field::Counts, sizeof(unsigned));
    atomicOr(active + a, 1u);
    const int zr = pfc::grain::detail::depth_radius(grid, radius);
    for (int dz = -zr; dz <= zr; ++dz)
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          if (!pfc::grain::detail::stencil(grid, dx, dy, dz, radius)) continue;
          std::size_t adjacent;
          if (!neighbor(grid, i % cells, dx, dy, dz, adjacent)) continue;
          for (Slot slot = 0; slot < slots; ++slot) {
            const Id other = dg::load<Observe>(
                labels, static_cast<std::size_t>(slot) * cells + adjacent, counts,
                Field::Labels);
            if (other == background || other == id) continue;
            const auto b = find_id<Observe>(ids, grains, other, counts);
            if (b == grains) {
              {
                if constexpr (Observe)
                  dg::read(counts, Field::Flags, sizeof(unsigned));
                if constexpr (Observe)
                  dg::write(counts, Field::Flags, sizeof(unsigned));
                atomicOr(error, 1u);
              }
              continue;
            }
            const auto low = a < b ? a : b, high = a < b ? b : a;
            if constexpr (Observe)
              dg::read(counts, Field::AdjacencyMatrix, sizeof(unsigned long long));
            if constexpr (Observe)
              dg::write(counts, Field::AdjacencyMatrix, sizeof(unsigned long long));
            if (!insert_contact_key(keys, table_slots, low, high))
              atomicOr(error, 2u);
          }
        }
  }
  if constexpr (Observe) dg::merge(totals, local);
}
constexpr unsigned blocks = 128, threads = 256;
} // namespace

template <bool Observe, class Grid>
PropagationResult propagate_impl(Tag, Grid grid, Slot slots,
                                 std::span<const std::uint8_t> occupancy,
                                 std::span<const Id> seeds, std::span<Id> output,
                                 std::size_t max_sweeps) {
  const auto n = storage_size(grid, slots);
  check_span(occupancy, n);
  check_span(seeds, n);
  check_span(output, n);
  dg::Observation<Tag> observation;
  dg::Interval interval(dg::Phase::Propagation);
  auto *counts = observation.data();
  Buffer<Id> first(n), second(n);
  Buffer<unsigned> flags(1);
  GPU_LAUNCH_KERNEL(initialize<Observe>, blocks, threads,
                    (occupancy.data(), seeds.data(), first.data, n, counts), 0);
  dg::scan(dg::host_accesses(), dg::Phase::Propagation, slots);
  Id *before = first.data, *after = second.data;
  PropagationResult result{Status::IterationLimit, 0};
  for (std::size_t step = 0; step < max_sweeps; ++step) {
    GPU_LAUNCH_KERNEL(zero<Observe>, 1, 1, (flags.data, 1, counts, Field::Flags), 0);
    GPU_LAUNCH_KERNEL((sweep<Observe, Grid>), blocks, threads,
                      (grid, n, occupancy.data(), before, after, flags.data, counts),
                      0);
    unsigned state = 0;
    GPU_CHECK(gpuMemcpy(&state, flags.data, sizeof(state), gpuMemcpyDeviceToHost));
    dg::copy(Direction::DeviceToHost, Field::Flags, sizeof(state));
    dg::scan(dg::host_accesses(), dg::Phase::Propagation, slots);
    ++result.sweeps;
    std::swap(before, after);
    if (!(state & 2u)) {
      result.status = Status::Success;
      break;
    }
    if (!(state & 1u)) {
      result.status = Status::Unseeded;
      break;
    }
  }
  if (result.status == Status::Success) {
    GPU_CHECK(
        gpuMemcpy(output.data(), before, n * sizeof(Id), gpuMemcpyDeviceToDevice));
    dg::copy(Direction::DeviceToDevice, Field::Labels, n * sizeof(Id));
  }
  GPU_CHECK(gpuDeviceSynchronize());
  return result;
}

template <bool Observe, class Grid>
AdjacencyResult
adjacency_impl(Tag, Grid grid, Slot slots, std::span<const Id> labels,
               std::span<const Id> identities, std::span<Contact> output,
               std::size_t contact_radius) {
  const auto n = storage_size(grid, slots);
  if (contact_radius >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "grain contact radius exceeds signed coordinate limit");
  check_span(labels, n);
  if (output.size() && !output.data())
    throw std::invalid_argument("null grain contact output");
  const auto grains = identities.size();
  // Reuse the independent canonical graph validation for the host registry.
  validate(ContactGraph{std::vector<Id>(identities.begin(), identities.end()), {}});
  if (grains > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("grain contact hash key exceeds uint32 index");
  const auto table_slots = contact_hash_slots(output.size());
  const auto table_bytes = contact_hash_bytes(output.size());
  dg::Observation<Tag> observation;
  dg::Interval interval(dg::Phase::Adjacency);
  auto *counts = observation.data();
  Buffer<Id> ids(grains);
  Buffer<unsigned long long> keys(table_slots);
  Buffer<unsigned> active(grains), error(1);
  Buffer<Contact> edges(output.size());
  if (grains) {
    GPU_CHECK(gpuMemcpy(ids.data, identities.data(), grains * sizeof(Id),
                        gpuMemcpyHostToDevice));
    dg::copy(Direction::HostToDevice, Field::Registry, grains * sizeof(Id));
  }
  GPU_LAUNCH_KERNEL((zero<Observe, unsigned long long>), blocks, threads,
                    (keys.data, table_slots, counts, Field::AdjacencyMatrix), 0);
  GPU_LAUNCH_KERNEL(zero<Observe>, blocks, threads,
                    (active.data, grains, counts, Field::Counts), 0);
  GPU_LAUNCH_KERNEL(zero<Observe>, 1, 1, (error.data, 1, counts, Field::Flags), 0);
  GPU_LAUNCH_KERNEL((contacts<Observe, Grid>), blocks, threads,
                    (grid, slots, labels.data(), ids.data, grains, keys.data,
                     table_slots, active.data, error.data,
                     static_cast<int>(contact_radius), counts),
                    0);
  unsigned bad = 0;
  GPU_CHECK(gpuMemcpy(&bad, error.data, sizeof(bad), gpuMemcpyDeviceToHost));
  dg::copy(Direction::DeviceToHost, Field::Flags, sizeof(bad));
  dg::scan(dg::host_accesses(), dg::Phase::Adjacency, slots);
  if (bad & 1u) return {Status::UnknownIdentity, 0, {}, table_bytes};
  std::vector<unsigned long long> packed(table_slots);
  if (table_slots) {
    GPU_CHECK(gpuMemcpy(packed.data(), keys.data, table_bytes, gpuMemcpyDeviceToHost));
    dg::copy(Direction::DeviceToHost, Field::AdjacencyMatrix, table_bytes);
  }
  std::vector<Contact> unique;
  unique.reserve(output.size());
  for (auto key : packed) {
    if (key == contact_hash_empty) continue;
    const auto ends = unpack_contact_key(key);
    unique.push_back({identities[ends.first], identities[ends.second]});
  }
  std::sort(unique.begin(), unique.end());
  const auto found = unique.size();
  if (found > output.size() || (bad & 2u))
    return {Status::CapacityOverflow, found, {}, table_bytes};
  std::vector<unsigned> observed(grains);
  if (grains) {
    GPU_CHECK(gpuMemcpy(observed.data(), active.data, grains * sizeof(unsigned),
                        gpuMemcpyDeviceToHost));
    dg::copy(Direction::DeviceToHost, Field::Counts, grains * sizeof(unsigned));
  }
  AdjacencyResult result{Status::Success, found, {}, table_bytes};
  for (std::size_t i = 0; i < grains; ++i)
    if (observed[i]) result.active.push_back(identities[i]);
  if (found) {
    GPU_CHECK(gpuMemcpy(output.data(), unique.data(), found * sizeof(Contact),
                        gpuMemcpyHostToDevice));
    dg::copy(Direction::HostToDevice, Field::Contacts, found * sizeof(Contact));
  }
  GPU_CHECK(gpuDeviceSynchronize());
  return result;
}
template <class Grid>
PropagationResult propagate_dispatch(Tag tag, Grid grid, Slot slots,
                                     std::span<const std::uint8_t> occupancy,
                                     std::span<const Id> seeds, std::span<Id> output,
                                     std::size_t sweeps) {
  if (dg::current) {
    try {
      dg::current->covered();
      return propagate_impl<true>(tag, grid, slots, occupancy, seeds, output,
                                  sweeps);
    } catch (...) {
      dg::current->fail();
      throw;
    }
  }
  return propagate_impl<false>(tag, grid, slots, occupancy, seeds, output, sweeps);
}
template <class Grid>
AdjacencyResult adjacency_dispatch(Tag tag, Grid grid, Slot slots,
                                   std::span<const Id> labels,
                                   std::span<const Id> ids,
                                   std::span<Contact> output, std::size_t radius) {
  if (dg::current) {
    try {
      dg::current->covered();
      return adjacency_impl<true>(tag, grid, slots, labels, ids, output, radius);
    } catch (...) {
      dg::current->fail();
      throw;
    }
  }
  return adjacency_impl<false>(tag, grid, slots, labels, ids, output, radius);
}
PropagationResult propagate(Tag tag, Grid2D grid, Slot slots,
                            std::span<const std::uint8_t> occupancy,
                            std::span<const Id> seeds, std::span<Id> output,
                            std::size_t sweeps) {
  return propagate_dispatch(tag, grid, slots, occupancy, seeds, output, sweeps);
}
AdjacencyResult adjacency(Tag tag, Grid2D grid, Slot slots,
                          std::span<const Id> labels, std::span<const Id> ids,
                          std::span<Contact> output, std::size_t radius) {
  return adjacency_dispatch(tag, grid, slots, labels, ids, output, radius);
}
PropagationResult detail::propagate(Tag tag, Grid3D grid, Slot slots,
                                    std::span<const std::uint8_t> occupancy,
                                    std::span<const Id> seeds, std::span<Id> output,
                                    std::size_t sweeps) {
  return propagate_dispatch(tag, grid, slots, occupancy, seeds, output, sweeps);
}
AdjacencyResult detail::adjacency(Tag tag, Grid3D grid, Slot slots,
                                  std::span<const Id> labels,
                                  std::span<const Id> ids, std::span<Contact> output,
                                  std::size_t radius) {
  return adjacency_dispatch(tag, grid, slots, labels, ids, output, radius);
}
} // namespace pfc::grain::tracking
