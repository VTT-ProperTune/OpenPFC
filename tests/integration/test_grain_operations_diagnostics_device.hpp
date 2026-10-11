// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "../fixtures/allocation_provider.hpp"
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/grain/contact_hash.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>
#include <openpfc/runtime/gpu/grain_transfer.hpp>
#if defined(__HIPCC__) || defined(__HIP__)
using Backend = pfc::backend::HIPTag;
#else
using Backend = pfc::backend::CUDATag;
#endif
using namespace pfc::grain;
namespace dg = pfc::grain::diagnostics;
namespace {
void available() {
  int count = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
#endif
  REQUIRE(count > 0);
}
std::size_t f(dg::Field field) { return static_cast<std::size_t>(field); }
std::size_t dir(dg::Direction direction) {
  return static_cast<std::size_t>(direction);
}
} // namespace
TEST_CASE("device allocation overlap and persistent baseline are exact",
          "[grain][diagnostics]") {
  available();
  Diagnostics d;
  dg::AllocationTotals both, after, baseline, finished;
  {
    dg::Scope scope(&d);
    pfc::core::DataBuffer<Backend, double> values(1);
    values.resize(2);
    both = d.allocations(dg::Space::Device);
    pfc::core::DataBuffer<Backend, double> moved(std::move(values));
    d.reset_interval();
    baseline = d.allocations(dg::Space::Device);
    moved.resize(0);
    after = d.allocations(dg::Space::Device);
  }
  finished = d.allocations(dg::Space::Device);
  REQUIRE(both.requests == 2);
  REQUIRE(both.requested_bytes == 24);
  REQUIRE(both.frees == 1);
  REQUIRE(both.live_bytes == 16);
  REQUIRE(both.peak_live_bytes == 24);
  REQUIRE(baseline.initial_live_bytes == 16);
  REQUIRE(baseline.peak_live_bytes == 16);
  REQUIRE(baseline.requests == 0);
  REQUIRE(after.frees == 1);
  REQUIRE(after.freed_bytes == 16);
  REQUIRE(finished.live_bytes == 0);
}
TEST_CASE("one-cell tracking counts source branches allocations and copies",
          "[grain][diagnostics]") {
  available();
  const Grid2D grid{1, 1, false, false, Connectivity::Eight};
  pfc::core::DataBuffer<Backend, std::uint8_t> occupied(1);
  occupied.copy_from_host(std::vector<std::uint8_t>{1});
  pfc::core::DataBuffer<Backend, Id> seed(1), out(1);
  seed.copy_from_host(std::vector<Id>{7});
  Diagnostics d;
  tracking::PropagationResult result;
  {
    dg::Scope scope(&d);
    result = tracking::propagate(Backend{}, grid, 1, {occupied.data(), 1},
                                 {seed.data(), 1}, {out.data(), 1}, 1);
  }
  REQUIRE(result.status == tracking::Status::Success);
  REQUIRE(out.to_host() == std::vector<Id>{7});
  const auto totals = d.allocations(dg::Space::Device);
  REQUIRE(totals.requests == 4);
  REQUIRE(totals.frees == 4);
  REQUIRE(totals.live_bytes == 0);
  REQUIRE(totals.peak_live_bytes == sizeof(dg::Accesses) + 20);
  REQUIRE(d.accesses.reads[f(dg::Field::Occupancy)] == 3);
  REQUIRE(d.accesses.reads[f(dg::Field::Labels)] == 1);
  REQUIRE(d.accesses.reads[f(dg::Field::StagedLabels)] == 2);
  REQUIRE(d.accesses.writes[f(dg::Field::StagedLabels)] == 2);
  REQUIRE(d.accesses.writes[f(dg::Field::Flags)] == 1);
  REQUIRE(d.copies.bytes[dir(dg::Direction::DeviceToHost)][f(dg::Field::Flags)] ==
          4);
  REQUIRE(d.copies.bytes[dir(dg::Direction::DeviceToDevice)][f(dg::Field::Labels)] ==
          8);
  REQUIRE(d.copies.bytes[dir(dg::Direction::HostToDevice)]
                        [f(dg::Field::Instrumentation)] == sizeof(dg::Accesses));
  REQUIRE(d.accesses
              .full_plane_scans[static_cast<std::size_t>(dg::Phase::Propagation)] ==
          2);
  REQUIRE(d.events[static_cast<std::size_t>(dg::Phase::Propagation)].available);
  d.reset_interval();
  {
    dg::Scope outer(&d);
    dg::Scope off(nullptr);
    result = tracking::propagate(Backend{}, grid, 1, {occupied.data(), 1},
                                 {seed.data(), 1}, {out.data(), 1}, 1);
  }
  REQUIRE(result.status == tracking::Status::Success);
  REQUIRE(d.allocations(dg::Space::Device).requests == 0);
  REQUIRE(d.accesses.reads[f(dg::Field::Occupancy)] == 0);
}
TEST_CASE("one-cell no-op transfer exposes full staging despite zero moves",
          "[grain][diagnostics]") {
  available();
  const Grid2D grid{1, 1, false, false, Connectivity::Eight};
  pfc::core::DataBuffer<Backend, double> values(1);
  values.copy_from_host(std::vector<double>{.5});
  pfc::core::DataBuffer<Backend, Id> ids(1);
  ids.copy_from_host(std::vector<Id>{7});
  Diagnostics d;
  {
    auto result = [&] {
      dg::Scope scope(&d);
      return transfer(grid, 1, values, ids, std::vector<Grain>{{7, 0, true}}, {});
    }();
    REQUIRE(result.status == TransferStatus::Success);
    REQUIRE(result.values.to_host() == std::vector<double>{.5});
    REQUIRE(d.allocations(dg::Space::Device).requests == 6);
    REQUIRE(d.allocations(dg::Space::Device).live_bytes == 16);
    REQUIRE(d.allocations(dg::Space::Device).peak_live_bytes ==
            sizeof(dg::Accesses) + 40);
    REQUIRE(d.accesses.reads[f(dg::Field::Values)] == 4);
    REQUIRE(d.accesses.reads[f(dg::Field::Labels)] == 9);
    REQUIRE(d.accesses.reads[f(dg::Field::Assignments)] == 11);
    REQUIRE(d.accesses.writes[f(dg::Field::StagedValues)] == 1);
    REQUIRE(d.accesses.writes[f(dg::Field::StagedLabels)] == 1);
    REQUIRE(d.copies.bytes[dir(dg::Direction::HostToDevice)]
                          [f(dg::Field::Assignments)] == sizeof(detail::Assignment));
  }
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
  REQUIRE(d.allocations(dg::Space::Device).frees == 6);
}

TEST_CASE("single-UID adjacency exposes exact auxiliary counts",
          "[grain][diagnostics]") {
  available();
  const Grid2D grid{1, 1, false, false, Connectivity::Eight};
  pfc::core::DataBuffer<Backend, Id> ids(1);
  ids.copy_from_host(std::vector<Id>{7});
  Diagnostics d;
  tracking::AdjacencyResult result;
  {
    dg::Scope scope(&d);
    result = tracking::adjacency(Backend{}, grid, 1, {ids.data(), 1},
                                 std::vector<Id>{7}, {}, 1);
  }
  REQUIRE(result.status == tracking::Status::Success);
  REQUIRE(result.edge_count == 0);
  REQUIRE(result.active == std::vector<Id>{7});
  REQUIRE(result.workspace_bytes == contact_hash_bytes(0));
  REQUIRE(d.source_accesses_complete);
  REQUIRE(d.allocations(dg::Space::Device).requests == 5);
  REQUIRE(d.allocations(dg::Space::Device).peak_live_bytes ==
          sizeof(dg::Accesses) + sizeof(Id) + contact_hash_bytes(0) +
              2 * sizeof(unsigned));
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
  REQUIRE(d.accesses.reads[f(dg::Field::Labels)] == 2);
  REQUIRE(d.accesses.reads[f(dg::Field::Registry)] == 2);
  REQUIRE(d.accesses.reads[f(dg::Field::Counts)] == 1);
  REQUIRE(d.accesses.writes[f(dg::Field::Counts)] == 2);
  REQUIRE(d.accesses.writes[f(dg::Field::AdjacencyMatrix)] == contact_hash_slots(0));
  REQUIRE(d.accesses.reads[f(dg::Field::AdjacencyMatrix)] == 0);
  REQUIRE(d.copies.bytes[dir(dg::Direction::HostToDevice)][f(dg::Field::Registry)] ==
          8);
  REQUIRE(d.copies.bytes[dir(dg::Direction::DeviceToHost)][f(dg::Field::Counts)] ==
          sizeof(unsigned));
  REQUIRE(d.copies.bytes[dir(dg::Direction::DeviceToHost)]
                        [f(dg::Field::AdjacencyMatrix)] == contact_hash_bytes(0));
  REQUIRE(
      d.copies.bytes[dir(dg::Direction::DeviceToDevice)][f(dg::Field::Contacts)] ==
      0);
}
namespace {
__global__ void produce_observed_value(double *values) {
  if (blockIdx.x == 0 && threadIdx.x == 0) values[0] = .5;
}
} // namespace
TEST_CASE(
    "observed nonblocking transfer orders metadata and preserves rejected originals",
    "[grain][diagnostics][stream]") {
  available();
  const Grid2D grid{1, 1, false, false, Connectivity::Eight};
  pfc::core::DataBuffer<Backend, double> values(2);
  values.copy_from_host(std::vector<double>{0, .25});
  pfc::core::DataBuffer<Backend, Id> ids(2);
  ids.copy_from_host(std::vector<Id>{7, 9});
  pfc::gpuStream_t stream;
  GPU_CHECK(pfc::gpuStreamCreateWithFlags(&stream, pfc::gpuStreamNonBlocking));
  GPU_LAUNCH_KERNEL(produce_observed_value, 1, 1, (values.data()), stream);
  Diagnostics d;
  auto result = [&] {
    dg::Scope scope(&d);
    return transfer(grid, 2, values, ids,
                    std::vector<Grain>{{7, 0, true}, {9, 1, true}},
                    std::vector<Transfer>{{9, 1, 0}}, 0, stream);
  }();
  GPU_CHECK(pfc::gpuStreamDestroy(stream));
  REQUIRE(result.status == TransferStatus::OccupiedDestination);
  REQUIRE(result.values.empty());
  REQUIRE(result.labels.empty());
  REQUIRE(values.to_host() == std::vector<double>{.5, .25});
  REQUIRE(ids.to_host() == std::vector<Id>{7, 9});
  REQUIRE(d.source_accesses_complete);
  REQUIRE_FALSE(d.observation_failed);
  REQUIRE(d.allocations(dg::Space::Device).live_bytes == 0);
  REQUIRE(d.accesses.reads[f(dg::Field::Values)] > 0);
  REQUIRE(d.accesses.writes[f(dg::Field::StagedValues)] == 2);
  REQUIRE(d.events[static_cast<std::size_t>(dg::Phase::Transfer)].available);
}
