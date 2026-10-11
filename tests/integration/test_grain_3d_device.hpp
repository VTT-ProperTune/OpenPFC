// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "fixtures/grain_3d_cases.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <iostream>
#include <openpfc/runtime/gpu/grain_remapping.hpp>
#if defined(__HIPCC__) || defined(__HIP__)
using ThreeBackend = pfc::backend::HIPTag;
#else
using ThreeBackend = pfc::backend::CUDATag;
#endif
namespace {
using namespace grain_3d_test;
using ThreeValues = pfc::core::DataBuffer<ThreeBackend, double>;
using ThreeLabels = pfc::core::DataBuffer<ThreeBackend, Id>;
void require_device_3d() {
  int count = 0;
#if defined(__HIPCC__) || defined(__HIP__)
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
  REQUIRE(count > 0);
  hipDeviceProp_t properties{};
  REQUIRE(hipGetDeviceProperties(&properties, 0) == hipSuccess);
  std::cout << "3D actual device: " << properties.name << " / "
            << properties.gcnArchName << '\n';
#else
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE(count > 0);
#endif
}
__global__ void produce_3d(double *values, std::size_t i, double q) {
  if (!blockIdx.x && !threadIdx.x) values[i] = q;
}
} // namespace
TEST_CASE("3D singleton graph scans every label once at large contact radius",
          "[grain][3d][device][singleton]") {
  require_device_3d();
  for (bool periodic : {false, true}) {
    const Grid3D g{3, 2, 2, periodic, periodic, periodic, Connectivity::TwentySix};
    const std::vector<Id> ids{9};
    for (bool occupied : {false, true}) {
      std::vector<Id> labels(24, occupied ? 9 : 0);
      labels[0] = 0;
      ThreeLabels input(labels.size());
      input.copy_from_host(labels);
      pfc::core::DataBuffer<ThreeBackend, Contact> output(1);
      const std::vector<Contact> sentinel{{80, 90}};
      output.copy_from_host(sentinel);
      const auto expected = graph(g, 2, labels, 2);
      Diagnostics d;
      {
        diagnostics::Scope scope(&d);
        const auto actual = tracking::adjacency(
            ThreeBackend{}, g, 2, std::span(input.data(), input.size()), ids,
            std::span(output.data(), output.size()), 64);
        REQUIRE(actual.status == tracking::Status::Success);
        REQUIRE(actual.active == expected.vertices);
        REQUIRE(actual.edge_count == expected.edges.size());
      }
      REQUIRE(d.source_accesses_complete);
      REQUIRE(
          d.accesses.reads[static_cast<std::size_t>(diagnostics::Field::Labels)] ==
          labels.size());
      REQUIRE(input.to_host() == labels);
      REQUIRE(output.to_host() == sentinel);
      labels.back() = 77;
      input.copy_from_host(labels);
      REQUIRE(tracking::adjacency(ThreeBackend{}, g, 2,
                                  std::span(input.data(), input.size()), ids,
                                  std::span(output.data(), output.size()), 64)
                  .status == tracking::Status::UnknownIdentity);
      REQUIRE(output.to_host() == sentinel);
    }
  }
}
TEST_CASE("3D device ownership matches independent exhaustive tiny masks",
          "[grain][3d][device]") {
  require_device_3d();
  for (auto stencil : {Connectivity::Six, Connectivity::TwentySix}) {
    Grid3D g{2, 2, 2, true, true, true, stencil};
    for (unsigned mask = 0; mask < 256; ++mask) {
      std::vector<std::uint8_t> occupied(8);
      std::vector<Id> seeds(8);
      for (unsigned i = 0; i < 8; ++i) occupied[i] = (mask >> i) & 1;
      if (occupied[0]) seeds[0] = 9;
      if (occupied[7]) seeds[7] = 3;
      auto expected = ownership(g, 1, occupied, seeds);
      bool complete = true;
      for (unsigned i = 0; i < 8; ++i)
        if (occupied[i] && !expected[i]) complete = false;
      pfc::core::DataBuffer<ThreeBackend, std::uint8_t> occupancy(8);
      occupancy.copy_from_host(occupied);
      ThreeLabels prior(8), output(8);
      prior.copy_from_host(seeds);
      output.copy_from_host(std::vector<Id>(8, 77));
      auto actual = tracking::propagate(
          ThreeBackend{}, g, 1, std::span(occupancy.data(), occupancy.size()),
          std::span(prior.data(), prior.size()),
          std::span(output.data(), output.size()), 8);
      REQUIRE(actual.status ==
              (complete ? tracking::Status::Success : tracking::Status::Unseeded));
      REQUIRE(output.to_host() == (complete ? expected : std::vector<Id>(8, 77)));
      REQUIRE(prior.to_host() == seeds);
    }
  }
  Grid3D g{3, 1, 1};
  pfc::core::DataBuffer<ThreeBackend, std::uint8_t> occupancy(3);
  occupancy.copy_from_host(std::vector<std::uint8_t>{1, 1, 1});
  ThreeLabels prior(3), output(3);
  prior.copy_from_host(std::vector<Id>{9, 0, 3});
  output.copy_from_host(std::vector<Id>{77, 77, 77});
  REQUIRE(tracking::propagate(ThreeBackend{}, g, 1,
                              std::span(occupancy.data(), occupancy.size()),
                              std::span(prior.data(), prior.size()),
                              std::span(output.data(), output.size()), 0)
              .status == tracking::Status::IterationLimit);
  REQUIRE(output.to_host() == std::vector<Id>{77, 77, 77});
  REQUIRE(tracking::propagate(ThreeBackend{}, g, 1,
                              std::span(occupancy.data(), occupancy.size()),
                              std::span(prior.data(), prior.size()),
                              std::span(output.data(), output.size()), 2)
              .status == tracking::Status::Success);
  REQUIRE(output.to_host() == std::vector<Id>{9, 3, 3});
}
TEST_CASE(
    "3D device complete contacts cover faces edges corners overlap and capacity",
    "[grain][3d][device]") {
  require_device_3d();
  for (auto stencil : {Connectivity::Six, Connectivity::TwentySix})
    for (unsigned periodic = 0; periodic < 8; ++periodic) {
      Grid3D g{
          3,      3, 3, bool(periodic & 1), bool(periodic & 2), bool(periodic & 4),
          stencil};
      std::vector<Id> labels(54), ids;
      for (std::size_t i = 0; i < 27; ++i) {
        labels[i] = i + 1;
        ids.push_back(i + 1);
      }
      labels[27] = 28;
      ids.push_back(28); // Same-cell cross-slot overlap.
      ThreeLabels input(labels.size());
      input.copy_from_host(labels);
      pfc::core::DataBuffer<ThreeBackend, Contact> output(378);
      for (std::size_t radius = 0; radius <= 3; ++radius) {
        const auto expected = graph(g, 2, labels, radius);
        auto actual = tracking::adjacency(
            ThreeBackend{}, g, 2, std::span(input.data(), input.size()), ids,
            std::span(output.data(), output.size()), radius);
        REQUIRE(actual.status == tracking::Status::Success);
        REQUIRE(actual.active == expected.vertices);
        REQUIRE(actual.edge_count == expected.edges.size());
        auto observed = output.to_host();
        observed.resize(actual.edge_count);
        REQUIRE(observed == expected.edges);
      }
      output.copy_from_host(std::vector<Contact>(378, {80, 90}));
      auto failed = tracking::adjacency(ThreeBackend{}, g, 2,
                                        std::span(input.data(), input.size()), ids,
                                        std::span<Contact>(output.data(), 1), 3);
      REQUIRE(failed.status == tracking::Status::CapacityOverflow);
      REQUIRE(failed.edge_count == graph(g, 2, labels, 3).edges.size());
      REQUIRE(output.to_host() == std::vector<Contact>(378, {80, 90}));
      auto registry = ids;
      registry.pop_back();
      REQUIRE(tracking::adjacency(ThreeBackend{}, g, 2,
                                  std::span(input.data(), input.size()), registry,
                                  std::span(output.data(), output.size()), 1)
                  .status == tracking::Status::UnknownIdentity);
      REQUIRE(output.to_host() == std::vector<Contact>(378, {80, 90}));
      REQUIRE(input.to_host() == labels);
    }
}
TEST_CASE("3D device full support cyclic transfer orders nonblocking producer",
          "[grain][3d][device]") {
  require_device_3d();
  Grid3D g{2, 2, 2};
  const auto n = cell_count(g);
  std::vector<double> q(3 * n);
  std::vector<Id> labels(3 * n);
  std::vector<Grain> grains{{11, 0, true}, {22, 1, true}, {33, 2, true}};
  for (std::size_t s = 0; s < 3; ++s)
    for (std::size_t i = 0; i < n; ++i) {
      q[s * n + i] = 0.125 * (s + 1) + 0.001 * i;
      labels[s * n + i] = grains[s].id;
    }
  std::vector<Transfer> cycle{{11, 0, 1}, {22, 1, 2}, {33, 2, 0}};
  for (bool enabled : {false, true}) {
    Diagnostics d;
    diagnostics::Scope scope(enabled ? &d : nullptr);
    ThreeValues values(q.size());
    ThreeLabels seeds(labels.size());
    values.copy_from_host(q);
    seeds.copy_from_host(labels);
    pfc::gpuStream_t stream;
    GPU_CHECK(pfc::gpuStreamCreateWithFlags(&stream, pfc::gpuStreamNonBlocking));
    auto changed = q;
    changed[0] = 0.875;
    GPU_LAUNCH_KERNEL(produce_3d, 1, 1, (values.data(), std::size_t{0}, changed[0]),
                      stream);
    auto actual =
        pfc::grain::transfer(g, 3, values, seeds, grains, cycle, 0, stream);
    GPU_CHECK(pfc::gpuStreamDestroy(stream));
    auto expected = pfc::grain::transfer(g, 3, changed, labels, grains, cycle);
    REQUIRE(actual.status == expected.status);
    REQUIRE(actual.values.to_host() == expected.values);
    REQUIRE(actual.labels.to_host() == expected.labels);
    REQUIRE(values.to_host() == changed);
    REQUIRE(seeds.to_host() == labels);
    std::vector<Transfer> collision{{11, 0, 1}};
    auto failed = pfc::grain::transfer(g, 3, values, seeds, grains, collision);
    REQUIRE(failed.status == TransferStatus::OccupiedDestination);
    REQUIRE(failed.values.empty());
    REQUIRE(values.to_host() == changed);
    if (enabled) {
      REQUIRE(d.source_accesses_complete);
      REQUIRE_FALSE(d.observation_failed);
      REQUIRE(d.events[static_cast<std::size_t>(diagnostics::Phase::Transfer)]
                  .available);
    }
  }
}
TEST_CASE("3D device remapping matches owning CPU transaction on and off",
          "[grain][3d][device]") {
  require_device_3d();
  auto fixtures = cases();
  for (std::size_t ci = 0; ci < fixtures.size(); ++ci)
    for (bool enabled : {false, true}) {
      auto c = fixtures[ci];
      auto expected = run(c);
      Diagnostics d;
      c.options.diagnostics = enabled ? &d : nullptr;
      ThreeValues values(c.values.size());
      ThreeLabels seeds(c.seeds.size());
      auto upload = c.values;
      const auto produced = index(c.grid, 4, 2, 2);
      if (ci == 0) upload[produced] = 0;
      values.copy_from_host(upload);
      seeds.copy_from_host(c.seeds);
      pfc::gpuStream_t stream;
      GPU_CHECK(pfc::gpuStreamCreateWithFlags(&stream, pfc::gpuStreamNonBlocking));
      if (ci == 0)
        GPU_LAUNCH_KERNEL(produce_3d, 1, 1,
                          (values.data(), produced, c.values[produced]), stream);
      auto actual = remapping::remap(c.grid, c.slots, values, seeds, c.grains,
                                     c.options, stream);
      GPU_CHECK(pfc::gpuStreamDestroy(stream));
      REQUIRE(actual.status == expected.status);
      REQUIRE(actual.transfer_status == expected.transfer_status);
      REQUIRE(actual.values.to_host() == expected.values);
      REQUIRE(actual.labels.to_host() == expected.labels);
      REQUIRE(actual.snapshot.graph.vertices == expected.snapshot.graph.vertices);
      REQUIRE(actual.snapshot.graph.edges == expected.snapshot.graph.edges);
      REQUIRE(actual.snapshot.epoch == expected.snapshot.epoch);
      REQUIRE(actual.snapshot.grains.size() == expected.snapshot.grains.size());
      for (std::size_t i = 0; i < actual.snapshot.grains.size(); ++i) {
        REQUIRE(actual.snapshot.grains[i].id == expected.snapshot.grains[i].id);
        REQUIRE(actual.snapshot.grains[i].slot == expected.snapshot.grains[i].slot);
        REQUIRE(actual.snapshot.grains[i].active ==
                expected.snapshot.grains[i].active);
      }
      REQUIRE(actual.statistics.propagation_sweeps ==
              expected.statistics.propagation_sweeps);
      REQUIRE(actual.statistics.edges == expected.statistics.edges);
      REQUIRE(actual.statistics.attempts == expected.statistics.attempts);
      REQUIRE(actual.statistics.changed_grains ==
              expected.statistics.changed_grains);
      REQUIRE(actual.statistics.moved_samples == expected.statistics.moved_samples);
      REQUIRE(actual.statistics.moved_value_bytes ==
              expected.statistics.moved_value_bytes);
      REQUIRE(actual.statistics.moved_label_bytes ==
              expected.statistics.moved_label_bytes);
      REQUIRE(actual.statistics.conflict == expected.statistics.conflict);
      auto original = values.to_host();
      REQUIRE(original.size() == c.values.size());
      REQUIRE(std::memcmp(original.data(), c.values.data(),
                          original.size() * sizeof(double)) == 0);
      REQUIRE(seeds.to_host() == c.seeds);
      if (enabled) {
        REQUIRE(d.source_accesses_complete);
        REQUIRE_FALSE(d.observation_failed);
        if (actual.status == remapping::Status::Success)
          for (auto phase :
               {diagnostics::Phase::Preflight, diagnostics::Phase::Propagation,
                diagnostics::Phase::Inspection, diagnostics::Phase::Adjacency,
                diagnostics::Phase::Decision, diagnostics::Phase::Transfer})
            REQUIRE(d.events[static_cast<std::size_t>(phase)].available);
      } else
        REQUIRE(d.allocations(diagnostics::Space::Device).requests == 0);
    }
}
