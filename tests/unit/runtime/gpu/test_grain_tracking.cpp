// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <openpfc/kernel/grain/contact_hash.hpp>
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/grain_tracking.hpp>

namespace gr = pfc::grain;
namespace tr = gr::tracking;
#if defined(OPENPFC_TEST_TRACKING_HIP)
using Tag = pfc::backend::HIPTag;
#else
using Tag = pfc::backend::CUDATag;
#endif
template <class T> using Buffer = pfc::core::DataBuffer<Tag, T>;
template <class T> std::span<T> view(Buffer<T> &buf) {
  return {buf.data(), buf.size()};
}
template <class T> std::span<const T> read(const Buffer<T> &buf) {
  return {buf.data(), buf.size()};
}

bool available() {
#if defined(OPENPFC_TEST_TRACKING_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}
TEST_CASE("singleton contacts validate labels without visiting neighbours",
          "[grain][tracking][singleton]") {
  if (!available()) SKIP("GPU unavailable");
  for (bool periodic : {false, true}) {
    const gr::Grid2D grid{3, 2, periodic, periodic, gr::Connectivity::Eight};
    const std::vector<gr::Id> ids{41};
    for (bool occupied : {false, true}) {
      std::vector<gr::Id> labels(12, occupied ? 41 : 0);
      labels[0] = 0;
      Buffer<gr::Id> input(labels.size());
      input.copy_from_host(labels);
      Buffer<gr::Contact> output(1);
      const std::vector<gr::Contact> sentinel{{555, 777}};
      output.copy_from_host(sentinel);
      const auto expected = tr::reference::contact_graph(grid, 2, labels, 2);
      gr::Diagnostics diagnostics;
      {
        gr::diagnostics::Scope scope(&diagnostics);
        const auto actual =
            tr::adjacency(Tag{}, grid, 2, read(input), ids, view(output), 64);
        REQUIRE(actual.status == tr::Status::Success);
        REQUIRE(actual.active == expected.vertices);
        REQUIRE(actual.edge_count == expected.edges.size());
      }
      REQUIRE(diagnostics.source_accesses_complete);
      REQUIRE(diagnostics.accesses
                  .reads[static_cast<std::size_t>(gr::diagnostics::Field::Labels)] ==
              labels.size());
      REQUIRE(input.to_host() == labels);
      REQUIRE(output.to_host() == sentinel);
      labels.back() = 99;
      input.copy_from_host(labels);
      REQUIRE(
          tr::adjacency(Tag{}, grid, 2, read(input), ids, view(output), 64).status ==
          tr::Status::UnknownIdentity);
      REQUIRE(output.to_host() == sentinel);
    }
  }
}
TEST_CASE("resident grain tracking matches shortest path and contact oracles",
          "[grain][tracking]") {
  if (!available()) SKIP("GPU unavailable");
  const gr::Slot slots = 2;
  for (bool periodic : {false, true})
    for (auto stencil : {gr::Connectivity::Four, gr::Connectivity::Eight}) {
      gr::Grid2D grid{5, 3, periodic, periodic, stencil};
      std::vector<std::uint8_t> occupancy(30, 1);
      std::vector<gr::Id> seeds(30, 0);
      seeds[0] = 41;
      seeds[4] = 77;
      seeds[15 + 7] = 99;
      // Removed support clears a prior identity instead of repainting inactive data.
      seeds[15 + 14] = 123;
      occupancy[15 + 14] = 0;
      Buffer<std::uint8_t> mask(30);
      mask.copy_from_host(occupancy);
      Buffer<gr::Id> prior(30), result(30);
      prior.copy_from_host(seeds);
      result.copy_from_host(std::vector<gr::Id>(30, 999));
      const auto propagated = tr::propagate(Tag{}, grid, slots, read(mask),
                                            read(prior), view(result), 15);
      REQUIRE(propagated.status == tr::Status::Success);
      REQUIRE(result.to_host() ==
              tr::reference::propagate(grid, slots, occupancy, seeds).labels);
      REQUIRE(result.to_host()[0] == 41);
      REQUIRE(result.to_host()[4] == 77);
      REQUIRE(result.to_host().back() == 0);
      const auto expected =
          tr::reference::contact_graph(grid, slots, result.to_host());
      Buffer<gr::Contact> contacts(10);
      const std::vector<gr::Id> registry{41, 77, 99, 123};
      const auto extracted =
          tr::adjacency(Tag{}, grid, slots, read(result), registry, view(contacts));
      REQUIRE(extracted.status == tr::Status::Success);
      REQUIRE(extracted.active == expected.vertices);
      auto actual = contacts.to_host();
      actual.resize(extracted.edge_count);
      REQUIRE(actual == expected.edges);
    }
}

TEST_CASE(
    "tracking preserves disconnected same-slot identities and thin periodic grids",
    "[grain][tracking]") {
  if (!available()) SKIP("GPU unavailable");
  for (auto grid :
       {gr::Grid2D{5, 1}, gr::Grid2D{1, 5, true, true, gr::Connectivity::Eight}}) {
    std::vector<std::uint8_t> occupied{1, 1, 0, 1, 1};
    std::vector<gr::Id> seeds{8, 0, 0, 0, 22};
    Buffer<std::uint8_t> mask(5);
    mask.copy_from_host(occupied);
    Buffer<gr::Id> labels(5);
    labels.copy_from_host(seeds);
    REQUIRE(tr::propagate(Tag{}, grid, 1, read(mask), read(labels), view(labels), 5)
                .status == tr::Status::Success);
    REQUIRE(labels.to_host() ==
            tr::reference::propagate(grid, 1, occupied, seeds).labels);
    REQUIRE(labels.to_host()[0] == 8);
    REQUIRE(labels.to_host()[4] == 22);
  }
}

TEST_CASE("tracking failures preserve previous label and graph publications",
          "[grain][tracking]") {
  if (!available()) SKIP("GPU unavailable");
  const gr::Grid2D grid{5, 1};
  Buffer<std::uint8_t> mask(5);
  mask.copy_from_host(std::vector<std::uint8_t>(5, 1));
  Buffer<gr::Id> prior(5), output(5);
  const std::vector<gr::Id> old(5, 555);
  output.copy_from_host(old);
  prior.copy_from_host(std::vector<gr::Id>{41, 0, 0, 0, 0});
  REQUIRE(tr::propagate(Tag{}, grid, 1, read(mask), read(prior), view(output), 1)
              .status == tr::Status::IterationLimit);
  REQUIRE(output.to_host() == old);
  prior.copy_from_host(std::vector<gr::Id>(5, 0));
  REQUIRE(tr::propagate(Tag{}, grid, 1, read(mask), read(prior), view(output), 5)
              .status == tr::Status::Unseeded);
  REQUIRE(output.to_host() == old);
  prior.copy_from_host(std::vector<gr::Id>{1, 2, 3, 4, 5});
  Buffer<gr::Contact> contacts(1);
  contacts.copy_from_host(std::vector<gr::Contact>{{555, 777}});
  const std::vector<gr::Id> ids{1, 2, 3, 4, 5};
  const auto overflow =
      tr::adjacency(Tag{}, grid, 1, read(prior), ids, view(contacts));
  REQUIRE(overflow.status == tr::Status::CapacityOverflow);
  REQUIRE(overflow.edge_count == 4);
  REQUIRE(contacts.to_host() == std::vector<gr::Contact>{{555, 777}});
  const std::vector<gr::Id> missing{1, 2, 3, 4};
  REQUIRE(
      tr::adjacency(Tag{}, grid, 1, read(prior), missing, view(contacts)).status ==
      tr::Status::UnknownIdentity);
  REQUIRE(contacts.to_host() == std::vector<gr::Contact>{{555, 777}});
  const std::vector<gr::Id> unsorted{2, 1};
  REQUIRE_THROWS_AS(
      tr::adjacency(Tag{}, grid, 1, read(prior), unsorted, view(contacts)),
      std::invalid_argument);
}

int main(int argc, char **argv) {
#if defined(OPENPFC_TEST_TRACKING_HIP)
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) == hipSuccess)
    std::cout << "tracking device: " << properties.name << " "
              << properties.gcnArchName << '\n';
#else
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
    std::cout << "tracking device: " << properties.name << '\n';
#endif
  return Catch::Session().run(argc, argv);
}

TEST_CASE("conservative radius-three contacts match periodic coordinate oracle",
          "[grain][tracking]") {
  if (!available()) SKIP("GPU unavailable");
  for (auto stencil : {gr::Connectivity::Four, gr::Connectivity::Eight}) {
    gr::Grid2D grid{7, 5, true, true, stencil};
    std::vector<gr::Id> labels(70, 0);
    labels[0] = 1;
    labels[3 + 7 * 3] = 2;
    labels[6 + 7 * 4] = 3;
    labels[35 + 4] = 4;
    Buffer<gr::Id> resident(labels.size());
    resident.copy_from_host(labels);
    Buffer<gr::Contact> output(6);
    const std::vector<gr::Id> ids{1, 2, 3, 4};
    const auto actual =
        tr::adjacency(Tag{}, grid, 2, read(resident), ids, view(output), 3);
    REQUIRE(actual.status == tr::Status::Success);
    const auto expected = tr::reference::contact_graph(grid, 2, labels, 3);
    auto edges = output.to_host();
    edges.resize(actual.edge_count);
    REQUIRE(edges == expected.edges);
    REQUIRE(actual.active == expected.vertices);
  }
}

TEST_CASE("resident graph has no fixed thirty-two neighbor capacity",
          "[grain][tracking]") {
  if (!available()) SKIP("GPU unavailable");
  std::vector<gr::Id> ids;
  for (gr::Id id = 1; id <= 65; ++id) ids.push_back(id);
  Buffer<gr::Id> labels(ids.size());
  labels.copy_from_host(ids);
  Buffer<gr::Contact> output(65 * 64 / 2);
  const auto result =
      tr::adjacency(Tag{}, {1, 1}, 65, read(labels), ids, view(output));
  REQUIRE(result.status == tr::Status::Success);
  REQUIRE(result.edge_count == 2080);
  REQUIRE(output.to_host() == tr::reference::contact_graph({1, 1}, 65, ids).edges);
}

TEST_CASE("ten thousand registered grains do not allocate a dense adjacency matrix",
          "[grain][tracking][sparse]") {
  if (!available()) SKIP("GPU unavailable");
  constexpr std::size_t grains = 10000;
  std::vector<gr::Id> ids(grains);
  for (std::size_t i = 0; i < grains; ++i) ids[i] = i + 1;
  gr::Grid2D grid{4, 1};
  std::vector<gr::Id> labels{1, 2, 0, 0};
  Buffer<gr::Id> resident(labels.size());
  resident.copy_from_host(labels);
  Buffer<gr::Contact> output(16);
  const auto t0 = std::chrono::steady_clock::now();
  const auto extracted =
      tr::adjacency(Tag{}, grid, 1, read(resident), ids, view(output));
  const auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  REQUIRE(extracted.status == tr::Status::Success);
  REQUIRE(extracted.edge_count == 1);
  auto edges = output.to_host();
  edges.resize(extracted.edge_count);
  REQUIRE(edges == std::vector<gr::Contact>{{1, 2}});
  REQUIRE(extracted.workspace_bytes == gr::contact_hash_bytes(output.size()));
  REQUIRE(extracted.workspace_bytes < gr::dense_adjacency_bytes(grains) / 100);
  std::cout << "sparse adjacency grains=" << grains
            << " workspace_bytes=" << extracted.workspace_bytes
            << " dense_bytes=" << gr::dense_adjacency_bytes(grains)
            << " adjacency_ms=" << ms << '\n';
}
