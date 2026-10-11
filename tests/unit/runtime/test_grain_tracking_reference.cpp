// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/grain/contact_hash.hpp>
#include <openpfc/runtime/cpu/detail/grain_tracking.hpp>

namespace gr = pfc::grain;
namespace ref = pfc::grain::tracking::reference;

TEST_CASE("host ownership chooses the nearest seed and stable identity tie",
          "[grain][tracking][reference]") {
  const std::vector<std::uint8_t> occupied(5, 1);
  const std::vector<gr::Id> seeds{91, 0, 0, 0, 7};
  const auto result = ref::propagate({5, 1}, 1, occupied, seeds);
  REQUIRE(result.complete);
  REQUIRE(result.labels == std::vector<gr::Id>{91, 91, 7, 7, 7});
  REQUIRE(seeds == std::vector<gr::Id>{91, 0, 0, 0, 7});
  const std::vector<std::uint8_t> seam{1, 0, 0, 0, 1};
  const std::vector<gr::Id> wrapped_seed{0, 0, 0, 0, 5};
  REQUIRE(ref::propagate({5, 1, true, false}, 1, seam, wrapped_seed).labels ==
          std::vector<gr::Id>{5, 0, 0, 0, 5});
  REQUIRE_FALSE(ref::propagate({5, 1}, 1, seam, wrapped_seed).complete);
}

TEST_CASE("host ownership clears disappearing support and reports unseeded regions",
          "[grain][tracking][reference]") {
  const std::vector<std::uint8_t> occupied{1, 1, 0, 1, 1};
  std::vector<gr::Id> seeds{4, 0, 88, 0, 0};
  const auto incomplete = ref::propagate({5, 1}, 1, occupied, seeds);
  REQUIRE_FALSE(incomplete.complete);
  REQUIRE(incomplete.labels == std::vector<gr::Id>{4, 4, 0, 0, 0});
  seeds[4] = 9;
  const auto complete = ref::propagate({5, 1}, 1, occupied, seeds);
  REQUIRE(complete.complete);
  REQUIRE(complete.labels == std::vector<gr::Id>{4, 4, 0, 9, 9});
  const std::vector<std::uint8_t> empty{0};
  const std::vector<gr::Id> retired{99};
  REQUIRE(ref::propagate({1, 1, true, true}, 1, empty, retired).complete);
  REQUIRE(ref::propagate({1, 1, true, true}, 1, empty, retired).labels ==
          std::vector<gr::Id>{0});
}

TEST_CASE("host contact radius distinguishes Manhattan and Chebyshev periodic edges",
          "[grain][tracking][reference]") {
  std::vector<gr::Id> labels(49, 0);
  labels[0] = 1;
  labels[2 + 7 * 2] = 2;
  labels[6] = 3;
  labels[3] = 4;
  const auto four =
      ref::contact_graph({7, 7, true, true, gr::Connectivity::Four}, 1, labels, 3);
  REQUIRE(four.vertices == std::vector<gr::Id>{1, 2, 3, 4});
  REQUIRE(four.edges == std::vector<gr::Contact>{{1, 3}, {1, 4}, {2, 4}, {3, 4}});
  const auto eight =
      ref::contact_graph({7, 7, true, true, gr::Connectivity::Eight}, 1, labels, 3);
  REQUIRE(eight.edges ==
          std::vector<gr::Contact>{{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}});
  const auto open = ref::contact_graph({7, 7, false, false, gr::Connectivity::Eight},
                                       1, labels, 3);
  REQUIRE(open.edges == std::vector<gr::Contact>{{1, 2}, {1, 4}, {2, 4}, {3, 4}});
  const std::vector<gr::Id> overlap{4, 9};
  REQUIRE(ref::contact_graph({1, 1}, 2, overlap, 0).edges ==
          std::vector<gr::Contact>{{4, 9}});
}

TEST_CASE("host reference validates geometry and storage at its boundary",
          "[grain][tracking][reference]") {
  const std::vector<std::uint8_t> occupied{1};
  const std::vector<gr::Id> labels{1};
  REQUIRE_THROWS_AS(ref::propagate({0, 1}, 1, occupied, labels),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(ref::propagate({1, 1}, 0, occupied, labels),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(ref::propagate({2, 1}, 1, occupied, labels),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(ref::contact_graph({2, 1}, 1, labels), std::invalid_argument);
  REQUIRE_THROWS_AS(
      ref::contact_graph({1, 1}, 1, labels, std::numeric_limits<std::size_t>::max()),
      std::invalid_argument);
  const std::vector<gr::Id> empty{0};
  REQUIRE(ref::contact_graph({1, 1}, 1, empty).vertices.empty());
}

TEST_CASE("contact hash workspace stays compact at ten thousand grains",
          "[grain][tracking][reference]") {
  constexpr std::size_t grains = 10000;
  constexpr std::size_t edges = 4096;
  const auto dense = gr::dense_adjacency_bytes(grains);
  const auto compact = gr::contact_hash_bytes(edges);
  REQUIRE(dense == grains * grains * sizeof(unsigned));
  REQUIRE(compact == gr::contact_hash_slots(edges) * sizeof(unsigned long long));
  REQUIRE(compact < dense / 100);
  REQUIRE(gr::contact_hash_slots(0) == 8);
  REQUIRE(gr::pack_contact_key(3, 7) != gr::contact_hash_empty);
  const auto ends = gr::unpack_contact_key(gr::pack_contact_key(3, 7));
  REQUIRE(ends.first == 3);
  REQUIRE(ends.second == 7);
  REQUIRE_THROWS_AS(gr::pack_contact_key(4, 4), std::invalid_argument);
}
