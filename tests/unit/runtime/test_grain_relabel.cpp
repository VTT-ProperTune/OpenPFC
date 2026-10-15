// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/runtime/cpu/grain_relabel.hpp>

namespace gr = pfc::grain;
namespace ref = pfc::grain::tracking::reference;

TEST_CASE("relabel gives each core component a new identity",
          "[grain][tracking][reference]") {
  //              core:  . X X . . X . .   support adds the cells next to cores
  const std::vector<std::uint8_t> core{0, 1, 1, 0, 0, 1, 0, 0};
  const std::vector<std::uint8_t> support{1, 1, 1, 1, 1, 1, 1, 0};
  gr::IdentitySequence seq;
  const auto first = ref::relabel(gr::Grid2D{8, 1}, core, support, {}, seq);
  REQUIRE(first.created == 2);
  REQUIRE(first.labels == std::vector<gr::Id>{1, 1, 1, 1, 2, 2, 2, 0});
}

TEST_CASE("relabel keeps two identities when their cores touch",
          "[grain][tracking][reference]") {
  gr::IdentitySequence seq;
  const gr::Grid2D grid{8, 1, true, false};
  const std::vector<std::uint8_t> core0{0, 1, 1, 0, 0, 1, 1, 0};
  const std::vector<std::uint8_t> support0{1, 1, 1, 1, 1, 1, 1, 1};
  const auto a = ref::relabel(grid, core0, support0, {}, seq);
  REQUIRE(a.created == 2);
  // The gap closes: one connected core of six cells.
  const std::vector<std::uint8_t> core1{0, 1, 1, 1, 1, 1, 1, 0};
  const auto b = ref::relabel(grid, core1, support0, a.labels, seq);
  REQUIRE(b.created == 0);
  REQUIRE(b.labels[1] == a.labels[1]);
  REQUIRE(b.labels[6] == a.labels[6]);
  REQUIRE(b.labels[1] != b.labels[6]);
  // A new piece elsewhere gets a third identity; the old ones persist.
  const std::vector<std::uint8_t> core2{0, 1, 1, 0, 0, 1, 1, 0};
  std::vector<std::uint8_t> support2 = support0;
  std::vector<gr::Id> previous = b.labels;
  previous[0] = gr::background;
  std::vector<std::uint8_t> core3 = core2;
  core3[0] = 1;  // joins the first piece through its neighbour, cell 1
  const auto c = ref::relabel(grid, core3, support2, previous, seq);
  REQUIRE(c.created == 0);
  REQUIRE(c.labels[0] == a.labels[1]);
}

TEST_CASE("relabel in 3D keeps two z-invariant columns apart",
          "[grain][tracking][reference]") {
  const gr::Grid3D grid{6, 3, 4, true, true, true, gr::Connectivity::TwentySix};
  const auto cells = gr::cell_count(grid);
  std::vector<std::uint8_t> core(cells, 0), support(cells, 1);
  auto at = [&](int x, int y, int z) { return x + 6 * (y + 3 * z); };
  for (int z = 0; z < 4; ++z) {
    core[at(1, 1, z)] = 1;
    core[at(4, 1, z)] = 1;
  }
  gr::IdentitySequence seq;
  const auto a = ref::relabel(grid, core, support, {}, seq);
  REQUIRE(a.created == 2);
  for (int z = 0; z < 4; ++z) {
    core[at(2, 1, z)] = 1;
    core[at(3, 1, z)] = 1;
  }
  const auto b = ref::relabel(grid, core, support, a.labels, seq);
  REQUIRE(b.created == 0);
  for (int z = 0; z < 4; ++z) {
    REQUIRE(b.labels[at(1, 1, z)] == a.labels[at(1, 1, 0)]);
    REQUIRE(b.labels[at(4, 1, z)] == a.labels[at(4, 1, 0)]);
  }
  REQUIRE(a.labels[at(1, 1, 0)] != a.labels[at(4, 1, 0)]);
}

TEST_CASE("relabel rejects a core cell outside the support", "[grain][tracking][reference]") {
  gr::IdentitySequence seq;
  const std::vector<std::uint8_t> core{1, 0}, support{0, 1};
  REQUIRE_THROWS_AS(ref::relabel(gr::Grid2D{2, 1}, core, support, {}, seq),
                    std::invalid_argument);
}
