// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

using namespace pfc;
using namespace pfc::types;

TEST_CASE("Decomposition::domain() reproduces the global World coordinate system",
          "[decomposition][domain][unit]") {
  const auto w =
      world::from_bounds({128, 96, 64}, {-1.0, -2.0, 0.0}, {1.0, 4.0, 8.0});
  const auto decomp = decomposition::create(w, Int3{2, 2, 1});

  const Domain d = decomposition::domain(decomp);
  REQUIRE(domain::get_size(d) == world::get_size(w));
  REQUIRE(domain::get_spacing(d) == world::get_spacing(w));
  REQUIRE(domain::get_origin(d) == world::get_origin(w));
  REQUIRE(domain::get_periodic(d) == world::get_periodic(w));
}

TEST_CASE("Decomposition::global_box() is the full [lower, upper] index box",
          "[decomposition][box3i][unit]") {
  const auto w = world::create(GridSize({128, 128, 128}));
  const auto decomp = decomposition::create(w, Int3{2, 2, 2});

  const Box3i g = decomposition::global_box(decomp);
  REQUIRE(g.low == world::get_lower(w));
  REQUIRE(g.high == world::get_upper(w));
  REQUIRE(g.is_consistent());
  REQUIRE(static_cast<size_t>(g.count()) == world::get_total_size(w));
}

TEST_CASE(
    "Decomposition::local_box() matches each subworld and tiles the global box",
    "[decomposition][box3i][unit]") {
  const auto w = world::create(GridSize({100, 80, 60}));
  const auto decomp = decomposition::create(w, Int3{2, 2, 1});
  const int n = decomposition::get_num_domains(decomp);
  REQUIRE(n == 4);

  long long summed = 0;
  for (int i = 0; i < n; ++i) {
    const auto &sub = decomposition::get_subworld(decomp, i);
    const Box3i b = decomposition::local_box(decomp, i);
    // Same index range as the legacy subworld accessor.
    REQUIRE(b.low == world::get_lower(sub));
    REQUIRE(b.high == world::get_upper(sub));
    REQUIRE(b.is_consistent());
    REQUIRE(static_cast<size_t>(b.count()) == world::get_total_size(sub));
    // Every local box lies within the global box.
    REQUIRE(decomposition::global_box(decomp).contains(b.low));
    REQUIRE(decomposition::global_box(decomp).contains(b.high));
    summed += b.count();
  }
  // Non-overlapping subdomains exactly cover the global domain.
  REQUIRE(summed == decomposition::global_box(decomp).count());
}
