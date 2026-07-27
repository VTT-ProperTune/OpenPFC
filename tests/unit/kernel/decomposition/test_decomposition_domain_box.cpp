// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

using namespace pfc;
using namespace pfc::types;

TEST_CASE("Decomposition::domain() reproduces the global Domain coordinate system",
          "[decomposition][domain][unit]") {
  const auto d =
      domain::from_bounds({128, 96, 64}, {-1.0, -2.0, 0.0}, {1.0, 4.0, 8.0});
  const auto decomp = decomposition::create(d, Int3{2, 2, 1});

  const Domain retrieved_d = decomposition::domain(decomp);
  REQUIRE(domain::get_size(retrieved_d) == domain::get_size(d));
  REQUIRE(domain::get_spacing(retrieved_d) == domain::get_spacing(d));
  REQUIRE(domain::get_origin(retrieved_d) == domain::get_origin(d));
  REQUIRE(domain::get_periodic(retrieved_d) == domain::get_periodic(d));
}

TEST_CASE("Decomposition::global_box() is the full [lower, upper] index box",
          "[decomposition][box3i][unit]") {
  const auto d = domain::create(Int3{128, 128, 128});
  const auto decomp = decomposition::create(d, Int3{2, 2, 2});

  const Box3i g = decomposition::global_box(decomp);
  REQUIRE(g.low == domain::index_box(d).low);
  REQUIRE(g.high == domain::index_box(d).high);
  REQUIRE(g.is_consistent());
  REQUIRE(static_cast<size_t>(g.count()) == domain::get_total_size(d));
}

TEST_CASE(
    "Decomposition::local_box() matches stored local boxes and tiles the global box",
    "[decomposition][box3i][unit]") {
  const auto d = domain::create(Int3{100, 80, 60});
  const auto decomp = decomposition::create(d, Int3{2, 2, 1});
  const int n = decomposition::get_num_domains(decomp);
  REQUIRE(n == 4);

  long long summed = 0;
  for (int i = 0; i < n; ++i) {
    const Box3i b = decomposition::local_box(decomp, i);
    // Boxes are consistent and have positive size.
    REQUIRE(b.is_consistent());
    // Every local box lies within the global box.
    REQUIRE(decomposition::global_box(decomp).contains(b.low));
    REQUIRE(decomposition::global_box(decomp).contains(b.high));
    summed += b.count();
  }
  // Non-overlapping subdomains exactly cover the global domain.
  REQUIRE(summed == decomposition::global_box(decomp).count());
}
