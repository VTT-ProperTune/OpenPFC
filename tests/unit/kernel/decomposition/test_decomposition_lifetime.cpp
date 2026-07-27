// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_decomposition_lifetime.cpp
 * @brief Lifetime invariants for `pfc::decomposition::Decomposition`.
 *
 * @details
 * `Decomposition::m_global_world` was historically a `const World&` to the
 * `World` passed into the constructor. That made it silently dangle when a
 * factory function returned a `Decomposition` whose source `World` only
 * existed in the factory's local scope:
 *
 *     // Buggy under the old reference-storing implementation.
 *     pfc::decomposition::Decomposition make_decomp(int N) {
 *       auto w = pfc::world::create(pfc::GridSize({N, N, N}), ...);
 *       return pfc::decomposition::create(w, /\*nproc=\*\/1);
 *     }
 *     auto decomp = make_decomp(16);
 *     // decomp.m_global_world is now a dangling reference to the
 *     // destroyed `w` inside make_decomp.
 *
 * After the fix, `m_global_world` is an owned `World` value and the
 * pattern above is safe. These tests exercise that contract directly so
 * any future regression (e.g. accidentally re-introducing a reference
 * member) is caught at the kernel-decomposition test layer rather than
 * surfacing as a downstream `LocalField` / heat3d failure.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

using Catch::Matchers::WithinAbs;

namespace {

/**
 * @brief Factory that returns a `Decomposition` by value, the source `Domain`
 *        of which is purely local to this function and is destroyed at
 *        function return.
 *
 * Under the historical implementation (where `Decomposition` stored
 * `const World&`), the returned `Decomposition::m_global_world` would
 * dangle. After the fix this is safe and the returned value is fully
 * self-contained.
 */
pfc::decomposition::Decomposition make_decomp(int N, int nproc, double origin,
                                              double spacing) {
  auto local_domain = pfc::domain::create(
      pfc::GridSize({N, N, N}), pfc::PhysicalOrigin({origin, origin, origin}),
      pfc::GridSpacing({spacing, spacing, spacing}));
  return pfc::decomposition::create(local_domain, nproc);
}

} // namespace

TEST_CASE("Decomposition: factory return-by-value keeps the global Domain intact",
          "[decomposition][lifetime][unit]") {
  // Source Domain existed only inside `make_decomp`; it has been destroyed
  // by the time we read the Decomposition's Domain below.
  auto decomp =
      make_decomp(/*N=*/16, /*nproc=*/1, /*origin=*/-3.5, /*spacing=*/0.25);

  const auto &domain = pfc::decomposition::domain(decomp);

  SECTION("size survives the factory's local Domain destruction") {
    const auto sz = pfc::domain::get_size(domain);
    REQUIRE(sz[0] == 16);
    REQUIRE(sz[1] == 16);
    REQUIRE(sz[2] == 16);
  }
  SECTION("origin survives") {
    const auto org = pfc::domain::get_origin(domain);
    REQUIRE_THAT(org[0], WithinAbs(-3.5, 1e-15));
    REQUIRE_THAT(org[1], WithinAbs(-3.5, 1e-15));
    REQUIRE_THAT(org[2], WithinAbs(-3.5, 1e-15));
  }
  SECTION("spacing survives") {
    const auto sp = pfc::domain::get_spacing(domain);
    REQUIRE_THAT(sp[0], WithinAbs(0.25, 1e-15));
    REQUIRE_THAT(sp[1], WithinAbs(0.25, 1e-15));
    REQUIRE_THAT(sp[2], WithinAbs(0.25, 1e-15));
  }
}

TEST_CASE("Decomposition: copy-construction yields a self-contained copy",
          "[decomposition][lifetime][unit]") {
  // Build the source decomp inside an inner scope so the source Domain is
  // destroyed before we look at the copy.
  auto make_copy = []() {
    auto src_domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({1.0, 1.0, 1.0}));
    auto src_decomp = pfc::decomposition::create(src_domain, /*nproc=*/1);
    // Copy the decomposition. After this scope ends, src_domain and
    // src_decomp are gone; the returned copy must remain valid.
    return pfc::decomposition::Decomposition(src_decomp);
  };

  auto copy = make_copy();
  const auto &domain = pfc::decomposition::domain(copy);
  const auto sz = pfc::domain::get_size(domain);
  REQUIRE(sz[0] == 8);
  REQUIRE(sz[1] == 8);
  REQUIRE(sz[2] == 8);
}

TEST_CASE("Decomposition: stored by value inside an aggregate stays valid "
          "after the source goes out of scope",
          "[decomposition][lifetime][unit]") {
  // Mirrors the FFTLayout pattern (`const Decomposition m_decomposition;`):
  // an aggregate type embeds a Decomposition by value. The source Domain
  // and source Decomposition both belong to the factory scope.
  struct Holder {
    pfc::decomposition::Decomposition decomp;
  };

  auto build_holder = []() {
    auto src_domain = pfc::domain::create(pfc::GridSize({12, 12, 12}),
                                          pfc::PhysicalOrigin({1.0, 2.0, 3.0}),
                                          pfc::GridSpacing({0.5, 0.5, 0.5}));
    return Holder{pfc::decomposition::create(src_domain, /*nproc=*/1)};
  };

  auto holder = build_holder();
  const auto &domain = pfc::decomposition::domain(holder.decomp);
  const auto org = pfc::domain::get_origin(domain);
  REQUIRE_THAT(org[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(org[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(org[2], WithinAbs(3.0, 1e-15));
}
