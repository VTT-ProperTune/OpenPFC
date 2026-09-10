// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_resolution.cpp
 * @brief The grid criterion behind the dealias question, pinned.
 *
 * `tungsten_dealias_study` takes minutes and is not run by CI. The arithmetic
 * it is built on is instant, so that is checked here: where the harmonics of
 * the crystal fall relative to Nyquist and to the 2/3 cut, and on which side
 * of the threshold the shipped presets sit.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numbers>

#include <tungsten/resolution.hpp>

using Catch::Matchers::WithinRel;
namespace res = tungsten::resolution;

TEST_CASE("Tungsten resolution: the two conditions meet at pi/3",
          "[tungsten][resolution][unit]") {
  // Both conditions -- third harmonic under Nyquist, and the 2/3 cut above the
  // second harmonic -- reduce to the same spacing. That coincidence is the
  // whole reason a single number characterises the grid.
  const double safe = res::dealias_safe_dx();
  REQUIRE_THAT(safe, WithinRel(std::numbers::pi / 3.0, 1e-15));
  REQUIRE_THAT(res::points_per_lattice(safe), WithinRel(6.0, 1e-12));

  // Just inside and just outside.
  CHECK(res::third_harmonic_resolved(safe * 0.999));
  CHECK(res::mask_spares_second_harmonic(safe * 0.999));
  CHECK_FALSE(res::third_harmonic_resolved(safe * 1.001));
  CHECK_FALSE(res::mask_spares_second_harmonic(safe * 1.001));
}

TEST_CASE("Tungsten resolution: the shipped presets sit on the wrong side",
          "[tungsten][resolution][unit]") {
  // Every shipped input uses this spacing. It fails both conditions, which is
  // why the mask cannot simply be switched on: with it off the third harmonic
  // folds back, with it on the crystal's own second harmonic is deleted.
  // Measured cost at this spacing is ~1.7% in spectral power; see the
  // resolution section of the tungsten chapter.
  constexpr double shipped = 1.1107207345395915;
  CHECK_FALSE(res::third_harmonic_resolved(shipped));
  CHECK_FALSE(res::mask_spares_second_harmonic(shipped));
  CHECK(shipped > res::dealias_safe_dx());
  INFO("shipped dx is "
       << 100.0 * (shipped / res::dealias_safe_dx() - 1.0)
       << "% coarser than the threshold");
  CHECK(shipped / res::dealias_safe_dx() < 1.10); // 6.1%, not a wild miss

  // Where the third harmonic lands when it folds back.
  const double k_ny = res::nyquist_k(shipped);
  const double folded = 2.0 * k_ny - 3.0 * res::kLatticeWavenumber;
  INFO("3k folds back to " << folded);
  CHECK(folded > 0.0);
  CHECK(folded < k_ny);
  REQUIRE_THAT(folded, WithinRel(2.6568542495, 1e-8));
}

TEST_CASE("Tungsten resolution: the 1/2 rule needs eight points",
          "[tungsten][resolution][unit]") {
  // Exact dealiasing of a cubic term is the 1/2 rule, not the 2/3 rule. This
  // is the spacing at which the measured mask-on/mask-off difference falls to
  // the 1e-4 level.
  const double clean = res::dealias_clean_dx();
  REQUIRE_THAT(res::points_per_lattice(clean), WithinRel(8.0, 1e-12));
  CHECK(clean < res::dealias_safe_dx());
  CHECK(res::third_harmonic_resolved(clean));
  CHECK(res::mask_spares_second_harmonic(clean));
}
