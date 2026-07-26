// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_spectral_gradient.cpp
 * @brief Unit tests for SpectralGradient factory functions including Box3i+Domain API.
 *
 * @details
 * Tests the Box3i+Domain factory functions for spectral gradient evaluators.
 * SpectralGradient requires FFT infrastructure (FFT plans, field data) so the
 * Box3i+Domain factory functions are designed to fail with clear error messages
 * directing users to use the full constructor or pfc::field::create().
 */

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
namespace {
struct OnlyXX {
  double xx{};
};
} // namespace

TEST_CASE("make_spectral_gradient with Box3i only throws informative error",
          "[kernel][field][spectral_gradient][box3i_domain]") {
  using namespace pfc;
  Box3i region = Box3i::from_bounds({0, 0, 0}, {7, 7, 7});

  REQUIRE_THROWS_AS(
      pfc::gradient::make_spectral_gradient<OnlyXX>(region),
      std::runtime_error);
}

TEST_CASE(
    "make_spectral_gradient with Box3i and Domain throws informative error",
    "[kernel][field][spectral_gradient][box3i_domain]") {
  using namespace pfc;
  Box3i region = Box3i::from_bounds({0, 0, 0}, {7, 7, 7});
  Domain domain = Domain{};

  REQUIRE_THROWS_AS(
      pfc::gradient::make_spectral_gradient<OnlyXX>(region, domain),
      std::runtime_error);
}

TEST_CASE(
    "make_spectral_gradient with Box3i, Domain, and spacing throws informative error",
    "[kernel][field][spectral_gradient][box3i_domain]") {
  using namespace pfc;
  Box3i region = Box3i::from_bounds({0, 0, 0}, {7, 7, 7});
  Domain domain = Domain{};
  std::array<double, 3> spacing{1.0, 1.0, 1.0};

  REQUIRE_THROWS_AS(pfc::gradient::make_spectral_gradient<OnlyXX>(region, domain,
                                                                    spacing),
                    std::runtime_error);
}
