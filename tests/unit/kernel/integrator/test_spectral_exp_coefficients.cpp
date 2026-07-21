// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace pfc::integrator;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

[[nodiscard]] double reference_phi1_nonzero(double L, double dt) {
  return std::expm1(L * dt) / L;
}

[[nodiscard]] double reference_phi1_taylor(double L, double dt) {
  return dt + 0.5 * L * dt * dt;
}

} // namespace

TEST_CASE("spectral_exp_coeffs nonzero finite modes match exp/expm1",
          "[integrator][spectral_exp]") {
  constexpr double dt = 0.25;
  constexpr double threshold = 1e-12;

  SECTION("negative L (dissipative)") {
    constexpr double L = -4.0;
    const auto c = spectral_exp_coeffs(L, dt, threshold);
    REQUIRE_THAT(c.exp_Ldt, WithinRel(std::exp(L * dt), 1e-14));
    REQUIRE_THAT(c.phi1_L, WithinRel(reference_phi1_nonzero(L, dt), 1e-14));
    REQUIRE(std::isfinite(c.exp_Ldt));
    REQUIRE(std::isfinite(c.phi1_L));
  }

  SECTION("positive L") {
    constexpr double L = 1.5;
    const auto c = spectral_exp_coeffs(L, dt, threshold);
    REQUIRE_THAT(c.exp_Ldt, WithinRel(std::exp(L * dt), 1e-14));
    REQUIRE_THAT(c.phi1_L, WithinRel(reference_phi1_nonzero(L, dt), 1e-14));
  }

  SECTION("dt == 0") {
    constexpr double L = -2.0;
    const auto c = spectral_exp_coeffs(L, 0.0, threshold);
    REQUIRE_THAT(c.exp_Ldt, WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(c.phi1_L, WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("spectral_exp_coeffs near-zero L uses Taylor and stays finite",
          "[integrator][spectral_exp]") {
  constexpr double dt = 0.1;
  constexpr double threshold = 1e-12;

  SECTION("L == 0") {
    const auto c = spectral_exp_coeffs(0.0, dt, threshold);
    REQUIRE_THAT(c.exp_Ldt, WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(c.phi1_L, WithinAbs(reference_phi1_taylor(0.0, dt), 1e-15));
    REQUIRE_THAT(c.phi1_L, WithinAbs(dt, 1e-15));
    REQUIRE(std::isfinite(c.phi1_L));
  }

  SECTION("|L| just below threshold") {
    constexpr double L = 1e-13;
    const auto c = spectral_exp_coeffs(L, dt, threshold);
    REQUIRE_THAT(c.exp_Ldt, WithinRel(std::exp(L * dt), 1e-14));
    REQUIRE_THAT(c.phi1_L, WithinAbs(reference_phi1_taylor(L, dt), 1e-18));
    REQUIRE(std::isfinite(c.phi1_L));
  }

  SECTION("|L| at threshold uses expm1 branch") {
    constexpr double L = 1e-12;
    const auto c = spectral_exp_coeffs(L, dt, threshold);
    REQUIRE_THAT(c.phi1_L, WithinRel(reference_phi1_nonzero(L, dt), 1e-12));
    REQUIRE(std::isfinite(c.phi1_L));
  }
}

TEST_CASE("fill_spectral_exp_coeffs multi-mode array fill",
          "[integrator][spectral_exp]") {
  constexpr double dt = 0.05;
  const std::vector<double> L{-8.0, -1.0, -1e-14, 0.0, 0.5, 1.0, 2.0, -3.5};
  std::vector<double> exp_out(L.size());
  std::vector<double> phi_out(L.size());

  fill_spectral_exp_coeffs(L, dt, exp_out, phi_out);

  REQUIRE(exp_out.size() >= 8);
  for (std::size_t i = 0; i < L.size(); ++i) {
    const auto ref = spectral_exp_coeffs(L[i], dt);
    REQUIRE_THAT(exp_out[i], WithinAbs(ref.exp_Ldt, 1e-15));
    REQUIRE_THAT(phi_out[i], WithinAbs(ref.phi1_L, 1e-15));
    REQUIRE(std::isfinite(exp_out[i]));
    REQUIRE(std::isfinite(phi_out[i]));
  }

  SECTION("size mismatch throws") {
    std::vector<double> bad(L.size() - 1);
    REQUIRE_THROWS_AS(fill_spectral_exp_coeffs(L, dt, bad, phi_out),
                      std::invalid_argument);
  }
}

TEST_CASE("SpectralExpCoefficientCache identity invalidation",
          "[integrator][spectral_exp]") {
  constexpr double dt = 0.2;
  const std::vector<double> L{-1.0, 0.0, 2.0, -1e-13, 0.5, -4.0, 1.0, 3.0};
  SpectralExpCoefficientCache cache;

  const auto op = SpectralExpOperatorId{.value = 1};
  const auto dt_id = SpectralExpDtId::from_bits(dt);
  const auto cfg = SpectralExpConfigId{.value = 7};

  cache.ensure(L, dt, op, dt_id, cfg);
  REQUIRE(cache.valid());
  REQUIRE(cache.rebuilt_last_call());
  REQUIRE(cache.exp_Ldt().size() == L.size());
  REQUIRE(cache.phi1_L().size() == L.size());

  for (std::size_t i = 0; i < L.size(); ++i) {
    const auto ref = spectral_exp_coeffs(L[i], dt);
    REQUIRE_THAT(cache.exp_Ldt()[i], WithinAbs(ref.exp_Ldt, 1e-15));
    REQUIRE_THAT(cache.phi1_L()[i], WithinAbs(ref.phi1_L, 1e-15));
  }

  SECTION("same identities skip rebuild") {
    cache.ensure(L, dt, op, dt_id, cfg);
    REQUIRE(cache.valid());
    REQUIRE_FALSE(cache.rebuilt_last_call());
  }

  SECTION("operator id change forces rebuild") {
    cache.ensure(L, dt, SpectralExpOperatorId{.value = 99}, dt_id, cfg);
    REQUIRE(cache.rebuilt_last_call());
  }

  SECTION("dt id change forces rebuild") {
    cache.ensure(L, dt, op, SpectralExpDtId{.value = 42}, cfg);
    REQUIRE(cache.rebuilt_last_call());
  }

  SECTION("config id change forces rebuild") {
    cache.ensure(L, dt, op, dt_id, SpectralExpConfigId{.value = 8});
    REQUIRE(cache.rebuilt_last_call());
  }

  SECTION("length change forces rebuild") {
    const std::vector<double> L2{-1.0, 0.0};
    cache.ensure(L2, dt, op, dt_id, cfg);
    REQUIRE(cache.rebuilt_last_call());
    REQUIRE(cache.exp_Ldt().size() == 2);
  }
}
