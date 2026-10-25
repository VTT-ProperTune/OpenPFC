// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_periodic_spectra.cpp
 * @brief Occupancy-matched periodic families and Parseval heat errors.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string_view>
#include <vector>

#include <openpfc/kernel/field/periodic_spectra.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace sp = pfc::field::spectra;

TEST_CASE("builtin families have stable ids and occupancy-1 amplitudes",
          "[field][periodic-spectra][unit]") {
  REQUIRE(sp::gaussian().amplitude(1.0) ==
          Catch::Approx(sp::kContentThreshold).epsilon(1e-14));
  REQUIRE(sp::isotropic_exponential().amplitude(1.0) ==
          Catch::Approx(sp::kContentThreshold).epsilon(1e-14));
  REQUIRE(sp::separable_exponential().amplitude(1.0) ==
          Catch::Approx(sp::kContentThreshold).epsilon(1e-14));
  REQUIRE(sp::isotropic_tophat().amplitude(1.0) == Catch::Approx(1.0));
  REQUIRE(sp::isotropic_tophat().amplitude(1.0 + 1e-12) == Catch::Approx(0.0));
  REQUIRE(sp::find_family("gaussian") != nullptr);
  REQUIRE(sp::find_family("isotropic_tophat") != nullptr);
  REQUIRE(sp::find_family("nope") == nullptr);
}

TEST_CASE("even and odd grids omit or keep Nyquist as documented",
          "[field][periodic-spectra][unit]") {
  const auto even = sp::mode_bounds(16, false);
  REQUIRE(even.first == -7);
  REQUIRE(even.second == 7);
  const auto even_nyq = sp::mode_bounds(16, true);
  REQUIRE(even_nyq.first == -7);
  REQUIRE(even_nyq.second == 8);
  const auto odd = sp::mode_bounds(15, false);
  REQUIRE(odd.first == -7);
  REQUIRE(odd.second == 7);
  REQUIRE(sp::nyquist_wavenumber(16) == Catch::Approx(8.0));
}

TEST_CASE("Gaussian product equals isotropic Gaussian amplitude",
          "[field][periodic-spectra][unit]") {
  const double k_c = 5.0;
  const auto iso = sp::gaussian();
  sp::SpectrumFamily sep{"tmp", "tmp", sp::WeightGeometry::SeparableProduct,
                         &sp::gaussian_amplitude};
  REQUIRE_THAT(sp::mode_amplitude(iso, 2, 3, 4, k_c, 3),
               WithinRel(sp::mode_amplitude(sep, 2, 3, 4, k_c, 3), 1e-14));
}

TEST_CASE("L1 and L2 exponential families differ off-axis at matched f",
          "[field][periodic-spectra][unit]") {
  const double k_c = 8.0;
  const double iso =
      sp::mode_amplitude(sp::isotropic_exponential(), 4, 4, 0, k_c, 2);
  const double sep =
      sp::mode_amplitude(sp::separable_exponential(), 4, 4, 0, k_c, 2);
  REQUIRE(iso > sep);
  REQUIRE(iso > 5.0 * sep);
}

TEST_CASE("tophat has more high-k energy than Gaussian at the same f",
          "[field][periodic-spectra][unit]") {
  constexpr double f = 0.5;
  constexpr int N = 48;
  const auto g = sp::diagnose_spectrum(sp::gaussian(), f, N, 2, 3);
  const auto t = sp::diagnose_spectrum(sp::isotropic_tophat(), f, N, 2, 3);
  REQUIRE(t.energy_near_cutoff > g.energy_near_cutoff);
  REQUIRE(t.moment2_over_nyquist2 > g.moment2_over_nyquist2);
}

TEST_CASE("spectral operator Parseval error is identically zero",
          "[field][periodic-spectra][unit]") {
  for (const auto &fam : sp::builtin_families()) {
    REQUIRE(sp::predict_heat_l2_error(fam, 0, 0.4, 32, sp::kDiffusionTime, 2) ==
            Catch::Approx(0.0));
  }
}

TEST_CASE("1D Parseval heat error recovers the FD-2 single-mode identity",
          "[field][periodic-spectra][unit]") {
  // One cosine mode n=1 on N=32, L=2pi: k=1, theta=2pi/N.
  // A top-hat with f such that k_c > 1 and k_c < 2 keeps only n=0 and |n|=1.
  // n=0 does not contribute to the error. Relative L2 uses energy of n=±1
  // plus DC. Use a large threshold-style tophat at f = 1.5/16 so k_c=1.5.
  const double f = 1.5 / 16.0;
  const int N = 32;
  const double e = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, f, N,
                                             sp::kDiffusionTime, 1);
  const double theta = 2.0 * std::acos(-1.0) / static_cast<double>(N);
  const double k_c = f * static_cast<double>(N) / 2.0;
  const double defect = pfc::field::fd::d2_symbol_defect(2, theta);
  const double tau = sp::kDiffusionTime;
  const double delta = defect * tau / (std::acos(-1.0) * std::acos(-1.0) * f * f);
  const double nu = 1.0 / k_c;
  const double decay = std::exp(-tau * nu * nu);
  const double diff = decay * std::expm1(delta);
  // Energy: DC amp=1, plus two side modes amp=1 each. err only from ±1.
  const double energy = 1.0 + 2.0;
  const double expected = std::sqrt(2.0 * diff * diff / energy);
  REQUIRE_THAT(e, WithinRel(expected, 1e-12));
}

TEST_CASE("Gaussian 3D map is independent of even N at fixed f",
          "[field][periodic-spectra][unit]") {
  const double reference =
      sp::predict_heat_l2_error(sp::gaussian(), 4, 0.3, 128, sp::kDiffusionTime, 3);
  for (int N : {48, 64, 96}) {
    REQUIRE_THAT(
        sp::predict_heat_l2_error(sp::gaussian(), 4, 0.3, N, sp::kDiffusionTime, 3),
        WithinRel(reference, 1e-6));
  }
}

TEST_CASE("odd-N 1D tophat is consistent with the neighbouring even grid",
          "[field][periodic-spectra][unit]") {
  const double e_even = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, 0.4, 32,
                                                  sp::kDiffusionTime, 1);
  const double e_odd = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, 0.4, 31,
                                                 sp::kDiffusionTime, 1);
  REQUIRE_THAT(e_odd, WithinRel(e_even, 0.15));
}

TEST_CASE("including even-grid Nyquist changes a cutoff-at-Nyquist tophat",
          "[field][periodic-spectra][unit]") {
  const double f = 1.0;
  const int N = 32;
  const double without = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, f, N,
                                                   sp::kDiffusionTime, 1, false);
  const double with_nyq = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, f, N,
                                                    sp::kDiffusionTime, 1, true);
  REQUIRE(with_nyq > without);
}

TEST_CASE("Gaussian FD-2 error scales near order 2 at small f in 1D",
          "[field][periodic-spectra][unit]") {
  const double e1 = sp::predict_heat_l2_error(
      sp::gaussian(), 2, 0.20, sp::auto_map_grid(0.20), sp::kDiffusionTime, 1);
  const double e2 = sp::predict_heat_l2_error(
      sp::gaussian(), 2, 0.10, sp::auto_map_grid(0.10), sp::kDiffusionTime, 1);
  const double p = std::log(e1 / e2) / std::log(2.0);
  REQUIRE(p > 1.6);
  REQUIRE(p < 2.4);
}

TEST_CASE("matched f does not imply matched FD-2 error across families",
          "[field][periodic-spectra][unit]") {
  constexpr double f = 0.5;
  constexpr int N = 48;
  const double g =
      sp::predict_heat_l2_error(sp::gaussian(), 2, f, N, sp::kDiffusionTime, 3);
  const double t = sp::predict_heat_l2_error(sp::isotropic_tophat(), 2, f, N,
                                             sp::kDiffusionTime, 3);
  REQUIRE(t > 2.0 * g);
}

TEST_CASE("a 1D top-hat with k_c < 1 is a constant field of amplitude 1",
          "[field][periodic-spectra][unit]") {
  const int N = 32;
  const double f = 0.5 / 16.0;
  const double u = sp::evaluate_periodic_field(sp::isotropic_tophat(), 0.3, 0.0, 0.0,
                                               0.0, f, N, 1);
  REQUIRE_THAT(u, WithinAbs(1.0, 1e-12));
}

TEST_CASE("crossover arithmetic is independent of the family",
          "[field][periodic-spectra][unit]") {
  REQUIRE_THAT(sp::crossover_fraction(6.73, 213.82), WithinRel(0.31573465, 1e-6));
  REQUIRE_THROWS(sp::crossover_fraction(-1.0, 1.0));
}

TEST_CASE("equal-accuracy search reports unattainable targets instead of f_min",
          "[field][periodic-spectra][unit]") {
  const auto &fam = sp::gaussian();
  const auto interior = sp::content_fraction_at(fam, 12, 1e-4);
  REQUIRE(interior.attainable);
  REQUIRE(interior.f > 1.0e-3);
  REQUIRE(interior.f < 1.0);
  REQUIRE(sp::predict_heat_l2_error(fam, 12, interior.f,
                                    sp::auto_map_grid(interior.f),
                                    sp::kDiffusionTime, 3) <= 1e-4);

  const auto at_nyquist = sp::content_fraction_at(fam, 12, 1e-2);
  REQUIRE(at_nyquist.attainable);
  REQUIRE(at_nyquist.f == Catch::Approx(1.0));

  const double f_min = 1.0e-3;
  const double best_fd2 = sp::predict_heat_l2_error(
      fam, 2, f_min, sp::auto_map_grid(f_min), sp::kDiffusionTime, 3);
  const double best_fd12 = sp::predict_heat_l2_error(
      fam, 12, f_min, sp::auto_map_grid(f_min), sp::kDiffusionTime, 3);
  const double too_tight = 0.1 * std::min(best_fd2, best_fd12);
  const auto miss = sp::content_fraction_at(fam, 2, too_tight);
  REQUIRE_FALSE(miss.attainable);

  const auto pick =
      sp::pick_equal_accuracy(fam, too_tight, {{2, 6.73}, {12, 24.69}}, 213.82);
  REQUIRE(pick.fd_order == 0);
  REQUIRE(pick.f_star == Catch::Approx(1.0));
  REQUIRE(pick.attainable);
}
