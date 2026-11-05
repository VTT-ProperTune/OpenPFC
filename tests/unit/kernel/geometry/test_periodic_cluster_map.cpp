// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/geometry/periodic_cluster_map.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

using Catch::Matchers::WithinAbs;
using pfc::geometry::cluster_computational;
using pfc::geometry::cluster_derivative;
using pfc::geometry::cluster_jacobian;
using pfc::geometry::cluster_physical;
using pfc::geometry::cluster_physical_integral;
using pfc::geometry::cluster_second_derivative;
using pfc::geometry::periodic_cluster_map;

namespace {

double max_abs_difference(const std::vector<double> &got,
                          const std::vector<double> &want) {
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::fabs(got[i] - want[i]));
  }
  return worst;
}

std::vector<double> sample_sine(const pfc::geometry::PeriodicClusterMap &map,
                                int count) {
  std::vector<double> samples(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const double xi = cluster_computational(map, i, count);
    const double x = cluster_physical(map, xi);
    samples[static_cast<std::size_t>(i)] =
        std::sin(2.0 * std::numbers::pi * x / map.period);
  }
  return samples;
}

std::vector<double> sine_derivative(const pfc::geometry::PeriodicClusterMap &map,
                                    int count) {
  std::vector<double> want(static_cast<std::size_t>(count));
  const double wave = 2.0 * std::numbers::pi / map.period;
  for (int i = 0; i < count; ++i) {
    const double x = cluster_physical(map, cluster_computational(map, i, count));
    want[static_cast<std::size_t>(i)] = wave * std::cos(wave * x);
  }
  return want;
}

double bump_second_error(const pfc::geometry::PeriodicClusterMap &map, int count,
                         double kappa) {
  std::vector<double> samples(static_cast<std::size_t>(count));
  std::vector<double> exact(static_cast<std::size_t>(count));
  const double wave = 2.0 * std::numbers::pi / map.period;
  for (int i = 0; i < count; ++i) {
    const double x = cluster_physical(map, cluster_computational(map, i, count));
    const double theta = wave * x;
    const double value = std::exp(kappa * std::cos(theta));
    const double sine = std::sin(theta);
    samples[static_cast<std::size_t>(i)] = value;
    exact[static_cast<std::size_t>(i)] =
        value * kappa * wave * wave * (kappa * sine * sine - std::cos(theta));
  }
  return max_abs_difference(cluster_second_derivative(map, samples), exact);
}

} // namespace

TEST_CASE("a periodic cluster map keeps a positive Jacobian",
          "[geometry][cluster]") {
  REQUIRE_THROWS_AS(periodic_cluster_map(0.0, 0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_cluster_map(1.0, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_cluster_map(1.0, -1.0), std::invalid_argument);
  const auto map = periodic_cluster_map(2.0, -0.8);
  REQUIRE(cluster_jacobian(map, 0.0) == 1.0 + map.amplitude);
  REQUIRE(cluster_physical(map, 0.0) == 0.0);
  for (int i = 0; i < 16; ++i) {
    REQUIRE(cluster_jacobian(map, cluster_computational(map, i, 16)) > 0.0);
  }
  REQUIRE_THROWS_AS(cluster_derivative(map, std::vector<double>{1.0}),
                    std::invalid_argument);
}

TEST_CASE("the identity cluster map recovers the uniform Fourier derivative",
          "[geometry][cluster]") {
  const auto map = periodic_cluster_map(1.0, 0.0);
  constexpr int count = 32;
  const auto samples = sample_sine(map, count);
  const auto got = cluster_derivative(map, samples);
  const auto want = sine_derivative(map, count);
  REQUIRE(max_abs_difference(got, want) < 1.0e-12);
  std::vector<double> ones(count, 1.0);
  REQUIRE(cluster_physical_integral(map, ones) == 1.0);
  REQUIRE_THAT(cluster_physical_integral(map, got), WithinAbs(0.0, 1.0e-12));
}

TEST_CASE("a smooth cluster map differentiates a sampled sine",
          "[geometry][cluster]") {
  const auto map = periodic_cluster_map(1.0, -0.6);
  constexpr int count = 64;
  const auto samples = sample_sine(map, count);
  REQUIRE(max_abs_difference(cluster_derivative(map, samples),
                             sine_derivative(map, count)) < 1.0e-10);
  std::vector<double> second_exact(count);
  const double wave = 2.0 * std::numbers::pi;
  for (int i = 0; i < count; ++i) {
    const double x = cluster_physical(map, cluster_computational(map, i, count));
    second_exact[static_cast<std::size_t>(i)] = -wave * wave * std::sin(wave * x);
  }
  REQUIRE(max_abs_difference(cluster_second_derivative(map, samples), second_exact) <
          1.0e-9);
  std::vector<double> ones(count, 1.0);
  REQUIRE_THAT(cluster_physical_integral(map, ones), WithinAbs(1.0, 1.0e-12));
  REQUIRE_THAT(cluster_physical_integral(map, cluster_derivative(map, samples)),
               WithinAbs(0.0, 1.0e-10));
}

TEST_CASE("clustering a periodic bump is measured at equal node count",
          "[geometry][cluster]") {
  const auto coarse = periodic_cluster_map(1.0, -0.65);
  const double coarse_error = bump_second_error(coarse, 32, 8.0);
  const double fine_error = bump_second_error(coarse, 64, 8.0);
  REQUIRE(fine_error < coarse_error);

  const auto uniform = periodic_cluster_map(1.0, 0.0);
  const auto packed = periodic_cluster_map(1.0, -0.65);
  constexpr double kappa = 12.0;
  const double uniform_32 = bump_second_error(uniform, 32, kappa);
  const double packed_32 = bump_second_error(packed, 32, kappa);
  const double uniform_48 = bump_second_error(uniform, 48, kappa);
  const double packed_48 = bump_second_error(packed, 48, kappa);
  const double uniform_64 = bump_second_error(uniform, 64, kappa);
  const double packed_64 = bump_second_error(packed, 64, kappa);
  REQUIRE(packed_32 < uniform_32);
  REQUIRE(uniform_32 > 1.0e4);
  REQUIRE(packed_32 > 1.0e3);
  REQUIRE(uniform_48 < packed_32);
  // The equal-error comparison is the decision gate, not merely same N.
  // Both N=48 cases reach error < 10; mapping has *no* DOF reduction here.
  // Uniform N=48 also beats the mapped result at the same N.
  REQUIRE(uniform_48 < 10.0);
  REQUIRE(packed_48 < 10.0);
  REQUIRE(uniform_48 < packed_48);
  REQUIRE(uniform_64 < 1.0e-3);
  REQUIRE(packed_64 < 1.0e-2);
  REQUIRE(uniform_64 < packed_64);
  // With target 1e-3, both variants need at most N=64, not fewer mapped
  // DOFs. The map's second derivative also uses two spectral gradients,
  // each rebuilding a line-FFT plan: there is no demonstrated work saving.
}
