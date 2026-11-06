// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/affine_laplacian.hpp>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <openpfc/kernel/geometry/affine_map.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using pfc::fft::kspace::k_laplacian_value;
using pfc::geometry::affine_map_from_columns;
using pfc::geometry::affine_map_from_matrix;
using pfc::geometry::AffineMap;
using pfc::geometry::identity_affine_map;
using pfc::geometry::physical_volume;
using pfc::geometry::Vec3;

namespace {

void require_vec(Vec3 got, Vec3 want) {
  REQUIRE(got.x == want.x);
  REQUIRE(got.y == want.y);
  REQUIRE(got.z == want.z);
}

double reference_k(int mode, int n, double spacing) {
  const double scale = pfc::two_pi / (spacing * static_cast<double>(n));
  return pfc::fft::kspace::k_component(mode, n, scale);
}

void fill_cosine(std::vector<double> &real, const std::array<int, 3> &n, int mx,
                 int my, int mz) {
  for (int z = 0; z < n[2]; ++z) {
    for (int y = 0; y < n[1]; ++y) {
      for (int x = 0; x < n[0]; ++x) {
        const auto index =
            static_cast<std::size_t>(x) +
            static_cast<std::size_t>(n[0]) *
                (static_cast<std::size_t>(y) +
                 static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(z));
        const double phase =
            pfc::two_pi * (static_cast<double>(mx * x) / static_cast<double>(n[0]) +
                           static_cast<double>(my * y) / static_cast<double>(n[1]) +
                           static_cast<double>(mz * z) / static_cast<double>(n[2]));
        real[index] = std::cos(phase);
      }
    }
  }
}

/// Forward unscaled, multiply by the shipped symbol, backward with 1/N.
double laplacian_field_error(const AffineMap &map, const std::array<int, 3> &n,
                             double spacing, int mx, int my, int mz, double lambda) {
  pfc::fft::LocalFFT fft(n);
  const std::array<double, 3> h{spacing, spacing, spacing};
  std::vector<double> real(fft.size_inbox());
  fill_cosine(real, n, mx, my, mz);
  std::vector<std::complex<double>> hat(fft.size_outbox());
  fft.forward(real, hat);
  pfc::fft::apply_affine_laplacian(map, fft.get_outbox_bounds(), n, h, hat.data(),
                                   hat.size());
  std::vector<double> got(real.size());
  fft.backward(hat, got);
  double error = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    error = std::max(error, std::abs(got[i] - lambda * real[i]));
  }
  return error;
}

} // namespace

TEST_CASE("identity affine map keeps the Cartesian Laplacian and volume",
          "[geometry][affine_map]") {
  const AffineMap map = identity_affine_map();
  REQUIRE(map.uses_cartesian_laplacian());
  REQUIRE(map.determinant == 1.0);
  REQUIRE(map.volume_scale() == 1.0);
  REQUIRE(physical_volume(map, 8.0) == 8.0);
  require_vec(map.to_physical({1.5, -2.0, 0.25}), {1.5, -2.0, 0.25});
  require_vec(map.to_reference({1.5, -2.0, 0.25}), {1.5, -2.0, 0.25});

  const double samples[][3] = {
      {0, 0, 0}, {1, -2, 3}, {0.2, -0.4, 0.6}, {-5, 0.5, 7}};
  for (const auto &k : samples) {
    REQUIRE(pfc::fft::laplacian_symbol(map, k[0], k[1], k[2]) ==
            k_laplacian_value(k[0], k[1], k[2]));
    REQUIRE(map.metric_laplacian_symbol(k[0], k[1], k[2]) ==
            k_laplacian_value(k[0], k[1], k[2]));
  }

  const double eye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const AffineMap shifted = affine_map_from_matrix(eye, {1, 2, 3});
  REQUIRE(shifted.uses_cartesian_laplacian());
  require_vec(shifted.to_physical({1, 0, -1}), {2, 2, 2});
  REQUIRE(pfc::fft::laplacian_symbol(shifted, 0.2, -0.4, 0.6) ==
          k_laplacian_value(0.2, -0.4, 0.6));

  constexpr int n = 16;
  constexpr double spacing = 1.0;
  const double kx = reference_k(1, n, spacing);
  const double lambda = k_laplacian_value(kx, 0.0, 0.0);
  pfc::fft::LocalFFT fft({n, n, n});
  std::vector<double> real(fft.size_inbox());
  fill_cosine(real, {n, n, n}, 1, 0, 0);
  std::vector<std::complex<double>> hat(fft.size_outbox());
  fft.forward(real, hat);
  auto cartesian = hat;
  pfc::fft::kspace::for_each_kpoint(
      fft.get_outbox_bounds(), {n, n, n}, {spacing, spacing, spacing},
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        cartesian[idx] *= k_laplacian_value(ki, kj, kk);
      });
  pfc::fft::apply_affine_laplacian(map, fft.get_outbox_bounds(), {n, n, n},
                                   {spacing, spacing, spacing}, hat.data(),
                                   hat.size());
  double coeff_gap = 0.0;
  for (std::size_t i = 0; i < hat.size(); ++i) {
    coeff_gap = std::max(coeff_gap, std::abs(hat[i] - cartesian[i]));
  }
  REQUIRE(coeff_gap == 0.0);
  REQUIRE(laplacian_field_error(map, {n, n, n}, spacing, 1, 0, 0, lambda) < 1.0e-9);
}

TEST_CASE("a rigid rotation preserves the isotropic Laplacian",
          "[geometry][affine_map]") {
  const AffineMap rotated =
      affine_map_from_columns({0, 1, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, 0});
  REQUIRE_FALSE(rotated.uses_cartesian_laplacian());
  REQUIRE(rotated.determinant == 1.0);
  REQUIRE(rotated.volume_scale() == 1.0);
  require_vec(rotated.to_physical({1, 0, 0}), {0, 1, 0});

  const double samples[][3] = {
      {0.3, -1.7, 2.2}, {1, 2, -3}, {4, 0, 0}, {0, -2, 0.5}};
  for (const auto &k : samples) {
    const double got = pfc::fft::laplacian_symbol(rotated, k[0], k[1], k[2]);
    const double cartesian = k_laplacian_value(k[0], k[1], k[2]);
    REQUIRE_THAT(got, WithinAbs(cartesian, 1.0e-12 * (1.0 + std::abs(cartesian))));
  }

  constexpr int n = 16;
  const std::array<int, 3> size{n, n, n};
  const std::array<double, 3> spacing{1, 1, 1};
  pfc::fft::LocalFFT fft(size);
  std::vector<double> mode_y(fft.size_inbox());
  std::vector<double> mode_z(fft.size_inbox());
  fill_cosine(mode_y, size, 0, 1, 0);
  fill_cosine(mode_z, size, 0, 0, 1);
  std::vector<double> real(fft.size_inbox());
  fill_cosine(real, size, 2, 1, 0);
  for (std::size_t i = 0; i < real.size(); ++i) {
    real[i] += 0.5 * mode_y[i] + 0.25 * mode_z[i];
  }
  std::vector<std::complex<double>> identity_hat(fft.size_outbox());
  std::vector<std::complex<double>> rotated_hat(fft.size_outbox());
  fft.forward(real, identity_hat);
  rotated_hat = identity_hat;
  pfc::fft::apply_affine_laplacian(identity_affine_map(), fft.get_outbox_bounds(),
                                   size, spacing, identity_hat.data(),
                                   identity_hat.size());
  pfc::fft::apply_affine_laplacian(rotated, fft.get_outbox_bounds(), size, spacing,
                                   rotated_hat.data(), rotated_hat.size());
  std::vector<double> identity_field(real.size());
  std::vector<double> rotated_field(real.size());
  fft.backward(identity_hat, identity_field);
  fft.backward(rotated_hat, rotated_field);
  double gap = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < identity_field.size(); ++i) {
    gap = std::max(gap, std::abs(identity_field[i] - rotated_field[i]));
    scale = std::max(scale, std::abs(identity_field[i]));
  }
  REQUIRE(gap <= 1.0e-9 * (1.0 + scale));
}

TEST_CASE("anisotropic scaling matches the analytical Fourier eigenvalue",
          "[geometry][affine_map]") {
  constexpr double sx = 2.0;
  constexpr double sy = 3.0;
  constexpr double sz = 4.0;
  const AffineMap map =
      affine_map_from_columns({sx, 0, 0}, {0, sy, 0}, {0, 0, sz}, {0, 0, 0});
  REQUIRE(map.determinant == sx * sy * sz);
  REQUIRE(map.volume_scale() == sx * sy * sz);
  double metric[9];
  double inverse_metric[9];
  map.metric(metric);
  map.inverse_metric(inverse_metric);
  REQUIRE(metric[0] == sx * sx);
  REQUIRE(metric[4] == sy * sy);
  REQUIRE(metric[8] == sz * sz);
  REQUIRE(inverse_metric[0] == 1.0 / (sx * sx));
  REQUIRE_THAT(inverse_metric[4], WithinAbs(1.0 / (sy * sy), 1.0e-15));
  REQUIRE(inverse_metric[8] == 1.0 / (sz * sz));
  const Vec3 gradient = map.physical_gradient({3.0, -1.0, 2.0});
  require_vec(gradient, {3.0 / sx, -1.0 / sy, 2.0 / sz});

  constexpr int n = 16;
  constexpr double spacing = 1.0;
  const double kx = reference_k(1, n, spacing);
  const double ky = reference_k(1, n, spacing);
  const double kz = reference_k(1, n, spacing);
  const double lambda_x = -((kx / sx) * (kx / sx));
  const double lambda_xyz =
      -((kx / sx) * (kx / sx) + (ky / sy) * (ky / sy) + (kz / sz) * (kz / sz));
  REQUIRE(laplacian_field_error(map, {n, n, n}, spacing, 1, 0, 0, lambda_x) <
          1.0e-9);
  REQUIRE(laplacian_field_error(map, {n, n, n}, spacing, 1, 1, 1, lambda_xyz) <
          1.0e-9);
  REQUIRE_THAT(pfc::fft::laplacian_symbol(map, kx, ky, kz),
               WithinAbs(lambda_xyz, 1.0e-12));
}

TEST_CASE("a shear cell matches the independent k^T G k symbol",
          "[geometry][affine_map]") {
  constexpr double gamma = 0.5;
  const AffineMap map =
      affine_map_from_columns({1, 0, 0}, {gamma, 1, 0}, {0, 0, 1}, {4, -2, 1});
  REQUIRE(map.determinant == 1.0);
  REQUIRE(map.volume_scale() == 1.0);
  REQUIRE_FALSE(map.uses_cartesian_laplacian());
  const double kx = 1.25;
  const double ky = -0.75;
  const double kz = 0.5;
  const double predicted =
      -((1.0 + gamma * gamma) * kx * kx - 2.0 * gamma * kx * ky + ky * ky + kz * kz);
  REQUIRE_THAT(pfc::fft::laplacian_symbol(map, kx, ky, kz),
               WithinAbs(predicted, 1.0e-12));
  double inverse_metric[9];
  map.inverse_metric(inverse_metric);
  const double form = kx * (inverse_metric[0] * kx + inverse_metric[1] * ky) +
                      ky * (inverse_metric[3] * kx + inverse_metric[4] * ky) +
                      kz * kz;
  REQUIRE_THAT(-form, WithinAbs(predicted, 1.0e-12));

  constexpr int n = 16;
  constexpr double spacing = 1.0;
  const double mode_kx = reference_k(1, n, spacing);
  const double mode_ky = reference_k(1, n, spacing);
  const double mode_lambda = -((1.0 + gamma * gamma) * mode_kx * mode_kx -
                               2.0 * gamma * mode_kx * mode_ky + mode_ky * mode_ky);
  REQUIRE(laplacian_field_error(map, {n, n, n}, spacing, 1, 1, 0, mode_lambda) <
          1.0e-9);
}

TEST_CASE("physical volume scales by the absolute Jacobian determinant",
          "[geometry][affine_map]") {
  const AffineMap expanding =
      affine_map_from_columns({-2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {1, 1, 1});
  REQUIRE(expanding.determinant == -24.0);
  REQUIRE(expanding.volume_scale() == 24.0);
  REQUIRE(physical_volume(expanding, 10.0) == 240.0);

  constexpr int n = 8;
  constexpr double spacing = 0.5;
  const double cell = spacing * spacing * spacing;
  double sum = 0.0;
  for (int i = 0; i < n * n * n; ++i) sum += expanding.volume_scale() * cell;
  const double reference_volume = static_cast<double>(n * n * n) * cell;
  REQUIRE(sum == 24.0 * reference_volume);
  REQUIRE(physical_volume(expanding, reference_volume) == 24.0 * reference_volume);
}

TEST_CASE("coordinate transforms use the stored Jacobian",
          "[geometry][affine_map]") {
  const AffineMap map =
      affine_map_from_columns({2, 0, 0}, {0.5, 1, 0}, {0, 0, 3}, {1, -2, 4});
  REQUIRE(map.determinant == 6.0);
  require_vec(map.column(0), {2, 0, 0});
  require_vec(map.column(1), {0.5, 1, 0});
  require_vec(map.column(2), {0, 0, 3});
  REQUIRE(map.inverse_column(0).x == 0.5);
  REQUIRE(map.inverse_column(1).x == -0.25);
  REQUIRE(map.inverse_column(1).y == 1.0);
  REQUIRE(map.inverse_column(2).z == 1.0 / 3.0);

  const Vec3 xi{1, 2, -1};
  const Vec3 physical{4, 0, 1};
  require_vec(map.to_physical(xi), physical);
  const Vec3 back = map.to_reference(physical);
  REQUIRE_THAT(back.x, WithinAbs(xi.x, 1.0e-15));
  REQUIRE_THAT(back.y, WithinAbs(xi.y, 1.0e-15));
  REQUIRE_THAT(back.z, WithinAbs(xi.z, 1.0e-15));

  double metric[9];
  map.metric(metric);
  REQUIRE(metric[0] == 4.0);
  REQUIRE(metric[3] == 1.0);
  REQUIRE(metric[4] == 1.25);
  REQUIRE(metric[8] == 9.0);
}

TEST_CASE("a singular or non-finite affine map is rejected",
          "[geometry][affine_map]") {
  const double singular[9] = {1, 0, 0, 2, 0, 0, 0, 0, 1};
  REQUIRE_THROWS_AS(affine_map_from_matrix(singular, {}), std::invalid_argument);
  const double bad[9] = {1, 0, 0, 0, 1, 0, 0, 0, NAN};
  REQUIRE_THROWS_AS(affine_map_from_matrix(bad, {}), std::invalid_argument);
  const double eye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  REQUIRE_THROWS_AS(affine_map_from_matrix(eye, {0, INFINITY, 0}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(affine_map_from_matrix(nullptr, {}), std::invalid_argument);
  pfc::fft::LocalFFT fft({4, 4, 4});
  std::vector<std::complex<double>> hat(fft.size_outbox());
  REQUIRE_THROWS_AS(pfc::fft::apply_affine_laplacian(
                        identity_affine_map(), fft.get_outbox_bounds(), {4, 4, 4},
                        {1, 1, 1}, hat.data(), hat.size() - 1),
                    std::invalid_argument);
}
