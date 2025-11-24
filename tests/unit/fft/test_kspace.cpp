// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/constants.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft/kspace.hpp>

using namespace pfc;
using namespace pfc::fft::kspace;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("k_frequency_scaling computes correct scaling factors", "[fft][kspace]") {

  SECTION("3D uniform cubic domain") {
    auto world = world::uniform(128, 0.1);
    auto [fx, fy, fz] = k_frequency_scaling(world);

    double expected = two_pi / (0.1 * 128);
    REQUIRE_THAT(fx, WithinRel(expected, 1e-15));
    REQUIRE_THAT(fy, WithinRel(expected, 1e-15));
    REQUIRE_THAT(fz, WithinRel(expected, 1e-15));
  }

  SECTION("3D non-uniform domain") {
    auto world = world::with_spacing({64, 128, 32}, {0.1, 0.05, 0.2});
    auto [fx, fy, fz] = k_frequency_scaling(world);

    REQUIRE_THAT(fx, WithinRel(two_pi / (0.1 * 64), 1e-15));
    REQUIRE_THAT(fy, WithinRel(two_pi / (0.05 * 128), 1e-15));
    REQUIRE_THAT(fz, WithinRel(two_pi / (0.2 * 32), 1e-15));
  }

  SECTION("2D domain (Lz = 1)") {
    auto world = world::with_spacing({128, 128, 1}, {0.1, 0.1, 1.0});
    auto [fx, fy, fz] = k_frequency_scaling(world);

    REQUIRE_THAT(fx, WithinRel(two_pi / (0.1 * 128), 1e-15));
    REQUIRE_THAT(fy, WithinRel(two_pi / (0.1 * 128), 1e-15));
    REQUIRE_THAT(fz, WithinRel(two_pi / (1.0 * 1), 1e-15));
  }

  SECTION("1D domain (Ly = Lz = 1)") {
    auto world = world::with_spacing({256, 1, 1}, {0.05, 1.0, 1.0});
    auto [fx, fy, fz] = k_frequency_scaling(world);

    REQUIRE_THAT(fx, WithinRel(two_pi / (0.05 * 256), 1e-15));
    REQUIRE_THAT(fy, WithinRel(two_pi, 1e-15));
    REQUIRE_THAT(fz, WithinRel(two_pi, 1e-15));
  }

  SECTION("Matches manual calculation") {
    auto world = world::uniform(64, two_pi / 8.0);
    auto [fx, fy, fz] = k_frequency_scaling(world);

    // Manual calculation (like in examples)
    auto spacing = world::get_spacing(world);
    auto size = world::get_size(world);
    double pi = std::atan(1.0) * 4.0;
    double fx_manual = 2.0 * pi / (spacing[0] * size[0]);
    double fy_manual = 2.0 * pi / (spacing[1] * size[1]);
    double fz_manual = 2.0 * pi / (spacing[2] * size[2]);

    REQUIRE_THAT(fx, WithinRel(fx_manual, 1e-15));
    REQUIRE_THAT(fy, WithinRel(fy_manual, 1e-15));
    REQUIRE_THAT(fz, WithinRel(fz_manual, 1e-15));
  }
}

TEST_CASE("k_component handles Nyquist folding correctly", "[fft][kspace]") {
  constexpr int size = 128;
  constexpr double freq_scale = 1.0;

  SECTION("DC component (index 0)") {
    double k0 = k_component(0, size, freq_scale);
    REQUIRE(k0 == 0.0);
  }

  SECTION("Low positive frequencies (i <= size/2)") {
    double k1 = k_component(1, size, freq_scale);
    double k10 = k_component(10, size, freq_scale);
    double k32 = k_component(32, size, freq_scale);

    REQUIRE(k1 == 1.0 * freq_scale);
    REQUIRE(k10 == 10.0 * freq_scale);
    REQUIRE(k32 == 32.0 * freq_scale);
  }

  SECTION("Nyquist frequency (i = size/2)") {
    double k_nyquist = k_component(size / 2, size, freq_scale);
    REQUIRE(k_nyquist == (size / 2) * freq_scale);
  }

  SECTION("High frequencies fold to negative (i > size/2)") {
    double k65 = k_component(65, size, freq_scale);
    double k100 = k_component(100, size, freq_scale);
    double k127 = k_component(127, size, freq_scale);

    REQUIRE(k65 == -63.0 * freq_scale);
    REQUIRE(k100 == -28.0 * freq_scale);
    REQUIRE(k127 == -1.0 * freq_scale);
  }

  SECTION("Matches manual calculation") {
    int Lx = 64;
    double fx = two_pi / (0.1 * Lx);

    // Manual calculation (like in examples)
    auto k_manual = [Lx, fx](int i) {
      return (i <= Lx / 2) ? i * fx : (i - Lx) * fx;
    };

    for (int i = 0; i < Lx; ++i) {
      double k_helper = k_component(i, Lx, fx);
      double k_manual_val = k_manual(i);
      REQUIRE_THAT(k_helper, WithinRel(k_manual_val, 1e-15));
    }
  }
}

TEST_CASE("k_laplacian_value computes -k² correctly", "[fft][kspace]") {

  SECTION("Zero wave vector (DC)") {
    double kLap = k_laplacian_value(0.0, 0.0, 0.0);
    REQUIRE(kLap == 0.0);
  }

  SECTION("Unit wave vector in x") {
    double kLap = k_laplacian_value(1.0, 0.0, 0.0);
    REQUIRE(kLap == -1.0);
  }

  SECTION("Unit wave vector in y") {
    double kLap = k_laplacian_value(0.0, 1.0, 0.0);
    REQUIRE(kLap == -1.0);
  }

  SECTION("Unit wave vector in z") {
    double kLap = k_laplacian_value(0.0, 0.0, 1.0);
    REQUIRE(kLap == -1.0);
  }

  SECTION("3D wave vector") {
    double kLap = k_laplacian_value(1.0, 2.0, 3.0);
    REQUIRE(kLap == -(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0));
    REQUIRE(kLap == -14.0);
  }

  SECTION("Returns negative values for non-zero k") {
    double kLap = k_laplacian_value(2.0, 2.0, 2.0);
    REQUIRE(kLap < 0.0);
    REQUIRE(kLap == -12.0);
  }

  SECTION("Matches manual calculation") {
    double ki = 1.5, kj = 2.5, kk = 3.5;
    double kLap_helper = k_laplacian_value(ki, kj, kk);
    double kLap_manual = -(ki * ki + kj * kj + kk * kk);
    REQUIRE_THAT(kLap_helper, WithinRel(kLap_manual, 1e-15));
  }
}

TEST_CASE("k_squared_value computes k² correctly", "[fft][kspace]") {

  SECTION("Zero wave vector") {
    double k2 = k_squared_value(0.0, 0.0, 0.0);
    REQUIRE(k2 == 0.0);
  }

  SECTION("Unit wave vector") {
    double k2 = k_squared_value(1.0, 0.0, 0.0);
    REQUIRE(k2 == 1.0);
  }

  SECTION("3D wave vector") {
    double k2 = k_squared_value(3.0, 4.0, 0.0);
    REQUIRE(k2 == 25.0); // 3² + 4² = 25
  }

  SECTION("Returns positive values") {
    double k2 = k_squared_value(2.0, 2.0, 2.0);
    REQUIRE(k2 > 0.0);
    REQUIRE(k2 == 12.0);
  }

  SECTION("Relationship to k_laplacian_value") {
    double ki = 1.5, kj = 2.5, kk = 3.5;
    double k2 = k_squared_value(ki, kj, kk);
    double kLap = k_laplacian_value(ki, kj, kk);
    REQUIRE_THAT(kLap, WithinRel(-k2, 1e-15));
  }
}

TEST_CASE("Integration with World API", "[fft][kspace][integration]") {

  SECTION("Typical diffusion model operator construction") {
    // Simulate the pattern from 04_diffusion_model.cpp
    auto world = world::uniform(64, two_pi / 8.0);
    auto size = world::get_size(world);
    auto [fx, fy, fz] = k_frequency_scaling(world);

    // Test a few specific points
    double k0 =
        k_laplacian_value(k_component(0, size[0], fx), k_component(0, size[1], fy),
                          k_component(0, size[2], fz));
    REQUIRE(k0 == 0.0); // DC component

    double k111 =
        k_laplacian_value(k_component(1, size[0], fx), k_component(1, size[1], fy),
                          k_component(1, size[2], fz));
    REQUIRE(k111 < 0.0); // Non-zero frequency

    // Verify diffusion operator construction
    double dt = 0.01;
    double D = 1.0;
    double opL = 1.0 / (1.0 - D * dt * k111);
    REQUIRE(opL > 0.0);
    REQUIRE(std::isfinite(opL));
  }

  SECTION("Compare against manual calculation across full domain") {
    auto world = world::uniform(32, 0.1);
    auto size = world::get_size(world);
    auto spacing = world::get_spacing(world);

    // Helper function results
    auto [fx, fy, fz] = k_frequency_scaling(world);

    // Manual calculation (from examples)
    double pi = std::atan(1.0) * 4.0;
    double fx_manual = 2.0 * pi / (spacing[0] * size[0]);
    double fy_manual = 2.0 * pi / (spacing[1] * size[1]);
    double fz_manual = 2.0 * pi / (spacing[2] * size[2]);

    REQUIRE_THAT(fx, WithinRel(fx_manual, 1e-15));
    REQUIRE_THAT(fy, WithinRel(fy_manual, 1e-15));
    REQUIRE_THAT(fz, WithinRel(fz_manual, 1e-15));

    // Test several points
    for (int k : {0, 1, 16, 31}) {
      for (int j : {0, 1, 16, 31}) {
        for (int i : {0, 1, 16, 31}) {
          // Helper
          double ki = k_component(i, size[0], fx);
          double kj = k_component(j, size[1], fy);
          double kk_val = k_component(k, size[2], fz);
          double kLap_helper = k_laplacian_value(ki, kj, kk_val);

          // Manual
          double ki_manual =
              (i <= size[0] / 2) ? i * fx_manual : (i - size[0]) * fx_manual;
          double kj_manual =
              (j <= size[1] / 2) ? j * fy_manual : (j - size[1]) * fy_manual;
          double kk_manual =
              (k <= size[2] / 2) ? k * fz_manual : (k - size[2]) * fz_manual;
          double kLap_manual = -(ki_manual * ki_manual + kj_manual * kj_manual +
                                 kk_manual * kk_manual);

          REQUIRE_THAT(kLap_helper, WithinAbs(kLap_manual, 1e-10));
        }
      }
    }
  }
}

TEST_CASE("Helper functions are noexcept", "[fft][kspace][static]") {
  STATIC_REQUIRE(noexcept(k_component(0, 128, 1.0)));
  STATIC_REQUIRE(noexcept(k_laplacian_value(1.0, 1.0, 1.0)));
  STATIC_REQUIRE(noexcept(k_squared_value(1.0, 1.0, 1.0)));

  // k_frequency_scaling is noexcept for all world types
  auto world = world::uniform(64, 0.1);
  STATIC_REQUIRE(noexcept(k_frequency_scaling(world)));
}
