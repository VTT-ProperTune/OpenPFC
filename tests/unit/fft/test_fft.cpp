// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("FFT - basic functionality", "[fft][unit]") {
  auto world = world::create(GridSize({8), PhysicalOrigin(1), GridSpacing(1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("FFT - forward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto world = world::create(GridSize({8), PhysicalOrigin(1), GridSpacing(1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  // Generate input data: sine wave samples at 8 equally spaced points
  // Input represents: sin(k*x) where x = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356,
                               3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  // Sum of input values should appear in DC component (k=0)
  // Expected: sum(input) ≈ 21.991
  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
}

TEST_CASE("FFT - backward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto world = world::create(GridSize({2), PhysicalOrigin(1), GridSpacing(1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  // Generate input data in frequency space
  // Two frequency components: DC=1.0 and first harmonic=2.0
  using complex = std::complex<double>;
  std::vector<complex> input = {complex(1.0, 0.0), complex(2.0, 0.0)};

  // Perform the backward transformation to real space
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Verify output size
  REQUIRE(output.size() == fft.size_inbox());

  // Average value should be (1.0 + 2.0) / 2 = 1.5
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
}
