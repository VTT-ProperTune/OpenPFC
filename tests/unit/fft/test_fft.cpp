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
  auto world = world::create({8, 1, 1});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("FFT - forward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto world = world::create({8, 1, 1});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  // Generate input data
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356,
                               3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
}

TEST_CASE("FFT - backward transformation", "[fft][unit]") {
  // Create an FFT object with a fixed decomposition
  auto world = world::create({2, 1, 1});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  // Generate input data
  using complex = std::complex<double>;
  std::vector<complex> input = {complex(1.0, 0.0), complex(2.0, 0.0)};

  // Perform the backward transformation
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Perform assertions on the output
  REQUIRE(output.size() == fft.size_inbox());
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
  // Add more assertions as needed
}
