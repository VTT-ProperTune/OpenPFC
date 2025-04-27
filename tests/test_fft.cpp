/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp> // Updated include for Catch2 v3
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/fft.hpp>
#include <vector>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("FFT forward transformation", "[FFT]") {
  // Create an FFT object with a fixed decomposition
  FFT fft(Decomposition(World({8, 1, 1}), 0, 1));

  // Generate input data
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
}

TEST_CASE("FFT backward transformation", "[FFT]") {
  // Create an FFT object with a fixed decomposition
  FFT fft(Decomposition(World({2, 1, 1}), 0, 1));

  // Generate input data
  std::vector<std::complex<double>> input = {std::complex<double>(1.0, 0.0), std::complex<double>(2.0, 0.0)};

  // Perform the backward transformation
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Perform assertions on the output
  REQUIRE(output.size() == fft.size_inbox());
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
  // Add more assertions as needed
}
