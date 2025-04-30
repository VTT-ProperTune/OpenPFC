// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include <catch2/catch_test_macros.hpp> // Updated include for Catch2 v3
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("FFT - Basic Functionality", "[fft]") {
  World world = create_world({8, 1, 1});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD, heffte::default_options<heffte::backend::fftw>(),
          world); // Provide all parameters

  REQUIRE(fft.size_inbox() > 0);
  REQUIRE(fft.size_outbox() > 0);
  REQUIRE(fft.size_workspace() > 0);
}

TEST_CASE("FFT forward transformation", "[FFT]") {
  // Create an FFT object with a fixed decomposition
  World world = create_world({8, 1, 1});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD, heffte::default_options<heffte::backend::fftw>(),
          world);

  // Generate input data
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356,
                               3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
}

TEST_CASE("FFT backward transformation", "[FFT]") {
  // Create an FFT object with a fixed decomposition
  World world = create_world({2, 1, 1});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD, heffte::default_options<heffte::backend::fftw>(),
          world);

  // Generate input data
  std::vector<std::complex<double>> input = {std::complex<double>(1.0, 0.0),
                                             std::complex<double>(2.0, 0.0)};

  // Perform the backward transformation
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Perform assertions on the output
  REQUIRE(output.size() == fft.size_inbox());
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
  // Add more assertions as needed
}
