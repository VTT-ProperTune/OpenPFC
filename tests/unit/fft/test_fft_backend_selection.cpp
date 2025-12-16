// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_backend_selection.cpp
 * @brief Tests for runtime FFT backend selection
 *
 * Tests the ability to select different FFT backends (FFTW, CUDA) at runtime
 * through the IFFT interface and configuration parsing.
 */

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/ui/from_json.hpp"

using namespace Catch::Matchers;
using namespace pfc;
using json = nlohmann::json;

TEST_CASE("FFT Backend - FFTW backend selection", "[fft][backend][unit]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);

  // Create FFT with FFTW backend explicitly
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::FFTW);

  REQUIRE(fft != nullptr);
  REQUIRE(fft->size_inbox() > 0);
  REQUIRE(fft->size_outbox() > 0);
  REQUIRE(fft->size_workspace() > 0);
}

TEST_CASE("FFT Backend - FFTW forward/backward transform", "[fft][backend][unit]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::FFTW);

  // Create input data
  std::vector<double> input(fft->size_inbox(), 1.0);
  std::vector<std::complex<double>> fourier(fft->size_outbox());
  std::vector<double> output(fft->size_inbox());

  // Forward transform
  fft->forward(input, fourier);

  // DC component should be sum of all input values
  double expected_dc = static_cast<double>(input.size());
  REQUIRE_THAT(std::real(fourier[0]), WithinAbs(expected_dc, 0.01));

  // Backward transform (should recover original)
  fft->backward(fourier, output);

  // Check round-trip accuracy
  for (size_t i = 0; i < input.size(); ++i) {
    REQUIRE_THAT(output[i], WithinAbs(input[i], 1e-10));
  }
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("FFT Backend - CUDA backend selection", "[fft][backend][cuda][unit]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);

  // Create FFT with CUDA backend explicitly
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::CUDA);

  REQUIRE(fft != nullptr);
  REQUIRE(fft->size_inbox() > 0);
  REQUIRE(fft->size_outbox() > 0);
  REQUIRE(fft->size_workspace() > 0);
}

TEST_CASE("FFT Backend - CUDA requires DataBuffer", "[fft][backend][cuda][unit]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::CUDA);

  // CUDA backend should throw when using std::vector
  std::vector<double> input(fft->size_inbox(), 1.0);
  std::vector<std::complex<double>> output(fft->size_outbox());

  REQUIRE_THROWS_AS(fft->forward(input, output), std::runtime_error);
}
#endif

TEST_CASE("FFT Backend - parse backend from JSON (FFTW)",
          "[fft][backend][config][unit]") {
  json config = {{"backend", "fftw"}};

  auto backend = ui::from_json<fft::Backend>(config);
  REQUIRE(backend == fft::Backend::FFTW);
}

TEST_CASE("FFT Backend - parse backend from JSON (case insensitive)",
          "[fft][backend][config][unit]") {
  json config1 = {{"backend", "FFTW"}};
  json config2 = {{"backend", "FfTw"}};

  auto backend1 = ui::from_json<fft::Backend>(config1);
  auto backend2 = ui::from_json<fft::Backend>(config2);

  REQUIRE(backend1 == fft::Backend::FFTW);
  REQUIRE(backend2 == fft::Backend::FFTW);
}

TEST_CASE("FFT Backend - default to FFTW if not specified",
          "[fft][backend][config][unit]") {
  json config = {}; // Empty config

  auto backend = ui::from_json<fft::Backend>(config);
  REQUIRE(backend == fft::Backend::FFTW);
}

TEST_CASE("FFT Backend - invalid backend throws", "[fft][backend][config][unit]") {
  json config = {{"backend", "invalid_backend"}};

  REQUIRE_THROWS_AS(ui::from_json<fft::Backend>(config), std::runtime_error);
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("FFT Backend - parse CUDA backend from JSON",
          "[fft][backend][config][cuda][unit]") {
  json config = {{"backend", "cuda"}};

  auto backend = ui::from_json<fft::Backend>(config);
  REQUIRE(backend == fft::Backend::CUDA);
}
#else
TEST_CASE("FFT Backend - CUDA backend throws if not compiled",
          "[fft][backend][config][unit]") {
  json config = {{"backend", "cuda"}};

  REQUIRE_THROWS_AS(ui::from_json<fft::Backend>(config), std::runtime_error);
}
#endif

TEST_CASE("FFT Backend - timing functions work", "[fft][backend][unit]") {
  auto world = world::create(GridSize({8}), PhysicalOrigin({8}), GridSpacing({8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::FFTW);

  // Reset timing
  fft->reset_fft_time();
  REQUIRE(fft->get_fft_time() == 0.0);

  // Perform some FFT operations
  std::vector<double> input(fft->size_inbox(), 1.0);
  std::vector<std::complex<double>> fourier(fft->size_outbox());
  std::vector<double> output(fft->size_inbox());

  fft->forward(input, fourier);
  fft->backward(fourier, output);

  // Timing should be non-zero after operations
  REQUIRE(fft->get_fft_time() > 0.0);

  // Reset should work
  fft->reset_fft_time();
  REQUIRE(fft->get_fft_time() == 0.0);
}

TEST_CASE("FFT Backend - size queries work through interface",
          "[fft][backend][unit]") {
  auto world =
      world::create(GridSize({16}), PhysicalOrigin({16}), GridSpacing({16}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create_with_backend(decomposition, 0, fft::Backend::FFTW);

  // Inbox size should be full grid (16*16*16 = 4096)
  REQUIRE(fft->size_inbox() == 4096);

  // Outbox size should be half-complex (16*16*9 = 2304)
  // Due to r2c symmetry: (Nz/2+1) = 9
  REQUIRE(fft->size_outbox() == 2304);

  // Workspace size should be non-zero
  REQUIRE(fft->size_workspace() > 0);
}
