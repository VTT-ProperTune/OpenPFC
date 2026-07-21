// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <complex>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>

using namespace pfc;

TEST_CASE("FFT_Impl vector forward rejects wrong buffer sizes",
          "[fft][buffer_size][unit]") {
  auto world = world::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  const auto n_in = fft.size_inbox();
  const auto n_out = fft.size_outbox();
  REQUIRE(n_in > 1);
  REQUIRE(n_out > 0);

  SECTION("undersized real inbox") {
    std::vector<double> in(n_in - 1, 0.0);
    std::vector<std::complex<double>> out(n_out);
    REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
  }
  SECTION("oversized real inbox") {
    std::vector<double> in(n_in + 1, 0.0);
    std::vector<std::complex<double>> out(n_out);
    REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
  }
  SECTION("undersized complex outbox") {
    std::vector<double> in(n_in, 0.0);
    std::vector<std::complex<double>> out(n_out - 1);
    REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
  }
  SECTION("oversized complex outbox") {
    std::vector<double> in(n_in, 0.0);
    std::vector<std::complex<double>> out(n_out + 1);
    REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
  }
}

TEST_CASE("FFT_Impl vector backward rejects wrong buffer sizes",
          "[fft][buffer_size][unit]") {
  auto world = world::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  const auto n_in = fft.size_inbox();
  const auto n_out = fft.size_outbox();
  REQUIRE(n_in > 1);
  REQUIRE(n_out > 0);

  SECTION("undersized complex outbox") {
    std::vector<std::complex<double>> in(n_out - 1);
    std::vector<double> out(n_in, 0.0);
    REQUIRE_THROWS_AS(fft.backward(in, out), std::invalid_argument);
  }
  SECTION("oversized complex outbox") {
    std::vector<std::complex<double>> in(n_out + 1);
    std::vector<double> out(n_in, 0.0);
    REQUIRE_THROWS_AS(fft.backward(in, out), std::invalid_argument);
  }
  SECTION("undersized real inbox") {
    std::vector<std::complex<double>> in(n_out);
    std::vector<double> out(n_in - 1, 0.0);
    REQUIRE_THROWS_AS(fft.backward(in, out), std::invalid_argument);
  }
  SECTION("oversized real inbox") {
    std::vector<std::complex<double>> in(n_out);
    std::vector<double> out(n_in + 1, 0.0);
    REQUIRE_THROWS_AS(fft.backward(in, out), std::invalid_argument);
  }
}

TEST_CASE("FFT_Impl DataBuffer forward/backward rejects wrong buffer sizes",
          "[fft][buffer_size][unit]") {
  auto world = world::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  const auto n_in = fft.size_inbox();
  const auto n_out = fft.size_outbox();
  REQUIRE(n_in > 1);
  REQUIRE(n_out > 0);

  SECTION("undersized real DataBuffer on forward") {
    fft::RealDataBuffer in(n_in - 1);
    fft::ComplexDataBuffer out(n_out);
    REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
  }
  SECTION("undersized complex DataBuffer on backward") {
    fft::ComplexDataBuffer in(n_out - 1);
    fft::RealDataBuffer out(n_in);
    REQUIRE_THROWS_AS(fft.backward(in, out), std::invalid_argument);
  }
}

TEST_CASE("FFT_Impl vector forward accepts correctly sized buffers",
          "[fft][buffer_size][unit]") {
  auto world = world::create(GridSize({8, 1, 1}), PhysicalOrigin({1.0, 1.0, 1.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  std::vector<double> in(fft.size_inbox(), 1.0);
  std::vector<std::complex<double>> out(fft.size_outbox());
  REQUIRE_NOTHROW(fft.forward(in, out));
}
