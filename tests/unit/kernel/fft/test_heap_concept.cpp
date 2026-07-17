// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <complex>

#include <openpfc/kernel/fft/heap_concept.hpp>

#include <heffte.h>

using namespace pfc::fft;

static_assert(!HeapBackend<heffte::backend::fftw>,
              "the CPU (FFTW) backend is not a device allocator and must not "
              "satisfy HeapBackend");

#if defined(OpenPFC_ENABLE_CUDA)

static_assert(HeapBackend<heffte::backend::cufft>);

TEST_CASE("cuFFT backend satisfies HeapBackend", "[fft][cuda][heap_concept]") {
  REQUIRE(HeapBackend<heffte::backend::cufft>);
}

TEST_CASE("cuFFT double-precision workspace constructs with a size and "
          "reports it back",
          "[fft][cuda][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::cufft>::buffer_container<
      std::complex<double>>;
  buffer_type buf(1024);
  REQUIRE(buf.size() == 1024);
  REQUIRE(buf.data() != nullptr);
}

TEST_CASE("cuFFT single-precision workspace constructs with a size and "
          "reports it back",
          "[fft][cuda][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::cufft>::buffer_container<
      std::complex<float>>;
  buffer_type buf(1024);
  REQUIRE(buf.size() == 1024);
  REQUIRE(buf.data() != nullptr);
}

TEST_CASE("cuFFT workspace accepts a zero size", "[fft][cuda][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::cufft>::buffer_container<
      std::complex<double>>;
  buffer_type buf(0);
  REQUIRE(buf.size() == 0);
}

#endif // OpenPFC_ENABLE_CUDA

#if defined(OpenPFC_ENABLE_HIP)

static_assert(HeapBackend<heffte::backend::rocfft>);

TEST_CASE("rocFFT backend satisfies HeapBackend", "[fft][hip][heap_concept]") {
  REQUIRE(HeapBackend<heffte::backend::rocfft>);
}

TEST_CASE("rocFFT double-precision workspace constructs with a size and "
          "reports it back",
          "[fft][hip][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::rocfft>::buffer_container<
      std::complex<double>>;
  buffer_type buf(1024);
  REQUIRE(buf.size() == 1024);
  REQUIRE(buf.data() != nullptr);
}

TEST_CASE("rocFFT single-precision workspace constructs with a size and "
          "reports it back",
          "[fft][hip][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::rocfft>::buffer_container<
      std::complex<float>>;
  buffer_type buf(1024);
  REQUIRE(buf.size() == 1024);
  REQUIRE(buf.data() != nullptr);
}

TEST_CASE("rocFFT workspace accepts a zero size", "[fft][hip][heap_concept]") {
  using buffer_type = heffte::fft3d_r2c<heffte::backend::rocfft>::buffer_container<
      std::complex<double>>;
  buffer_type buf(0);
  REQUIRE(buf.size() == 0);
}

#endif // OpenPFC_ENABLE_HIP

TEST_CASE("HeapBackend correctly rejects the CPU backend",
          "[fft][cpu][heap_concept]") {
  REQUIRE_FALSE(HeapBackend<heffte::backend::fftw>);
}
