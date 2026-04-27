// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/runtime/common/backend_from_string.hpp>

TEST_CASE("backend_from_string maps fftw backend", "[runtime][backend]") {
  const auto backend = pfc::runtime::backend_from_string("fftw");

  REQUIRE(backend.has_value());
  REQUIRE(*backend == pfc::fft::Backend::FFTW);
}

TEST_CASE("backend_from_string rejects unsupported names", "[runtime][backend]") {
  REQUIRE_FALSE(pfc::runtime::backend_from_string("").has_value());
  REQUIRE_FALSE(pfc::runtime::backend_from_string("FFTW").has_value());
  REQUIRE_FALSE(pfc::runtime::backend_from_string("unknown").has_value());
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("backend_from_string maps cuda when compiled in",
          "[runtime][backend][cuda]") {
  const auto backend = pfc::runtime::backend_from_string("cuda");

  REQUIRE(backend.has_value());
  REQUIRE(*backend == pfc::fft::Backend::CUDA);
}
#else
TEST_CASE("backend_from_string rejects cuda when not compiled in",
          "[runtime][backend]") {
  REQUIRE_FALSE(pfc::runtime::backend_from_string("cuda").has_value());
}
#endif
