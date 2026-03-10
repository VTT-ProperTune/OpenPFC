// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("GPU device detection", "[gpu][device]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available - skipping GPU tests");
  }

  // If we get here, CUDA is available
  REQUIRE(true);
}

// Only compile CUDA-specific tests if CUDA is enabled
// CMake only defines OpenPFC_ENABLE_CUDA if CUDA was found
#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>

TEST_CASE("GPU memory allocation", "[gpu][memory]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  double *d_ptr = nullptr;
  size_t n = 100;

  cudaError_t err = cudaMalloc(&d_ptr, n * sizeof(double));
  REQUIRE(err == cudaSuccess);
  REQUIRE(d_ptr != nullptr);

  cudaFree(d_ptr);
}
#endif

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
