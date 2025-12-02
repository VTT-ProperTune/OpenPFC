// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <openpfc/gpu/gpu_vector.hpp>
#include <openpfc/gpu/kernels_simple.hpp>
#include <vector>

using Catch::Approx;

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

TEST_CASE("GPU kernel: add scalar", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::gpu::GPUVector<double> gpu_vec(input.size());
  gpu_vec.copy_from_host(input);

  // Add 10.0 to each element
  pfc::gpu::add_scalar(gpu_vec, 10.0);

  // Verify
  std::vector<double> result = gpu_vec.to_host();
  std::vector<double> expected = {11.0, 12.0, 13.0, 14.0, 15.0};

  REQUIRE(result == expected);
}

TEST_CASE("GPU kernel: add scalar to large vector", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  const size_t n = 10000;
  std::vector<double> input(n);
  std::vector<double> expected(n);
  for (size_t i = 0; i < n; ++i) {
    input[i] = static_cast<double>(i);
    expected[i] = static_cast<double>(i) + 5.5;
  }

  pfc::gpu::GPUVector<double> gpu_vec(n);
  gpu_vec.copy_from_host(input);

  // Add 5.5 to each element
  pfc::gpu::add_scalar(gpu_vec, 5.5);

  // Verify
  std::vector<double> result = gpu_vec.to_host();
  REQUIRE(result.size() == n);
  REQUIRE(result == expected);
}

TEST_CASE("GPU kernel: multiply scalar", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::gpu::GPUVector<double> gpu_vec(input.size());
  gpu_vec.copy_from_host(input);

  // Multiply each element by 2.0
  pfc::gpu::multiply_scalar(gpu_vec, 2.0);

  // Verify
  std::vector<double> result = gpu_vec.to_host();
  std::vector<double> expected = {2.0, 4.0, 6.0, 8.0, 10.0};

  REQUIRE(result == expected);
}

TEST_CASE("GPU kernel: multiply scalar by zero", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::gpu::GPUVector<double> gpu_vec(input.size());
  gpu_vec.copy_from_host(input);

  // Multiply by zero
  pfc::gpu::multiply_scalar(gpu_vec, 0.0);

  // Verify all elements are zero
  std::vector<double> result = gpu_vec.to_host();
  std::vector<double> expected(input.size(), 0.0);

  REQUIRE(result == expected);
}

TEST_CASE("GPU kernel: chained operations", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::gpu::GPUVector<double> gpu_vec(input.size());
  gpu_vec.copy_from_host(input);

  // Chain: add 10, then multiply by 2
  pfc::gpu::add_scalar(gpu_vec, 10.0);
  pfc::gpu::multiply_scalar(gpu_vec, 2.0);

  // Verify: (1+10)*2 = 22, (2+10)*2 = 24, etc.
  std::vector<double> result = gpu_vec.to_host();
  std::vector<double> expected = {22.0, 24.0, 26.0, 28.0, 30.0};

  REQUIRE(result == expected);
}

TEST_CASE("GPU kernel: empty vector", "[gpu][kernel]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  // Should not crash on empty vector
  pfc::gpu::GPUVector<double> gpu_vec(0);
  REQUIRE_NOTHROW(pfc::gpu::add_scalar(gpu_vec, 10.0));
  REQUIRE_NOTHROW(pfc::gpu::multiply_scalar(gpu_vec, 2.0));
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
