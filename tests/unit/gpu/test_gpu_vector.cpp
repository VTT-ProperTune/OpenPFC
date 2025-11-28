// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <openpfc/gpu/gpu_vector.hpp>
#include <vector>

using Catch::Approx;

TEST_CASE("GPUVector construction", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::gpu::GPUVector<double> vec(100);
  REQUIRE(vec.size() == 100);
  REQUIRE(vec.data() != nullptr);
  REQUIRE(!vec.empty());
}

TEST_CASE("GPUVector empty construction", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::gpu::GPUVector<double> vec(0);
  REQUIRE(vec.size() == 0);
  REQUIRE(vec.empty());
}

TEST_CASE("GPUVector CPU-GPU round trip", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::gpu::GPUVector<double> gpu_vec(input.size());

  gpu_vec.copy_from_host(input);
  std::vector<double> output = gpu_vec.to_host();

  REQUIRE(output == input);
}

TEST_CASE("GPUVector large data round trip", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  const size_t n = 10000;
  std::vector<double> input(n);
  for (size_t i = 0; i < n; ++i) {
    input[i] = static_cast<double>(i);
  }

  pfc::gpu::GPUVector<double> gpu_vec(n);
  gpu_vec.copy_from_host(input);

  std::vector<double> output = gpu_vec.to_host();

  REQUIRE(output.size() == input.size());
  REQUIRE(std::equal(input.begin(), input.end(), output.begin()));
}

TEST_CASE("GPUVector move semantics", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::gpu::GPUVector<double> vec1(100);
  void *ptr1 = vec1.data();
  size_t size1 = vec1.size();

  pfc::gpu::GPUVector<double> vec2 = std::move(vec1);
  REQUIRE(vec2.data() == ptr1);
  REQUIRE(vec2.size() == size1);
  REQUIRE(vec1.data() == nullptr); // Moved from
  REQUIRE(vec1.size() == 0);
}

TEST_CASE("GPUVector copy_from_host size mismatch", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::gpu::GPUVector<double> vec(100);
  std::vector<double> wrong_size(50);

  REQUIRE_THROWS_AS(vec.copy_from_host(wrong_size), std::runtime_error);
}

TEST_CASE("GPUVector complex type", "[gpu][vector]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  using Complex = std::complex<double>;
  std::vector<Complex> input = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};

  pfc::gpu::GPUVector<Complex> gpu_vec(input.size());
  gpu_vec.copy_from_host(input);

  std::vector<Complex> output = gpu_vec.to_host();

  REQUIRE(output.size() == input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    REQUIRE(output[i].real() == Approx(input[i].real()));
    REQUIRE(output[i].imag() == Approx(input[i].imag()));
  }
}

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
