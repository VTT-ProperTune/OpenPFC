// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/runtime/gpu/deep_copy_gpu.hpp>
#include <openpfc/runtime/gpu/fill_gpu.hpp>

#include <vector>

using Catch::Approx;

#if defined(OPENPFC_TEST_DEEP_COPY_CUDA)
#include <cuda_runtime.h>
TEST_CASE("deep_copy CUDA DataBuffer scalar fill", "[gpu][deep_copy][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf(5);
  pfc::deep_copy(buf, 1.5);
  const std::vector<double> host = buf.to_host();
  REQUIRE(host.size() == 5);
  for (double x : host) {
    REQUIRE(x == Approx(1.5));
  }
}

TEST_CASE("CUDA fill_cuda_impl on a raw device pointer", "[gpu][deep_copy][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  double *ptr = nullptr;
  REQUIRE(cudaMalloc(&ptr, 4 * sizeof(double)) == cudaSuccess);
  pfc::fill_cuda_impl(ptr, 4, 9.0);
  std::vector<double> host(4);
  REQUIRE(cudaMemcpy(host.data(), ptr, 4 * sizeof(double), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  REQUIRE(cudaFree(ptr) == cudaSuccess);
  for (double x : host) {
    REQUIRE(x == Approx(9.0));
  }
}
#endif

#if defined(OPENPFC_TEST_DEEP_COPY_HIP)
#include <hip/hip_runtime.h>
TEST_CASE("deep_copy HIP DataBuffer scalar fill", "[gpu][deep_copy][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  pfc::core::DataBuffer<pfc::backend::HipTag, double> buf(5);
  pfc::deep_copy(buf, 1.5);
  const std::vector<double> host = buf.to_host();
  REQUIRE(host.size() == 5);
  for (double x : host) {
    REQUIRE(x == Approx(1.5));
  }
}

TEST_CASE("HIP fill_hip_impl on a raw device pointer", "[gpu][deep_copy][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  double *ptr = nullptr;
  REQUIRE(hipMalloc(&ptr, 4 * sizeof(double)) == hipSuccess);
  pfc::fill_hip_impl(ptr, 4, 9.0);
  std::vector<double> host(4);
  REQUIRE(hipMemcpy(host.data(), ptr, 4 * sizeof(double), hipMemcpyDeviceToHost) ==
          hipSuccess);
  REQUIRE(hipFree(ptr) == hipSuccess);
  for (double x : host) {
    REQUIRE(x == Approx(9.0));
  }
}
#endif

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }
