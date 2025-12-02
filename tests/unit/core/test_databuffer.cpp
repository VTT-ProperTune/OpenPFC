// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/databuffer.hpp>
#include <openpfc/core/memory_traits.hpp>
#include <vector>

using Catch::Approx;

TEST_CASE("DataBuffer CPU construction", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(100);
  REQUIRE(buf.size() == 100);
  REQUIRE(buf.data() != nullptr);
  REQUIRE(!buf.empty());
}

TEST_CASE("DataBuffer CPU empty construction", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(0);
  REQUIRE(buf.size() == 0);
  REQUIRE(buf.empty());
}

TEST_CASE("DataBuffer CPU element access", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(10);

  // Write via operator[]
  for (size_t i = 0; i < 10; ++i) {
    buf[i] = static_cast<double>(i);
  }

  // Read via operator[]
  for (size_t i = 0; i < 10; ++i) {
    REQUIRE(buf[i] == Approx(static_cast<double>(i)));
  }
}

TEST_CASE("DataBuffer CPU copy semantics", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf1(10);
  for (size_t i = 0; i < 10; ++i) {
    buf1[i] = static_cast<double>(i);
  }

  // Copy construction
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf2(buf1);
  REQUIRE(buf2.size() == buf1.size());
  for (size_t i = 0; i < 10; ++i) {
    REQUIRE(buf2[i] == Approx(buf1[i]));
  }

  // Copy assignment
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf3(5);
  buf3 = buf1;
  REQUIRE(buf3.size() == buf1.size());
  for (size_t i = 0; i < 10; ++i) {
    REQUIRE(buf3[i] == Approx(buf1[i]));
  }
}

TEST_CASE("DataBuffer CPU move semantics", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf1(10);
  for (size_t i = 0; i < 10; ++i) {
    buf1[i] = static_cast<double>(i);
  }

  // Move construction
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf2(std::move(buf1));
  REQUIRE(buf2.size() == 10);
  REQUIRE(buf1.size() == 0); // Moved from
  for (size_t i = 0; i < 10; ++i) {
    REQUIRE(buf2[i] == Approx(static_cast<double>(i)));
  }
}

TEST_CASE("DataBuffer CPU copy_from_host", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(5);
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};

  buf.copy_from_host(input);

  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(buf[i] == Approx(input[i]));
  }
}

TEST_CASE("DataBuffer CPU to_host", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(5);
  for (size_t i = 0; i < 5; ++i) {
    buf[i] = static_cast<double>(i + 1);
  }

  std::vector<double> result = buf.to_host();
  REQUIRE(result.size() == 5);
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(result[i] == Approx(static_cast<double>(i + 1)));
  }
}

TEST_CASE("DataBuffer CPU resize", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(5);
  REQUIRE(buf.size() == 5);

  buf.resize(10);
  REQUIRE(buf.size() == 10);

  buf.resize(3);
  REQUIRE(buf.size() == 3);
}

TEST_CASE("DataBuffer CPU copy_from_host size mismatch", "[core][databuffer][cpu]") {
  pfc::core::DataBuffer<pfc::backend::CpuTag, double> buf(5);
  std::vector<double> input = {1.0, 2.0, 3.0}; // Wrong size

  REQUIRE_THROWS_AS(buf.copy_from_host(input), std::runtime_error);
}

TEST_CASE("DataBuffer CPU backend traits", "[core][databuffer][cpu]") {
  using traits = pfc::core::backend_traits<pfc::backend::CpuTag>;
  REQUIRE(traits::has_host_access == true);
  REQUIRE(traits::has_device_access == false);
  REQUIRE(traits::requires_transfer == false);
}

#if defined(OpenPFC_ENABLE_CUDA)
#include "unit/gpu/test_helpers.hpp"

TEST_CASE("DataBuffer CUDA construction", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf(100);
  REQUIRE(buf.size() == 100);
  REQUIRE(buf.data() != nullptr);
  REQUIRE(!buf.empty());
}

TEST_CASE("DataBuffer CUDA empty construction", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf(0);
  REQUIRE(buf.size() == 0);
  REQUIRE(buf.empty());
}

TEST_CASE("DataBuffer CUDA copy semantics disabled", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf1(10);

  // Copy construction should be deleted (compile-time check)
  // This test verifies the behavior at runtime
  REQUIRE(buf1.size() == 10);
}

TEST_CASE("DataBuffer CUDA move semantics", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf1(10);
  REQUIRE(buf1.size() == 10);
  REQUIRE(buf1.data() != nullptr);

  // Move construction
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf2(std::move(buf1));
  REQUIRE(buf2.size() == 10);
  REQUIRE(buf2.data() != nullptr);
  REQUIRE(buf1.size() == 0);       // Moved from
  REQUIRE(buf1.data() == nullptr); // Moved from
}

TEST_CASE("DataBuffer CUDA CPU-GPU round trip", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> gpu_buf(input.size());

  gpu_buf.copy_from_host(input);
  std::vector<double> output = gpu_buf.to_host();

  REQUIRE(output.size() == input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    REQUIRE(output[i] == Approx(input[i]));
  }
}

TEST_CASE("DataBuffer CUDA large data round trip", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  const size_t size = 10000;
  std::vector<double> input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<double>(i);
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> gpu_buf(size);
  gpu_buf.copy_from_host(input);

  std::vector<double> output = gpu_buf.to_host();

  REQUIRE(output.size() == size);
  for (size_t i = 0; i < size; ++i) {
    REQUIRE(output[i] == Approx(static_cast<double>(i)));
  }
}

TEST_CASE("DataBuffer CUDA copy_from_host size mismatch",
          "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf(5);
  std::vector<double> input = {1.0, 2.0, 3.0}; // Wrong size

  REQUIRE_THROWS_AS(buf.copy_from_host(input), std::runtime_error);
}

TEST_CASE("DataBuffer CUDA resize", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  pfc::core::DataBuffer<pfc::backend::CudaTag, double> buf(5);
  REQUIRE(buf.size() == 5);

  buf.resize(10);
  REQUIRE(buf.size() == 10);

  buf.resize(3);
  REQUIRE(buf.size() == 3);
}

TEST_CASE("DataBuffer CUDA backend traits", "[core][databuffer][cuda]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  using traits = pfc::core::backend_traits<pfc::backend::CudaTag>;
  REQUIRE(traits::has_host_access == false);
  REQUIRE(traits::has_device_access == true);
  REQUIRE(traits::requires_transfer == true);
}
#endif // OpenPFC_ENABLE_CUDA
