// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_exchange_hip_fail_closed.cpp
 * @brief Fail-closed HIP SparseVector isend/irecv without GPU-aware MPI
 */

#if !defined(OpenPFC_ENABLE_HIP)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include <openpfc/runtime/hip/exchange_hip.hpp>
#include <openpfc/runtime/hip/hip_check.hpp>
#include <openpfc/runtime/hip/sparse_vector_hip.hpp>

TEST_CASE("HIP isend/irecv fail closed without OpenPFC_MPI_HIP_AWARE",
          "[gpu][exchange][fail-closed][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if !defined(OpenPFC_MPI_HIP_AWARE)
  const std::vector<size_t> indices = {0, 2, 4};
  const std::vector<double> values = {1.0, 3.0, 5.0};
  auto sparse =
      pfc::sparsevector::create<double, pfc::backend::HipTag>(indices, values);
  auto empty =
      pfc::sparsevector::create<double, pfc::backend::HipTag>({}, {});

  MPI_Request req = MPI_REQUEST_NULL;

  REQUIRE_THROWS_AS(
      pfc::exchange::isend_data(sparse, rank, rank, MPI_COMM_WORLD, &req),
      std::runtime_error);
  REQUIRE_THROWS_AS(
      pfc::exchange::irecv_data(sparse, rank, rank, MPI_COMM_WORLD, &req),
      std::runtime_error);

  req = MPI_REQUEST_NULL;
  REQUIRE_NOTHROW(
      pfc::exchange::isend_data(empty, rank, rank, MPI_COMM_WORLD, &req));
  REQUIRE(req == MPI_REQUEST_NULL);

  req = MPI_REQUEST_NULL;
  REQUIRE_NOTHROW(
      pfc::exchange::irecv_data(empty, rank, rank, MPI_COMM_WORLD, &req));
  REQUIRE(req == MPI_REQUEST_NULL);

  const int other = rank + 1;
  req = MPI_REQUEST_NULL;
  REQUIRE_NOTHROW(
      pfc::exchange::isend_data(sparse, other, rank, MPI_COMM_WORLD, &req));
  REQUIRE(req == MPI_REQUEST_NULL);

  req = MPI_REQUEST_NULL;
  REQUIRE_NOTHROW(
      pfc::exchange::irecv_data(sparse, rank, other, MPI_COMM_WORLD, &req));
  REQUIRE(req == MPI_REQUEST_NULL);

  REQUIRE_THROWS_AS(
      pfc::hip::detail::hip_check(hipErrorInvalidValue, "injected"),
      std::runtime_error);
#else
  SKIP("fail-closed path requires OpenPFC_MPI_HIP_AWARE=OFF");
  (void)rank;
#endif

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }

  int result = Catch::Session().run(argc, argv);

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }

  return result;
}

#endif // OpenPFC_ENABLE_HIP
