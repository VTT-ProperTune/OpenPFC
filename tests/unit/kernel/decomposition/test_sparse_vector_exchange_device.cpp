// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_vector_exchange_device.cpp
 * @brief Catch2 coverage for CudaTag/HipTag SparseVector MPI exchange
 *
 * Non-blocking isend_data/irecv_data must fail closed (throw) when MPI is not
 * device-aware — never silently succeed with MPI_REQUEST_NULL. Blocking
 * send_data/receive_data host-stage when unaware.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using Catch::Approx;

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/exchange_cuda.hpp>
#include <openpfc/runtime/cuda/sparse_vector_cuda.hpp>

namespace {

std::vector<double> cuda_sparse_data_host(
    const pfc::core::SparseVector<pfc::backend::CudaTag, double> &sv) {
  std::vector<double> host(sv.size());
  if (sv.size() == 0) {
    return host;
  }
  REQUIRE(cudaMemcpy(host.data(), sv.data().data(), sv.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

} // namespace

TEST_CASE("CudaTag isend/irecv fail closed when MPI is not CUDA-aware",
          "[SparseVector][MPI][Exchange][CUDA]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::vector<size_t> indices = {0, 2, 4};
  const std::vector<double> values = {1.0, 3.0, 5.0};
  auto sparse =
      pfc::sparsevector::create<double, pfc::backend::CudaTag>(indices, values);

  MPI_Request req = MPI_REQUEST_NULL;

  if (!pfc::exchange::detail::runtime_mpi_cuda_aware()) {
    if (rank == 0) {
      REQUIRE_THROWS_AS(
          pfc::exchange::isend_data(sparse, 0, 1, MPI_COMM_WORLD, &req),
          std::runtime_error);
      REQUIRE(req == MPI_REQUEST_NULL);
    } else {
      // Non-sender early-out: no throw, NULL request
      pfc::exchange::isend_data(sparse, 0, 1, MPI_COMM_WORLD, &req);
      REQUIRE(req == MPI_REQUEST_NULL);
    }

    req = MPI_REQUEST_NULL;
    if (rank == 1) {
      REQUIRE_THROWS_AS(
          pfc::exchange::irecv_data(sparse, 0, 1, MPI_COMM_WORLD, &req),
          std::runtime_error);
      REQUIRE(req == MPI_REQUEST_NULL);
    } else {
      pfc::exchange::irecv_data(sparse, 0, 1, MPI_COMM_WORLD, &req);
      REQUIRE(req == MPI_REQUEST_NULL);
    }
  } else {
    // Device-aware path is exercised separately when a 2-rank MPI job is
    // available; single-rank builds still lock the non-silent unaware contract
    // above when the probe reports false.
    SUCCEED("runtime MPI reports CUDA-aware; fail-closed path not taken");
  }
}

TEST_CASE("CudaTag blocking send_data/receive_data host-stage roundtrip",
          "[SparseVector][MPI][Exchange][CUDA]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 MPI processes");
  }

  const std::vector<size_t> indices = {1, 3, 5};
  auto send_sv =
      pfc::sparsevector::create<double, pfc::backend::CudaTag>(indices, {10.0, 30.0, 50.0});
  auto recv_sv =
      pfc::sparsevector::create<double, pfc::backend::CudaTag>(indices, {0.0, 0.0, 0.0});

  if (rank == 0) {
    pfc::exchange::send_data(send_sv, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    pfc::exchange::receive_data(recv_sv, 0, 1, MPI_COMM_WORLD);
    const auto got = cuda_sparse_data_host(recv_sv);
    REQUIRE(got[0] == Approx(10.0).margin(1e-10));
    REQUIRE(got[1] == Approx(30.0).margin(1e-10));
    REQUIRE(got[2] == Approx(50.0).margin(1e-10));
  }
}

#endif // OpenPFC_ENABLE_CUDA

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/exchange_hip.hpp>
#include <openpfc/runtime/hip/sparse_vector_hip.hpp>

namespace {

std::vector<double> hip_sparse_data_host(
    const pfc::core::SparseVector<pfc::backend::HipTag, double> &sv) {
  std::vector<double> host(sv.size());
  if (sv.size() == 0) {
    return host;
  }
  REQUIRE(hipMemcpy(host.data(), sv.data().data(), sv.size() * sizeof(double),
                    hipMemcpyDeviceToHost) == hipSuccess);
  return host;
}

} // namespace

TEST_CASE("HipTag isend/irecv fail closed when MPI is not HIP-aware",
          "[SparseVector][MPI][Exchange][HIP]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::vector<size_t> indices = {0, 2, 4};
  const std::vector<double> values = {1.0, 3.0, 5.0};
  auto sparse =
      pfc::sparsevector::create<double, pfc::backend::HipTag>(indices, values);

  MPI_Request req = MPI_REQUEST_NULL;

  if (!pfc::exchange::detail::runtime_mpi_hip_aware()) {
    if (rank == 0) {
      REQUIRE_THROWS_AS(
          pfc::exchange::isend_data(sparse, 0, 1, MPI_COMM_WORLD, &req),
          std::runtime_error);
      REQUIRE(req == MPI_REQUEST_NULL);
    } else {
      pfc::exchange::isend_data(sparse, 0, 1, MPI_COMM_WORLD, &req);
      REQUIRE(req == MPI_REQUEST_NULL);
    }

    req = MPI_REQUEST_NULL;
    if (rank == 1) {
      REQUIRE_THROWS_AS(
          pfc::exchange::irecv_data(sparse, 0, 1, MPI_COMM_WORLD, &req),
          std::runtime_error);
      REQUIRE(req == MPI_REQUEST_NULL);
    } else {
      pfc::exchange::irecv_data(sparse, 0, 1, MPI_COMM_WORLD, &req);
      REQUIRE(req == MPI_REQUEST_NULL);
    }
  } else {
    SUCCEED("runtime MPI reports HIP-aware; fail-closed path not taken");
  }
}

TEST_CASE("HipTag blocking send_data/receive_data host-stage roundtrip",
          "[SparseVector][MPI][Exchange][HIP]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 MPI processes");
  }

  const std::vector<size_t> indices = {1, 3, 5};
  auto send_sv =
      pfc::sparsevector::create<double, pfc::backend::HipTag>(indices, {10.0, 30.0, 50.0});
  auto recv_sv =
      pfc::sparsevector::create<double, pfc::backend::HipTag>(indices, {0.0, 0.0, 0.0});

  if (rank == 0) {
    pfc::exchange::send_data(send_sv, 0, 1, MPI_COMM_WORLD);
  } else if (rank == 1) {
    pfc::exchange::receive_data(recv_sv, 0, 1, MPI_COMM_WORLD);
    const auto got = hip_sparse_data_host(recv_sv);
    REQUIRE(got[0] == Approx(10.0).margin(1e-10));
    REQUIRE(got[1] == Approx(30.0).margin(1e-10));
    REQUIRE(got[2] == Approx(50.0).margin(1e-10));
  }
}

#endif // OpenPFC_ENABLE_HIP
