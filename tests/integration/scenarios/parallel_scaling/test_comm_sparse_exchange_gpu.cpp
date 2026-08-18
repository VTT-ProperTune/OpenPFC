// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Device SparseExchange (M4). HIP runs on LUMI. CUDA case compiles when
// CUDA is enabled; execute that half on tohtori.

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/comm_sparse_exchange_gpu.hpp>

using namespace pfc;

namespace {

Box3i whole_box(int nx, int ny, int nz) {
  return Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
}

template <typename Space> bool device_runtime_available() {
#if defined(OpenPFC_ENABLE_HIP)
  if constexpr (std::is_same_v<Space, HipSpace>) {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
  }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
  if constexpr (std::is_same_v<Space, CudaSpace>) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }
#endif
  return false;
}

} // namespace

#if defined(OpenPFC_ENABLE_HIP)
TEST_CASE("SparseExchange HipSpace: custom RemoteHalo scatter stays on device",
          "[sparse_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<HipSpace>()) {
    return;
  }
  if (!pfc::gpu::runtime_mpi_gpu_aware()) {
    return;
  }

  data::Field<double, HipSpace> u(domain::create({8, 1, 1}), whole_box(8, 1, 1),
                                  0);
  u.with_host_view([&](double *data, std::size_t) {
    for (int i = 0; i < 8; ++i) {
      data[u.idx(i, 0, 0)] = static_cast<double>(i + 1);
    }
  });

  halo::RemoteHalo<double> h;
  h.peer_rank = rank;
  h.send_tag = 7;
  h.recv_tag = 7;
  h.send_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{2, 5});
  h.recv_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{6, 7});
  h.scatter_after_recv = true;

  comm::SparseExchange<HipSpace, double> ex(u, {std::move(h)}, rank,
                                            MPI_COMM_WORLD);
  REQUIRE(ex.num_halos() == 1);
  REQUIRE_THROWS_AS(ex.start(), std::logic_error);
  ex.exchange();

  u.with_host_view([&](double *data, std::size_t) {
    REQUIRE(data[u.idx(6, 0, 0)] == 3.0);
    REQUIRE(data[u.idx(7, 0, 0)] == 6.0);
    REQUIRE(data[u.idx(2, 0, 0)] == 3.0);
  });
}

TEST_CASE("SparseExchange HipSpace: structured face_recv_ptrs are device-side",
          "[sparse_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<HipSpace>()) {
    return;
  }

  auto domain = domain::create({8, 6, 1});
  auto decomp = decomposition::create(domain, 1);
  data::Field<double, HipSpace> u(domain, decomposition::local_box(decomp, rank),
                                  /*storage_halo=*/0, /*iteration_halo=*/1);
  comm::SparseExchange<HipSpace, double> ex(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(ex.num_halos() == 6);
  const auto faces = ex.face_recv_ptrs();
  for (int f = 0; f < 6; ++f) {
    REQUIRE(faces[static_cast<std::size_t>(f)] != nullptr);
  }
}
#endif

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("SparseExchange CudaSpace: custom RemoteHalo scatter stays on device",
          "[sparse_exchange][cuda]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<CudaSpace>()) {
    return;
  }
  if (!pfc::gpu::runtime_mpi_gpu_aware()) {
    return;
  }

  data::Field<double, CudaSpace> u(domain::create({8, 1, 1}), whole_box(8, 1, 1),
                                   0);
  u.with_host_view([&](double *data, std::size_t) {
    for (int i = 0; i < 8; ++i) {
      data[u.idx(i, 0, 0)] = static_cast<double>(i + 1);
    }
  });

  halo::RemoteHalo<double> h;
  h.peer_rank = rank;
  h.send_tag = 7;
  h.recv_tag = 7;
  h.send_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{2, 5});
  h.recv_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{6, 7});
  h.scatter_after_recv = true;

  comm::SparseExchange<CudaSpace, double> ex(u, {std::move(h)}, rank,
                                             MPI_COMM_WORLD);
  ex.exchange();
  u.with_host_view([&](double *data, std::size_t) {
    REQUIRE(data[u.idx(6, 0, 0)] == 3.0);
  });
}
#endif

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
