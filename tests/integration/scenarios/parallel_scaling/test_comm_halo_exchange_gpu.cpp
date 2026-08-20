// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Device HaloExchange facade (M4). HIP runs on LUMI. CUDA cases compile when
// CUDA is enabled but cannot execute here — verify that half on tohtori.

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

using namespace pfc;

namespace {

template <typename Space> bool device_runtime_available() {
#if defined(OpenPFC_ENABLE_HIP)
  if constexpr (std::is_same_v<Space, HIPSpace>) {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
  }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
  if constexpr (std::is_same_v<Space, CUDASpace>) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }
#endif
  return false;
}

template <typename Space>
void fill_owned_host(data::Field<double, Space> &u, double val) {
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          data[u.idx(i, j, k)] = val;
  });
}

template <typename Space>
bool halo_x_matches(data::Field<double, Space> &u, int i, double expected) {
  bool matches = true;
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        matches &= data[u.idx(i, j, k)] == expected;
  });
  return matches;
}

bool cray_path_should_be_aware() {
  const char *assume = std::getenv("OPENPFC_ASSUME_GPU_AWARE_MPI");
  if (assume != nullptr && assume[0] == '0') {
    return false;
  }
  const char *cray = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
  return cray != nullptr && cray[0] == '1';
}

template <typename Space> data::Field<double, Space> make_padded_field(
    const decomposition::Decomposition &decomp, int rank, int halo) {
  return data::Field<double, Space>(decomposition::domain(decomp),
                                    decomposition::local_box(decomp, rank), halo);
}

} // namespace

#if defined(OpenPFC_ENABLE_HIP)
TEST_CASE("HaloExchange HIPSpace Faces: single-rank periodic wrap",
          "[halo_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<HIPSpace>()) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  fill_owned_host(u, 7.0);

  comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.connectivity() == comm::HaloConnectivity::Faces);
  REQUIRE(halo.num_fields() == 1);
  if (cray_path_should_be_aware()) {
    REQUIRE(halo.uses_gpu_aware_mpi());
    REQUIRE(halo.uses_contiguous_device_mpi());
  }
  halo.exchange();

  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, -1, 7.0));
  REQUIRE(halo_x_matches(u, n[0], 7.0));
}

TEST_CASE("HaloExchange HIPSpace: start() and persistent are rejected",
          "[halo_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<HIPSpace>()) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  fill_owned_host(u, 1.0);

  comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE_THROWS_AS(halo.start(), std::logic_error);
  REQUIRE_THROWS_AS(halo.finish(), std::logic_error);

  comm::HaloExchangeOptions opt;
  opt.persistent = true;
  REQUIRE_THROWS_AS(
      (comm::HaloExchange<HIPSpace, double>(u, decomp, rank, MPI_COMM_WORLD, opt)),
      std::invalid_argument);
}

TEST_CASE("HaloExchange HIPSpace Faces: two fields wrap",
          "[halo_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 || !device_runtime_available<HIPSpace>()) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  auto v = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  fill_owned_host(u, 3.0);
  fill_owned_host(v, 5.0);

  comm::HaloExchange<HIPSpace, double> halo({&u, &v}, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.num_fields() == 2);
  halo.exchange();
  REQUIRE(halo_x_matches(u, -1, 3.0));
  REQUIRE(halo_x_matches(v, -1, 5.0));
  REQUIRE(halo::field_tag_base(0, 1) == halo::kCanonicalTagCount);
}

TEST_CASE("HaloExchange HIPSpace Faces: 2-rank X-neighbor pack+device MPI",
          "[MPI][halo_exchange][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2 || !device_runtime_available<HIPSpace>()) {
    return;
  }

  auto domain = domain::create({16, 8, 4});
  auto decomp = decomposition::create(domain, {2, 1, 1});
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned_host(u, mine);

  comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  if (cray_path_should_be_aware()) {
    REQUIRE(halo.uses_contiguous_device_mpi());
  }
  halo.exchange();

  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, -1, other));
  REQUIRE(halo_x_matches(u, n[0], other));
}

TEST_CASE("HaloExchange HIPSpace 4-rank Faces: X and Y neighbors",
          "[MPI][halo_exchange][hip][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4 || !device_runtime_available<HIPSpace>()) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  const double mine = static_cast<double>(rank);
  fill_owned_host(u, mine);

  comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();

  const auto n = u.local_size();
  // 2x2x1, x-fastest: rank = cy*2 + cx. Opposite X/Y ranks flip one bit.
  const int west = rank ^ 1;
  const int south = rank ^ 2;
  REQUIRE(halo_x_matches(u, -1, static_cast<double>(west)));
  REQUIRE(halo_x_matches(u, n[0], static_cast<double>(west)));
  bool y_lo = true;
  bool y_hi = true;
  u.with_host_view([&](double *data, std::size_t) {
    for (int k = 0; k < n[2]; ++k) {
      for (int i = 0; i < n[0]; ++i) {
        y_lo &= data[u.idx(i, -1, k)] == static_cast<double>(south);
        y_hi &= data[u.idx(i, n[1], k)] == static_cast<double>(south);
      }
    }
  });
  REQUIRE(y_lo);
  REQUIRE(y_hi);
}

TEST_CASE("HaloExchange HIPSpace 4-rank Full: corner hash wrap",
          "[MPI][halo_exchange][hip][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4 || !device_runtime_available<HIPSpace>()) {
    return;
  }
  if (!pfc::gpu::runtime_mpi_gpu_aware()) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u = make_padded_field<HIPSpace>(decomp, rank, /*halo=*/1);
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    for (int k = 0; k < n[2]; ++k) {
      for (int j = 0; j < n[1]; ++j) {
        for (int i = 0; i < n[0]; ++i) {
          const auto g = u.global(i, j, k);
          data[u.idx(i, j, k)] = 1.0 + static_cast<double>(g[0]) +
                                 1024.0 * static_cast<double>(g[1]) +
                                 1048576.0 * static_cast<double>(g[2]);
        }
      }
    }
  });

  comm::HaloExchangeOptions opt;
  opt.connectivity = comm::HaloConnectivity::Full;
  comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  halo.exchange();

  const auto n = u.size3();
  const auto gsz = u.global_size();
  const int hw = u.storage_halo();
  bool ok = true;
  u.with_host_view([&](double *data, std::size_t) {
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const auto g = u.global(i, j, k);
          const int gx = ((g[0] % gsz[0]) + gsz[0]) % gsz[0];
          const int gy = ((g[1] % gsz[1]) + gsz[1]) % gsz[1];
          const int gz = ((g[2] % gsz[2]) + gsz[2]) % gsz[2];
          const double expect = 1.0 + static_cast<double>(gx) +
                                1024.0 * static_cast<double>(gy) +
                                1048576.0 * static_cast<double>(gz);
          ok &= data[u.idx(i, j, k)] == expect;
        }
      }
    }
  });
  REQUIRE(ok);
}
#endif // OpenPFC_ENABLE_HIP

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("HaloExchange CUDASpace Faces: single-rank periodic wrap",
          "[halo_exchange][cuda]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // CUDA: not testable on LUMI — verify on tohtori.
  if (size != 1 || !device_runtime_available<CUDASpace>()) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = make_padded_field<CUDASpace>(decomp, rank, /*halo=*/1);
  fill_owned_host(u, 7.0);

  comm::HaloExchange<CUDASpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();
  REQUIRE(halo_x_matches(u, -1, 7.0));
}

TEST_CASE("HaloExchange CUDASpace Faces: 4-rank X/Y neighbors",
          "[MPI][halo_exchange][cuda][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // CUDA: not testable on LUMI — verify on tohtori.
  if (size != 4 || !device_runtime_available<CUDASpace>()) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u = make_padded_field<CUDASpace>(decomp, rank, /*halo=*/1);
  fill_owned_host(u, static_cast<double>(rank));
  comm::HaloExchange<CUDASpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();
  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, -1, static_cast<double>(rank ^ 1)));
  REQUIRE(halo_x_matches(u, n[0], static_cast<double>(rank ^ 1)));
}
#endif // OpenPFC_ENABLE_CUDA

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
