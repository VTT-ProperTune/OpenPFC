// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_padded_device_halo_self_wrap_hip.hip
 * @brief HIP twin of `test_padded_device_halo_self_wrap.cpp` for
 *        `pfc::comm::HaloExchange<HIPSpace>` (`HaloConnectivity::Faces`).
 *
 * `OPENPFC_HIP_FORCE_PACKED_HALO` is read in the exchanger constructor.
 * Execute on LUMI.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstdlib>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

namespace {

using pfc::HIPSpace;

inline double cell_hash(int gx, int gy, int gz) {
  return 1.0 + static_cast<double>(gx) + 1024.0 * static_cast<double>(gy) +
         1048576.0 * static_cast<double>(gz);
}

inline int wrap(int g, int n) { return ((g % n) + n) % n; }

pfc::data::Field<double, HIPSpace>
make_padded_field(const pfc::decomposition::Decomposition &decomp, int rank,
                  int hw) {
  return pfc::data::Field<double, HIPSpace>(
      pfc::decomposition::domain(decomp),
      pfc::decomposition::local_box(decomp, rank), hw);
}

bool hip_runtime_available() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

void fill_owned_hash_zero_halo(pfc::data::Field<double, HIPSpace> &u) {
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    const int hw = u.storage_halo();
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const bool owned =
              i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
          if (owned) {
            const auto g = u.global(i, j, k);
            data[u.idx(i, j, k)] = cell_hash(g[0], g[1], g[2]);
          } else {
            data[u.idx(i, j, k)] = 0.0;
          }
        }
      }
    }
  });
}

void assert_face_halos(pfc::data::Field<double, HIPSpace> &u) {
  bool ok = true;
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    const auto gsz = u.global_size();
    const int hw = u.storage_halo();
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const bool in_x = i >= 0 && i < n[0];
          const bool in_y = j >= 0 && j < n[1];
          const bool in_z = k >= 0 && k < n[2];
          const int axis_inside = static_cast<int>(in_x) + static_cast<int>(in_y) +
                                  static_cast<int>(in_z);
          if (axis_inside < 2) {
            continue;
          }
          const auto g = u.global(i, j, k);
          const double expect =
              cell_hash(wrap(g[0], gsz[0]), wrap(g[1], gsz[1]), wrap(g[2], gsz[2]));
          ok = ok && (data[u.idx(i, j, k)] == expect);
        }
      }
    }
  });
  REQUIRE(ok);
}

void run_self_wrap_check(const pfc::decomposition::Decomposition &decomp, int rank,
                         int hw) {
  auto u = make_padded_field(decomp, rank, hw);
  fill_owned_hash_zero_halo(u);
  pfc::comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.connectivity() == pfc::comm::HaloConnectivity::Faces);
  halo.exchange();
  assert_face_halos(u);
}

} // namespace

TEST_CASE("HaloExchange HIPSpace Faces: self-wrap face halos hw=1",
          "[gpu][hip][padded_halo][self_wrap]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!hip_runtime_available()) {
    SKIP("No HIP runtime / device available on this host");
  }

  auto domain = pfc::domain::create({8, 6, 4});
  auto decomp = pfc::decomposition::create(domain, 1);
  run_self_wrap_check(decomp, rank, /*hw=*/1);
}

TEST_CASE("HaloExchange HIPSpace Faces: self-wrap hw=2 packed",
          "[gpu][hip][padded_halo][self_wrap]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!hip_runtime_available()) {
    SKIP("No HIP runtime / device available on this host");
  }

  REQUIRE(::setenv("OPENPFC_HIP_FORCE_PACKED_HALO", "1", /*overwrite=*/1) == 0);

  auto domain = pfc::domain::create({6, 6, 4});
  auto decomp = pfc::decomposition::create(domain, 1);
  {
    auto u = make_padded_field(decomp, rank, /*hw=*/2);
    fill_owned_hash_zero_halo(u);
    pfc::comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
    REQUIRE_FALSE(halo.uses_gpu_aware_mpi());
    halo.exchange();
    assert_face_halos(u);
  }

  ::unsetenv("OPENPFC_HIP_FORCE_PACKED_HALO");
}

#endif // OpenPFC_ENABLE_HIP
