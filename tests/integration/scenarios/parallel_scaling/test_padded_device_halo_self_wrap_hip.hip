// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_padded_device_halo_self_wrap_hip.hip
 * @brief HIP twin of the CUDA `PaddedDeviceHaloExchanger` self-wrap face-halo
 *        regression (`test_padded_device_halo_self_wrap.cpp`).
 *
 * `OPENPFC_HIP_FORCE_PACKED_HALO` is read in the constructor; the packed case
 * setenv's before constructing.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/hip/padded_device_halo_exchange.hpp>

namespace {

using pfc::types::Int3;

inline double cell_hash(int gx, int gy, int gz) {
  return 1.0 + static_cast<double>(gx) + 1024.0 * static_cast<double>(gy) +
         1048576.0 * static_cast<double>(gz);
}

inline int periodic_wrap(int g, int N) { return ((g % N) + N) % N; }

inline std::size_t lin(int pi, int pj, int pk, int nxp, int nyp) {
  return static_cast<std::size_t>(pi) +
         static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
         static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

struct FaceHaloRef {
  std::vector<double> expected_face;
  std::vector<double> initial;
  int nx = 0, ny = 0, nz = 0;
  int nxp = 0, nyp = 0, nzp = 0;
};

FaceHaloRef build_face_reference(int rank,
                                 const pfc::decomposition::Decomposition &decomp,
                                 const Int3 &global_size, int hw) {
  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  const auto local_lower = pfc::world::get_lower(local_world);
  const auto local_size = pfc::world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);

  FaceHaloRef ref;
  ref.nx = nx;
  ref.ny = ny;
  ref.nz = nz;
  ref.nxp = nxp;
  ref.nyp = nyp;
  ref.nzp = nzp;
  ref.expected_face.assign(total, 0.0);
  ref.initial.assign(total, 0.0);

  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const int gx = periodic_wrap(local_lower[0] + (pi - hw), global_size[0]);
        const int gy = periodic_wrap(local_lower[1] + (pj - hw), global_size[1]);
        const int gz = periodic_wrap(local_lower[2] + (pk - hw), global_size[2]);
        const double v = cell_hash(gx, gy, gz);
        const std::size_t l = lin(pi, pj, pk, nxp, nyp);
        const bool in_x = pi >= hw && pi < hw + nx;
        const bool in_y = pj >= hw && pj < hw + ny;
        const bool in_z = pk >= hw && pk < hw + nz;
        const int axis_inside =
            static_cast<int>(in_x) + static_cast<int>(in_y) + static_cast<int>(in_z);
        if (axis_inside >= 2) {
          ref.expected_face[l] = v;
        }
        if (axis_inside == 3) {
          ref.initial[l] = v;
        }
      }
    }
  }
  return ref;
}

bool hip_runtime_available() {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  return e == hipSuccess && n > 0;
}

void assert_face_halos(const std::vector<double> &host_after, const FaceHaloRef &ref,
                       int hw) {
  bool ok = true;
  for (int pk = 0; pk < ref.nzp; ++pk) {
    for (int pj = 0; pj < ref.nyp; ++pj) {
      for (int pi = 0; pi < ref.nxp; ++pi) {
        const bool in_x = pi >= hw && pi < hw + ref.nx;
        const bool in_y = pj >= hw && pj < hw + ref.ny;
        const bool in_z = pk >= hw && pk < hw + ref.nz;
        const int axis_inside =
            static_cast<int>(in_x) + static_cast<int>(in_y) + static_cast<int>(in_z);
        if (axis_inside < 2) {
          continue;
        }
        const std::size_t l = lin(pi, pj, pk, ref.nxp, ref.nyp);
        ok = ok && (host_after[l] == ref.expected_face[l]);
      }
    }
  }
  REQUIRE(ok);
}

} // namespace

TEST_CASE("HIP PaddedDeviceHaloExchanger self-wrap face halos hw=1",
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

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  const int hw = 1;
  const FaceHaloRef ref = build_face_reference(rank, decomp, global_size, hw);
  const std::size_t total = ref.initial.size();
  const std::size_t bytes = total * sizeof(double);

  double *d_field = nullptr;
  REQUIRE(hipMalloc(reinterpret_cast<void **>(&d_field), bytes) == hipSuccess);
  REQUIRE(hipMemcpy(d_field, ref.initial.data(), bytes, hipMemcpyHostToDevice) ==
          hipSuccess);

  pfc::hip::PaddedDeviceHaloExchanger exchanger(decomp, rank, hw, MPI_COMM_WORLD,
                                                /*base_tag=*/0);
  exchanger.exchange_halos_device(d_field, total, /*stream=*/nullptr);

  std::vector<double> host_after(total);
  REQUIRE(hipMemcpy(host_after.data(), d_field, bytes, hipMemcpyDeviceToHost) ==
          hipSuccess);
  REQUIRE(hipFree(d_field) == hipSuccess);
  assert_face_halos(host_after, ref, hw);
}

TEST_CASE("HIP PaddedDeviceHaloExchanger self-wrap face halos hw=2 packed",
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

  const Int3 global_size{6, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  {
    const int hw = 2;
    const FaceHaloRef ref = build_face_reference(rank, decomp, global_size, hw);
    const std::size_t total = ref.initial.size();
    const std::size_t bytes = total * sizeof(double);

    double *d_field = nullptr;
    REQUIRE(hipMalloc(reinterpret_cast<void **>(&d_field), bytes) == hipSuccess);
    REQUIRE(hipMemcpy(d_field, ref.initial.data(), bytes, hipMemcpyHostToDevice) ==
            hipSuccess);

    pfc::hip::PaddedDeviceHaloExchanger exchanger(decomp, rank, hw, MPI_COMM_WORLD,
                                                  /*base_tag=*/0);
    REQUIRE_FALSE(exchanger.uses_gpu_aware_mpi());
    exchanger.exchange_halos_device(d_field, total, /*stream=*/nullptr);

    std::vector<double> host_after(total);
    REQUIRE(hipMemcpy(host_after.data(), d_field, bytes, hipMemcpyDeviceToHost) ==
            hipSuccess);
    REQUIRE(hipFree(d_field) == hipSuccess);
    assert_face_halos(host_after, ref, hw);
  }

  ::unsetenv("OPENPFC_HIP_FORCE_PACKED_HALO");
}

#endif // OpenPFC_ENABLE_HIP
