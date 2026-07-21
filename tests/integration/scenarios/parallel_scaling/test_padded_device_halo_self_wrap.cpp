// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_padded_device_halo_self_wrap.cpp
 * @brief Face-halo self-wrap regression for `pfc::cuda::PaddedDeviceHaloExchanger`.
 *
 * On a single-rank Full3D / Axes3D decomp every face neighbor is this rank.
 * After `exchange_halos_device`, each of the six face halo slabs must equal the
 * periodic opposite-side owned slab (not the same-slot near-edge duplicate).
 *
 * Edge and corner cells are intentionally unchecked — this exchanger fills only
 * the six faces. `OPENPFC_CUDA_FORCE_PACKED_HALO` is read in the constructor; the
 * packed case setenv's before constructing (or use process-level env).
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/runtime/cuda/padded_device_halo_exchange.hpp>

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
  std::vector<double> expected_face; // face + owned; edges/corners unused
  std::vector<double> initial;       // owned filled, halos zero
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
        // Owned (3) or face (2) — edges/corners stay 0 in expected_face.
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

bool cuda_runtime_available() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  return e == cudaSuccess && n > 0;
}

/// Assert owned + six face halos; leave edges/corners unchecked.
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
          continue; // edge or corner — exchanger does not fill these
        }
        const std::size_t l = lin(pi, pj, pk, ref.nxp, ref.nyp);
        ok = ok && (host_after[l] == ref.expected_face[l]);
      }
    }
  }
  REQUIRE(ok);
}

void run_self_wrap_check(const pfc::decomposition::Decomposition &decomp, int rank,
                         const Int3 &global_size, int hw) {
  const FaceHaloRef ref = build_face_reference(rank, decomp, global_size, hw);
  const std::size_t total = ref.initial.size();
  const std::size_t bytes = total * sizeof(double);

  double *d_field = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_field), bytes) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_field, ref.initial.data(), bytes, cudaMemcpyHostToDevice) ==
          cudaSuccess);

  // Force packed / GPU-aware selection happens in the constructor (getenv).
  pfc::cuda::PaddedDeviceHaloExchanger exchanger(decomp, rank, hw, MPI_COMM_WORLD,
                                                 /*base_tag=*/0);
  exchanger.exchange_halos_device(d_field, total, /*stream=*/nullptr);

  std::vector<double> host_after(total);
  REQUIRE(cudaMemcpy(host_after.data(), d_field, bytes, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  REQUIRE(cudaFree(d_field) == cudaSuccess);

  assert_face_halos(host_after, ref, hw);
}

} // namespace

TEST_CASE("PaddedDeviceHaloExchanger self-wrap face halos hw=1",
          "[gpu][padded_halo][self_wrap]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!cuda_runtime_available()) {
    SKIP("No CUDA runtime / device available on this host");
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  run_self_wrap_check(decomp, rank, global_size, /*hw=*/1);
}

TEST_CASE("PaddedDeviceHaloExchanger self-wrap face halos hw=2 packed",
          "[gpu][padded_halo][self_wrap]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!cuda_runtime_available()) {
    SKIP("No CUDA runtime / device available on this host");
  }

  // Env is read in the constructor — set before building the exchanger.
  REQUIRE(::setenv("OPENPFC_CUDA_FORCE_PACKED_HALO", "1", /*overwrite=*/1) == 0);

  const Int3 global_size{6, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  {
    const FaceHaloRef ref = build_face_reference(rank, decomp, global_size, /*hw=*/2);
    const std::size_t total = ref.initial.size();
    const std::size_t bytes = total * sizeof(double);

    double *d_field = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_field), bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_field, ref.initial.data(), bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    pfc::cuda::PaddedDeviceHaloExchanger exchanger(decomp, rank, /*hw=*/2,
                                                   MPI_COMM_WORLD, /*base_tag=*/0);
    REQUIRE_FALSE(exchanger.uses_gpu_aware_mpi());
    exchanger.exchange_halos_device(d_field, total, /*stream=*/nullptr);

    std::vector<double> host_after(total);
    REQUIRE(cudaMemcpy(host_after.data(), d_field, bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaFree(d_field) == cudaSuccess);
    assert_face_halos(host_after, ref, /*hw=*/2);
  }

  ::unsetenv("OPENPFC_CUDA_FORCE_PACKED_HALO");
}

#endif // OpenPFC_ENABLE_CUDA
