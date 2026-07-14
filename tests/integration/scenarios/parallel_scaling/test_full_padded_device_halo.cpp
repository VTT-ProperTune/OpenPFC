// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_full_padded_device_halo.cpp
 * @brief Exhaustive correctness test for `pfc::cuda::FullPaddedDeviceHalo`.
 *
 * Covers the full **26-direction** halo (faces + edges + corners) on
 * `1`, `2 (2x1x1)`, and `4 (2x2x1)` ranks. The goal is **bit-identical
 * agreement** between the post-exchange padded brick and a host-side
 * reference where every padded cell is set to `hash(periodic_global_coord)`.
 *
 * Each test:
 *   1. Allocates a `(nx+2hw) x (ny+2hw) x (nz+2hw)` device buffer per field.
 *   2. Initialises the **owned** region to `hash(global_coord, field_idx)`,
 *      and clears the halo ring to `0`.
 *   3. Runs `FullPaddedDeviceHalo::exchange()`.
 *   4. Asserts **every** padded cell — including all 8 corners and 12 edge
 *      strips — matches the periodic-wrap reference pattern.
 *
 * The hash is unique per global cell so a single mismatch pinpoints which
 * neighbour data was missing or duplicated.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/runtime/cuda/full_padded_device_halo.hpp>

namespace {

using pfc::types::Int3;

/// Distinct double per (field, global coord) — fits exactly in IEEE 754 double
/// for the small grids used here, so equality compare is sound.
inline double cell_hash(int field, int gx, int gy, int gz) {
  return 1.0 + 0.5 * static_cast<double>(field) + static_cast<double>(gx) +
         1024.0 * static_cast<double>(gy) + 1048576.0 * static_cast<double>(gz);
}

inline int periodic_wrap(int g, int N) { return ((g % N) + N) % N; }

inline std::size_t lin(int pi, int pj, int pk, int nxp, int nyp) {
  return static_cast<std::size_t>(pi) +
         static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
         static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
             static_cast<std::size_t>(nyp);
}

/// Build the reference padded brick (every cell = periodic-wrap hash) and the
/// initial state (owned region from the reference, halo cells = 0).
struct PaddedFieldRef {
  std::vector<double> expected; // size nxp*nyp*nzp
  std::vector<double> initial;  // owned cells from `expected`, halos = 0
};

PaddedFieldRef build_reference(int field_idx, int rank,
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

  PaddedFieldRef ref;
  ref.expected.assign(total, 0.0);
  ref.initial.assign(total, 0.0);

  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const int gx = periodic_wrap(local_lower[0] + (pi - hw), global_size[0]);
        const int gy = periodic_wrap(local_lower[1] + (pj - hw), global_size[1]);
        const int gz = periodic_wrap(local_lower[2] + (pk - hw), global_size[2]);
        const double v = cell_hash(field_idx, gx, gy, gz);
        const std::size_t l = lin(pi, pj, pk, nxp, nyp);
        ref.expected[l] = v;
        const bool owned = pi >= hw && pi < hw + nx && pj >= hw && pj < hw + ny &&
                           pk >= hw && pk < hw + nz;
        if (owned) {
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

/// Drive one exchange-and-verify scenario for a given decomposition and run on
/// `n_fields` fields concurrently.
void run_full_halo_check(const pfc::decomposition::Decomposition &decomp, int rank,
                         const Int3 &global_size, int hw, std::size_t n_fields) {
  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  const auto local_size = pfc::world::get_size(local_world);
  const int nxp = local_size[0] + 2 * hw;
  const int nyp = local_size[1] + 2 * hw;
  const int nzp = local_size[2] + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);
  const std::size_t bytes = total * sizeof(double);

  // Build host references and per-field device buffers.
  std::vector<PaddedFieldRef> refs;
  refs.reserve(n_fields);
  std::vector<double *> d_fields(n_fields, nullptr);

  for (std::size_t f = 0; f < n_fields; ++f) {
    refs.push_back(
        build_reference(static_cast<int>(f), rank, decomp, global_size, hw));
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_fields[f]), bytes) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(d_fields[f], refs[f].initial.data(), bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }

  pfc::cuda::FullPaddedDeviceHalo halo(decomp, rank, hw, MPI_COMM_WORLD, n_fields,
                                       /*base_tag=*/0);
  halo.exchange(d_fields.data(), /*stream=*/nullptr);

  std::vector<double> host_after(total);
  std::size_t total_mismatches = 0;
  for (std::size_t f = 0; f < n_fields; ++f) {
    REQUIRE(cudaMemcpy(host_after.data(), d_fields[f], bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t l = 0; l < total; ++l) {
      if (host_after[l] != refs[f].expected[l]) {
        ++total_mismatches;
      }
    }
    REQUIRE(cudaFree(d_fields[f]) == cudaSuccess);
  }
  REQUIRE(total_mismatches == 0);
}

} // namespace

TEST_CASE("FullPaddedDeviceHalo: 1-rank periodic full-fill (all 26 halos)",
          "[gpu][padded_halo][full_halo]") {
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

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("FullPaddedDeviceHalo: 2-rank 2x1x1 full-fill (X real, Y/Z self)",
          "[MPI][gpu][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }
  if (!cuda_runtime_available()) {
    SKIP("No CUDA runtime / device available on this host");
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, {2, 1, 1});

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("FullPaddedDeviceHalo: 4-rank 2x2x1 full-fill (X+Y real, Z self)",
          "[MPI][gpu][padded_halo][full_halo][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }
  if (!cuda_runtime_available()) {
    SKIP("No CUDA runtime / device available on this host");
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, {2, 2, 1});

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("FullPaddedDeviceHalo: hw=2 1-rank widened halo correctness",
          "[gpu][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!cuda_runtime_available()) {
    SKIP("No CUDA runtime / device available on this host");
  }

  const Int3 global_size{6, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  run_full_halo_check(decomp, rank, global_size, /*hw=*/2, /*n_fields=*/1);
}

TEST_CASE("FullPaddedDeviceHalo: Axes3D set fills only the 6 axis faces",
          "[gpu][padded_halo][full_halo][halo_directions]") {
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

  const int hw = 1;
  const std::size_t n_fields = 1;
  const int field_idx = 0;
  const auto ref = build_reference(field_idx, rank, decomp, global_size, hw);

  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  const auto local_size = pfc::world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const int nxp = nx + 2 * hw, nyp = ny + 2 * hw, nzp = nz + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);
  const std::size_t bytes = total * sizeof(double);

  // Initial buffer: sentinel everywhere, owned cells overwritten with the
  // reference hash.
  const double sentinel = -1.0;
  std::vector<double> initial(total, sentinel);
  for (int pk = hw; pk < hw + nz; ++pk)
    for (int pj = hw; pj < hw + ny; ++pj)
      for (int pi = hw; pi < hw + nx; ++pi)
        initial[lin(pi, pj, pk, nxp, nyp)] = ref.expected[lin(pi, pj, pk, nxp, nyp)];

  double *d_field = nullptr;
  REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_field), bytes) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_field, initial.data(), bytes, cudaMemcpyHostToDevice) ==
          cudaSuccess);

  pfc::cuda::FullPaddedDeviceHalo halo(decomp, rank, hw, MPI_COMM_WORLD, n_fields,
                                       pfc::halo::presets::Axes3D(),
                                       /*base_tag=*/0);
  REQUIRE(halo.direction_set() == pfc::halo::presets::Axes3D());
  halo.exchange(&d_field, /*stream=*/nullptr);

  std::vector<double> host_after(total);
  REQUIRE(cudaMemcpy(host_after.data(), d_field, bytes, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  REQUIRE(cudaFree(d_field) == cudaSuccess);

  // Axes3D should fill ±X / ±Y / ±Z face slabs with the reference, but leave
  // edges and corners at the sentinel value (no widening passes).
  bool values_match = true;
  for (int pk = 0; pk < nzp; ++pk) {
    for (int pj = 0; pj < nyp; ++pj) {
      for (int pi = 0; pi < nxp; ++pi) {
        const bool in_x = pi >= hw && pi < hw + nx;
        const bool in_y = pj >= hw && pj < hw + ny;
        const bool in_z = pk >= hw && pk < hw + nz;
        const int axis_inside =
            static_cast<int>(in_x) + static_cast<int>(in_y) + static_cast<int>(in_z);
        const std::size_t l = lin(pi, pj, pk, nxp, nyp);
        if (axis_inside == 3) {
          // Owned cell — already had the reference value.
          values_match &= host_after[l] == ref.expected[l];
        } else if (axis_inside == 2) {
          // Face cell — Axes3D fills it.
          values_match &= host_after[l] == ref.expected[l];
        } else {
          // Edge or corner — Axes3D narrow passes do not fill these.
          values_match &= host_after[l] == sentinel;
        }
      }
    }
  }
  REQUIRE(values_match);
}

#endif // OpenPFC_ENABLE_CUDA
