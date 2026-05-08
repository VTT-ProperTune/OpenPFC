// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_sparse_halo_exchange.cpp
 * @brief Tests for `pfc::SparseHaloExchanger<T>`,
 *        `pfc::halo::make_structured_halos<T>`, and
 *        `pfc::halo::copy_to_face_layout<T>`.
 *
 * Covers:
 *   1. Single-peer round-trip with a hand-crafted `RemoteHalo` list (no
 *      grid / face geometry; verifies the gather → MPI → scatter contract).
 *   2. Single-rank `Axes3D()` self-wrap: the periodic neighbour is the
 *      rank itself; recv buffers must hold the cells from the opposite
 *      face of the owned domain.
 *   3. Single-rank `Full3D()` self-wrap: an edge-direction recv buffer
 *      must hold the diagonally opposite edge cells, exercising the
 *      non-axis branch of `direction_to_canonical_tag`.
 *   4. Multi-rank `Axes3D()` (`nproc == 2`) X-split: each rank's recv
 *      buffer matches the peer's send slab (regression for tag pairing
 *      and the in-place periodic recv index ordering).
 *   5. `Connectivity::Edges` no longer aliases `Faces`: dispatched halo
 *      pattern map contains 18 entries on a 3×3×3 decomposition (faces
 *      + edges, no corners).
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/decomposition/sparse_halo_exchange.hpp>

using namespace pfc;
using Int3 = pfc::types::Int3;

namespace {

/// Build a deterministic field value `f(gx, gy, gz)` so we can assert
/// recv contents from any rank and any direction without hand-rolled
/// neighbour bookkeeping.
double periodic_field_value(int gx, int gy, int gz) {
  return static_cast<double>(gx) + 1000.0 * static_cast<double>(gy) +
         1'000'000.0 * static_cast<double>(gz);
}

} // namespace

TEST_CASE("SparseHaloExchanger: single-peer custom RemoteHalo round-trip",
          "[MPI][sparse_halo][custom]") {
  // Single-rank only: build a self-pointing exchange where we lift two
  // arbitrary indices, ship them through MPI, scatter them into a
  // different region of the same field, and verify the values arrived.
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return; // Test is single-rank by design.
  }

  std::vector<double> field(8);
  for (std::size_t i = 0; i < field.size(); ++i) {
    field[i] = static_cast<double>(i + 1);
  }

  halo::RemoteHalo<double> h;
  h.peer_rank = rank;
  h.send_tag = 7;
  h.recv_tag = 7;
  h.send_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{2, 5});
  h.recv_values =
      core::SparseVector<backend::CpuTag, double>(std::vector<std::size_t>{6, 7});
  h.scatter_after_recv = true;

  std::vector<halo::RemoteHalo<double>> halos;
  halos.push_back(std::move(h));
  SparseHaloExchanger<double> ex(MPI_COMM_WORLD, rank, std::move(halos));
  ex.exchange_halos(field.data(), field.size());

  // After the exchange, indices 6 and 7 of `field` should now hold the
  // values originally at indices 2 and 5 (sorted ascending — SparseVector
  // sorts both index lists).
  REQUIRE(field[6] == 3.0); // was index 2 → value 3
  REQUIRE(field[7] == 6.0); // was index 5 → value 6
  // Sanity: original cells unchanged.
  REQUIRE(field[2] == 3.0);
  REQUIRE(field[5] == 6.0);
}

TEST_CASE("SparseHaloExchanger: Axes3D self-wrap on a single rank",
          "[MPI][sparse_halo][structured][axes3d]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  constexpr int N = 4;
  constexpr int hw = 1;
  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);

  const std::size_t nlocal = static_cast<std::size_t>(N) *
                             static_cast<std::size_t>(N) *
                             static_cast<std::size_t>(N);
  std::vector<double> u(nlocal);
  for (int z = 0; z < N; ++z) {
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        u[static_cast<std::size_t>(x) +
          static_cast<std::size_t>(y) * static_cast<std::size_t>(N) +
          static_cast<std::size_t>(z) * static_cast<std::size_t>(N * N)] =
            periodic_field_value(x, y, z);
      }
    }
  }

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  SparseHaloExchanger<double> ex(
      MPI_COMM_WORLD, rank, halo::make_structured_halos<double>(decomp, rank, hw));
  ex.exchange_halos(u.data(), u.size());
  halo::copy_to_face_layout(ex, face_halos);

  // For a periodic single-rank cube the +X face buffer holds the LEFT
  // edge values (x=0 plane) of the same field, and the -X face buffer
  // holds the RIGHT edge values (x=N-1). This mirrors the
  // create_recv_halo / create_send_halo conventions exercised by the
  // create_send_halo / create_recv_halo conventions and the templated
  // brick Laplacians.
  // The face buffer layout produced by SparseVector(sorted recv indices)
  // is the row-major slab in the same order as create_recv_halo, so we
  // can index it as a 3D buffer of size hw * N * N (or N * hw * N, etc.).
  REQUIRE(face_halos[0].size() == static_cast<std::size_t>(hw) *
                                      static_cast<std::size_t>(N) *
                                      static_cast<std::size_t>(N)); // +X
  REQUIRE(face_halos[1].size() == static_cast<std::size_t>(hw) *
                                      static_cast<std::size_t>(N) *
                                      static_cast<std::size_t>(N)); // -X

  // Spot-check one cell from each axis: take the natural face buffer
  // ordering produced by sparsevector::get_index ascending (z slowest,
  // then y, then x) and confirm it carries the periodic-wrap value.
  // For +X recv we expect the x=0 slab of the field, in (z, y) order.
  for (int z = 0; z < N; ++z) {
    for (int y = 0; y < N; ++y) {
      const std::size_t off =
          static_cast<std::size_t>(y) +
          static_cast<std::size_t>(z) * static_cast<std::size_t>(N);
      REQUIRE(face_halos[0][off] == periodic_field_value(0, y, z));
      REQUIRE(face_halos[1][off] == periodic_field_value(N - 1, y, z));
    }
  }
}

TEST_CASE("SparseHaloExchanger: Full3D self-wrap delivers edge data",
          "[MPI][sparse_halo][structured][full3d]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  constexpr int N = 4;
  constexpr int hw = 1;
  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);

  const std::size_t nlocal = static_cast<std::size_t>(N) *
                             static_cast<std::size_t>(N) *
                             static_cast<std::size_t>(N);
  std::vector<double> u(nlocal);
  for (int z = 0; z < N; ++z) {
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        u[static_cast<std::size_t>(x) +
          static_cast<std::size_t>(y) * static_cast<std::size_t>(N) +
          static_cast<std::size_t>(z) * static_cast<std::size_t>(N * N)] =
            periodic_field_value(x, y, z);
      }
    }
  }

  SparseHaloExchanger<double> ex(MPI_COMM_WORLD, rank,
                                 halo::make_structured_halos<double>(
                                     decomp, rank, hw, halo::presets::Full3D()));
  ex.exchange_halos(u.data(), u.size());

  // Find the (1, 1, 0) edge entry: send slab is the (x=N-1, y=N-1) line,
  // recv slab is the (x=0, y=0) line of the field (z varies).
  bool checked_edge = false;
  for (const auto &h : ex.halos()) {
    if (h.direction == Int3{1, 1, 0}) {
      REQUIRE(h.recv_values.size() == static_cast<std::size_t>(N) *
                                          static_cast<std::size_t>(hw) *
                                          static_cast<std::size_t>(hw));
      const double *recv = h.recv_values.data().data();
      // Recv indices were sorted, so the recv buffer holds (z varies
      // slowest) values at (x=0, y=0, z=0..N-1).
      for (int z = 0; z < N; ++z) {
        REQUIRE(recv[static_cast<std::size_t>(z)] == periodic_field_value(0, 0, z));
      }
      checked_edge = true;
      break;
    }
  }
  REQUIRE(checked_edge);
}

TEST_CASE("SparseHaloExchanger: Axes3D X-split on np=2 delivers neighbour data",
          "[MPI][sparse_halo][structured][np2]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 2) {
    return;
  }

  constexpr int Nx = 8;
  constexpr int Ny = 4;
  constexpr int Nz = 1;
  constexpr int hw = 1;
  auto world = world::create(GridSize({Nx, Ny, Nz}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_lower = world::get_lower(local_world);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];

  std::vector<double> u(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz));
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const int gx = local_lower[0] + x;
        const int gy = local_lower[1] + y;
        const int gz = local_lower[2] + z;
        u[static_cast<std::size_t>(x) +
          static_cast<std::size_t>(y) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(z) * static_cast<std::size_t>(nx * ny)] =
            periodic_field_value(gx, gy, gz);
      }
    }
  }

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  SparseHaloExchanger<double> ex(
      MPI_COMM_WORLD, rank, halo::make_structured_halos<double>(decomp, rank, hw));
  ex.exchange_halos(u.data(), u.size());
  halo::copy_to_face_layout(ex, face_halos);

  // +X face: data from the neighbour's leftmost column (gx = (rank+1)%2 * nx).
  // -X face: data from the neighbour's rightmost column.
  const int peer_x_plus_lo = ((rank + 1) % 2) * nx;
  const int peer_x_minus_hi = ((rank + 1) % 2) * nx + nx - 1;
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      const std::size_t off =
          static_cast<std::size_t>(y) +
          static_cast<std::size_t>(z) * static_cast<std::size_t>(ny);
      const int gy = local_lower[1] + y;
      const int gz = local_lower[2] + z;
      REQUIRE(face_halos[0][off] == periodic_field_value(peer_x_plus_lo, gy, gz));
      REQUIRE(face_halos[1][off] == periodic_field_value(peer_x_minus_hi, gy, gz));
    }
  }
}

TEST_CASE("Connectivity::Edges yields faces+edges, no corners",
          "[MPI][sparse_halo][edges_fix]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  constexpr int N = 6;
  constexpr int hw = 1;
  auto world = world::create(GridSize({N, N, N}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, nproc);

  auto patterns = halo::create_halo_patterns<backend::CpuTag>(
      decomp, rank, halo::Connectivity::Edges, hw);
  // 6 face directions + 12 edge directions = 18 entries; no corners.
  REQUIRE(patterns.size() == 18);
  for (const auto &kv : patterns) {
    const auto &d = kv.first;
    const int nz_components =
        (d[0] != 0 ? 1 : 0) + (d[1] != 0 ? 1 : 0) + (d[2] != 0 ? 1 : 0);
    REQUIRE(nz_components >= 1);
    REQUIRE(nz_components <= 2);
  }
}
