// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_full_padded_halo_exchange.cpp
 * @brief Exhaustive correctness test for
 *        `pfc::communication::FullPaddedHaloExchanger`.
 *
 * Host-only twin of `test_full_padded_device_halo.cpp`. Covers the full
 * **26-direction** halo (faces + edges + corners) on `1`, `2 (2x1x1)`, and
 * `4 (2x2x1)` ranks. Asserts bit-identical agreement between the
 * post-exchange padded brick and a periodic `hash(global_coord)` reference.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/full_padded_halo_exchange.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace {

using pfc::types::Int3;

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

struct PaddedFieldRef {
  std::vector<double> expected;
  std::vector<double> initial;
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

void run_full_halo_check(const pfc::decomposition::Decomposition &decomp, int rank,
                         const Int3 &global_size, int hw) {
  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  const auto local_size = pfc::world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;
  const std::size_t total = static_cast<std::size_t>(nxp) *
                            static_cast<std::size_t>(nyp) *
                            static_cast<std::size_t>(nzp);

  auto ref = build_reference(/*field_idx=*/0, rank, decomp, global_size, hw);

  pfc::field::PaddedBrick<double> u(decomp, rank, hw);
  REQUIRE(u.size() == total);
  std::copy(ref.initial.begin(), ref.initial.end(), u.data());

  pfc::communication::FullPaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD);
  REQUIRE(halo.is_bound());
  halo.exchange();

  std::size_t total_mismatches = 0;
  for (std::size_t l = 0; l < total; ++l) {
    if (u.data()[l] != ref.expected[l]) {
      ++total_mismatches;
    }
  }
  REQUIRE(total_mismatches == 0);

  // Explicit corner cells in brick local coordinates (owned origin at 0).
  const auto corner_val = [&](int i, int j, int k) {
    const int pi = i + hw;
    const int pj = j + hw;
    const int pk = k + hw;
    return u.data()[lin(pi, pj, pk, nxp, nyp)];
  };
  const auto expect_at = [&](int i, int j, int k) {
    const int pi = i + hw;
    const int pj = j + hw;
    const int pk = k + hw;
    return ref.expected[lin(pi, pj, pk, nxp, nyp)];
  };
  REQUIRE(corner_val(-hw, -hw, -hw) == expect_at(-hw, -hw, -hw));
  REQUIRE(corner_val(nx, ny, nz) == expect_at(nx, ny, nz));
  // One edge cell (not a face centre): (-hw, -hw, 0).
  REQUIRE(corner_val(-hw, -hw, 0) == expect_at(-hw, -hw, 0));
}

} // namespace

TEST_CASE("FullPaddedHaloExchanger: 1-rank periodic full-fill (all 26 halos)",
          "[MPI][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1);
}

TEST_CASE("FullPaddedHaloExchanger: 2-rank 2x1x1 full-fill (X real, Y/Z self)",
          "[MPI][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, {2, 1, 1});

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1);
}

TEST_CASE("FullPaddedHaloExchanger: 4-rank 2x2x1 full-fill (X+Y real, Z self)",
          "[MPI][padded_halo][full_halo][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }

  const Int3 global_size{8, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, {2, 2, 1});

  run_full_halo_check(decomp, rank, global_size, /*hw=*/1);
}

TEST_CASE("FullPaddedHaloExchanger: hw=2 1-rank widened halo correctness",
          "[MPI][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  const Int3 global_size{6, 6, 4};
  auto world = pfc::world::create(
      pfc::GridSize({global_size[0], global_size[1], global_size[2]}));
  auto decomp = pfc::decomposition::create(world, 1);

  run_full_halo_check(decomp, rank, global_size, /*hw=*/2);
}
