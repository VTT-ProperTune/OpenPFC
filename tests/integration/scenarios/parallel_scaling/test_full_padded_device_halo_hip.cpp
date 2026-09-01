// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_full_padded_device_halo_hip.cpp
 * @brief HIP twin of `test_full_padded_device_halo.cpp` for
 *        `pfc::comm::HaloExchange<HIPSpace>` (`HaloConnectivity::Full`).
 *
 * Covers 1, 2 (`2x1x1`), and 4 (`2x2x1`) ranks. Every padded cell must match
 * `hash(periodic_global_coord)` after `exchange()`. Execute on LUMI.
 */

#include <catch2/catch_all.hpp>
#include <mpi.h>

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstddef>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>

namespace {

using pfc::HIPSpace;

inline double cell_hash(int field, int gx, int gy, int gz) {
  return 1.0 + 0.5 * static_cast<double>(field) + static_cast<double>(gx) +
         1024.0 * static_cast<double>(gy) + 1048576.0 * static_cast<double>(gz);
}

inline int wrap(int g, int n) { return ((g % n) + n) % n; }

pfc::data::Field<double, HIPSpace>
make_padded_field(const pfc::decomposition::Decomposition &decomp, int rank,
                  int hw) {
  return pfc::data::Field<double, HIPSpace>(
      pfc::decomposition::domain(decomp),
      pfc::decomposition::local_box(decomp, rank), hw);
}

void fill_owned_hash(pfc::data::Field<double, HIPSpace> &u, int field) {
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
            data[u.idx(i, j, k)] = cell_hash(field, g[0], g[1], g[2]);
          } else {
            data[u.idx(i, j, k)] = 0.0;
          }
        }
      }
    }
  });
}

bool full_periodic_hash_matches(pfc::data::Field<double, HIPSpace> &u, int field) {
  bool ok = true;
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    const auto gsz = u.global_size();
    const int hw = u.storage_halo();
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const auto g = u.global(i, j, k);
          const double expect = cell_hash(field, wrap(g[0], gsz[0]),
                                          wrap(g[1], gsz[1]), wrap(g[2], gsz[2]));
          ok &= data[u.idx(i, j, k)] == expect;
        }
      }
    }
  });
  return ok;
}

bool hip_runtime_available() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

void run_full_halo_check(const pfc::decomposition::Decomposition &decomp, int rank,
                         int hw, int n_fields) {
  auto u = make_padded_field(decomp, rank, hw);
  fill_owned_hash(u, 0);
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  if (n_fields == 1) {
    pfc::comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                   opt);
    REQUIRE(halo.num_fields() == 1);
    halo.exchange();
    REQUIRE(full_periodic_hash_matches(u, 0));
    return;
  }
  auto v = make_padded_field(decomp, rank, hw);
  fill_owned_hash(v, 1);
  pfc::comm::HaloExchange<HIPSpace, double> halo({&u, &v}, decomp, rank,
                                                 MPI_COMM_WORLD, opt);
  REQUIRE(halo.connectivity() == pfc::comm::HaloConnectivity::Full);
  REQUIRE(halo.num_fields() == 2);
  halo.exchange();
  REQUIRE(full_periodic_hash_matches(u, 0));
  REQUIRE(full_periodic_hash_matches(v, 1));
}

} // namespace

TEST_CASE("HaloExchange HIPSpace Full: 1-rank periodic fill (all 26 halos)",
          "[gpu][hip][padded_halo][full_halo]") {
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
  run_full_halo_check(decomp, rank, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("HaloExchange HIPSpace Full: 2-rank 2x1x1 fill (X real, Y/Z self)",
          "[MPI][gpu][hip][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }
  if (!hip_runtime_available()) {
    SKIP("No HIP runtime / device available on this host");
  }

  auto domain = pfc::domain::create({8, 6, 4});
  auto decomp = pfc::decomposition::create(domain, {2, 1, 1});
  run_full_halo_check(decomp, rank, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("HaloExchange HIPSpace Full: 4-rank 2x2x1 fill (X+Y real, Z self)",
          "[MPI][gpu][hip][padded_halo][full_halo][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }
  if (!hip_runtime_available()) {
    SKIP("No HIP runtime / device available on this host");
  }

  auto domain = pfc::domain::create({8, 6, 4});
  auto decomp = pfc::decomposition::create(domain, {2, 2, 1});
  run_full_halo_check(decomp, rank, /*hw=*/1, /*n_fields=*/2);
}

TEST_CASE("HaloExchange HIPSpace Full: hw=2 1-rank widened halo",
          "[gpu][hip][padded_halo][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }
  if (!hip_runtime_available()) {
    SKIP("No HIP runtime / device available on this host");
  }

  auto domain = pfc::domain::create({6, 6, 4});
  auto decomp = pfc::decomposition::create(domain, 1);
  run_full_halo_check(decomp, rank, /*hw=*/2, /*n_fields=*/1);
}

TEST_CASE("HaloExchange HIPSpace Full+Axes3D fills only the 6 axis faces",
          "[gpu][hip][padded_halo][full_halo][halo_directions]") {
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
  constexpr int hw = 1;
  auto u = make_padded_field(decomp, rank, hw);
  const double sentinel = -1.0;
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const bool owned =
              i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
          if (owned) {
            const auto g = u.global(i, j, k);
            data[u.idx(i, j, k)] = cell_hash(0, g[0], g[1], g[2]);
          } else {
            data[u.idx(i, j, k)] = sentinel;
          }
        }
      }
    }
  });

  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  opt.directions = pfc::halo::presets::Axes3D();
  pfc::comm::HaloExchange<HIPSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                 opt);
  halo.exchange();

  bool values_match = true;
  u.with_host_view([&](double *data, std::size_t) {
    const auto n = u.size3();
    const auto gsz = u.global_size();
    for (int k = -hw; k < n[2] + hw; ++k) {
      for (int j = -hw; j < n[1] + hw; ++j) {
        for (int i = -hw; i < n[0] + hw; ++i) {
          const bool in_x = i >= 0 && i < n[0];
          const bool in_y = j >= 0 && j < n[1];
          const bool in_z = k >= 0 && k < n[2];
          const int axis_inside = static_cast<int>(in_x) + static_cast<int>(in_y) +
                                  static_cast<int>(in_z);
          const auto g = u.global(i, j, k);
          const double expect = cell_hash(0, wrap(g[0], gsz[0]), wrap(g[1], gsz[1]),
                                          wrap(g[2], gsz[2]));
          if (axis_inside >= 2) {
            values_match &= data[u.idx(i, j, k)] == expect;
          } else {
            values_match &= data[u.idx(i, j, k)] == sentinel;
          }
        }
      }
    }
  });
  REQUIRE(values_match);
}

#endif // OpenPFC_ENABLE_HIP
