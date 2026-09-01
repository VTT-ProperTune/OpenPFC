// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// 4-rank HaloExchange mode comparison (M4 required test). Host blocking
// (one Waitall across fields), split-phase, and multi-field batching must
// agree bitwise. Persistent
// multi-rank remains red on LUMI (same as test_fd_heat_mpi); that path is
// still covered on one rank in test_comm_halo_exchange.cpp. Full 26-direction
// fill is checked on a corner-dependent periodic hash. HIP 4-rank cases live
// in test_comm_halo_exchange_gpu.cpp. CUDA: verify on tohtori.

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <algorithm>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {

double cell_hash(int field, int gx, int gy, int gz) {
  return 1.0 + 0.5 * static_cast<double>(field) + static_cast<double>(gx) +
         1024.0 * static_cast<double>(gy) + 1048576.0 * static_cast<double>(gz);
}

int wrap(int g, int n) { return ((g % n) + n) % n; }

void fill_owned_hash(data::Field<double, HostSpace> &u, int field) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = u.global(i, j, k);
        u(i, j, k) = cell_hash(field, g[0], g[1], g[2]);
      }
    }
  }
}

void copy_field(const data::Field<double, HostSpace> &src,
                data::Field<double, HostSpace> &dst) {
  REQUIRE(src.size() == dst.size());
  std::copy(src.data(), src.data() + src.size(), dst.data());
}

bool fields_equal(const data::Field<double, HostSpace> &a,
                  const data::Field<double, HostSpace> &b) {
  return a.size() == b.size() && std::equal(a.data(), a.data() + a.size(), b.data());
}

void require_full_periodic_hash(const data::Field<double, HostSpace> &u, int field) {
  const auto n = u.size3();
  const auto gsz = u.global_size();
  const int hw = u.storage_halo();
  for (int k = -hw; k < n[2] + hw; ++k) {
    for (int j = -hw; j < n[1] + hw; ++j) {
      for (int i = -hw; i < n[0] + hw; ++i) {
        const auto g = u.global(i, j, k);
        const double expect = cell_hash(field, wrap(g[0], gsz[0]),
                                        wrap(g[1], gsz[1]), wrap(g[2], gsz[2]));
        REQUIRE(u(i, j, k) == expect);
      }
    }
  }
}

void run_full_periodic_hash(const decomposition::Decomposition &decomp, int rank,
                            int hw) {
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  fill_owned_hash(u, /*field=*/0);
  comm::HaloExchangeOptions opt;
  opt.connectivity = comm::HaloConnectivity::Full;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  halo.exchange();
  require_full_periodic_hash(u, /*field=*/0);
}

} // namespace

TEST_CASE("HaloExchange 4-rank: blocking equals start/finish",
          "[MPI][halo_exchange][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto block = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  auto split = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned_hash(block, /*field=*/0);
  copy_field(block, split);

  comm::HaloExchange<HostSpace, double> halo_block(block, decomp, rank,
                                                   MPI_COMM_WORLD);
  comm::HaloExchange<HostSpace, double> halo_split(split, decomp, rank,
                                                   MPI_COMM_WORLD);
  halo_block.exchange();
  halo_split.start();
  halo_split.finish();
  REQUIRE(fields_equal(block, split));
}

TEST_CASE("HaloExchange 4-rank: multi-field batch equals two singles",
          "[MPI][halo_exchange][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u_batch = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  auto v_batch = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  auto u_one = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  auto v_one = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned_hash(u_batch, 0);
  fill_owned_hash(v_batch, 1);
  copy_field(u_batch, u_one);
  copy_field(v_batch, v_one);

  comm::HaloExchange<HostSpace, double> batched({&u_batch, &v_batch}, decomp, rank,
                                                MPI_COMM_WORLD);
  comm::HaloExchange<HostSpace, double> only_u(u_one, decomp, rank, MPI_COMM_WORLD);
  comm::HaloExchangeOptions v_opt;
  v_opt.exchange_base = 1;
  comm::HaloExchange<HostSpace, double> only_v(v_one, decomp, rank, MPI_COMM_WORLD,
                                               v_opt);
  batched.exchange();
  only_u.exchange();
  only_v.exchange();
  REQUIRE(fields_equal(u_batch, u_one));
  REQUIRE(fields_equal(v_batch, v_one));
}

TEST_CASE("HaloExchange 4-rank Full: corners and edges match periodic hash",
          "[MPI][halo_exchange][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }

  auto domain = domain::create({16, 12, 8});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned_hash(u, /*field=*/0);

  comm::HaloExchangeOptions opt;
  opt.connectivity = comm::HaloConnectivity::Full;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  REQUIRE(halo.connectivity() == comm::HaloConnectivity::Full);
  halo.exchange();
  require_full_periodic_hash(u, /*field=*/0);
}

TEST_CASE("HaloExchange Full: 1-rank periodic fill (all 26 halos)",
          "[MPI][halo_exchange][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  run_full_periodic_hash(decomp, rank, /*hw=*/1);
}

TEST_CASE("HaloExchange Full: 2-rank 2x1x1 fill (X real, Y/Z self)",
          "[MPI][halo_exchange][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, {2, 1, 1});
  run_full_periodic_hash(decomp, rank, /*hw=*/1);
}

TEST_CASE("HaloExchange Full: hw=2 1-rank widened halo correctness",
          "[MPI][halo_exchange][full_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({6, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  run_full_periodic_hash(decomp, rank, /*hw=*/2);
}
