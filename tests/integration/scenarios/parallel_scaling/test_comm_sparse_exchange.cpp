// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Host SparseExchange facade (M4). Device SparseExchange is in
// test_comm_sparse_exchange_gpu.cpp.

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

TEST_CASE("SparseExchange HostSpace: custom RemoteHalo scatter",
          "[sparse_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 1, 1});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain_unpadded<double>(decomp, rank, 0);
  REQUIRE(u.size() == 8);
  for (int i = 0; i < 8; ++i) {
    u(i, 0, 0) = static_cast<double>(i + 1);
  }

  halo::RemoteHalo<double> h;
  h.peer_rank = rank;
  h.send_tag = 7;
  h.recv_tag = 7;
  h.send_values =
      core::SparseVector<backend::CPUTag, double>(std::vector<std::size_t>{2, 5});
  h.recv_values =
      core::SparseVector<backend::CPUTag, double>(std::vector<std::size_t>{6, 7});
  h.scatter_after_recv = true;

  comm::SparseExchange<HostSpace, double> ex(u, {std::move(h)}, rank,
                                             MPI_COMM_WORLD);
  REQUIRE(ex.num_halos() == 1);
  ex.exchange();
  REQUIRE(u(6, 0, 0) == 3.0);
  REQUIRE(u(7, 0, 0) == 6.0);
  REQUIRE(u(2, 0, 0) == 3.0);
}

TEST_CASE("SparseExchange HostSpace: Axes3D self-wrap fills face buffers",
          "[sparse_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  constexpr int N = 4;
  constexpr int hw = 1;
  auto domain = domain::create({N, N, N});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain_unpadded<double>(decomp, rank, hw);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x)
        u(x, y, z) = static_cast<double>(x) + 1000.0 * y + 1'000'000.0 * z;

  comm::SparseExchangeOptions opt;
  opt.dirs = halo::presets::Axes3D();
  comm::SparseExchange<HostSpace, double> ex(u, decomp, rank, MPI_COMM_WORLD, opt);
  REQUIRE(ex.num_halos() == 6);
  ex.exchange();

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, hw);
  halo::copy_to_face_layout(ex.halos(), face_halos);

  // +X recv holds the left plane (x=0); -X recv holds the right plane (x=N-1).
  REQUIRE(face_halos[0][0] == 0.0);
  REQUIRE(face_halos[1][0] == static_cast<double>(N - 1));
}

TEST_CASE("SparseExchange HostSpace: start/finish and unbound start throw",
          "[sparse_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  halo::RemoteHalo<double> h;
  h.peer_rank = rank;
  h.send_tag = 1;
  h.recv_tag = 1;
  comm::SparseExchange<HostSpace, double> unbound({std::move(h)}, rank,
                                                  MPI_COMM_WORLD);
  REQUIRE_THROWS_AS(unbound.start(), std::logic_error);

  auto domain = domain::create({8, 1, 1});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain_unpadded<double>(decomp, rank, 0);
  u(2, 0, 0) = 3.0;
  u(5, 0, 0) = 6.0;

  halo::RemoteHalo<double> h2;
  h2.peer_rank = rank;
  h2.send_tag = 3;
  h2.recv_tag = 3;
  h2.send_values =
      core::SparseVector<backend::CPUTag, double>(std::vector<std::size_t>{2, 5});
  h2.recv_values =
      core::SparseVector<backend::CPUTag, double>(std::vector<std::size_t>{6, 7});
  h2.scatter_after_recv = true;
  comm::SparseExchange<HostSpace, double> ex(u, {std::move(h2)}, rank,
                                             MPI_COMM_WORLD);
  ex.start();
  ex.finish();
  REQUIRE(u(6, 0, 0) == 3.0);
  REQUIRE(u(7, 0, 0) == 6.0);
}
