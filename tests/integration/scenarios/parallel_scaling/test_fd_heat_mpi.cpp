// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_persistent.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

using namespace pfc;

TEST_CASE("Laplacian of constant field is zero after halo exchange", "[MPI][fd]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return;
  }

  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u(nlocal, 1.0);
  std::vector<double> lap(nlocal, 0.0);

  constexpr int halo_width = 1;
  HaloExchanger<double> exchanger(decomp, rank, halo_width, MPI_COMM_WORLD);
  exchanger.exchange_halos(u.data(), u.size());

  const double inv = 1.0;
  field::fd::laplacian_7point_interior(u.data(), lap.data(), nx, ny, nz, inv, inv, inv,
                                       halo_width);

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(lap[i] == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("PersistentHaloExchanger matches HaloExchanger face sync", "[MPI][fd][persistent]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 4) {
    return;
  }

  auto world = world::uniform(16, 1.0);
  auto decomp = decomposition::create(world, {2, 2, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> a(nlocal);
  std::vector<double> b(nlocal);
  const double fill = static_cast<double>(rank);
  for (size_t i = 0; i < nlocal; ++i) {
    a[i] = fill;
    b[i] = fill;
  }

  constexpr int halo_width = 1;
  HaloExchanger<double> hex(decomp, rank, halo_width, MPI_COMM_WORLD);
  PersistentHaloExchanger<double> pex(decomp, rank, halo_width, MPI_COMM_WORLD, b.data());

  hex.exchange_halos(a.data(), a.size());
  pex.exchange_halos();

  for (size_t i = 0; i < nlocal; ++i) {
    REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-12));
  }
}

TEST_CASE("Separated face halos match in-place Laplacian on interior", "[MPI][fd][separated]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return;
  }

  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  const int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  const size_t nlocal =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

  std::vector<double> u_inplace(nlocal);
  std::vector<double> u_core(nlocal);
  for (size_t i = 0; i < nlocal; ++i) {
    const double v = static_cast<double>(rank) + 0.01 * static_cast<double>(i);
    u_inplace[i] = v;
    u_core[i] = v;
  }

  constexpr int halo_width = 1;
  std::vector<double> lap_in(nlocal, 0.0);
  std::vector<double> lap_sep(nlocal, 0.0);

  HaloExchanger<double> hex(decomp, rank, halo_width, MPI_COMM_WORLD);
  hex.exchange_halos(u_inplace.data(), u_inplace.size());
  const double inv = 1.0;
  field::fd::laplacian_7point_interior(u_inplace.data(), lap_in.data(), nx, ny, nz, inv,
                                       inv, inv, halo_width);

  auto face_halos = halo::allocate_face_halos<double>(decomp, rank, halo_width);
  SeparatedFaceHaloExchanger<double> sex(decomp, rank, halo_width, MPI_COMM_WORLD);
  sex.exchange_halos(u_core.data(), u_core.size(), face_halos);

  std::array<const double *, 6> face_ptrs;
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<size_t>(i)] = face_halos[static_cast<size_t>(i)].data();
  }
  field::fd::laplacian_7point_interior_separated(u_core.data(), face_ptrs, lap_sep.data(),
                                                 nx, ny, nz, inv, inv, inv, halo_width);

  const int imin = halo_width;
  const int imax = nx - halo_width;
  const int jmin = halo_width;
  const int jmax = ny - halo_width;
  const int kmin = halo_width;
  const int kmax = nz - halo_width;
  const int sxy = nx * ny;
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const size_t c = static_cast<size_t>(ix) +
                         static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                         static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        REQUIRE(lap_sep[c] == Catch::Approx(lap_in[c]).margin(1e-11));
      }
    }
  }
}
