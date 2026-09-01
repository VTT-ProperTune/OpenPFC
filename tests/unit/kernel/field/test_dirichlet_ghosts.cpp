// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <numbers>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/dirichlet_ghosts.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>

using Catch::Matchers::WithinAbs;
using pfc::HostSpace;
using pfc::comm::HaloExchange;
using pfc::communication::apply_dirichlet_ghosts;

TEST_CASE("Dirichlet ghosts odd-reflect about node walls on a non-periodic axis",
          "[kernel][field][bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  constexpr int Nx = 8;
  auto domain = pfc::domain::create(
      pfc::GridSize({Nx, 4, 4}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, true, true});
  auto decomp = pfc::decomposition::create(domain, 1);
  constexpr int hw = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
  const auto sz0 = u.local_size();
  for (int k = 0; k < sz0[2]; ++k) {
    for (int j = 0; j < sz0[1]; ++j) {
      for (int i = 0; i < sz0[0]; ++i) {
        u(i, j, k) = static_cast<double>(i + 1);
      }
    }
  }
  apply_dirichlet_ghosts(u, /*axis=*/0, 0.0, 0.0);
  REQUIRE_THAT(u(-1, 0, 0), WithinAbs(-u(1, 0, 0), 1e-12));
  REQUIRE_THAT(u(Nx, 0, 0), WithinAbs(-u(Nx - 2, 0, 0), 1e-12));
}

TEST_CASE("StagePreparationService Dirichlet sine is a discrete Laplacian eigenmode",
          "[kernel][field][bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  constexpr int Nx = 17;
  constexpr int Ny = 5;
  constexpr int Nz = 5;
  constexpr int hw = 1;
  // Periodic Domain so HaloExchange can construct; Dirichlet overwrites the
  // x-wrap (same mixed-BC pattern as wave2d).
  auto domain = pfc::domain::create(pfc::GridSize({Nx, Ny, Nz}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
  auto lap = pfc::data::field_from_subdomain<double>(decomp, 0, hw);

  const double k = std::numbers::pi / static_cast<double>(Nx - 1);
  const auto sz = u.local_size();
  for (int kk = 0; kk < sz[2]; ++kk) {
    for (int jj = 0; jj < sz[1]; ++jj) {
      for (int ii = 0; ii < sz[0]; ++ii) {
        u(ii, jj, kk) = std::sin(k * static_cast<double>(ii));
      }
    }
  }

  HaloExchange<HostSpace, double> halo(u, decomp, 0, MPI_COMM_WORLD);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", halo);
  prep.set_boundary_hook([&](std::string_view) { apply_dirichlet_ghosts(u, 0); });

  pfc::integrator::StageContext ctx{.needs_halo_exchange = true};
  auto req = pfc::integrator::requirements_from(ctx, /*needs_boundary=*/true);
  req.ordering = pfc::communication::BoundaryHaloOrder::HaloThenBoundary;
  const std::string_view fields[] = {"u"};
  prep.prepare(req, fields);

  REQUIRE_THAT(u(-1, 0, 0), WithinAbs(-u(1, 0, 0), 1e-12));
  REQUIRE_THAT(u(Nx, 0, 0), WithinAbs(-u(Nx - 2, 0, 0), 1e-12));

  const int npx = u.padded_extent(0);
  const int npy = u.padded_extent(1);
  const int npz = u.padded_extent(2);
  pfc::field::fd::laplacian_interior<2>(u.data(), lap.data(), npx, npy, npz, 1.0,
                                        1.0, 1.0, hw);
  const double lambda = 2.0 * std::cos(k) - 2.0;
  for (int kk = 0; kk < Nz; ++kk) {
    for (int jj = 0; jj < Ny; ++jj) {
      for (int ii = 0; ii < Nx; ++ii) {
        const double expect = lambda * u(ii, jj, kk);
        REQUIRE_THAT(lap(ii, jj, kk), WithinAbs(expect, 1e-10));
      }
    }
  }
}

TEST_CASE("StagePreparationService spectral penalty hook writes owned cells",
          "[kernel][field][bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  auto domain = pfc::domain::create({8, 4, 4});
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, 1);
  const auto szp = u.local_size();
  for (int k = 0; k < szp[2]; ++k) {
    for (int j = 0; j < szp[1]; ++j) {
      for (int i = 0; i < szp[0]; ++i) {
        u(i, j, k) = 1.0;
      }
    }
  }

  HaloExchange<HostSpace, double> halo(u, decomp, 0, MPI_COMM_WORLD);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("psi", halo);
  prep.set_boundary_hook([&](std::string_view name) {
    REQUIRE(name == "psi");
    for (int k = 0; k < szp[2]; ++k) {
      for (int j = 0; j < szp[1]; ++j) {
        u(0, j, k) = 0.0;
      }
    }
  });

  pfc::communication::StagePreparationRequirements req{
      .needs_halo_exchange = false,
      .needs_boundary_update = true,
  };
  const std::string_view fields[] = {"psi"};
  prep.prepare(req, fields);
  REQUIRE(u(0, 0, 0) == 0.0);
  REQUIRE(u(1, 0, 0) == 1.0);
}
