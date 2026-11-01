// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/dirichlet_boundary.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pfc::HostSpace;
using pfc::communication::CartesianFace;
using pfc::communication::dirichlet_stage_requirements;
using pfc::communication::DirichletBoundary;
using pfc::communication::DirichletCentering;

namespace {

pfc::data::Field<double, HostSpace> make_field(pfc::Int3 size, pfc::Real3 upper,
                                               pfc::Bool3 periodic, pfc::Int3 grid,
                                               int rank, int halo) {
  const auto domain =
      pfc::domain::from_bounds(size, {0.0, 0.0, 0.0}, upper, periodic);
  const auto decomp = pfc::decomposition::create(domain, grid);
  return pfc::data::field_from_subdomain<double>(decomp, rank, halo);
}

pfc::decomposition::Decomposition make_decomp(const pfc::Domain &domain,
                                              pfc::Int3 grid) {
  return pfc::decomposition::create(domain, grid);
}

/// Physical Dirichlet faces have no MPI neighbour. Direction agreement
/// rejects that peer, so the exchange keeps only ranks that exist.
pfc::comm::HaloExchangeOptions
face_exchange_options(const pfc::decomposition::Decomposition &decomp) {
  pfc::comm::HaloExchangeOptions opt;
  opt.selector = [&decomp](int r) {
    std::vector<pfc::Int3> kept;
    for (const auto &d : pfc::halo::presets::Axes3D().dirs) {
      if (pfc::decomposition::get_neighbor_rank(decomp, r, d) >= 0) {
        kept.push_back(d);
      }
    }
    return pfc::halo::HaloDirectionSet(std::move(kept));
  };
  return opt;
}

template <int Order> double d2_eigenvalue(double theta, double h) {
  using Stencil = pfc::field::fd::EvenCentralD2<Order>;
  double acc = static_cast<double>(Stencil::coeffs[0]);
  for (int k = 1; k <= Stencil::half_width; ++k) {
    acc += 2.0 * static_cast<double>(Stencil::coeffs[k]) * std::cos(k * theta);
  }
  return acc / (static_cast<double>(Stencil::denom) * h * h);
}

} // namespace

TEST_CASE("Dirichlet stage order is boundary then halo", "[dirichlet_bc][unit]") {
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const auto req = dirichlet_stage_requirements(ctx);
  REQUIRE(req.needs_halo_exchange);
  REQUIRE(req.needs_boundary_update);
  REQUIRE(req.ordering == pfc::communication::BoundaryHaloOrder::BoundaryThenHalo);
}

TEST_CASE("Node and cell-face Dirichlet ghosts use different mirrors",
          "[dirichlet_bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  constexpr int n = 8;
  auto fill = [](pfc::data::Field<double, HostSpace> &u) {
    u.for_each_owned([&](int i, int j, int k) {
      u(i, j, k) = 0.5 * static_cast<double>(i + 1);
      (void)j;
      (void)k;
    });
  };
  auto node =
      make_field({n, 4, 4}, {7.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  auto cell =
      make_field({n, 4, 4}, {7.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  fill(node);
  fill(cell);
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Xm, 1.25);
  walls.set_constant(CartesianFace::Xp, -0.5);
  walls.enforce_owned_nodes(node, 0.0);
  walls.apply_ghosts(node, 0.0);
  REQUIRE_THAT(node(0, 1, 1), WithinAbs(1.25, 1e-12));
  REQUIRE_THAT(node(-1, 1, 1), WithinAbs(2.0 * 1.25 - node(1, 1, 1), 1e-12));
  REQUIRE_THAT(node(-2, 1, 1), WithinAbs(2.0 * 1.25 - node(2, 1, 1), 1e-12));
  REQUIRE_THAT(node(n, 1, 1), WithinAbs(2.0 * -0.5 - node(n - 2, 1, 1), 1e-12));

  walls.centering = DirichletCentering::CellFace;
  walls.apply_ghosts(cell, 0.0);
  REQUIRE_THAT(cell(0, 1, 1), WithinAbs(0.5, 1e-12));
  REQUIRE_THAT(cell(-1, 1, 1), WithinAbs(2.0 * 1.25 - cell(0, 1, 1), 1e-12));
  REQUIRE_THAT(cell(-2, 1, 1), WithinAbs(2.0 * 1.25 - cell(1, 1, 1), 1e-12));
}

TEST_CASE("Coordinate and time callbacks set the node wall",
          "[dirichlet_bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  auto u =
      make_field({6, 5, 4}, {5.0, 5.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  u.for_each_owned(
      [&](int i, int j, int k) { u(i, j, k) = 10.0 + static_cast<double>(i); });
  DirichletBoundary<double> walls;
  walls.set_coordinate_dependent(
      CartesianFace::Xm, [](pfc::Real3 x, double) { return 1.5 + 0.25 * x[1]; });
  walls.set_time_dependent(CartesianFace::Xp,
                           [](double time) { return 2.0 * time; });
  const double time = 1.5;
  walls.enforce_owned_nodes(u, time);
  walls.apply_ghosts(u, time);
  for (int j = 0; j < 5; ++j) {
    const double g = 1.5 + 0.25 * static_cast<double>(j);
    REQUIRE_THAT(u(0, j, 0), WithinAbs(g, 1e-12));
    REQUIRE_THAT(u(-1, j, 0), WithinAbs(2.0 * g - u(1, j, 0), 1e-12));
    REQUIRE_THAT(u(-2, j, 0), WithinAbs(2.0 * g - u(2, j, 0), 1e-12));
  }
  REQUIRE_THAT(u(5, 0, 0), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(u(6, 0, 0), WithinAbs(2.0 * 3.0 - u(4, 0, 0), 1e-12));
}

TEST_CASE("A periodic Dirichlet face is rejected", "[dirichlet_bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  auto u =
      make_field({4, 4, 4}, {4.0, 4.0, 4.0}, {true, true, true}, {1, 1, 1}, 0, 1);
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Ym, 0.0);
  REQUIRE_THROWS_WITH(walls.apply_ghosts(u, 0.0), ContainsSubstring("periodic"));
}

TEST_CASE("Order-4 Dirichlet sine is a discrete Laplacian eigenmode",
          "[dirichlet_bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  constexpr int n = 17;
  constexpr int halo = 2;
  auto u = make_field({n, 5, 5}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1}, 0,
                      halo);
  auto lap = make_field({n, 5, 5}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1},
                        0, halo);
  const double dx = u.spacing()[0];
  const double theta = std::numbers::pi / static_cast<double>(n - 1);
  u.for_each_owned([&](int i, int j, int k) {
    u(i, j, k) = std::sin(theta * static_cast<double>(i));
    (void)j;
    (void)k;
  });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Xm, 0.0);
  walls.set_constant(CartesianFace::Xp, 0.0);
  walls.enforce_owned_nodes(u, 0.0);
  walls.apply_ghosts(u, 0.0);
  REQUIRE_THAT(u(-1, 0, 0), WithinAbs(-u(1, 0, 0), 1e-12));
  REQUIRE_THAT(u(-2, 0, 0), WithinAbs(-u(2, 0, 0), 1e-12));
  pfc::field::fd::laplacian_interior<4>(
      u.data(), lap.data(), u.padded_extent(0), u.padded_extent(1),
      u.padded_extent(2), 1.0 / (dx * dx), 1.0 / (u.spacing()[1] * u.spacing()[1]),
      1.0 / (u.spacing()[2] * u.spacing()[2]), halo);
  const double lambda = d2_eigenvalue<4>(theta, dx);
  for (int i = 0; i < n; ++i) {
    REQUIRE_THAT(lap(i, 2, 2), WithinAbs(lambda * u(i, 2, 2), 1e-10));
  }
}

TEST_CASE("Nonzero Dirichlet walls hold a steady linear profile",
          "[dirichlet_bc][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  constexpr int n = 9;
  constexpr int halo = 1;
  auto u = make_field({n, 4, 4}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1}, 0,
                      halo);
  auto lap = make_field({n, 4, 4}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1},
                        0, halo);
  constexpr double g0 = 1.5;
  constexpr double g1 = -0.5;
  u.for_each_owned([&](int i, int j, int k) {
    u(i, j, k) =
        g0 + (g1 - g0) * static_cast<double>(i) / static_cast<double>(n - 1);
    (void)j;
    (void)k;
  });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Xm, g0);
  walls.set_constant(CartesianFace::Xp, g1);
  walls.enforce_owned_nodes(u, 0.0);
  walls.apply_ghosts(u, 0.0);
  const double dx = u.spacing()[0];
  pfc::field::fd::laplacian_interior<2>(u.data(), lap.data(), u.padded_extent(0),
                                        u.padded_extent(1), u.padded_extent(2),
                                        1.0 / (dx * dx), 1.0, 1.0, halo);
  for (int i = 0; i < n; ++i) {
    REQUIRE_THAT(lap(i, 1, 1), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(u(i, 1, 1), WithinAbs(g0 + (g1 - g0) * static_cast<double>(i) /
                                                static_cast<double>(n - 1),
                                       1e-12));
  }
  u(0, 1, 1) = 9.0;
  walls.enforce_owned_nodes(u, 0.0);
  REQUIRE_THAT(u(0, 1, 1), WithinAbs(g0, 1e-12));
}

TEST_CASE("One-rank heat mode decays at the discrete Dirichlet rate",
          "[dirichlet_bc][unit]") {
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1) {
    return;
  }
  constexpr int n = 17;
  constexpr int halo = 1;
  constexpr int steps = 20;
  const auto domain = pfc::domain::from_bounds(
      {n, 4, 4}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, pfc::Bool3{false, true, true});
  const auto decomp = make_decomp(domain, {1, 1, 1});
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, halo);
  auto lap = pfc::data::field_from_subdomain<double>(decomp, 0, halo);
  const double dx = u.spacing()[0];
  const double theta = std::numbers::pi / static_cast<double>(n - 1);
  const double lambda = d2_eigenvalue<2>(theta, dx);
  // Stays inside the forward-Euler limit of the order-2 stencil on this dx.
  const double dt = 0.001;
  u.for_each_owned([&](int i, int j, int k) {
    u(i, j, k) = std::sin(theta * static_cast<double>(i));
    (void)j;
    (void)k;
  });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Xm, 0.0);
  walls.set_constant(CartesianFace::Xp, 0.0);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, 0, MPI_COMM_WORLD,
                                                      halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  walls.bind_dirichlet(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const auto req = dirichlet_stage_requirements(ctx);
  const std::string_view names[] = {"u"};
  for (int step = 0; step < steps; ++step) {
    prep.prepare(req, names);
    pfc::field::fd::laplacian_interior<2>(
        u.data(), lap.data(), u.padded_extent(0), u.padded_extent(1),
        u.padded_extent(2), 1.0 / (dx * dx), 1.0 / (u.spacing()[1] * u.spacing()[1]),
        1.0 / (u.spacing()[2] * u.spacing()[2]), halo);
    u.for_each_owned([&](int i, int j, int k) { u(i, j, k) += dt * lap(i, j, k); });
    walls.enforce_owned_nodes(u, time);
  }
  const double factor = std::pow(1.0 + dt * lambda, steps);
  for (int i = 0; i < n; ++i) {
    const double expect = factor * std::sin(theta * static_cast<double>(i));
    REQUIRE_THAT(u(i, 1, 1), WithinAbs(expect, 1e-10));
  }
}

TEST_CASE("Two ranks publish a Dirichlet wall before filling ghosts",
          "[dirichlet_bc][MPI]") {
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) {
    return;
  }
  constexpr int nx = 8;
  constexpr int nz = 6;
  const auto domain = pfc::domain::from_bounds(
      {nx, 4, nz}, {0.0, 0.0, 0.0}, {8.0, 4.0, 5.0}, pfc::Bool3{true, true, false});
  const auto decomp = make_decomp(domain, {2, 1, 1});
  constexpr int halo = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  u.for_each_owned([&](int i, int j, int k) { u(i, j, k) = 7.0; });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Zm, 1.25);
  walls.set_constant(CartesianFace::Zp, -0.5);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, rank,
                                                      MPI_COMM_WORLD, halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  walls.bind_dirichlet(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const std::string_view names[] = {"u"};
  prep.prepare(dirichlet_stage_requirements(ctx), names);

  int ok = 1;
  const int z0 = u.lower_global()[2];
  for (int i = 0; i < u.local_size()[0]; ++i) {
    if (z0 == 0 && std::abs(u(i, 0, 0) - 1.25) > 1e-12) {
      ok = 0;
    }
    if (z0 + u.local_size()[2] == nz &&
        std::abs(u(i, 0, u.local_size()[2] - 1) + 0.5) > 1e-12) {
      ok = 0;
    }
  }
  if (std::abs(u(-1, 0, 0) - 1.25) > 1e-12) {
    ok = 0;
  }
  if (z0 == 0 && std::abs(u(0, 0, -1) - (2.0 * 1.25 - u(0, 0, 1))) > 1e-12) {
    ok = 0;
  }
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(all_ok == 1);
}

TEST_CASE("Two-rank periodic-x Dirichlet-z heat matches the discrete mode",
          "[dirichlet_bc][MPI]") {
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) {
    return;
  }
  constexpr int nz = 9;
  constexpr int halo = 1;
  constexpr int steps = 6;
  const auto domain = pfc::domain::from_bounds(
      {8, 4, nz}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, pfc::Bool3{true, true, false});
  const auto decomp = make_decomp(domain, {2, 1, 1});
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  auto lap = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  const double dz = u.spacing()[2];
  const double theta = std::numbers::pi / static_cast<double>(nz - 1);
  const double lambda = d2_eigenvalue<2>(theta, dz);
  const double dt = 0.02;
  u.for_each_owned([&](int i, int j, int k) {
    const int gz = u.global(i, j, k)[2];
    u(i, j, k) = std::sin(theta * static_cast<double>(gz));
  });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Zm, 0.0);
  walls.set_constant(CartesianFace::Zp, 0.0);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, rank,
                                                      MPI_COMM_WORLD, halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  walls.bind_dirichlet(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const auto req = dirichlet_stage_requirements(ctx);
  const std::string_view names[] = {"u"};
  const double inv_z = 1.0 / (dz * dz);
  for (int step = 0; step < steps; ++step) {
    prep.prepare(req, names);
    pfc::field::fd::laplacian_interior<2>(u.data(), lap.data(), u.padded_extent(0),
                                          u.padded_extent(1), u.padded_extent(2),
                                          1.0, 1.0, inv_z, halo);
    u.for_each_owned([&](int i, int j, int k) { u(i, j, k) += dt * lap(i, j, k); });
    walls.enforce_owned_nodes(u, time);
  }
  const double factor = std::pow(1.0 + dt * lambda, steps);
  int ok = 1;
  u.for_each_owned([&](int i, int j, int k) {
    const int gz = u.global(i, j, k)[2];
    const double expect = factor * std::sin(theta * static_cast<double>(gz));
    if (std::abs(u(i, j, k) - expect) > 1e-9) {
      ok = 0;
    }
  });
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(all_ok == 1);
}

TEST_CASE("An interior rank does not treat a partition face as Dirichlet",
          "[dirichlet_bc][MPI]") {
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 3) {
    return;
  }
  constexpr int nz = 12;
  const auto domain = pfc::domain::from_bounds(
      {4, 4, nz}, {0.0, 0.0, 0.0}, {4.0, 4.0, 11.0}, pfc::Bool3{true, true, false});
  const auto decomp = make_decomp(domain, {1, 1, 3});
  constexpr int halo = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  u.for_each_owned([&](int i, int j, int k) {
    u(i, j, k) = static_cast<double>(u.global(i, j, k)[2]);
  });
  DirichletBoundary<double> walls;
  walls.set_constant(CartesianFace::Zm, 0.0);
  walls.set_constant(CartesianFace::Zp, 0.0);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, rank,
                                                      MPI_COMM_WORLD, halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  walls.bind_dirichlet(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const std::string_view names[] = {"u"};
  prep.prepare(dirichlet_stage_requirements(ctx), names);

  const int z0 = u.lower_global()[2];
  const int local_nz = u.local_size()[2];
  int ok = 1;
  if (z0 == 0) {
    if (std::abs(u(0, 0, 0)) > 1e-12) {
      ok = 0;
    }
    if (std::abs(u(0, 0, -1) + u(0, 0, 1)) > 1e-12) {
      ok = 0;
    }
  } else if (z0 + local_nz == nz) {
    if (std::abs(u(0, 0, local_nz - 1)) > 1e-12) {
      ok = 0;
    }
  } else {
    for (int k = 0; k < local_nz; ++k) {
      if (std::abs(u(0, 0, k) - static_cast<double>(z0 + k)) > 1e-12) {
        ok = 0;
      }
    }
    const double expected_low = static_cast<double>(z0 - 1);
    if (std::abs(u(0, 0, -1) - expected_low) > 1e-12) {
      ok = 0;
    }
  }
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(all_ok == 1);
}
