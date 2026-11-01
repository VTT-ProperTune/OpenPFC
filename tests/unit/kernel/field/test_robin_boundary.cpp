// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
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
#include <openpfc/kernel/decomposition/robin_boundary.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pfc::HostSpace;
using pfc::communication::BoundaryCentering;
using pfc::communication::CartesianFace;
using pfc::communication::DirichletBoundary;
using pfc::communication::ReflectionParity;
using pfc::communication::robin_stage_requirements;
using pfc::communication::RobinBoundary;

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

int world_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int world_size() {
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  return ranks;
}

template <int Order>
double d1_x(const pfc::data::Field<double, HostSpace> &u, int i, double h) {
  using Stencil = pfc::field::fd::EvenCentralD1<Order>;
  double acc = 0.0;
  for (int layer = 1; layer <= Stencil::half_width; ++layer) {
    acc += static_cast<double>(Stencil::coeffs[static_cast<std::size_t>(layer)]) *
           (u(i + layer, 0, 0) - u(i - layer, 0, 0));
  }
  return acc / (static_cast<double>(Stencil::denom) * h);
}

template <int Order>
double d2_x(const pfc::data::Field<double, HostSpace> &u, int i, double h) {
  using Stencil = pfc::field::fd::EvenCentralD2<Order>;
  double acc = static_cast<double>(Stencil::coeffs[0]) * u(i, 0, 0);
  for (int layer = 1; layer <= Stencil::half_width; ++layer) {
    acc += static_cast<double>(Stencil::coeffs[static_cast<std::size_t>(layer)]) *
           (u(i + layer, 0, 0) + u(i - layer, 0, 0));
  }
  return acc / (static_cast<double>(Stencil::denom) * h * h);
}

void paint_x(pfc::data::Field<double, HostSpace> &u,
             const std::vector<double> &owned) {
  const int halo = u.storage_halo();
  const int nx = u.local_size()[0];
  const int ny = u.local_size()[1];
  const int nz = u.local_size()[2];
  for (int i = -halo; i < nx + halo; ++i) {
    const double value =
        (i >= 0 && i < nx) ? owned[static_cast<std::size_t>(i)] : 0.0;
    for (int k = -halo; k < nz + halo; ++k) {
      for (int j = -halo; j < ny + halo; ++j) {
        u(i, j, k) = value;
      }
    }
  }
}

void eliminate(std::vector<double> &matrix, std::vector<double> &rhs) {
  const int n = static_cast<int>(rhs.size());
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(matrix[static_cast<std::size_t>(col * n + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double value = std::abs(matrix[static_cast<std::size_t>(row * n + col)]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    REQUIRE(best > 0.0);
    if (pivot != col) {
      for (int j = col; j < n; ++j) {
        std::swap(matrix[static_cast<std::size_t>(col * n + j)],
                  matrix[static_cast<std::size_t>(pivot * n + j)]);
      }
      std::swap(rhs[static_cast<std::size_t>(col)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    const double diag = matrix[static_cast<std::size_t>(col * n + col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor = matrix[static_cast<std::size_t>(row * n + col)] / diag;
      for (int j = col; j < n; ++j) {
        matrix[static_cast<std::size_t>(row * n + j)] -=
            factor * matrix[static_cast<std::size_t>(col * n + j)];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(col)];
    }
  }
  for (int row = n - 1; row >= 0; --row) {
    double acc = rhs[static_cast<std::size_t>(row)];
    for (int j = row + 1; j < n; ++j) {
      acc -= matrix[static_cast<std::size_t>(row * n + j)] *
             rhs[static_cast<std::size_t>(j)];
    }
    rhs[static_cast<std::size_t>(row)] =
        acc / matrix[static_cast<std::size_t>(row * n + row)];
  }
}

template <int Order>
std::vector<double> laplacian_x(pfc::data::Field<double, HostSpace> &u,
                                const RobinBoundary<double> &robin, double time,
                                const std::vector<double> &owned) {
  paint_x(u, owned);
  robin.apply_ghosts(u, time);
  const int n = static_cast<int>(owned.size());
  const double h = u.spacing()[0];
  std::vector<double> out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = d2_x<Order>(u, i, h);
  }
  return out;
}

template <int Order>
double poisson_error(int n, double (*exact)(double), double (*slope)(double),
                     double (*second)(double), double alpha, double beta) {
  constexpr int halo = Order / 2;
  auto u = make_field({n, 4, 4}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1}, 0,
                      halo);
  const double h = u.spacing()[0];
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Xm, alpha, beta,
                     alpha * exact(0.0) + beta * (-slope(0.0)));
  robin.set_constant(CartesianFace::Xp, alpha, beta,
                     alpha * exact(1.0) + beta * slope(1.0));
  const auto apply = [&](const std::vector<double> &owned) {
    return laplacian_x<Order>(u, robin, 0.0, owned);
  };
  const std::vector<double> affine =
      apply(std::vector<double>(static_cast<std::size_t>(n), 0.0));
  std::vector<double> matrix(static_cast<std::size_t>(n * n));
  std::vector<double> rhs(static_cast<std::size_t>(n));
  for (int col = 0; col < n; ++col) {
    std::vector<double> basis(static_cast<std::size_t>(n), 0.0);
    basis[static_cast<std::size_t>(col)] = 1.0;
    const std::vector<double> column = apply(basis);
    for (int row = 0; row < n; ++row) {
      matrix[static_cast<std::size_t>(row * n + col)] =
          column[static_cast<std::size_t>(row)] -
          affine[static_cast<std::size_t>(row)];
    }
  }
  for (int row = 0; row < n; ++row) {
    rhs[static_cast<std::size_t>(row)] =
        second(static_cast<double>(row) * h) - affine[static_cast<std::size_t>(row)];
  }
  eliminate(matrix, rhs);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    error = std::max(error, std::abs(rhs[static_cast<std::size_t>(i)] -
                                     exact(static_cast<double>(i) * h)));
  }
  return error;
}

} // namespace

TEST_CASE("Robin stage order is boundary then halo", "[robin_bc][unit]") {
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const auto req = robin_stage_requirements(ctx);
  REQUIRE(req.needs_halo_exchange);
  REQUIRE(req.needs_boundary_update);
  REQUIRE(req.ordering == pfc::communication::BoundaryHaloOrder::BoundaryThenHalo);
}

TEST_CASE("Outward normals on the low and high faces have opposite signs",
          "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  constexpr int n = 9;
  auto u =
      make_field({n, 4, 4}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  const double h = u.spacing()[0];
  const double slope = 1.7;
  u.for_each_owned([&](int i, int j, int k) {
    const double x = static_cast<double>(u.global(i, j, k)[0]) * h;
    u(i, j, k) = 0.4 + slope * x;
    (void)j;
    (void)k;
  });
  constexpr double alpha = 1.25;
  constexpr double beta = 0.5;
  const double u_low = 0.4;
  const double u_high = 0.4 + slope;
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Xm, alpha, beta,
                     alpha * u_low + beta * (-slope));
  robin.set_constant(CartesianFace::Xp, alpha, beta, alpha * u_high + beta * slope);
  robin.apply_ghosts(u, 0.0);
  REQUIRE_THAT(d1_x<2>(u, 0, h), WithinAbs(slope, 1e-12));
  REQUIRE_THAT(d1_x<4>(u, 0, h), WithinAbs(slope, 1e-12));
  REQUIRE_THAT(d1_x<2>(u, n - 1, h), WithinAbs(slope, 1e-12));
  REQUIRE_THAT(d1_x<4>(u, n - 1, h), WithinAbs(slope, 1e-12));
  const double outward_low = -d1_x<2>(u, 0, h);
  const double outward_high = d1_x<2>(u, n - 1, h);
  REQUIRE(outward_low * outward_high < 0.0);
  REQUIRE_THAT(alpha * u(0, 0, 0) + beta * outward_low,
               WithinAbs(alpha * u_low + beta * (-slope), 1e-12));
  REQUIRE_THAT(alpha * u(n - 1, 0, 0) + beta * outward_high,
               WithinAbs(alpha * u_high + beta * slope, 1e-12));
  REQUIRE_THAT(u(-1, 0, 0), WithinAbs(0.4 - slope * h, 1e-12));
  REQUIRE_THAT(u(-2, 0, 0), WithinAbs(0.4 - slope * 2.0 * h, 1e-12));
  REQUIRE_THAT(u(n, 0, 0), WithinAbs(0.4 + slope * (1.0 + h), 1e-12));
}

TEST_CASE("Homogeneous Neumann matches the even mirror", "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  constexpr int n = 8;
  auto original =
      make_field({n, 4, 4}, {7.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  original.for_each_owned([&](int i, int j, int k) {
    original(i, j, k) = 0.2 * static_cast<double>(i) + 0.05 * static_cast<double>(j);
    (void)k;
  });
  auto reflected = original;
  pfc::communication::ReflectionWalls<double> walls;
  walls.xm.enabled = true;
  walls.xm.parity = ReflectionParity::Even;
  walls.xm.centering = BoundaryCentering::Node;
  walls.xp = walls.xm;
  pfc::communication::apply_reflection_ghosts(reflected, walls);
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Xm, 0.0, 1.0, 0.0);
  robin.set_constant(CartesianFace::Xp, 0.0, 1.0, 0.0);
  robin.apply_ghosts(original, 0.0);
  for (int layer = 1; layer <= 2; ++layer) {
    REQUIRE_THAT(original(-layer, 1, 1), WithinAbs(reflected(-layer, 1, 1), 1e-12));
    REQUIRE_THAT(original(n - 1 + layer, 1, 1),
                 WithinAbs(reflected(n - 1 + layer, 1, 1), 1e-12));
  }
  REQUIRE_THAT(original(3, 1, 1), WithinAbs(reflected(3, 1, 1), 1e-12));
}

TEST_CASE("A zero beta reuses the node Dirichlet mirror", "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  constexpr int n = 8;
  auto robin_field =
      make_field({n, 4, 4}, {7.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 2);
  auto dirichlet_field = robin_field;
  auto fill = [](pfc::data::Field<double, HostSpace> &u) {
    u.for_each_owned([&](int i, int j, int k) {
      u(i, j, k) = 0.3 * static_cast<double>(i + j);
      (void)k;
    });
  };
  fill(robin_field);
  fill(dirichlet_field);
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Xm, 2.0, 0.0, 2.5);
  robin.set_constant(CartesianFace::Xp, 2.0, 0.0, -1.0);
  DirichletBoundary<double> dirichlet;
  dirichlet.set_constant(CartesianFace::Xm, 1.25);
  dirichlet.set_constant(CartesianFace::Xp, -0.5);
  robin.enforce_dirichlet_nodes(robin_field, 0.0);
  robin.apply_ghosts(robin_field, 0.0);
  dirichlet.enforce_owned_nodes(dirichlet_field, 0.0);
  dirichlet.apply_ghosts(dirichlet_field, 0.0);
  const int halo = robin_field.storage_halo();
  for (int i = -halo; i < n + halo; ++i) {
    REQUIRE_THAT(robin_field(i, 1, 1), WithinAbs(dirichlet_field(i, 1, 1), 1e-12));
  }
}

TEST_CASE("A disabled face does not overwrite its ghost", "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  constexpr int n = 6;
  auto u =
      make_field({n, 4, 4}, {5.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1}, 0, 1);
  u.for_each_owned([&](int i, int j, int k) { u(i, j, k) = 1.0 + 0.2 * i; });
  u(-1, 0, 0) = 42.0;
  u(n, 0, 0) = 42.0;
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Xm, 1.0, 0.5, -0.25);
  robin.apply_ghosts(u, 0.0);
  REQUIRE(u(n, 0, 0) == 42.0);
  REQUIRE(u(-1, 0, 0) != 42.0);
}

TEST_CASE("Singular Robin coefficients fail closed", "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  RobinBoundary<double> robin;
  REQUIRE_THROWS_WITH(robin.set_constant(CartesianFace::Ym, 0.0, 0.0, 1.0),
                      ContainsSubstring("singular"));
  REQUIRE_THROWS_WITH(robin.set_constant(CartesianFace::Ym, std::nan(""), 1.0, 0.0),
                      ContainsSubstring("finite"));
  robin.coefficient_tolerance = 1e-6;
  REQUIRE_THROWS_WITH(robin.set_constant(CartesianFace::Ym, 0.0, 1e-9, 0.0),
                      ContainsSubstring("singular"));
  robin.coefficient_tolerance = 0.0;
  robin.set_constant(CartesianFace::Xm, 0.0, 1e-4, 0.0);
  robin.coefficient_tolerance = 1e-6;
  auto u = make_field({4, 4, 4}, {3000.0, 4.0, 4.0}, {false, true, true}, {1, 1, 1},
                      0, 1);
  REQUIRE(u.spacing()[0] == 1000.0);
  REQUIRE_THROWS_WITH(robin.apply_ghosts(u, 0.0), ContainsSubstring("singular"));
  auto periodic =
      make_field({4, 4, 4}, {4.0, 4.0, 4.0}, {true, true, true}, {1, 1, 1}, 0, 1);
  RobinBoundary<double> wrapped;
  wrapped.set_constant(CartesianFace::Xm, 1.0, 0.2, 0.0);
  REQUIRE_THROWS_WITH(wrapped.apply_ghosts(periodic, 0.0),
                      ContainsSubstring("periodic"));
}

TEST_CASE("A quadratic heat solution is an exact discrete Robin mode",
          "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  constexpr int n = 9;
  constexpr int halo = 2;
  auto u = make_field({n, 4, 4}, {1.0, 1.0, 1.0}, {false, true, true}, {1, 1, 1}, 0,
                      halo);
  const double h = u.spacing()[0];
  constexpr double alpha = 1.5;
  constexpr double beta = 0.5;
  constexpr double dt = 0.05;
  const auto value = [&](double x, double time) {
    return (1.0 + time) + 0.25 * x + 0.5 * x * x;
  };
  const auto slope = [](double x) { return 0.25 + x; };
  RobinBoundary<double> robin;
  robin.set_time_dependent_gamma(CartesianFace::Xm, alpha, beta, [=](double time) {
    return alpha * value(0.0, time) + beta * (-slope(0.0));
  });
  robin.set_time_dependent_gamma(CartesianFace::Xp, alpha, beta, [=](double time) {
    return alpha * value(1.0, time) + beta * slope(1.0);
  });
  double time = 0.0;
  for (int i = -halo; i < n + halo; ++i) {
    const double x = static_cast<double>(i) * h;
    for (int k = 0; k < 4; ++k) {
      for (int j = 0; j < 4; ++j) {
        u(i, j, k) = value(x, time);
      }
    }
  }
  for (int step = 0; step < 4; ++step) {
    robin.apply_ghosts(u, time);
    for (int i = 0; i < n; ++i) {
      REQUIRE_THAT(d2_x<2>(u, i, h), WithinAbs(1.0, 1e-10));
      REQUIRE_THAT(d2_x<4>(u, i, h), WithinAbs(1.0, 1e-10));
      const double helmholtz = -d2_x<2>(u, i, h) + 1.5 * u(i, 0, 0);
      const double x = static_cast<double>(i) * h;
      REQUIRE_THAT(helmholtz, WithinAbs(-1.0 + 1.5 * value(x, time), 1e-10));
    }
    for (int i = 0; i < n; ++i) {
      u(i, 0, 0) += dt * 1.0;
    }
    time += dt;
  }
  for (int i = 0; i < n; ++i) {
    REQUIRE_THAT(u(i, 0, 0),
                 WithinAbs(value(static_cast<double>(i) * h, time), 1e-10));
  }
}

TEST_CASE("Robin Poisson converges under grid refinement", "[robin_bc][unit]") {
  if (world_rank() != 0) {
    return;
  }
  const auto quadratic = [](double x) { return x * x; };
  const auto quadratic_slope = [](double x) { return 2.0 * x; };
  const auto quadratic_second = [](double) { return 2.0; };
  REQUIRE(poisson_error<2>(17, quadratic, quadratic_slope, quadratic_second, 1.0,
                           0.5) < 1e-8);
  REQUIRE(poisson_error<4>(17, quadratic, quadratic_slope, quadratic_second, 1.0,
                           0.5) < 1e-8);

  const auto sine = [](double x) { return std::sin(1.3 * x); };
  const auto sine_slope = [](double x) { return 1.3 * std::cos(1.3 * x); };
  const auto sine_second = [](double x) { return -1.69 * std::sin(1.3 * x); };
  const double coarse =
      poisson_error<2>(17, sine, sine_slope, sine_second, 0.8, 0.3);
  const double mid = poisson_error<2>(33, sine, sine_slope, sine_second, 0.8, 0.3);
  const double fine = poisson_error<2>(65, sine, sine_slope, sine_second, 0.8, 0.3);
  REQUIRE(mid < coarse * 0.6);
  REQUIRE(fine < mid * 0.6);
  REQUIRE(fine < 1e-3);
}

TEST_CASE("Two ranks exchange through a Robin wall", "[robin_bc][MPI]") {
  if (world_size() != 2) {
    return;
  }
  const int rank = world_rank();
  constexpr int nz = 6;
  const auto domain = pfc::domain::from_bounds(
      {8, 4, nz}, {0.0, 0.0, 0.0}, {8.0, 4.0, 5.0}, pfc::Bool3{true, true, false});
  const auto decomp = make_decomp(domain, {2, 1, 1});
  constexpr int halo = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  const double owned_value = 100.0 + static_cast<double>(rank);
  u.for_each_owned([&](int i, int j, int k) { u(i, j, k) = owned_value; });
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Zm, 0.0, 1.0, 0.5);
  robin.set_constant(CartesianFace::Zp, 0.0, 1.0, -0.25);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, rank,
                                                      MPI_COMM_WORLD, halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  robin.bind_robin(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const std::string_view names[] = {"u"};
  prep.prepare(robin_stage_requirements(ctx), names);

  const double dz = u.spacing()[2];
  const int local_nz = u.local_size()[2];
  int ok = 1;
  if (std::abs(u(0, 0, 0) - owned_value) > 1e-12) {
    ok = 0;
  }
  if (std::abs(u(0, 0, -1) - (u(0, 0, 1) + dz)) > 1e-12) {
    ok = 0;
  }
  if (std::abs(u(0, 0, local_nz) - (u(0, 0, local_nz - 2) - 0.5 * dz)) > 1e-12) {
    ok = 0;
  }
  const double neighbor = 100.0 + static_cast<double>(1 - rank);
  if (std::abs(u(-1, 0, 0) - neighbor) > 1e-12) {
    ok = 0;
  }
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(all_ok == 1);
}

TEST_CASE("An interior rank does not treat a partition face as Robin",
          "[robin_bc][MPI]") {
  if (world_size() != 3) {
    return;
  }
  const int rank = world_rank();
  constexpr int nz = 12;
  const auto domain = pfc::domain::from_bounds(
      {4, 4, nz}, {0.0, 0.0, 0.0}, {4.0, 4.0, 11.0}, pfc::Bool3{true, true, false});
  const auto decomp = make_decomp(domain, {1, 1, 3});
  constexpr int halo = 1;
  auto u = pfc::data::field_from_subdomain<double>(decomp, rank, halo);
  u.for_each_owned([&](int i, int j, int k) {
    u(i, j, k) = static_cast<double>(u.global(i, j, k)[2]);
  });
  RobinBoundary<double> robin;
  robin.set_constant(CartesianFace::Zm, 0.0, 1.0, 0.0);
  robin.set_constant(CartesianFace::Zp, 0.0, 1.0, 0.0);
  const auto halo_opt = face_exchange_options(decomp);
  pfc::comm::HaloExchange<HostSpace, double> exchange(u, decomp, rank,
                                                      MPI_COMM_WORLD, halo_opt);
  pfc::communication::StagePreparationService<double> prep;
  prep.bind("u", exchange);
  const double time = 0.0;
  robin.bind_robin(prep, "u", u, time);
  pfc::integrator::StageContext ctx;
  ctx.needs_halo_exchange = true;
  const std::string_view names[] = {"u"};
  prep.prepare(robin_stage_requirements(ctx), names);

  const int z0 = u.lower_global()[2];
  const int local_nz = u.local_size()[2];
  int ok = 1;
  if (z0 == 0) {
    if (std::abs(u(0, 0, 0)) > 1e-12 || std::abs(u(0, 0, -1) - u(0, 0, 1)) > 1e-12) {
      ok = 0;
    }
  } else if (z0 + local_nz == nz) {
    if (std::abs(u(0, 0, local_nz) - u(0, 0, local_nz - 2)) > 1e-12) {
      ok = 0;
    }
  } else if (std::abs(u(0, 0, -1) - static_cast<double>(z0 - 1)) > 1e-12) {
    ok = 0;
  }
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(all_ok == 1);
}
