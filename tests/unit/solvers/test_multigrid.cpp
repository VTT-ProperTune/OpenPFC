// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/structured_operators.hpp>
using namespace pfc::solvers;
namespace {
std::vector<double> random_vector(int n) {
  std::vector<double> v(n);
  unsigned state = 27;
  for (auto &x : v) {
    state = 1664525u * state + 1013904223u;
    x = static_cast<double>(state >> 8) / 16777216. - .5;
  }
  return v;
}
} // namespace
TEST_CASE("physical grid transfers preserve constants and are adjoints",
          "[multigrid]") {
  for (auto boundary : {Boundary::Periodic, Boundary::Dirichlet}) {
    StructuredGrid fine{{9, 9, 9}, 3, boundary}, coarse{{5, 5, 5}, 3, boundary};
    GridTransfer transfer(fine, coarse);
    VectorOperations ops;
    auto a = random_vector(fine.size()), b = random_vector(coarse.size());
    fine.project(a);
    coarse.project(b);
    std::vector<double> ra, pb;
    transfer.restrict(a, ra);
    transfer.prolong(b, pb);
    REQUIRE(std::abs(ops.dot(ra, b) - ops.dot(a, pb)) < 1e-12);
    if (boundary == Boundary::Periodic) {
      b.assign(coarse.size(), 1);
      transfer.prolong(b, pb);
      for (double v : pb) REQUIRE(std::abs(v - 1) < 1e-14);
    } else {
      for (int z = 0; z < 9; ++z)
        for (int y = 0; y < 9; ++y)
          for (int x = 0; x < 9; ++x) {
            b.assign(coarse.size(), 0);
            if (fine.constrained(x, y, z)) REQUIRE(pb[3 * fine.node(x, y, z)] == 0);
          }
    }
  }
}
TEST_CASE("Galerkin operator equals independent R A P", "[multigrid]") {
  StructuredGrid f{{9, 9, 9}, 3}, c{{5, 5, 5}, 3};
  GridTransfer transfer(f, c);
  auto a = elastic_auxiliary(f);
  auto ac = transfer.galerkin(a);
  auto x = random_vector(c.size());
  std::vector<double> px, apx, rapx, ax;
  transfer.prolong(x, px);
  a(px, apx);
  transfer.restrict(apx, rapx);
  ac(x, ax);
  VectorOperations ops;
  ops.axpy(ax, -1, rapx);
  REQUIRE(ops.norm(ax) < 1e-12 * ops.norm(rapx));
}
TEST_CASE("Poisson and coupled elasticity multigrid reduce random loads",
          "[multigrid]") {
  VectorOperations ops;
  for (auto boundary : {Boundary::Periodic, Boundary::Dirichlet})
    for (int components : {1, 3})
      for (auto coarse : {CoarseOperator::Rediscretized, CoarseOperator::Galerkin}) {
        int previous = 0;
        for (int n : {5, 9, 17}) {
          StructuredGrid g{{n, n, n}, components, boundary};
          auto factory = [&](const StructuredGrid &grid) {
            return components == 1 ? poisson_auxiliary(grid)
                                   : elastic_auxiliary(grid);
          };
          MultigridControls mc;
          mc.coarse_operator = coarse;
          Multigrid mg(g, factory, mc);
          auto a = factory(g);
          auto b = random_vector(g.size());
          g.project(b);
          std::vector<double> x(g.size()), z, az;
          mg(b, z);
          a(z, az);
          ops.axpy(az, -1, b);
          REQUIRE(ops.norm(az) < ops.norm(b));
          KrylovControls c;
          c.relative_tolerance = 1e-9;
          c.maximum_iterations = 200;
          auto result = gmres(ops, a, b, x, c,
                              [&](const auto &r, auto &w, int i) { mg(r, w, i); });
          INFO("n=" << n << " components=" << components
                    << " boundary=" << int(boundary) << " coarse=" << int(coarse)
                    << " iterations=" << result.iterations);
          REQUIRE(result.converged());
          REQUIRE(result.iterations < 60);
          if (previous) REQUIRE(result.iterations < previous + 15);
          previous = result.iterations;
          if (boundary == Boundary::Periodic) {
            std::vector<double> constant(g.size(), 1), null;
            a(constant, null);
            REQUIRE(ops.norm(null) < 1e-12);
            mg(constant, null);
            REQUIRE(ops.norm(null) < 1e-12);
          }
        }
      }
}
TEST_CASE("elastic auxiliary couples components and preserves symmetry",
          "[multigrid]") {
  StructuredGrid g{{5, 5, 5}, 3};
  auto a = elastic_auxiliary(g);
  auto x = random_vector(g.size()), y = random_vector(g.size());
  g.project(x);
  g.project(y);
  std::vector<double> ax, ay;
  a(x, ax);
  a(y, ay);
  VectorOperations ops;
  REQUIRE(std::abs(ops.dot(x, ay) - ops.dot(y, ax)) < 1e-12);
  REQUIRE(ops.dot(x, ax) > 0);
  x.assign(g.size(), 0);
  x[3 * g.node(2, 2, 2)] = 1;
  a(x, ax);
  double cross = 0;
  for (std::size_t i = 0; i < ax.size(); ++i)
    if (i % 3 != 0) cross += ax[i] * ax[i];
  REQUIRE(cross > 0);
}
TEST_CASE("manufactured Poisson has second-order physical convergence",
          "[multigrid]") {
  VectorOperations ops;
  double previous = 0;
  for (int n : {9, 17, 33}) {
    StructuredGrid g{{n, n, n}, 1, Boundary::Dirichlet};
    auto a = poisson_auxiliary(g);
    double h = 1.0 / (n - 1);
    std::vector<double> exact(g.size()), b(g.size()), x(g.size());
    const double pi = std::acos(-1.);
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int i = 0; i < n; ++i) {
          int k = g.node(i, y, z);
          exact[k] =
              std::sin(pi * i * h) * std::sin(pi * y * h) * std::sin(pi * z * h);
          b[k] = h * h * h * 3 * pi * pi * exact[k];
        }
    g.project(b);
    g.project(exact);
    Multigrid mg(g, poisson_auxiliary);
    KrylovControls c;
    c.relative_tolerance = 1e-11;
    auto r =
        gmres(ops, a, b, x, c, [&](const auto &r, auto &z, int i) { mg(r, z, i); });
    REQUIRE(r.converged());
    ops.axpy(x, -1, exact);
    double error = ops.norm(x) / ops.norm(exact);
    if (previous) REQUIRE(previous / error > 3.9);
    previous = error;
  }
}

#include <openpfc/solvers/multigrid_mpi.hpp>
#include <openpfc/solvers/poisson_action.hpp>
TEST_CASE("halo Poisson with replicated MG matches serial owned reference",
          "[multigrid][MPI]") {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  StructuredGrid g{{9, 9, 9}, 1};
  auto domain = pfc::domain::create(pfc::Int3{9, 9, 9});
  auto decomp = pfc::decomposition::create(domain, ranks);
  PeriodicPoissonAction action(decomp, rank, MPI_COMM_WORLD);
  auto owned = action.owned_indices();
  Multigrid mg(g, poisson_auxiliary);
  ReplicatedMultigrid precondition(mg, owned, MPI_COMM_WORLD);
  auto global = random_vector(g.size());
  g.project(global);
  std::vector<double> expected_global;
  mg(global, expected_global);
  std::vector<double> b(owned.size()), x(owned.size()), z, ap, global_ap;
  for (std::size_t i = 0; i < owned.size(); ++i) b[i] = global[owned[i]];
  precondition(b, z);
  for (std::size_t i = 0; i < owned.size(); ++i)
    REQUIRE(std::abs(z[i] - expected_global[owned[i]]) < 1e-12);
  poisson_auxiliary(g)(global, global_ap);
  action(b, ap);
  for (std::size_t i = 0; i < owned.size(); ++i)
    REQUIRE(std::abs(ap[i] - global_ap[owned[i]]) < 1e-13);
  VectorOperations ops(MPI_COMM_WORLD);
  KrylovControls c;
  c.relative_tolerance = 1e-10;
  auto result = gmres(ops, action, b, x, c,
                      [&](const auto &r, auto &w, int k) { precondition(r, w, k); });
  REQUIRE(result.converged());
  REQUIRE(result.iterations < 30);
  REQUIRE(action.communication_seconds > 0);
  REQUIRE(precondition.gathers > 0);
}

TEST_CASE("coupled vector elasticity matches a continuum manufactured field",
          "[multigrid]") {
  VectorOperations ops;
  double previous = 0;
  const double w = 2 * std::acos(-1.);
  for (int n : {9, 17, 33}) {
    StructuredGrid g{{n, n, n}, 3};
    auto a = elastic_auxiliary(g);
    std::vector<double> exact(g.size()), b(g.size()), x(g.size());
    for (int z = 0; z < n; ++z)
      for (int y = 0; y < n; ++y)
        for (int i = 0; i < n; ++i) {
          int k = 3 * g.node(i, y, z);
          exact[k] = std::sin(w * i / n) * std::sin(w * y / n);
          exact[k + 1] = std::cos(w * i / n) * std::cos(w * y / n);
          exact[k + 2] = std::sin(w * z / n);
          b[k] = w * w * exact[k] / g.nodes();
          b[k + 1] = w * w * exact[k + 1] / g.nodes();
          b[k + 2] = 2 * w * w * exact[k + 2] / g.nodes();
        }
    g.project(b);
    g.project(exact);
    Multigrid mg(g,
                 [](const StructuredGrid &grid) { return elastic_auxiliary(grid); });
    KrylovControls c;
    c.relative_tolerance = 1e-11;
    auto r =
        gmres(ops, a, b, x, c, [&](const auto &r, auto &z, int k) { mg(r, z, k); });
    REQUIRE(r.converged());
    ops.axpy(x, -1, exact);
    double error = ops.norm(x) / ops.norm(exact);
    if (previous) REQUIRE(previous / error > 3.0);
    previous = error;
  }
}
