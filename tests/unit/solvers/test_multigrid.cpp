// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <openpfc/solvers/coarse_correction.hpp>
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

TEST_CASE("auxiliary operators preserve physical box metrics",
          "[multigrid][metrics]") {
  StructuredGrid g{{7, 5, 9}, 3};
  auto a = elastic_auxiliary(g, 1, .5),
       dilated = elastic_auxiliary(g, 1, .5, {2, 2, 2});
  REQUIRE(a.columns == dilated.columns);
  for (std::size_t i = 0; i < a.values.size(); ++i)
    REQUIRE(std::abs(dilated.values[i] - 2 * a.values[i]) < 1e-14);
  std::array<double, 3> lengths{2, 3, 5}, amplitude{.7, -.2, .3};
  auto anisotropic = elastic_auxiliary(g, 1, .5, lengths);
  std::vector<double> u(g.size()), expected(g.size()), result;
  double volume = 1, alpha[3], beta[3], total = 0;
  for (int d = 0; d < 3; ++d) {
    double h = lengths[d] / g.shape[d], theta = 2 * std::acos(-1.) / g.shape[d];
    volume *= h;
    alpha[d] = 4 * std::pow(std::sin(theta / 2), 2) / (h * h);
    beta[d] = std::sin(theta) / h;
    total += alpha[d];
  }
  for (int z = 0; z < g.shape[2]; ++z)
    for (int y = 0; y < g.shape[1]; ++y)
      for (int x = 0; x < g.shape[0]; ++x) {
        double value = std::sin(2 * std::acos(-1.) *
                                (double(x) / g.shape[0] + double(y) / g.shape[1] +
                                 double(z) / g.shape[2]));
        for (int c = 0; c < 3; ++c) {
          int row = 3 * g.node(x, y, z) + c;
          u[row] = amplitude[c] * value;
          double v = .5 * total * amplitude[c] + 1.5 * alpha[c] * amplitude[c];
          for (int d = 0; d < 3; ++d)
            if (d != c) v += 1.5 * beta[c] * beta[d] * amplitude[d];
          expected[row] = volume * value * v;
        }
      }
  anisotropic(u, result);
  for (std::size_t i = 0; i < u.size(); ++i)
    REQUIRE(std::abs(result[i] - expected[i]) < 1e-12);
  std::fill(u.begin(), u.end(), 1);
  anisotropic(u, result);
  for (double v : result) REQUIRE(std::abs(v) < 1e-13);
  REQUIRE_THROWS_AS(elastic_auxiliary(g, 1, .5, {1, 0, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(
      elastic_auxiliary(g, 1, .5, {1, std::numeric_limits<double>::infinity(), 1}),
      std::invalid_argument);
  StructuredGrid scalar{{5, 5, 5}, 1, Boundary::Dirichlet};
  auto dirichlet = elastic_auxiliary(scalar, 0, 1, lengths);
  for (int row = 0; row < dirichlet.size(); ++row)
    for (int k = dirichlet.offsets[row]; k < dirichlet.offsets[row + 1]; ++k) {
      int col = dirichlet.columns[k];
      double opposite = 0;
      for (int j = dirichlet.offsets[col]; j < dirichlet.offsets[col + 1]; ++j)
        if (dirichlet.columns[j] == row) opposite = dirichlet.values[j];
      REQUIRE(std::abs(dirichlet.values[k] - opposite) < 1e-14);
    }
}

namespace {
Multigrid::Factory auxiliary_factory(int components, std::array<double, 3> lengths) {
  return [=](const StructuredGrid &grid) {
    return components == 1 ? poisson_auxiliary(grid)
                           : elastic_auxiliary(grid, 1, .5, lengths);
  };
}
std::vector<int> every_index(int n) {
  std::vector<int> owned(n);
  for (int i = 0; i < n; ++i) owned[i] = i;
  return owned;
}
void require_true_residual(VectorOperations<> &ops, const std::vector<double> &b,
                           const std::vector<double> &ax) {
  auto residual = ax;
  ops.axpy(residual, -1, b);
  REQUIRE(ops.norm(residual) <= 1e-8 * ops.norm(b));
}
} // namespace

TEST_CASE("shared coarse correction is adjoint and admits a true residual",
          "[multigrid][coarse]") {
  VectorOperations ops;
  for (int components : {1, 3})
    for (auto boundary : {Boundary::Periodic, Boundary::Dirichlet}) {
      StructuredGrid grid{{9, 7, 5}, components, boundary};
      auto factory = auxiliary_factory(components, {2, 3, 5});
      SharedCoarseCorrection correction(grid, every_index(grid.size()), factory,
                                        MPI_COMM_NULL);
      std::vector<double> coarse(correction.coarse().size(), 1), fine, restricted;
      correction.prolong_owned(coarse, fine);
      for (int i = 0; i < grid.size(); ++i) {
        int node = i / components, x = node % grid.shape[0],
            y = node / grid.shape[0] % grid.shape[1],
            z = node / (grid.shape[0] * grid.shape[1]);
        if (boundary == Boundary::Periodic)
          REQUIRE(std::abs(fine[i] - 1) < 1e-12);
        else if (grid.constrained(x, y, z))
          REQUIRE(fine[i] == 0);
        else
          REQUIRE(fine[i] >= -1e-12);
      }
      auto field = random_vector(grid.size()),
           mode = random_vector(correction.coarse().size());
      grid.project(field);
      correction.coarse().project(mode);
      correction.restrict_owned(field, restricted);
      correction.prolong_owned(mode, fine);
      REQUIRE(std::abs(ops.dot(restricted, mode) - ops.dot(field, fine)) < 1e-9);
      if (boundary == Boundary::Periodic) {
        std::vector<double> biased(grid.size(), 1), update;
        correction(biased, update);
        double sum = 0;
        for (double value : update) sum += value;
        REQUIRE(std::abs(sum) < 1e-8);
      }
      auto rhs = random_vector(grid.size());
      std::vector<double> unknown(grid.size()), product;
      grid.project(rhs);
      auto action = factory(grid);
      KrylovControls controls;
      controls.relative_tolerance = 1e-10;
      controls.maximum_iterations = 200;
      auto solved = gmres(ops, action, rhs, unknown, controls,
                          [&](const auto &r, auto &z, int) {
                            two_level(correction, action, action.diagonal, r, z);
                          });
      REQUIRE(solved.converged());
      REQUIRE(solved.iterations < 80);
      action(unknown, product);
      require_true_residual(ops, rhs, product);
      REQUIRE(correction.coarse().size() <= correction.fine().size() / 2);
    }
  StructuredGrid grid{{9, 9, 9}, 1};
  auto incomplete = every_index(grid.size());
  incomplete.pop_back();
  REQUIRE_THROWS_AS(
      SharedCoarseCorrection(grid, incomplete, poisson_auxiliary, MPI_COMM_NULL),
      std::invalid_argument);
}

TEST_CASE("two ranks share one coarse correction under unequal ownership",
          "[multigrid][MPI][coarse]") {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) SKIP("two-rank coarse correction");
  VectorOperations ops(MPI_COMM_WORLD);
  for (int components : {1, 3}) {
    StructuredGrid grid{{9, 9, 9}, components};
    auto factory = auxiliary_factory(components, {2, 3, 5});
    std::vector<int> owned;
    for (int i = 0; i < grid.size(); ++i)
      if (i % ranks == rank) owned.push_back(i);
    int count = owned.size(), counts[2];
    MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    REQUIRE(counts[0] != counts[1]);
    SharedCoarseCorrection serial(grid, every_index(grid.size()), factory,
                                  MPI_COMM_NULL);
    SharedCoarseCorrection distributed(grid, owned, factory, MPI_COMM_WORLD);
    auto global = random_vector(grid.size());
    grid.project(global);
    std::vector<double> serial_update, local(owned.size()), distributed_update;
    for (std::size_t i = 0; i < owned.size(); ++i) local[i] = global[owned[i]];
    serial(global, serial_update);
    distributed(local, distributed_update);
    for (std::size_t i = 0; i < owned.size(); ++i)
      REQUIRE(std::abs(distributed_update[i] - serial_update[owned[i]]) < 1e-8);
    REQUIRE(distributed.collective_bytes ==
            static_cast<std::size_t>(distributed.applications) * 2 *
                distributed.coarse().size() * sizeof(double));
    REQUIRE(distributed.coarse().size() < grid.size() / 4);
    auto rhs = random_vector(grid.size());
    grid.project(rhs);
    auto full = factory(grid);
    std::vector<int> counts_owned(ranks), offsets(ranks), indices;
    MPI_Allgather(&count, 1, MPI_INT, counts_owned.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);
    int total = 0;
    for (int i = 0; i < ranks; ++i) {
      offsets[i] = total;
      total += counts_owned[i];
    }
    indices.resize(total);
    MPI_Allgatherv(owned.data(), count, MPI_INT, indices.data(), counts_owned.data(),
                   offsets.data(), MPI_INT, MPI_COMM_WORLD);
    auto action = [&](const std::vector<double> &x, std::vector<double> &y) {
      std::vector<double> gathered(indices.size()), complete(grid.size()), product;
      MPI_Allgatherv(x.data(), x.size(), MPI_DOUBLE, gathered.data(),
                     counts_owned.data(), offsets.data(), MPI_DOUBLE,
                     MPI_COMM_WORLD);
      for (std::size_t i = 0; i < indices.size(); ++i)
        complete[indices[i]] = gathered[i];
      full(complete, product);
      y.resize(owned.size());
      for (std::size_t i = 0; i < owned.size(); ++i) y[i] = product[owned[i]];
    };
    std::vector<double> b(owned.size()), x(owned.size()), product,
        diagonal(owned.size());
    for (std::size_t i = 0; i < owned.size(); ++i) {
      b[i] = rhs[owned[i]];
      diagonal[i] = full.diagonal[owned[i]];
    }
    KrylovControls controls;
    controls.relative_tolerance = 1e-9;
    controls.maximum_iterations = 200;
    auto solved =
        gmres(ops, action, b, x, controls, [&](const auto &r, auto &z, int) {
          two_level(distributed, action, diagonal, r, z);
        });
    REQUIRE(solved.converged());
    REQUIRE(solved.iterations < 80);
    action(x, product);
    require_true_residual(ops, b, product);
  }
  StructuredGrid grid{{9, 9, 9}, 1};
  auto domain = pfc::domain::create(pfc::Int3{9, 9, 9});
  auto decomposition = pfc::decomposition::create(domain, ranks);
  PeriodicPoissonAction poisson(decomposition, rank, MPI_COMM_WORLD);
  SharedCoarseCorrection correction(grid, poisson.owned_indices(), poisson_auxiliary,
                                    MPI_COMM_WORLD);
  auto global = random_vector(grid.size());
  grid.project(global);
  auto owned = poisson.owned_indices();
  auto reference = poisson_auxiliary(grid);
  std::vector<double> b(owned.size()), x(owned.size()), product,
      diagonal(owned.size());
  for (std::size_t i = 0; i < owned.size(); ++i) {
    b[i] = global[owned[i]];
    diagonal[i] = reference.diagonal[owned[i]];
  }
  KrylovControls controls;
  controls.relative_tolerance = 1e-8;
  controls.maximum_iterations = 200;
  auto solved =
      gmres(ops, poisson, b, x, controls, [&](const auto &r, auto &z, int) {
        two_level(correction, poisson, diagonal, r, z);
      });
  REQUIRE(solved.converged());
  REQUIRE(solved.iterations < 80);
  poisson(x, product);
  require_true_residual(ops, b, product);
}
