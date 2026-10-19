// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_dirichlet_multigrid.hpp>

namespace {

using pfc::HostSpace;
using MG = pfc::solvers::NodalDirichletMultigrid;
using Field = MG::FieldT;
using Ops = pfc::solvers::FieldOperations<HostSpace>;

using Function = std::function<double(int, int, int)>;

std::size_t count(const pfc::Int3 &s) {
  return static_cast<std::size_t>(s[0]) * s[1] * s[2];
}

std::size_t id(const pfc::Int3 &s, int i, int j, int k) {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(s[0]) *
             (static_cast<std::size_t>(j) + static_cast<std::size_t>(s[1]) * k);
}

template <typename Callback> void owned(const Field &field, Callback fun) {
  const auto n = field.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i)
        fun(i, j, k);
}

void fill(Field &field, const Function &fn) {
  owned(field, [&](int i, int j, int k) {
    const auto g = field.global(i, j, k);
    field(i, j, k) = fn(g[0], g[1], g[2]);
  });
}

double sample_a(int i, int j, int k) {
  return std::sin(.191 * (i + 1) + .117 * (j + 1)) +
         0.31 * std::cos(.137 * (k + 1)) +
         0.1 * std::sin(.023 * (i + 1) * (k + 1));
}
double sample_b(int i, int j, int k) {
  return std::cos(.083 * (i + 1) * (j + 1)) +
         0.3 * std::sin(.217 * (k + 1) + .037 * (j + 1)) -
         0.13 * std::cos(.079 * (i + 1));
}

std::vector<double> gather(const Field &f, MPI_Comm comm) {
  const auto global = pfc::domain::get_size(f.domain());
  std::vector<double> local(count(global), 0.0), total(count(global), 0.0);
  owned(f, [&](int i, int j, int k) {
    const auto g = f.global(i, j, k);
    local[id(global, g[0], g[1], g[2])] = f(i, j, k);
  });
  MPI_Allreduce(local.data(), total.data(), static_cast<int>(total.size()),
                MPI_DOUBLE, MPI_SUM, comm);
  return total;
}

double owned_dot(const Field &a, const Field &b, MPI_Comm comm) {
  double local = 0.0;
  owned(a, [&](int i, int j, int k) { local += a(i, j, k) * b(i, j, k); });
  double result = 0.0;
  MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
  return result;
}

std::vector<pfc::Int3> process_grids(int nproc) {
  if (nproc == 1) return {{1, 1, 1}};
  if (nproc == 2) return {{2, 1, 1}, {1, 1, 2}};
  if (nproc == 4) return {{2, 2, 1}, {2, 1, 2}};
  return {};
}

struct Fixture {
  pfc::Int3 nodes;
  pfc::Real3 h;
  pfc::Int3 proc_grid;
  MPI_Comm comm;
  int rank;
  pfc::Domain domain;
  pfc::decomposition::Decomposition decomposition;

  Fixture(pfc::Int3 n, pfc::Real3 spacing, pfc::Int3 grid, MPI_Comm c)
      : nodes(n), h(spacing), proc_grid(grid), comm(c), rank([&]() {
          int value = 0;
          MPI_Comm_rank(c, &value);
          return value;
        }()),
        domain(pfc::domain::with_spacing(n, spacing,
                                        {false, false, false})),
        decomposition(pfc::decomposition::create(domain, grid)) {}

  Field field() const {
    return Field(domain, pfc::decomposition::local_box(decomposition, rank), 1);
  }
};

/// Independently integrate one Q1 element with two Gauss points per axis.
/// No row-stencil formula and no reference to nmg::set_element_data.
std::array<std::array<double, 8>, 8> element_matrix(
    const pfc::Real3 &h, double diffusion, double reaction) {
  const double p[2] = {0.5 - 0.5 / std::sqrt(3.0),
                       0.5 + 0.5 / std::sqrt(3.0)};
  const double volume = h[0] * h[1] * h[2];
  std::array<std::array<double, 8>, 8> ke{};
  for (int qz = 0; qz < 2; ++qz)
    for (int qy = 0; qy < 2; ++qy)
      for (int qx = 0; qx < 2; ++qx) {
        const double q[3] = {p[qx], p[qy], p[qz]};
        double basis[8], gradients[8][3];
        for (int a = 0; a < 8; ++a) {
          double shape[3], derivative[3];
          for (int d = 0; d < 3; ++d) {
            const bool high = (a >> d) & 1;
            shape[d] = high ? q[d] : 1.0 - q[d];
            derivative[d] = (high ? 1.0 : -1.0) / h[d];
          }
          basis[a] = shape[0] * shape[1] * shape[2];
          gradients[a][0] = derivative[0] * shape[1] * shape[2];
          gradients[a][1] = shape[0] * derivative[1] * shape[2];
          gradients[a][2] = shape[0] * shape[1] * derivative[2];
        }
        for (int a = 0; a < 8; ++a)
          for (int b = 0; b < 8; ++b) {
            double stiffness = 0.0;
            for (int d = 0; d < 3; ++d)
              stiffness += gradients[a][d] * gradients[b][d];
            ke[a][b] += (volume / 8.0) *
                        (diffusion * stiffness +
                         reaction * basis[a] * basis[b]);
          }
      }
  return ke;
}

/// Independent assembled-action test: physically there are (free+1) elements
/// on each axis, and fixed Dirichlet nodes carry value zero.
std::vector<double> independent_action(const pfc::Int3 &n,
                                       const pfc::Real3 &h,
                                       const std::vector<double> &field,
                                       double diffusion, double reaction) {
  const auto ke = element_matrix(h, diffusion, reaction);
  std::vector<double> result(count(n), 0.0);
  for (int ez = 0; ez <= n[2]; ++ez)
    for (int ey = 0; ey <= n[1]; ++ey)
      for (int ex = 0; ex <= n[0]; ++ex) {
        int physical[8][3];
        bool free[8];
        double values[8]{};
        for (int a = 0; a < 8; ++a) {
          for (int d = 0; d < 3; ++d) {
            const int e = d == 0 ? ex : (d == 1 ? ey : ez);
            physical[a][d] = e + ((a >> d) & 1) - 1;
          }
          free[a] = true;
          for (int d = 0; d < 3; ++d)
            free[a] = free[a] && physical[a][d] >= 0 &&
                      physical[a][d] < n[d];
          if (free[a])
            values[a] = field[id(n, physical[a][0], physical[a][1],
                                physical[a][2])];
        }
        for (int a = 0; a < 8; ++a) {
          if (!free[a]) continue;
          double sum = 0.0;
          for (int b = 0; b < 8; ++b) sum += ke[a][b] * values[b];
          result[id(n, physical[a][0], physical[a][1], physical[a][2])] += sum;
        }
      }
  return result;
}

void check_problem(const Fixture &f, double diffusion, double reaction) {
  MG::Controls opts;
  opts.diffusion = diffusion;
  opts.reaction = reaction;
  MG mg(f.decomposition, f.rank, f.comm, opts);
  REQUIRE(mg.levels() >= 2);
  const std::size_t expected_coarse =
      f.nodes[2] == f.nodes[0] ? 343 : 735;
  REQUIRE(mg.coarse_unknowns() == expected_coarse);
  REQUIRE(mg.local_unknowns() == count(
      pfc::decomposition::local_box(f.decomposition, f.rank).size));

  Field x = f.field(), y = f.field();
  fill(x, sample_a);
  mg.apply_operator(x, y);
  const auto gathered_x = gather(x, f.comm);
  const auto gathered_y = gather(y, f.comm);
  const auto oracle = independent_action(f.nodes, f.h, gathered_x,
                                         diffusion, reaction);
  double difference = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < oracle.size(); ++i) {
    difference = std::max(difference, std::abs(oracle[i] - gathered_y[i]));
    scale = std::max(scale, std::abs(oracle[i]));
  }
  INFO("grid " << f.nodes[0] << "x" << f.nodes[1] << "x" << f.nodes[2]
               << " ranks " << f.proc_grid[0] << "x" << f.proc_grid[1] << "x"
               << f.proc_grid[2] << " sigma=" << reaction);
  REQUIRE(difference <= 2e-11 * std::max(scale, 1.0));

  Field a = f.field(), b = f.field(), ma = f.field(), mb = f.field();
  fill(a, sample_a);
  fill(b, sample_b);
  mg.apply(a, ma);
  mg.apply(b, mb);
  const double ab = owned_dot(a, mb, f.comm);
  const double ba = owned_dot(b, ma, f.comm);
  REQUIRE(std::abs(ab - ba) <=
          1e-9 * std::max({1.0, std::abs(ab), std::abs(ba)}));
  REQUIRE(owned_dot(a, ma, f.comm) > 0.0);
  REQUIRE(owned_dot(b, mb, f.comm) > 0.0);

  // Use OpenPFC's standard owned-only, MPI-reduced vector/Krylov interfaces.
  const auto box = pfc::decomposition::local_box(f.decomposition, f.rank);
  Ops ops(f.domain, box, 1, 1, f.comm);
  auto rhs = ops.make(ops.size()), solution = ops.make(ops.size());
  mg.apply_operator(x, rhs[0]);
  pfc::solvers::KrylovControls controls;
  controls.relative_tolerance = 1e-10;
  controls.maximum_iterations = 90;
  auto status = pfc::solvers::cg(
      ops, [&](const Ops::Vector &v, Ops::Vector &out) {
        mg.apply_operator(v[0], out[0]);
      }, rhs, solution, controls,
      [&](const Ops::Vector &r, Ops::Vector &out, int) {
        mg.apply(r[0], out[0]);
      });
  REQUIRE(status.converged());
  REQUIRE(status.iterations <= 40);
  const auto computed = gather(solution[0], f.comm);
  double err2 = 0.0, ref2 = 0.0;
  for (std::size_t i = 0; i < computed.size(); ++i) {
    err2 += (computed[i] - gathered_x[i]) * (computed[i] - gathered_x[i]);
    ref2 += gathered_x[i] * gathered_x[i];
  }
  REQUIRE(std::sqrt(err2 / ref2) <= 1e-8);
  INFO("PCG iterations " << status.iterations);
}

} // namespace

TEST_CASE("Dirichlet Q1 OpenPFC V-cycle matches independent FEM and PCG",
          "[nodal_dirichlet][MPI]") {
  int nproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  for (const pfc::Int3 grid : process_grids(nproc)) {
    // Same physical box, independent element spacings on each axis.
    Fixture cubic({15, 15, 15},
                  {1.0 / 16, 1.3 / 16, 0.7 / 16},
                  grid, MPI_COMM_WORLD);
    for (double reaction : {0.0, 0.4})
      check_problem(cubic, 1.0, reaction);
    Fixture tall({15, 15, 31},
                 {1.0 / 16, 1.3 / 16, 0.7 / 32},
                 grid, MPI_COMM_WORLD);
    check_problem(tall, 1.0, 0.0);
  }
}

TEST_CASE("Dirichlet Q1 OpenPFC rejects unsupported geometries",
          "[nodal_dirichlet][MPI]") {
  int rank = 0, nproc = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const auto grids = process_grids(nproc);
  if (grids.empty()) return;
  const auto grid = grids.front();
  const auto make = [&](pfc::Int3 n, pfc::Bool3 periodic) {
    return pfc::decomposition::create(
        pfc::domain::with_spacing(n, {0.1, 0.1, 0.1}, periodic), grid);
  };
  auto wrong = make({16, 15, 15}, {false, false, false});
  REQUIRE_THROWS_AS(MG(wrong, rank, MPI_COMM_WORLD), std::invalid_argument);
  auto periodic = make({15, 15, 15}, {true, false, false});
  REQUIRE_THROWS_AS(MG(periodic, rank, MPI_COMM_WORLD), std::invalid_argument);

  MG::Controls cap;
  cap.max_coarse_unknowns = 4;
  auto fine = make({15, 15, 15}, {false, false, false});
  REQUIRE_THROWS_AS(MG(fine, rank, MPI_COMM_WORLD, cap), std::invalid_argument);
}
