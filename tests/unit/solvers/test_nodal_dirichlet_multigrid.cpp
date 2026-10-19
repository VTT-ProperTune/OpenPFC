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
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/solvers/nodal_dirichlet_nested_geometry.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_dirichlet_multigrid.hpp>
#include <openpfc/solvers/nodal_dirichlet_coarse.hpp>

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
  if (nproc == 3) return {{1, 1, 3}};
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
  controls.maximum_iterations = 120;
  auto status = pfc::solvers::cg(
      ops, [&](const Ops::Vector &v, Ops::Vector &out) {
        mg.apply_operator(v[0], out[0]);
      }, rhs, solution, controls,
      [&](const Ops::Vector &r, Ops::Vector &out, int) {
        mg.apply(r[0], out[0]);
      });
  REQUIRE(status.converged());
  REQUIRE(status.iterations <= 75);
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
    // A second, genuinely distributed Q1 hierarchy level exercises
    // P and P^T across more than one nonperiodic partition boundary.
    Fixture refined({31, 31, 31},
                    {1.0 / 32, 1.3 / 32, 0.7 / 32},
                    grid, MPI_COMM_WORLD);
    check_problem(refined, 1.0, 0.0);
    if (nproc <= 2) {
      Fixture refined_tall({31, 31, 63},
                           {1.0 / 32, 1.3 / 32, 0.7 / 64},
                           grid, MPI_COMM_WORLD);
      check_problem(refined_tall, 1.0, 0.0);
    }
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


TEST_CASE("Dirichlet shared Q1 row kernels ignore poisoned physical halos",
          "[nodal_dirichlet][kernel]") {
  namespace k = pfc::solvers::nmg;
  static_assert(std::is_trivially_copyable_v<k::dirichlet::Weights>);
  const pfc::Int3 n{15, 15, 15};
  const pfc::Real3 h{1.0 / 16, 1.3 / 16, 0.7 / 16};
  constexpr double diffusion = 0.73, reaction = 0.41;
  const double spacing[3] = {h[0], h[1], h[2]};
  const auto w = k::dirichlet::make_weights(spacing, diffusion, reaction);
  k::Level L;
  for (int d = 0; d < 3; ++d) {
    L.n[d] = n[d];
    L.gn[d] = n[d];
    L.low[d] = 0;
  }
  // Poison even the physical exterior ghost cells. A Dirichlet row must not
  // read them: fixed boundary nodes are eliminated, not MPI ghost unknowns.
  std::vector<double> padded(k::padded_size(L, k::dirichlet::kHalo), 9999.0);
  std::vector<double> input(count(n));
  for (int z = 0; z < n[2]; ++z)
    for (int y = 0; y < n[1]; ++y)
      for (int x = 0; x < n[0]; ++x) {
        const auto v = sample_a(x, y, z);
        input[id(n, x, y, z)] = v;
        padded[k::index(L, k::dirichlet::kHalo, x, y, z)] = v;
      }
  const auto assembled =
      independent_action(n, h, input, diffusion, reaction);
  double err = 0.0, scale = 0.0;
  for (int z = 0; z < n[2]; ++z)
    for (int y = 0; y < n[1]; ++y)
      for (int x = 0; x < n[0]; ++x) {
        const auto a = k::dirichlet::action(L, w, padded.data(), x, y, z);
        const auto b = assembled[id(n, x, y, z)];
        err = std::max(err, std::abs(a - b));
        scale = std::max(scale, std::abs(b));
        const auto updated = k::dirichlet::jacobi(
            L, w, padded.data(), input[id(n, x, y, z)], 0.25, 0.17,
            x, y, z);
        REQUIRE(std::abs(updated -
                         (input[id(n, x, y, z)] + 0.17 * (0.25 - a))) <
                1e-12 * std::max(1.0, std::abs(updated)));
      }
  REQUIRE(err <= 2e-12 * std::max(1.0, scale));
}

TEST_CASE("Dirichlet shared Q1 transfer kernels satisfy exact P transpose",
          "[nodal_dirichlet][kernel]") {
  namespace k = pfc::solvers::nmg;
  for (const pfc::Int3 nf : {pfc::Int3{15, 15, 15},
                              pfc::Int3{15, 15, 31}}) {
    k::Level F, C;
    for (int d = 0; d < 3; ++d) {
      F.n[d] = nf[d];
      F.gn[d] = nf[d];
      C.n[d] = (nf[d] - 1) / 2;
      C.gn[d] = C.n[d];
    }
    std::vector<double> f(k::padded_size(F, k::dirichlet::kHalo), 3333.0);
    std::vector<double> c(k::padded_size(C, k::dirichlet::kHalo), 4444.0);
    for (int z = 0; z < F.n[2]; ++z)
      for (int y = 0; y < F.n[1]; ++y)
        for (int x = 0; x < F.n[0]; ++x)
          f[k::index(F, k::dirichlet::kHalo, x, y, z)] =
              sample_a(x, y, z);
    for (int z = 0; z < C.n[2]; ++z)
      for (int y = 0; y < C.n[1]; ++y)
        for (int x = 0; x < C.n[0]; ++x)
          c[k::index(C, k::dirichlet::kHalo, x, y, z)] =
              sample_b(x, y, z);
    double fine_dot = 0.0, coarse_dot = 0.0;
    for (int z = 0; z < F.n[2]; ++z)
      for (int y = 0; y < F.n[1]; ++y)
        for (int x = 0; x < F.n[0]; ++x)
          fine_dot += sample_a(x, y, z) *
                      k::dirichlet::prolong_p(C, c.data(), x, y, z);
    for (int z = 0; z < C.n[2]; ++z)
      for (int y = 0; y < C.n[1]; ++y)
        for (int x = 0; x < C.n[0]; ++x)
          coarse_dot += sample_b(x, y, z) *
                        k::dirichlet::restrict_pt(F, f.data(), x, y, z);
    REQUIRE(std::abs(fine_dot - coarse_dot) <=
            1e-11 * std::max({1.0, std::abs(fine_dot),
                               std::abs(coarse_dot)}));
  }
}

TEST_CASE("Dirichlet shared Q1 row kernel honors shifted local bricks",
          "[nodal_dirichlet][kernel]") {
  namespace k = pfc::solvers::nmg;
  const pfc::Int3 global{15, 15, 31};
  const pfc::Real3 h{0.0625, 0.08125, 0.021875};
  std::vector<double> input(count(global));
  for (int z = 0; z < global[2]; ++z)
    for (int y = 0; y < global[1]; ++y)
      for (int x = 0; x < global[0]; ++x)
        input[id(global, x, y, z)] = sample_a(x, y, z);
  const auto assembled = independent_action(global, h, input, 1.0, 0.0);
  k::Level L;
  const int low[3] = {3, 4, 9}, local[3] = {7, 6, 5};
  for (int d = 0; d < 3; ++d) {
    L.n[d] = local[d];
    L.gn[d] = global[d];
    L.low[d] = low[d];
  }
  std::vector<double> ghost(k::padded_size(L, k::dirichlet::kHalo), 7777.0);
  for (int z = -2; z < local[2] + 2; ++z)
    for (int y = -2; y < local[1] + 2; ++y)
      for (int x = -2; x < local[0] + 2; ++x) {
        const int gx = low[0] + x, gy = low[1] + y, gz = low[2] + z;
        if (gx < 0 || gy < 0 || gz < 0 || gx >= global[0] ||
            gy >= global[1] || gz >= global[2]) continue;
        ghost[k::index(L, k::dirichlet::kHalo, x, y, z)] =
            input[id(global, gx, gy, gz)];
      }
  const double spacing[3] = {h[0], h[1], h[2]};
  const auto weights = k::dirichlet::make_weights(spacing, 1.0, 0.0);
  for (int z = 0; z < local[2]; ++z)
    for (int y = 0; y < local[1]; ++y)
      for (int x = 0; x < local[0]; ++x)
        REQUIRE(std::abs(k::dirichlet::action(
                             L, weights, ghost.data(), x, y, z) -
                         assembled[id(global, low[0] + x, low[1] + y,
                                      low[2] + z)]) < 2e-12);
}


TEST_CASE("Dirichlet FieldHalo MPI stage matches independent element Q1",
          "[nodal_dirichlet][MPI]") {
  // HostSpace executes the identical owned-field and halo staging contract
  // required by the GPU FieldHalo adapter, but this CI test needs no GPU.
  // The reference is independent element-by-element Gauss Q1 integration.
  namespace k = pfc::solvers::nmg;
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  const pfc::Int3 nodes{31, 31, 31};
  const pfc::Real3 h{1.0 / 32, 1.3 / 32, 0.7 / 32};
  auto domain = pfc::domain::with_spacing(
      nodes, h, {false, false, false});

  for (const auto process_grid : process_grids(ranks)) {
    const auto decomp = pfc::decomposition::create(domain, process_grid);
    const auto box = pfc::decomposition::local_box(decomp, rank);
    Ops operations(domain, box, 1, k::dirichlet::kHalo, MPI_COMM_WORLD);
    pfc::comm::HaloExchangeOptions options;
    options.connectivity = pfc::comm::HaloConnectivity::Full;
    pfc::solvers::FieldHalo<pfc::HostSpace> halo(
        operations, decomp, rank, MPI_COMM_WORLD, options);
    Field input(domain, box, 1), output(domain, box, 1);
    auto &padded = halo.scratch()[0];
    k::Level shape;
    for (int axis = 0; axis < 3; ++axis) {
      shape.n[axis] = box.size[axis];
      shape.low[axis] = box.low[axis];
      shape.gn[axis] = nodes[axis];
    }
    const double element_spacing[3] = {h[0], h[1], h[2]};
    const auto weights = k::dirichlet::make_weights(
        element_spacing, 0.75, 0.32);

    for (const auto sample_kind : {0, 1}) {
      fill(input, sample_kind == 0 ? Function(sample_a)
                                   : Function(sample_b));
      // Physical exterior halos deliberately contain stale values. Even
      // after an MPI exchange, the Q1 row must mask physical boundaries.
      std::fill(padded.vec().begin(), padded.vec().end(), 31415.0);
      owned(input, [&](int i, int j, int kk) {
        padded(i, j, kk) = input(i, j, kk);
      });
      halo.exchange();
      owned(input, [&](int i, int j, int kk) {
        output(i, j, kk) = k::dirichlet::action(
            shape, weights, padded.data(), i, j, kk);
      });

      const auto global_input = gather(input, MPI_COMM_WORLD);
      const auto global_output = gather(output, MPI_COMM_WORLD);
      const auto assembled = independent_action(
          nodes, h, global_input, 0.75, 0.32);
      double worst = 0.0, scale = 0.0;
      for (std::size_t id = 0; id < assembled.size(); ++id) {
        worst = std::max(worst, std::abs(global_output[id] - assembled[id]));
        scale = std::max(scale, std::abs(assembled[id]));
      }
      INFO("MPI process grid " << process_grid[0] << "x"
                               << process_grid[1] << "x" << process_grid[2]
                               << ", input kind " << sample_kind);
      REQUIRE(worst <= 2e-11 * std::max(1.0, scale));
    }
  }
}


TEST_CASE("Dirichlet nested OpenPFC FieldHalo P/Pt preserve adjointness on MPI",
          "[nodal_dirichlet][MPI]") {
  namespace k = pfc::solvers::nmg;
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  // Qualify both isotropic node counts and 2:1 element-count geometry.
  // These cover the same Cartesian aspect family as DiscretizationLab's
  // independent physical-Galerkin acceptance without allocating HPC grids.
  for (const pfc::Int3 fine_size :
       {pfc::Int3{31, 31, 31}, pfc::Int3{31, 31, 63}}) {
    const pfc::Int3 coarse_size{(fine_size[0] - 1) / 2,
                                (fine_size[1] - 1) / 2,
                                (fine_size[2] - 1) / 2};
    const pfc::Real3 hf{1.0 / 32, 1.3 / 32,
                        0.7 / static_cast<double>(fine_size[2] + 1)};
    const pfc::Real3 hc{2 * hf[0], 2 * hf[1], 2 * hf[2]};
    const auto fine_domain = pfc::domain::with_spacing(
        fine_size, hf, {false, false, false});
    const auto coarse_domain = pfc::domain::with_spacing(
        coarse_size, hc, {false, false, false});

  for (const auto process_grid : process_grids(ranks)) {
    const auto df = pfc::decomposition::create(fine_domain, process_grid);
    const auto dc = pfc::decomposition::create(coarse_domain, process_grid);
    REQUIRE_NOTHROW(k::dirichlet::validate_nested_transfer_reach(df, dc));
    const auto bf = pfc::decomposition::local_box(df, rank);
    const auto bc = pfc::decomposition::local_box(dc, rank);
    Ops fops(fine_domain, bf, 1, 2, MPI_COMM_WORLD);
    Ops cops(coarse_domain, bc, 1, 2, MPI_COMM_WORLD);
    pfc::comm::HaloExchangeOptions options;
    options.connectivity = pfc::comm::HaloConnectivity::Full;
    pfc::solvers::FieldHalo<pfc::HostSpace> fhalo(
        fops, df, rank, MPI_COMM_WORLD, options);
    pfc::solvers::FieldHalo<pfc::HostSpace> chalo(
        cops, dc, rank, MPI_COMM_WORLD, options);
    Field rf(fine_domain, bf, 0), xc(coarse_domain, bc, 0);
    Field restricted(coarse_domain, bc, 0), prolonged(fine_domain, bf, 0);
    fill(rf, sample_a);
    fill(xc, sample_b);

    auto &fh = fhalo.scratch()[0];
    std::fill(fh.vec().begin(), fh.vec().end(), 9999.0);
    owned(rf, [&](int i, int j, int z) { fh(i, j, z) = rf(i, j, z); });
    fhalo.exchange();

    auto &ch = chalo.scratch()[0];
    std::fill(ch.vec().begin(), ch.vec().end(), 11111.0);
    owned(xc, [&](int i, int j, int z) { ch(i, j, z) = xc(i, j, z); });
    chalo.exchange();

    k::Level F, C;
    for (int d = 0; d < 3; ++d) {
      F.n[d] = bf.size[d];
      F.low[d] = bf.low[d];
      F.gn[d] = fine_size[d];
      C.n[d] = bc.size[d];
      C.low[d] = bc.low[d];
      C.gn[d] = coarse_size[d];
    }
    owned(restricted, [&](int i, int j, int z) {
      restricted(i, j, z) = k::dirichlet::restrict_pt(
          F, fh.data(), bc.low[0] + i, bc.low[1] + j, bc.low[2] + z);
    });
    owned(prolonged, [&](int i, int j, int z) {
      prolonged(i, j, z) = k::dirichlet::prolong_p(
          C, ch.data(), bf.low[0] + i, bf.low[1] + j, bf.low[2] + z);
    });

    const auto all_f = gather(rf, MPI_COMM_WORLD);
    const auto all_c = gather(xc, MPI_COMM_WORLD);
    const auto all_r = gather(restricted, MPI_COMM_WORLD);
    const auto all_p = gather(prolonged, MPI_COMM_WORLD);

    // Independent Galerkin transfer oracle: assemble R=P^T as a scatter
    // from each fine-node coefficient into up to eight coarse nodes.
    std::vector<double> expected_r(count(coarse_size), 0.0);
    double largest_p_error = 0.0;
    for (int fk = 0; fk < fine_size[2]; ++fk)
      for (int fj = 0; fj < fine_size[1]; ++fj)
        for (int fi = 0; fi < fine_size[0]; ++fi) {
          const int f[3] = {fi + 1, fj + 1, fk + 1};
          const int base[3] = {f[0] / 2 - 1, f[1] / 2 - 1, f[2] / 2 - 1};
          const int count_d[3] = {(f[0] & 1) ? 2 : 1,
                                  (f[1] & 1) ? 2 : 1,
                                  (f[2] & 1) ? 2 : 1};
          const double weight = ((f[0] & 1) ? 0.5 : 1.0) *
                                ((f[1] & 1) ? 0.5 : 1.0) *
                                ((f[2] & 1) ? 0.5 : 1.0);
          double p_ref = 0.0;
          const double fine_value = all_f[id(fine_size, fi, fj, fk)];
          for (int dz = 0; dz < count_d[2]; ++dz)
            for (int dy = 0; dy < count_d[1]; ++dy)
              for (int dx = 0; dx < count_d[0]; ++dx) {
                const int ci = base[0] + dx, cj = base[1] + dy,
                          ck = base[2] + dz;
                if (ci < 0 || cj < 0 || ck < 0 ||
                    ci >= coarse_size[0] || cj >= coarse_size[1] ||
                    ck >= coarse_size[2])
                  continue;
                const auto ci_index = id(coarse_size, ci, cj, ck);
                p_ref += weight * all_c[ci_index];
                expected_r[ci_index] += weight * fine_value;
              }
          largest_p_error = std::max(
              largest_p_error,
              std::abs(p_ref - all_p[id(fine_size, fi, fj, fk)]));
        }

    double largest_r_error = 0.0, fine_dot = 0.0, coarse_dot = 0.0;
    for (std::size_t i = 0; i < all_r.size(); ++i) {
      largest_r_error = std::max(
          largest_r_error, std::abs(all_r[i] - expected_r[i]));
      coarse_dot += all_c[i] * all_r[i];
    }
    for (std::size_t i = 0; i < all_f.size(); ++i)
      fine_dot += all_f[i] * all_p[i];

    INFO("OpenPFC nested rank layout " << process_grid[0] << "x"
         << process_grid[1] << "x" << process_grid[2]);
    REQUIRE(largest_r_error <= 1e-11);
    REQUIRE(largest_p_error <= 1e-12);
    REQUIRE(std::abs(fine_dot - coarse_dot) <=
            1e-11 * std::max({1.0, std::abs(fine_dot),
                               std::abs(coarse_dot)}));
  }
  }
}

TEST_CASE("Dirichlet nested transfer rejects physically mismatched geometry",
          "[nodal_dirichlet][MPI]") {
  namespace k = pfc::solvers::nmg;
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const pfc::Int3 dims{1, 1, ranks};
  const auto fine = pfc::decomposition::create(
      pfc::domain::with_spacing({31, 31, 31},
                                {0.03125, 0.03125, 0.03125},
                                {false, false, false}), dims);
  const auto invalid_coarse = pfc::decomposition::create(
      pfc::domain::with_spacing({15, 15, 15},
                                {0.0625, 0.075, 0.0625},
                                {false, false, false}), dims);
  REQUIRE_THROWS_AS(
      k::dirichlet::validate_nested_transfer_reach(fine, invalid_coarse),
      std::invalid_argument);
}


TEST_CASE("Dirichlet coarse Q1 Cholesky matches independent Gauss FEM",
          "[nodal_dirichlet][coarse]") {
  namespace k = pfc::solvers::nmg;
  using Factor = k::dirichlet::DenseCoarseCholesky;
  for (const pfc::Int3 n : {pfc::Int3{3, 3, 3},
                             pfc::Int3{3, 3, 7},
                             pfc::Int3{7, 7, 7}}) {
    const pfc::Real3 h{1.0 / (n[0] + 1), 1.3 / (n[1] + 1),
                       0.7 / (n[2] + 1)};
    const double spacing[3] = {h[0], h[1], h[2]};
    const auto N = count(n);
    for (const double reaction : {0.0, 0.3}) {
      const double diffusion = 1.17;
      const auto weights = k::dirichlet::make_weights(
          spacing, diffusion, reaction);
      Factor coarse(n, weights, N);
      REQUIRE(coarse.unknowns() == N);
      REQUIRE(coarse.free_nodes() == n);

      std::vector<double> rhs(N), x;
      for (int sample = 0; sample < 2; ++sample) {
        for (int kz = 0; kz < n[2]; ++kz)
          for (int jy = 0; jy < n[1]; ++jy)
            for (int ix = 0; ix < n[0]; ++ix)
              rhs[id(n, ix, jy, kz)] =
                  sample == 0 ? sample_a(ix, jy, kz) :
                                sample_b(ix, jy, kz);

        coarse.solve(rhs, x);
        const auto assembled = independent_action(
            n, h, x, diffusion, reaction);
        double err2 = 0.0, rhs2 = 0.0, energy = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
          err2 += (assembled[i] - rhs[i]) *
                  (assembled[i] - rhs[i]);
          rhs2 += rhs[i] * rhs[i];
          energy += x[i] * assembled[i];
        }
        INFO("independent coarse FEM, nodes " << n[0] << "x"
             << n[1] << "x" << n[2] << ", reaction " << reaction
             << ", sample " << sample);
        REQUIRE(energy > 0.0);
        REQUIRE(std::sqrt(err2 / rhs2) < 2e-11);
      }
      REQUIRE_THROWS_AS(coarse.solve(rhs, rhs),
                        std::invalid_argument);
    }
  }
}

TEST_CASE("Dirichlet coarse factor fails closed on bad SPD or memory",
          "[nodal_dirichlet][coarse]") {
  namespace k = pfc::solvers::nmg;
  using Factor = k::dirichlet::DenseCoarseCholesky;
  const double spacing[3] = {0.1, 0.12, 0.08};
  const auto w = k::dirichlet::make_weights(spacing, 1.0, 0.0);
  REQUIRE_THROWS_AS(Factor(pfc::Int3{7, 7, 7}, w, 342),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Factor(pfc::Int3{0, 7, 7}, w, 1024),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Factor(pfc::Int3{3, 3, 3}, w, 0),
                    std::invalid_argument);

  auto asymmetric = w;
  asymmetric.v[k::offset_index(1, 0, 0)] += 0.01;
  REQUIRE_THROWS_AS(Factor(pfc::Int3{3, 3, 3}, asymmetric, 100),
                    std::invalid_argument);
  auto not_spd = w;
  for (double &v : not_spd.v) v = 0.0;
  REQUIRE_THROWS_AS(Factor(pfc::Int3{3, 3, 3}, not_spd, 100),
                    std::runtime_error);
}
