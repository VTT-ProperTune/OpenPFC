// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_nodal_multigrid.cpp
 * @brief Host NodalMultigrid: the Q1 action against an element-by-element
 *        assembly by Gauss quadrature, the Galerkin coarse correction against an
 *        independent P^T, V-cycle symmetry, grid-independent PCG, a 10^3 jump,
 *        sigma = 0, and decomposition invariance. Runs on any number of ranks;
 *        the `[MPI]` cases compare against a serial hierarchy on each rank.
 */

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/nodal_multigrid.hpp>

namespace {

using pfc::HostSpace;
using MG = pfc::solvers::NodalMultigrid<HostSpace>;
using FieldT = MG::FieldT;
using Ops = pfc::solvers::FieldOperations<HostSpace>;
using Coefficient = std::function<double(int, int, int)>;

struct Grid {
  pfc::Int3 n;
  pfc::Real3 h;
  MPI_Comm comm;
  int rank = 0, size = 1;
  pfc::decomposition::Decomposition decomp;
  Grid(pfc::Int3 n_, pfc::Real3 h_, MPI_Comm c = MPI_COMM_WORLD)
      : n(n_), h(h_), comm(c), decomp([&] {
          int p = 1;
          MPI_Comm_size(c, &p);
          return pfc::decomposition::create(
              pfc::domain::with_spacing(n_, h_, {true, true, true}), p);
        }()) {
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &size);
  }
  FieldT field() const {
    return FieldT(pfc::decomposition::domain(decomp),
                  pfc::decomposition::local_box(decomp, rank), 1);
  }
  std::size_t global_nodes() const {
    return static_cast<std::size_t>(n[0]) * n[1] * n[2];
  }
  int id(int i, int j, int k) const {
    const auto w = [](int v, int m) { return ((v % m) + m) % m; };
    return w(i, n[0]) + n[0] * (w(j, n[1]) + n[1] * w(k, n[2]));
  }
};

template <class Fn> void each(const FieldT &f, Fn &&fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) fn(i, j, k);
}

double dot(const Grid &g, const FieldT &a, const FieldT &b) {
  double s = 0.0;
  each(a, [&](int i, int j, int k) { s += a(i, j, k) * b(i, j, k); });
  double t = 0.0;
  MPI_Allreduce(&s, &t, 1, MPI_DOUBLE, MPI_SUM, g.comm);
  return t;
}

double random_value(unsigned seed, int gi, int gj, int gk) {
  std::mt19937 gen(seed +
                   7919u * static_cast<unsigned>(gi + 1000 * gj + 1000000 * gk));
  return std::uniform_real_distribution<double>(-1.0, 1.0)(gen);
}

/// Fill owned values from a function of the global index.
void set(FieldT &f, const Coefficient &fn) {
  each(f, [&](int i, int j, int k) {
    const auto g = f.global(i, j, k);
    f(i, j, k) = fn(g[0], g[1], g[2]);
  });
}

/// Global vector (every rank) from owned values.
std::vector<double> gather(const Grid &g, const FieldT &f) {
  std::vector<double> local(g.global_nodes(), 0.0), out(g.global_nodes(), 0.0);
  each(f, [&](int i, int j, int k) {
    const auto q = f.global(i, j, k);
    local[static_cast<std::size_t>(g.id(q[0], q[1], q[2]))] = f(i, j, k);
  });
  MPI_Allreduce(local.data(), out.data(), static_cast<int>(out.size()), MPI_DOUBLE,
                MPI_SUM, g.comm);
  return out;
}

/// Dense periodic Q1 matrix: element matrices by 2-point Gauss quadrature (exact
/// for the trilinear stiffness and mass), assembled element by element.
std::vector<double> assemble(const Grid &g, const Coefficient &a, double sigma) {
  const std::size_t N = g.global_nodes();
  std::vector<double> A(N * N, 0.0);
  const double gp[2] = {0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
  const double V = g.h[0] * g.h[1] * g.h[2];
  for (int ez = 0; ez < g.n[2]; ++ez)
    for (int ey = 0; ey < g.n[1]; ++ey)
      for (int ex = 0; ex < g.n[0]; ++ex) {
        double Ke[8][8] = {};
        for (int qz = 0; qz < 2; ++qz)
          for (int qy = 0; qy < 2; ++qy)
            for (int qx = 0; qx < 2; ++qx) {
              const double x[3] = {gp[qx], gp[qy], gp[qz]};
              double phi[8], grad[8][3];
              for (int p = 0; p < 8; ++p) {
                const int b[3] = {p & 1, (p >> 1) & 1, (p >> 2) & 1};
                double v[3], dv[3];
                for (int d = 0; d < 3; ++d) {
                  v[d] = b[d] ? x[d] : 1.0 - x[d];
                  dv[d] = (b[d] ? 1.0 : -1.0) / g.h[d];
                }
                phi[p] = v[0] * v[1] * v[2];
                grad[p][0] = dv[0] * v[1] * v[2];
                grad[p][1] = v[0] * dv[1] * v[2];
                grad[p][2] = v[0] * v[1] * dv[2];
              }
              const double w = V / 8.0;
              for (int p = 0; p < 8; ++p)
                for (int q = 0; q < 8; ++q)
                  Ke[p][q] += w * (a(ex, ey, ez) * (grad[p][0] * grad[q][0] +
                                                    grad[p][1] * grad[q][1] +
                                                    grad[p][2] * grad[q][2]) +
                                   sigma * phi[p] * phi[q]);
            }
        for (int p = 0; p < 8; ++p)
          for (int q = 0; q < 8; ++q) {
            const int r =
                g.id(ex + (p & 1), ey + ((p >> 1) & 1), ez + ((p >> 2) & 1));
            const int c =
                g.id(ex + (q & 1), ey + ((q >> 1) & 1), ez + ((q >> 2) & 1));
            A[static_cast<std::size_t>(r) * N + c] += Ke[p][q];
          }
      }
  return A;
}

/// Independent trilinear restriction P^T on a global vector (fine -> coarse).
std::vector<double> restrict_global(const Grid &g, const std::vector<double> &r) {
  const int m[3] = {g.n[0] / 2, g.n[1] / 2, g.n[2] / 2};
  std::vector<double> out(static_cast<std::size_t>(m[0]) * m[1] * m[2], 0.0);
  for (int k = 0; k < g.n[2]; ++k)
    for (int j = 0; j < g.n[1]; ++j)
      for (int i = 0; i < g.n[0]; ++i) {
        // Fine node i contributes to the coarse nodes it interpolates from.
        const int f[3] = {i, j, k};
        std::vector<std::pair<int, double>> c[3];
        for (int d = 0; d < 3; ++d) {
          if (f[d] % 2 == 0)
            c[d] = {{f[d] / 2, 1.0}};
          else
            c[d] = {{f[d] / 2, 0.5}, {(f[d] / 2 + 1) % m[d], 0.5}};
        }
        const double v = r[static_cast<std::size_t>(g.id(i, j, k))];
        for (auto [cz, wz] : c[2])
          for (auto [cy, wy] : c[1])
            for (auto [cx, wx] : c[0])
              out[static_cast<std::size_t>(cx + m[0] * (cy + m[1] * cz))] +=
                  wx * wy * wz * v;
      }
  return out;
}

FieldT coefficient_field(const Grid &g, const Coefficient &a) {
  FieldT c = g.field();
  set(c, a);
  return c;
}

Coefficient smooth_coefficient(const Grid &g, double alpha) {
  return [=](int i, int j, int k) {
    const double pi = 3.14159265358979323846;
    const double x = (i + 0.5) / g.n[0], y = (j + 0.5) / g.n[1],
                 z = (k + 0.5) / g.n[2];
    return 1.0 + alpha * std::cos(2 * pi * x) * std::cos(2 * pi * y) *
                     std::cos(2 * pi * z);
  };
}

Coefficient jump_coefficient(const Grid &g, double contrast) {
  return [=](int i, int j, int k) {
    const double x = (i + 0.5) / g.n[0], y = (j + 0.5) / g.n[1],
                 z = (k + 0.5) / g.n[2];
    const bool in =
        x > 0.25 && x < 0.75 && y > 0.25 && y < 0.75 && z > 0.25 && z < 0.75;
    return in ? contrast : 1.0;
  };
}

struct Solve {
  int iterations = 0;
  bool converged = false;
  std::vector<double> x;
};

/// PCG with the multigrid preconditioner on A x = b (b random, mean-free if
/// sigma = 0).
Solve pcg(const Grid &g, MG &mg, unsigned seed, bool mean_free, double rtol = 1e-8) {
  const auto dom = pfc::decomposition::domain(g.decomp);
  const auto box = pfc::decomposition::local_box(g.decomp, g.rank);
  Ops ops(dom, box, 1, 1, g.comm);
  auto b = ops.make(ops.size()), x = ops.make(ops.size());
  set(b[0], [&](int i, int j, int k) { return random_value(seed, i, j, k); });
  if (mean_free) {
    FieldT one = g.field();
    set(one, [](int, int, int) { return 1.0; });
    const double m = dot(g, b[0], one) / static_cast<double>(g.global_nodes());
    each(b[0], [&](int i, int j, int k) { b[0](i, j, k) -= m; });
  }
  pfc::solvers::KrylovControls c;
  c.relative_tolerance = rtol;
  c.maximum_iterations = 300;
  auto action = [&](const Ops::Vector &u, Ops::Vector &y) {
    mg.apply_operator(u[0], y[0]);
  };
  auto pre = [&](const Ops::Vector &r, Ops::Vector &z, int) {
    mg.apply(r[0], z[0]);
  };
  const auto res = pfc::solvers::cg(ops, action, b, x, c, pre);
  return Solve{res.iterations, res.converged(), gather(g, x[0])};
}

MG::Controls controls(double sigma, MG::Smoother s = MG::Smoother::Chebyshev) {
  MG::Controls c;
  c.sigma = sigma;
  c.smoother = s;
  return c;
}

} // namespace

TEST_CASE("NodalMultigrid action equals the assembled Q1 matrix",
          "[nodal_multigrid][MPI]") {
  for (double sigma : {0.0, 0.7}) {
    Grid g({8, 6, 4}, {0.3, 0.5, 0.2});
    const Coefficient a = [](int i, int j, int k) {
      return 1.0 + 0.9 * (0.5 + 0.5 * random_value(11u, i, j, k));
    };
    MG mg(g.decomp, g.rank, g.comm, controls(sigma));
    mg.update(coefficient_field(g, a));
    FieldT u = g.field(), y = g.field();
    set(u, [](int i, int j, int k) { return random_value(5u, i, j, k); });
    mg.apply_operator(u, y);
    const auto A = assemble(g, a, sigma);
    const auto U = gather(g, u), Y = gather(g, y);
    const std::size_t N = g.global_nodes();
    double err = 0, ref = 0;
    for (std::size_t r = 0; r < N; ++r) {
      double v = 0;
      for (std::size_t c = 0; c < N; ++c) v += A[r * N + c] * U[c];
      err = std::max(err, std::abs(v - Y[r]));
      ref = std::max(ref, std::abs(v));
    }
    REQUIRE(err <= 1e-12 * ref);
  }
}

TEST_CASE("NodalMultigrid coarse correction is the Galerkin projection",
          "[nodal_multigrid][MPI]") {
  // Two levels, no smoothing, exact coarse solve: z = P A_c^{-1} P^T b. The
  // restricted residual P^T (b - A z) vanishes iff A_c = P^T A P.
  Grid g({8, 8, 8}, {0.125, 0.1, 0.15});
  struct Case {
    Coefficient a;
    int galerkin_from;
    bool exact;
  };
  const Coefficient varying = [](int i, int j, int k) {
    return 1.0 + 0.8 * (0.5 + 0.5 * random_value(3u, i, j, k));
  };
  const Coefficient constant = [](int, int, int) { return 2.5; };
  for (const Case &cs : {Case{varying, 1, true}, Case{constant, 99, true},
                         Case{varying, 99, false}}) {
    auto c = controls(0.4);
    c.pre = c.post = 0;
    c.coarsest_extent = 4;
    c.galerkin_from_level = cs.galerkin_from;
    MG mg(g.decomp, g.rank, g.comm, c);
    REQUIRE(mg.levels() == 2);
    REQUIRE(mg.coarse_exact());
    mg.update(coefficient_field(g, cs.a));
    FieldT b = g.field(), z = g.field(), az = g.field();
    set(b, [](int i, int j, int k) { return random_value(9u, i, j, k); });
    mg.apply(b, z);
    mg.apply_operator(z, az);
    const auto B = gather(g, b), AZ = gather(g, az);
    std::vector<double> r(B.size());
    for (std::size_t q = 0; q < r.size(); ++q) r[q] = B[q] - AZ[q];
    const auto rc = restrict_global(g, r), bc = restrict_global(g, B);
    double num = 0, den = 0;
    for (std::size_t q = 0; q < rc.size(); ++q) {
      num += rc[q] * rc[q];
      den += bc[q] * bc[q];
    }
    const double rel = std::sqrt(num / den);
    if (cs.exact)
      REQUIRE(rel < 1e-12);
    else
      REQUIRE(rel > 1e-6);
  }
}

TEST_CASE("NodalMultigrid V-cycle is symmetric", "[nodal_multigrid][MPI]") {
  Grid g({16, 16, 16}, {1.0 / 16, 1.0 / 16, 1.0 / 16});
  for (auto sm : {MG::Smoother::Chebyshev, MG::Smoother::ColouredGaussSeidel})
    for (double sigma : {0.0, 1.0})
      for (int gal : {1, 99}) {
        auto c = controls(sigma, sm);
        c.galerkin_from_level = gal;
        c.agglomerate_max = 64;
        MG mg(g.decomp, g.rank, g.comm, c);
        mg.update(coefficient_field(g, jump_coefficient(g, 100.0)));
        FieldT r1 = g.field(), r2 = g.field(), z1 = g.field(), z2 = g.field();
        set(r1, [](int i, int j, int k) { return random_value(21u, i, j, k); });
        set(r2, [](int i, int j, int k) { return random_value(22u, i, j, k); });
        if (sigma == 0.0) { // the projected cycle is symmetric on mean-free vectors
          FieldT one = g.field();
          set(one, [](int, int, int) { return 1.0; });
          const double n = static_cast<double>(g.global_nodes());
          const double m1 = dot(g, r1, one) / n, m2 = dot(g, r2, one) / n;
          each(r1, [&](int i, int j, int k) {
            r1(i, j, k) -= m1;
            r2(i, j, k) -= m2;
          });
        }
        mg.apply(r1, z1);
        mg.apply(r2, z2);
        const double a = dot(g, z1, r2), b = dot(g, r1, z2);
        INFO("smoother " << static_cast<int>(sm) << " sigma " << sigma);
        REQUIRE(std::abs(a - b) <= 1e-11 * std::abs(a));
        REQUIRE(dot(g, z1, r1) > 0.0);
      }
}

TEST_CASE("NodalMultigrid PCG iterations do not grow with the grid",
          "[nodal_multigrid]") {
  std::vector<int> smooth, jump;
  for (int n : {16, 32, 64}) {
    Grid g({n, n, n}, {1.0 / n, 1.0 / n, 1.0 / n});
    MG ms(g.decomp, g.rank, g.comm, controls(1.0));
    ms.update(coefficient_field(g, smooth_coefficient(g, 0.9)));
    const auto s = pcg(g, ms, 1u, false);
    REQUIRE(s.converged);
    smooth.push_back(s.iterations);
    MG mj(g.decomp, g.rank, g.comm, controls(1.0));
    mj.update(coefficient_field(g, jump_coefficient(g, 1000.0)));
    const auto j = pcg(g, mj, 1u, false);
    REQUIRE(j.converged);
    jump.push_back(j.iterations);
  }
  WARN("PCG iterations, 16/32/64: smooth " << smooth[0] << " " << smooth[1] << " "
                                           << smooth[2] << " jump " << jump[0] << " "
                                           << jump[1] << " " << jump[2]);
  REQUIRE(smooth[2] <= smooth[0] + 2);
  REQUIRE(smooth[2] <= 15);
  REQUIRE(jump[2] <= jump[0] + 6);
}

TEST_CASE("NodalMultigrid handles the constant nullspace for sigma = 0",
          "[nodal_multigrid][MPI]") {
  Grid g({16, 16, 16}, {1.0 / 16, 1.0 / 16, 1.0 / 16});
  MG mg(g.decomp, g.rank, g.comm, controls(0.0));
  mg.update(coefficient_field(g, smooth_coefficient(g, 0.5)));
  const auto s = pcg(g, mg, 4u, true);
  REQUIRE(s.converged);
  REQUIRE(s.iterations <= 15);
  double mean = 0;
  for (double v : s.x) mean += v;
  REQUIRE(std::abs(mean / static_cast<double>(s.x.size())) < 1e-10);
}

TEST_CASE("NodalMultigrid is independent of the decomposition",
          "[nodal_multigrid][MPI]") {
  for (auto sm : {MG::Smoother::Chebyshev, MG::Smoother::ColouredGaussSeidel}) {
    Grid g({32, 32, 16}, {1.0 / 32, 1.0 / 32, 1.0 / 16});
    Grid serial({32, 32, 16}, {1.0 / 32, 1.0 / 32, 1.0 / 16}, MPI_COMM_SELF);
    auto c = controls(1.0, sm);
    c.agglomerate_max = 512; // force distributed levels and agglomeration
    MG mg(g.decomp, g.rank, g.comm, c);
    MG ms(serial.decomp, serial.rank, serial.comm, c);
    const auto a = jump_coefficient(g, 100.0);
    mg.update(coefficient_field(g, a));
    ms.update(coefficient_field(serial, a));
    REQUIRE(mg.levels() == ms.levels());
    for (int l = 0; l < mg.levels(); ++l)
      REQUIRE(std::abs(mg.level_info(l).lambda_max - ms.level_info(l).lambda_max) <=
              1e-10 * ms.level_info(l).lambda_max);
    const auto d = pcg(g, mg, 2u, false, 1e-10),
               s = pcg(serial, ms, 2u, false, 1e-10);
    REQUIRE(d.converged);
    REQUIRE(d.iterations == s.iterations);
    double err = 0, ref = 0;
    for (std::size_t q = 0; q < s.x.size(); ++q) {
      err = std::max(err, std::abs(d.x[q] - s.x[q]));
      ref = std::max(ref, std::abs(s.x[q]));
    }
    REQUIRE(err <= 1e-9 * ref);
    if (g.size > 1) {
      REQUIRE(mg.level_info(0).replicated == false);
      // Only the agglomerated coarse levels are replicated, and they are bounded.
      for (int l = 0; l < mg.levels(); ++l)
        if (mg.level_info(l).replicated) {
          const auto gl = mg.level_info(l).global;
          REQUIRE(static_cast<long>(gl[0]) * gl[1] * gl[2] <= c.agglomerate_max);
        }
    }
  }
}

TEST_CASE("NodalMultigrid gives the same result through the staging path",
          "[nodal_multigrid][MPI]") {
  Grid g({16, 16, 8}, {1.0 / 16, 1.0 / 16, 1.0 / 8});
  MG mg(g.decomp, g.rank, g.comm, controls(1.0));
  mg.update(coefficient_field(g, jump_coefficient(g, 100.0)));
  const auto dom = pfc::decomposition::domain(g.decomp);
  const auto box = pfc::decomposition::local_box(g.decomp, g.rank);
  FieldT r1 = g.field(), z1 = g.field(), y1 = g.field();
  FieldT r2(dom, box, 2), z2(dom, box, 2), y2(dom, box, 2);
  const Coefficient v = [](int i, int j, int k) {
    return random_value(31u, i, j, k);
  };
  set(r1, v);
  set(r2, v);
  mg.apply(r1, z1);
  mg.apply(r2, z2);
  mg.apply_operator(z1, y1);
  mg.apply_operator(z2, y2);
  double e = 0, n = 0;
  each(z1, [&](int i, int j, int k) {
    e = std::max(e, std::abs(z1(i, j, k) - z2(i, j, k)) +
                        std::abs(y1(i, j, k) - y2(i, j, k)));
    n = std::max(n, std::abs(z1(i, j, k)) + std::abs(y1(i, j, k)));
  });
  REQUIRE(e <= 1e-14 * n);
}
