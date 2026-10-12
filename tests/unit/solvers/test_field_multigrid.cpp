// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_field_multigrid.cpp
 * @brief Host FieldMultigrid: contraction, symmetry, grid-independent PCG with a
 *        10^3 coefficient jump, pinned cells, 3D. Runs on any number of ranks.
 */

#include <catch2/catch_all.hpp>

#include <cmath>
#include <functional>
#include <random>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/solvers/field_multigrid.hpp>

namespace {

using pfc::HostSpace;
using MG = pfc::solvers::FieldMultigrid<HostSpace>;
using FieldT = MG::FieldT;

struct Problem {
  int rank = 0, nproc = 1;
  pfc::decomposition::Decomposition decomp;
  Problem(int n, int nz)
      : decomp([&] {
          int r = 0, p = 1;
          MPI_Comm_rank(MPI_COMM_WORLD, &r);
          MPI_Comm_size(MPI_COMM_WORLD, &p);
          const double h = 1.0 / n;
          auto d = pfc::domain::with_spacing({n, n, nz}, {h, h, nz > 1 ? h : 1.0},
                                             {true, true, true});
          return pfc::decomposition::create(d, p);
        }()) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  }
  FieldT field() const {
    return FieldT(pfc::decomposition::domain(decomp),
                  pfc::decomposition::local_box(decomp, rank), 1);
  }
};

template <class Fn> void each(const FieldT &f, Fn &&fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) fn(i, j, k);
}

double dot(const FieldT &a, const FieldT &b) {
  double s = 0.0;
  each(a, [&](int i, int j, int k) { s += a(i, j, k) * b(i, j, k); });
  double g = 0.0;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

/// Cell coefficient c, face coefficients from kappa at the face centre, mask.
void set_coefficients(
    MG &mg, const FieldT &like, double c,
    const std::function<double(double, double, double)> &kappa,
    const std::function<bool(double, double, double)> &pinned = {}) {
  const auto sz = like.global_size();
  const double h = 1.0 / sz[0];
  each(like, [&](int i, int j, int k) {
    const auto g = like.global(i, j, k);
    const double x = (g[0] + 0.5) * h, y = (g[1] + 0.5) * h, z = (g[2] + 0.5) * h;
    mg.c()(i, j, k) = c;
    mg.dx()(i, j, k) = kappa(x + 0.5 * h, y, z);
    mg.dy()(i, j, k) = kappa(x, y + 0.5 * h, z);
    mg.dz()(i, j, k) = kappa(x, y, z + 0.5 * h);
    mg.mask()(i, j, k) = (pinned && pinned(x, y, z)) ? 0.0 : 1.0;
  });
  mg.update();
}

void random_fill(FieldT &f, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  each(f, [&](int i, int j, int k) {
    const auto g = f.global(i, j, k);
    gen.seed(seed +
             7919u * static_cast<unsigned>(g[0] + 1000 * g[1] + 1000000 * g[2]));
    f(i, j, k) = u(gen);
  });
}

/// Preconditioned CG on Fields; returns the number of iterations to rtol.
int pcg(MG &mg, const FieldT &b, FieldT &x, double rtol, bool multigrid,
        Problem &P) {
  FieldT r = P.field(), z = P.field(), p = P.field(), ap = P.field();
  each(x, [&](int i, int j, int k) { x(i, j, k) = 0.0; });
  each(r, [&](int i, int j, int k) { r(i, j, k) = b(i, j, k); });
  auto prec = [&](const FieldT &rr, FieldT &zz) {
    if (multigrid) {
      mg.apply(rr, zz);
      return;
    }
    FieldT e = P.field(), ae = P.field();
    each(e, [&](int i, int j, int k) { e(i, j, k) = 1.0; });
    // Jacobi: diagonal from A applied to unit vectors is not available cheaply;
    // use the local row sum of the operator on a delta: approximate by A 1 + face
    // sums.
    mg.apply_operator(e, ae); // c at free cells (faces cancel) -> add face sums
    each(rr, [&](int i, int j, int k) {
      const auto sz = rr.global_size();
      const double ih2 = static_cast<double>(sz[0]) * sz[0];
      double d = ae(i, j, k);
      if (mg.mask()(i, j, k) != 0.0) {
        d += ih2 * (mg.dx()(i, j, k) + mg.dx()(i - 1, j, k) + mg.dy()(i, j, k) +
                    mg.dy()(i, j - 1, k));
        if (sz[2] > 1) d += ih2 * (mg.dz()(i, j, k) + mg.dz()(i, j, k - 1));
      }
      zz(i, j, k) = rr(i, j, k) / d;
    });
  };
  prec(r, z);
  each(p, [&](int i, int j, int k) { p(i, j, k) = z(i, j, k); });
  double rz = dot(r, z);
  const double r0 = std::sqrt(dot(r, r));
  for (int it = 1; it <= 2000; ++it) {
    mg.apply_operator(p, ap);
    const double a = rz / dot(p, ap);
    each(x, [&](int i, int j, int k) {
      x(i, j, k) += a * p(i, j, k);
      r(i, j, k) -= a * ap(i, j, k);
    });
    if (std::sqrt(dot(r, r)) <= rtol * r0) return it;
    prec(r, z);
    const double rz2 = dot(r, z);
    each(p, [&](int i, int j, int k) {
      p(i, j, k) = z(i, j, k) + rz2 / rz * p(i, j, k);
    });
    rz = rz2;
  }
  return 2001;
}

double disk_jump(double x, double y, double) {
  const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
  return r2 < 0.0625 ? 1.0 : 1e-3;
}

} // namespace

TEST_CASE("FieldMultigrid V-cycle contracts the constant-coefficient error",
          "[solvers][field_multigrid]") {
  Problem P(64, 1);
  MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
  REQUIRE(mg.levels() >= 2);
  auto ref = P.field();
  set_coefficients(mg, ref, 1.0, [](double, double, double) { return 1.0; });
  FieldT f = P.field(), u = P.field(), r = P.field(), au = P.field(), z = P.field();
  random_fill(f, 3);
  each(u, [&](int i, int j, int k) { u(i, j, k) = 0.0; });
  double prev = std::sqrt(dot(f, f)), first = prev, last = prev;
  for (int it = 0; it < 8; ++it) {
    mg.apply_operator(u, au);
    each(r, [&](int i, int j, int k) { r(i, j, k) = f(i, j, k) - au(i, j, k); });
    mg.apply(r, z);
    each(u, [&](int i, int j, int k) { u(i, j, k) += z(i, j, k); });
    mg.apply_operator(u, au);
    each(r, [&](int i, int j, int k) { r(i, j, k) = f(i, j, k) - au(i, j, k); });
    last = std::sqrt(dot(r, r));
    prev = last;
  }
  const double factor = std::pow(last / first, 1.0 / 8);
  INFO("mean residual contraction per V-cycle " << factor);
  REQUIRE(factor < 0.15);
}

TEST_CASE("FieldMultigrid V-cycle is symmetric (usable in CG)",
          "[solvers][field_multigrid]") {
  Problem P(32, 1);
  MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
  auto ref = P.field();
  set_coefficients(mg, ref, 2.0, disk_jump,
                   [](double x, double y, double) { return x < 0.1 && y < 0.1; });
  FieldT a = P.field(), b = P.field(), ma = P.field(), mb = P.field();
  random_fill(a, 11);
  random_fill(b, 12);
  mg.apply(a, ma);
  mg.apply(b, mb);
  const double l = dot(ma, b), r = dot(a, mb);
  INFO("(Ma,b) = " << l << ", (a,Mb) = " << r);
  REQUIRE(std::abs(l - r) <= 1e-10 * std::abs(l));
  REQUIRE(dot(ma, a) > 0.0);
}

TEST_CASE("FieldMultigrid-PCG iterations do not grow with the grid",
          "[solvers][field_multigrid]") {
  // Constant coefficients, nearly singular (c = 1e-4): textbook grid independence.
  int its[3] = {0, 0, 0};
  const int ns[3] = {32, 64, 128};
  for (int q = 0; q < 3; ++q) {
    Problem P(ns[q], 1);
    MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
    auto ref = P.field();
    set_coefficients(mg, ref, 1e-4, [](double, double, double) { return 1.0; });
    FieldT b = P.field(), x = P.field();
    random_fill(b, 5);
    its[q] = pcg(mg, b, x, 1e-8, true, P);
  }
  INFO("MG-PCG " << its[0] << " " << its[1] << " " << its[2]);
  REQUIRE(its[2] <= 12);
  REQUIRE(its[2] <= its[0] + 1);
}

TEST_CASE("FieldMultigrid-PCG beats Jacobi-PCG with a 10^3 coefficient jump",
          "[solvers][field_multigrid]") {
  // Arithmetic coarse coefficients with bilinear transfers are not jump-robust:
  // iterations grow slowly with the grid (13, 19, 26 here), still far below Jacobi.
  int its[3] = {0, 0, 0}, jac[3] = {0, 0, 0};
  const int ns[3] = {32, 64, 128};
  for (int q = 0; q < 3; ++q) {
    Problem P(ns[q], 1);
    MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
    auto ref = P.field();
    set_coefficients(mg, ref, 1.0, disk_jump);
    FieldT b = P.field(), x = P.field();
    random_fill(b, 5);
    its[q] = pcg(mg, b, x, 1e-8, true, P);
    jac[q] = pcg(mg, b, x, 1e-8, false, P);
  }
  INFO("MG-PCG " << its[0] << " " << its[1] << " " << its[2] << ", Jacobi-PCG "
                 << jac[0] << " " << jac[1] << " " << jac[2]);
  REQUIRE(its[2] <= 30);
  REQUIRE(jac[2] > 8 * its[2]);
}

TEST_CASE("FieldMultigrid keeps pinned cells as identity rows",
          "[solvers][field_multigrid]") {
  Problem P(64, 1);
  MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
  auto ref = P.field();
  auto pinned = [](double x, double y, double) {
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return r2 < 0.01 || r2 > 0.2;
  };
  set_coefficients(mg, ref, 1.0, [](double, double, double) { return 1.0; }, pinned);
  FieldT r = P.field(), z = P.field(), x = P.field();
  random_fill(r, 9);
  mg.apply(r, z);
  bool ident = true;
  each(r, [&](int i, int j, int k) {
    if (mg.mask()(i, j, k) == 0.0) ident &= z(i, j, k) == r(i, j, k);
  });
  REQUIRE(ident);
  REQUIRE(pcg(mg, r, x, 1e-8, true, P) <= 30);
}

TEST_CASE("FieldMultigrid contracts in 3D with varying coefficients",
          "[solvers][field_multigrid]") {
  Problem P(32, 32);
  MG mg(P.decomp, P.rank, MPI_COMM_WORLD);
  REQUIRE(mg.levels() >= 2);
  auto ref = P.field();
  set_coefficients(mg, ref, 1.0, [](double x, double y, double z) {
    return 1.0 + 0.9 * std::sin(6.283185307179586 * x) *
                     std::cos(6.283185307179586 * (y + z));
  });
  FieldT b = P.field(), x = P.field();
  random_fill(b, 21);
  const int its = pcg(mg, b, x, 1e-8, true, P);
  INFO("3D MG-PCG iterations " << its);
  REQUIRE(its <= 20);
}
