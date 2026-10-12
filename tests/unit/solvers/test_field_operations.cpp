// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_field_operations.cpp
 * @brief Field-backed cg / gmres / newton on a variable-coefficient pair.
 *
 * The operator on a periodic unit square (two fields, owned cells only) is
 *   (A u)_0 = c u_0 - div(kappa grad u_0) + g u_1
 *   (A u)_1 = c u_1 - div(kappa grad u_1) + g u_0
 * with kappa = 1 + 0.5 sin(2 pi x) cos(2 pi y). A flat std::vector path uses
 * the same stencil. Runs on however many ranks launched the test.
 */

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/krylov.hpp>
#include <openpfc/solvers/newton.hpp>

namespace {

using pfc::HostSpace;
using Field = pfc::data::Field<double, HostSpace>;
using FVec = pfc::solvers::FieldVector<HostSpace>;
using FOps = pfc::solvers::FieldOperations<HostSpace>;
using Flat = pfc::solvers::VectorOperations<double>;

constexpr int kN = 16;
constexpr double kC = 1.0;
constexpr double kG = 0.2;

struct Grid {
  int rank = 0;
  int nproc = 1;
  pfc::decomposition::Decomposition decomp;
  pfc::Domain domain;
  pfc::Box3i box;
  double h = 1.0 / kN;
  double ih2 = double(kN) * kN;
  Grid()
      : decomp([&] {
          int r = 0, p = 1;
          MPI_Comm_rank(MPI_COMM_WORLD, &r);
          MPI_Comm_size(MPI_COMM_WORLD, &p);
          auto d = pfc::domain::with_spacing({kN, kN, 1}, {1.0 / kN, 1.0 / kN, 1.0},
                                             {true, true, true});
          return pfc::decomposition::create(d, p);
        }()),
        domain(pfc::decomposition::domain(decomp)),
        box(pfc::decomposition::local_box(decomp, rank_of())) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  }
  static int rank_of() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  }
  Field field() const { return Field(domain, box, 1); }
};

double kappa(double x, double y) {
  constexpr double pi = 3.14159265358979323846;
  return 1.0 + 0.5 * std::sin(2 * pi * x) * std::cos(2 * pi * y);
}

int wrap(int a, int n) {
  const int m = a % n;
  return m < 0 ? m + n : m;
}

void each(const Field &f, const std::function<void(int, int, int)> &fn) {
  const auto n = f.local_size();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) fn(i, j, k);
}

void fill_kappa(const Grid &g, Field &dx, Field &dy) {
  each(dx, [&](int i, int j, int k) {
    const auto q = dx.global(i, j, k);
    const double x = q[0] * g.h, y = q[1] * g.h;
    dx(i, j, k) = kappa(x + g.h, y + 0.5 * g.h);
    dy(i, j, k) = kappa(x + 0.5 * g.h, y + g.h);
  });
  pfc::comm::HaloExchangeOptions opt;
  opt.connectivity = pfc::comm::HaloConnectivity::Full;
  opt.directions = pfc::halo::presets::Full2D();
  pfc::comm::HaloExchange<HostSpace, double> halo({&dx, &dy}, g.decomp, g.rank,
                                                  MPI_COMM_WORLD, opt);
  halo.exchange();
}

/// (A u)_which. u0 and u1 halos are current; dx and dy halos are current.
void apply_at(const Grid &g, const Field &u0, const Field &u1, const Field &dx,
              const Field &dy, Field &out, int which) {
  const Field &u = which == 0 ? u0 : u1;
  const Field &other = which == 0 ? u1 : u0;
  each(out, [&](int i, int j, int k) {
    const double uc = u(i, j, k);
    const double diff = dx(i, j, k) * (uc - u(i + 1, j, k)) +
                        dx(i - 1, j, k) * (uc - u(i - 1, j, k)) +
                        dy(i, j, k) * (uc - u(i, j + 1, k)) +
                        dy(i, j - 1, k) * (uc - u(i, j - 1, k));
    out(i, j, k) = g.ih2 * diff + kC * uc + kG * other(i, j, k);
  });
}

std::size_t pack(int comp, int i, int j, int k, const pfc::Int3 &n) {
  return (static_cast<std::size_t>(comp) * n[2] + k) * n[1] * n[0] +
         static_cast<std::size_t>(j) * n[0] + i;
}

void pack_owned(const FVec &u, std::vector<double> &x) {
  x.assign(u.size(), 0.0);
  for (int c = 0; c < u.count(); ++c) {
    const auto n = u[c].local_size();
    each(u[c], [&](int i, int j, int k) { x[pack(c, i, j, k, n)] = u[c](i, j, k); });
  }
}

void unpack_owned(const std::vector<double> &x, FVec &u) {
  for (int c = 0; c < u.count(); ++c) {
    const auto n = u[c].local_size();
    each(u[c], [&](int i, int j, int k) { u[c](i, j, k) = x[pack(c, i, j, k, n)]; });
  }
}

double owned_sum(const Field &f) {
  double s = 0;
  each(f, [&](int i, int j, int k) { s += f(i, j, k); });
  double g = 0;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

struct Problem {
  Grid grid;
  FOps ops;
  Flat flat;
  pfc::solvers::FieldHalo<HostSpace> halo;
  Field dx;
  Field dy;
  Problem()
      : ops(grid.domain, grid.box, 2, 1, MPI_COMM_WORLD), flat(MPI_COMM_WORLD),
        halo(ops, grid.decomp, grid.rank, MPI_COMM_WORLD, full2d()),
        dx(grid.field()), dy(grid.field()) {
    fill_kappa(grid, dx, dy);
  }
  static pfc::comm::HaloExchangeOptions full2d() {
    pfc::comm::HaloExchangeOptions opt;
    opt.connectivity = pfc::comm::HaloConnectivity::Full;
    opt.directions = pfc::halo::presets::Full2D();
    return opt;
  }
  void apply(const FVec &x, FVec &y) {
    const FVec &u = halo(x);
    apply_at(grid, u[0], u[1], dx, dy, y[0], 0);
    apply_at(grid, u[0], u[1], dx, dy, y[1], 1);
  }
  void apply_flat(const std::vector<double> &x, std::vector<double> &y) {
    FVec u = ops.make(ops.size());
    unpack_owned(x, u);
    FVec w = ops.make(ops.size());
    apply(u, w);
    pack_owned(w, y);
  }
};

pfc::solvers::KrylovControls tight() {
  pfc::solvers::KrylovControls c;
  c.preconditioning = pfc::solvers::Preconditioning::None;
  c.relative_tolerance = 1e-10;
  c.maximum_iterations = 400;
  c.restart = 40;
  return c;
}

} // namespace

TEST_CASE("FieldOperations dot ignores halo junk and pinned cells", "[field_ops]") {
  Problem p;
  FVec x = p.ops.make(p.ops.size());
  FVec y = p.ops.make(p.ops.size());
  double expect = 0;
  for (int c = 0; c < 2; ++c) {
    each(x[c], [&](int i, int j, int k) {
      const auto g = x[c].global(i, j, k);
      x[c](i, j, k) = 0.5 * (c + 1) + 0.01 * g[0];
      y[c](i, j, k) = 0.25 * (g[1] + 1);
      expect += x[c](i, j, k) * y[c](i, j, k);
    });
    const auto n = x[c].local_size();
    const int hw = x[c].storage_halo();
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i)
        if (i < 0 || j < 0 || i >= n[0] || j >= n[1]) {
          x[c](i, j, 0) = 1e6;
          y[c](i, j, 0) = -1e6;
        }
  }
  double global_expect = 0;
  MPI_Allreduce(&expect, &global_expect, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(p.ops.dot(x, y) - global_expect) <=
          1e-12 * std::abs(global_expect));

  Field mask = p.grid.field();
  each(mask, [&](int i, int j, int k) { mask(i, j, k) = 1; });
  bool pinned = false;
  each(mask, [&](int i, int j, int k) {
    if (!pinned && i == 0 && j == 0 && k == 0) {
      mask(i, j, k) = 0;
      pinned = true;
    }
  });
  p.ops.set_mask(0, &mask);
  const double masked = p.ops.dot(x, y);
  p.ops.set_mask(0, nullptr);
  double dropped = 0;
  if (pinned) dropped = x[0](0, 0, 0) * y[0](0, 0, 0);
  double global_drop = 0;
  MPI_Allreduce(&dropped, &global_drop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(masked - (global_expect - global_drop)) <=
          1e-12 * std::abs(global_expect));

  double batch[2] = {};
  FVec columns[2] = {x, y};
  const auto calls = p.ops.reductions.calls;
  p.ops.dots(columns, 2, x, batch);
  REQUIRE(p.ops.reductions.calls == calls + 1);
  REQUIRE(std::abs(batch[0] - p.ops.dot(x, x)) <= 1e-12 * std::abs(batch[0] + 1));
  REQUIRE(std::abs(batch[1] - p.ops.dot(y, x)) <= 1e-12 * std::abs(batch[1] + 1));
}

TEST_CASE("FieldHalo Full2D copies periodic owned values into the halo",
          "[field_ops]") {
  Problem p;
  FVec u = p.ops.make(p.ops.size());
  for (int c = 0; c < 2; ++c) {
    each(u[c], [&](int i, int j, int k) {
      const auto g = u[c].global(i, j, k);
      u[c](i, j, k) = 0.3 * (c + 1) * (g[0] + 1) - 0.1 * (g[1] + 2);
    });
    const auto n = u[c].local_size();
    const int hw = u[c].storage_halo();
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i)
        if (i < 0 || j < 0 || i >= n[0] || j >= n[1]) u[c](i, j, 0) = 42;
  }
  const FVec &h = p.halo(u);
  bool ok = true;
  for (int c = 0; c < 2; ++c) {
    const auto n = h[c].local_size();
    const auto gsz = h[c].global_size();
    const int hw = h[c].storage_halo();
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i) {
        const auto g = h[c].global(i, j, 0);
        const double want = 0.3 * (c + 1) * (wrap(g[0], gsz[0]) + 1) -
                            0.1 * (wrap(g[1], gsz[1]) + 2);
        ok = ok && h[c](i, j, 0) == want;
      }
  }
  int all = ok ? 1 : 0, glob = 0;
  MPI_Allreduce(&all, &glob, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  REQUIRE(glob == 1);
}

TEST_CASE("Field cg and gmres match the flat vector path", "[field_ops]") {
  Problem p;
  FVec b = p.ops.make(p.ops.size());
  for (int c = 0; c < 2; ++c)
    each(b[c], [&](int i, int j, int k) {
      const auto g = b[c].global(i, j, k);
      b[c](i, j, k) = std::sin(0.17 * (g[0] + 1) + c) * std::cos(0.11 * (g[1] + 3));
    });
  std::vector<double> bflat;
  pack_owned(b, bflat);
  auto controls = tight();

  FVec x = p.ops.make(p.ops.size());
  const auto cg = pfc::solvers::cg(
      p.ops, [&](const FVec &v, FVec &w) { p.apply(v, w); }, b, x, controls);
  std::vector<double> xflat(bflat.size(), 0.0);
  const auto cg_flat = pfc::solvers::cg(
      p.flat,
      [&](const std::vector<double> &v, std::vector<double> &w) {
        p.apply_flat(v, w);
      },
      bflat, xflat, controls);
  REQUIRE(cg.converged());
  REQUIRE(cg_flat.converged());
  REQUIRE(cg.iterations == cg_flat.iterations);
  FVec from_flat = p.ops.make(p.ops.size());
  unpack_owned(xflat, from_flat);
  p.ops.axpy(from_flat, -1, x);
  const double diff = p.ops.norm(from_flat);
  const double scale = p.ops.norm(x);
  REQUIRE(diff <= 1e-8 * std::max(1.0, scale));

  FVec xg = p.ops.make(p.ops.size());
  const auto gm = pfc::solvers::gmres(
      p.ops, [&](const FVec &v, FVec &w) { p.apply(v, w); }, b, xg, controls);
  REQUIRE(gm.converged());
  REQUIRE(gm.residual <= 1e-10 * p.ops.norm(b));
  p.ops.axpy(xg, -1, x);
  REQUIRE(p.ops.norm(xg) <= 1e-8 * std::max(1.0, scale));
}

TEST_CASE("Field newton solves a mild cubic on the same operator", "[field_ops]") {
  Problem p;
  FVec exact = p.ops.make(p.ops.size());
  for (int c = 0; c < 2; ++c)
    each(exact[c], [&](int i, int j, int k) {
      const auto g = exact[c].global(i, j, k);
      exact[c](i, j, k) =
          0.05 * std::sin(0.4 * (g[0] + 1 + c)) * std::cos(0.2 * g[1]);
    });
  FVec f = p.ops.make(p.ops.size());
  p.apply(exact, f);
  constexpr double cubic = 0.05;
  for (int c = 0; c < 2; ++c)
    each(f[c], [&](int i, int j, int k) {
      const double u = exact[c](i, j, k);
      f[c](i, j, k) += cubic * u * u * u;
    });

  struct Model {
    Problem *p;
    double cubic = 0.05;
    FVec at;
    void residual(const FVec &u, FVec &r) {
      p->apply(u, r);
      for (int c = 0; c < u.count(); ++c)
        each(r[c], [&](int i, int j, int k) {
          const double v = u[c](i, j, k);
          r[c](i, j, k) += cubic * v * v * v - f_of(c, i, j, k);
        });
    }
    void linearize(const FVec &u) { at = u; }
    void jacobian(const FVec &v, FVec &w) {
      p->apply(v, w);
      for (int c = 0; c < v.count(); ++c)
        each(w[c], [&](int i, int j, int k) {
          const double s = at[c](i, j, k);
          w[c](i, j, k) += 3 * cubic * s * s * v[c](i, j, k);
        });
    }
    void precondition(const FVec &r, FVec &z, int) { p->ops.copy(r, z); }
    void begin_trial(const FVec &) {}
    void accept_trial(const FVec &) {}
    void reject_trial() {}
    const FVec *rhs = nullptr;
    double f_of(int c, int i, int j, int k) const { return (*rhs)[c](i, j, k); }
  } model;
  model.p = &p;
  model.rhs = &f;
  model.at = p.ops.make(p.ops.size());

  FVec u = p.ops.make(p.ops.size());
  pfc::solvers::NewtonControls nc;
  nc.linear = tight();
  nc.linear.relative_tolerance = 1e-10;
  nc.relative_tolerance = 1e-8;
  const auto got = pfc::solvers::newton(p.ops, model, u, nc);
  REQUIRE(got.converged());
  p.ops.axpy(u, -1, exact);
  REQUIRE(p.ops.norm(u) <= 1e-6 * std::max(1.0, p.ops.norm(exact)));
}
