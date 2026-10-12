// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <numeric>
#include <openpfc/solvers/krylov.hpp>
using namespace pfc::solvers;
namespace {
using Vec = std::vector<double>;
using Matrix = std::vector<Vec>;
Vec dense_solve(Matrix a, Vec b) {
  const int n = b.size();
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(a[i][k]) > std::abs(a[pivot][k])) pivot = i;
    std::swap(a[k], a[pivot]);
    std::swap(b[k], b[pivot]);
    REQUIRE(std::abs(a[k][k]) > 1e-15);
    for (int i = k + 1; i < n; ++i) {
      const double factor = a[i][k] / a[k][k];
      for (int j = k; j < n; ++j) a[i][j] -= factor * a[k][j];
      b[i] -= factor * b[k];
    }
  }
  Vec x(n);
  for (int i = n - 1; i >= 0; --i) {
    double v = b[i];
    for (int j = i + 1; j < n; ++j) v -= a[i][j] * x[j];
    x[i] = v / a[i][i];
  }
  return x;
}
auto multiply(const Matrix &a) {
  return [&a](const Vec &x, Vec &y) {
    y.assign(a.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = 0; j < x.size(); ++j) y[i] += a[i][j] * x[j];
  };
}
} // namespace
TEST_CASE("restarted GMRES agrees with independent pivoted elimination",
          "[krylov]") {
  Matrix a(9, Vec(9));
  Vec b(9);
  for (int i = 0; i < 9; ++i) {
    a[i][i] = 3 + .1 * i;
    b[i] = std::sin(1.3 * i) + .2;
    if (i < 8) a[i][i + 1] = 1.4;
    if (i > 0) a[i][i - 1] = -.6;
  }
  VectorOperations ops;
  const auto reference = dense_solve(a, b);
  for (auto side :
       {Preconditioning::None, Preconditioning::Left, Preconditioning::Right}) {
    Vec x(9, .07);
    KrylovControls c;
    c.restart = 3;
    c.relative_tolerance = 1e-12;
    c.preconditioning = side;
    c.flexible = false;
    auto diagonal = [&](const Vec &r, Vec &z, int) {
      z = r;
      for (int i = 0; i < 9; ++i) z[i] /= a[i][i];
    };
    const auto result = gmres(ops, multiply(a), b, x, c, diagonal);
    REQUIRE(result.converged());
    REQUIRE(result.restarts > 0);
    REQUIRE(result.residual <= 1e-12 * ops.norm(b));
    for (int i = 0; i < 9; ++i) REQUIRE(std::abs(x[i] - reference[i]) < 1e-11);
    Vec residual;
    multiply(a)(x, residual);
    ops.axpy(residual, -1, b);
    REQUIRE(ops.norm(residual) <= 1e-12 * ops.norm(b));
  }
}
TEST_CASE("reorthogonalized GMRES solves a widely scaled tridiagonal matrix",
          "[krylov]") {
  Matrix a(40, Vec(40));
  Vec b(40, 1);
  for (int i = 0; i < 40; ++i) {
    a[i][i] = std::pow(10., -3. + 6. * i / 39.);
    if (i) a[i][i - 1] = .1;
    if (i + 1 < 40) a[i][i + 1] = -.1;
  }
  VectorOperations ops;
  Vec x(40);
  KrylovControls c;
  // The dimension is the restart. GMRES(20) leaves a residual near 4e-2
  // after 200 iterations on this matrix with either orthogonalization,
  // because the restarted subspace is too small, not because the basis
  // lost orthogonality.
  c.restart = 40;
  c.relative_tolerance = 1e-10;
  c.maximum_iterations = 40;
  auto result = gmres(ops, multiply(a), b, x, c);
  REQUIRE(result.converged());
  Vec residual;
  multiply(a)(x, residual);
  ops.axpy(residual, -1, b);
  REQUIRE(ops.norm(residual) <= 1e-10 * ops.norm(b));
}
TEST_CASE("FGMRES retains independently varying right directions", "[krylov]") {
  Matrix a{{4, 2, -1}, {-1, 3, 1}, {0, -2, 5}};
  Vec b{1, -4, 2}, x(3);
  VectorOperations ops;
  KrylovControls c;
  c.restart = 3;
  c.relative_tolerance = 1e-12;
  auto variable = [](const Vec &r, Vec &z, int k) {
    z = r;
    for (int i = 0; i < 3; ++i) z[i] /= 1 + ((k + 2 * i) % 4);
  };
  auto result = gmres(ops, multiply(a), b, x, c, variable);
  REQUIRE(result.converged());
  auto reference = dense_solve(a, b);
  for (int i = 0; i < 3; ++i) REQUIRE(std::abs(x[i] - reference[i]) < 1e-11);
}
TEST_CASE("Krylov limits singular loads and nonfinite data fail closed",
          "[krylov]") {
  VectorOperations ops;
  KrylovControls c;
  Vec b{1, 2}, x(2);
  Matrix a{{2, 1}, {-1, 3}};
  c.maximum_iterations = 1;
  c.restart = 1;
  REQUIRE_FALSE(gmres(ops, multiply(a), b, x, c).converged());
  c.maximum_iterations = 0;
  x.assign(2, 0);
  REQUIRE(gmres(ops, multiply(a), b, x, c).iterations == 0);
  c.maximum_iterations = 20;
  Matrix singular{{1, 0}, {0, 0}};
  x.assign(2, 0);
  REQUIRE_FALSE(gmres(ops, multiply(singular), b, x, c).converged());
  c.preconditioning = Preconditioning::Left;
  c.flexible = true;
  REQUIRE_THROWS_AS(gmres(ops, multiply(a), b, x, c), std::invalid_argument);
  c.flexible = false;
  auto zero = [](const Vec &r, Vec &z, int) { z.assign(r.size(), 0); };
  REQUIRE(gmres(ops, multiply(a), b, x, c, zero).status == LinearStatus::Breakdown);
  c.preconditioning = Preconditioning::None;
  b[0] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(gmres(ops, multiply(a), b, x, c).status == LinearStatus::NonFinite);
}
TEST_CASE("verified SPD CG retains true residual checks", "[krylov]") {
  Matrix a{{4, -1, 0}, {-1, 4, -1}, {0, -1, 3}};
  Vec b{1, 3, 2}, x(3);
  VectorOperations ops;
  KrylovControls c;
  c.relative_tolerance = 1e-12;
  REQUIRE(cg(ops, multiply(a), b, x, c).converged());
  const auto reference = dense_solve(a, b);
  for (int i = 0; i < 3; ++i) REQUIRE(std::abs(x[i] - reference[i]) < 1e-11);
  Matrix negative{{-1, 0}, {0, -2}};
  x.assign(2, 0);
  b = {1, 2};
  REQUIRE(cg(ops, multiply(negative), b, x, c).status == LinearStatus::Breakdown);
}
TEST_CASE("Krylov MPI reductions use all owned entries", "[krylov][MPI]") {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  VectorOperations ops(MPI_COMM_WORLD);
  Matrix a{{2., 1.}, {-.5, 3.}};
  Vec reference{1. + rank, 2. - .2 * rank}, b, x(2);
  multiply(a)(reference, b);
  KrylovControls c;
  c.restart = 2;
  c.relative_tolerance = 1e-12;
  const auto result = gmres(ops, multiply(a), b, x, c);
  REQUIRE(result.converged());
  REQUIRE(std::abs(x[0] - reference[0]) < 1e-11);
  REQUIRE(std::abs(x[1] - reference[1]) < 1e-11);
  double expected = 0;
  for (int r = 0; r < ranks; ++r) {
    Vec q{1. + r, 2. - .2 * r}, v;
    multiply(a)(q, v);
    expected += v[0] * v[0] + v[1] * v[1];
  }
  REQUIRE(std::abs(result.rhs_norm - std::sqrt(expected)) < 1e-12);
  REQUIRE(ops.reductions.calls > 0);
}

TEST_CASE("unpreconditioned CG bypasses a supplied inverse", "[krylov]") {
  Matrix a{{4, -1}, {-1, 3}};
  Vec b{1, 2}, x(2);
  VectorOperations ops;
  KrylovControls c;
  c.preconditioning = Preconditioning::None;
  auto forbidden = [](const Vec &, Vec &, int) {
    throw std::runtime_error("inverse must be bypassed");
  };
  auto result = cg(ops, multiply(a), b, x, c, forbidden);
  REQUIRE(result.converged());
  REQUIRE(result.preconditioner_calls == 0);
}
