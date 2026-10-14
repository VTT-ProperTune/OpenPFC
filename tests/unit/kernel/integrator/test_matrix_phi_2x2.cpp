// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * φ0, φ1, φ2 of real 2 × 2 matrices against an independent oracle.
 *
 * The oracle is the block exponential of Saad (1992) / Higham (2008,
 * Thm. 10.13): for the 6 × 6 matrix [[M, I, 0], [0, 0, I], [0, 0, 0]],
 * exp gives [[exp M, φ1(M), φ2(M)], ...]. It is evaluated in long double
 * by Taylor series with scaling and squaring of the whole block, with no
 * eigenvalues and no φ doubling formula, so it shares no code path with
 * the implementation under test.
 */

#include <array>
#include <cmath>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/integrator/matrix_phi_2x2.hpp>

using pfc::integrator::etd2_coefficients;
using pfc::integrator::Matrix2x2;
using pfc::integrator::matrix_phi;
using pfc::integrator::matrix_phi_by_squaring;
using pfc::integrator::MatrixPhi2x2;

namespace {

using Block = std::array<std::array<long double, 6>, 6>;

Block block_multiply(const Block &a, const Block &b) {
  Block c{};
  for (int i = 0; i < 6; ++i) {
    for (int k = 0; k < 6; ++k) {
      for (int j = 0; j < 6; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

/// exp of the 6 × 6 augmented block, long double, Taylor + squaring.
MatrixPhi2x2 oracle(const Matrix2x2 &m) {
  Block a{};
  a[0][0] = m.m00;
  a[0][1] = m.m01;
  a[1][0] = m.m10;
  a[1][1] = m.m11;
  a[0][2] = 1.0L;
  a[1][3] = 1.0L;
  a[2][4] = 1.0L;
  a[3][5] = 1.0L;
  long double nrm = 0.0L;
  for (int j = 0; j < 6; ++j) {
    long double col = 0.0L;
    for (int i = 0; i < 6; ++i) {
      col += std::fabs(a[i][j]);
    }
    nrm = std::max(nrm, col);
  }
  int s = 0;
  while (nrm > 0.125L) {
    nrm *= 0.5L;
    ++s;
  }
  for (auto &row : a) {
    for (auto &v : row) {
      v = std::ldexp(v, -s);
    }
  }
  Block sum{};
  Block term{};
  for (int i = 0; i < 6; ++i) {
    sum[i][i] = 1.0L;
    term[i][i] = 1.0L;
  }
  for (int n = 1; n <= 30; ++n) {
    term = block_multiply(term, a);
    for (auto &row : term) {
      for (auto &v : row) {
        v /= static_cast<long double>(n);
      }
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        sum[i][j] += term[i][j];
      }
    }
  }
  for (int i = 0; i < s; ++i) {
    sum = block_multiply(sum, sum);
  }
  auto pick = [&](int c) {
    return Matrix2x2{
        static_cast<double>(sum[0][c]), static_cast<double>(sum[0][c + 1]),
        static_cast<double>(sum[1][c]), static_cast<double>(sum[1][c + 1])};
  };
  return {pick(0), pick(2), pick(4)};
}

double max_abs(const Matrix2x2 &a) {
  return std::max(
      {std::abs(a.m00), std::abs(a.m01), std::abs(a.m10), std::abs(a.m11)});
}

double max_diff(const Matrix2x2 &a, const Matrix2x2 &b) {
  return max_abs(
      Matrix2x2{a.m00 - b.m00, a.m01 - b.m01, a.m10 - b.m10, a.m11 - b.m11});
}

/// Largest error relative to the largest oracle entry of each function.
double phi_error(const MatrixPhi2x2 &got, const MatrixPhi2x2 &want) {
  auto rel = [](const Matrix2x2 &g, const Matrix2x2 &w) {
    return max_diff(g, w) / std::max(max_abs(w), 1e-300);
  };
  return std::max({rel(got.phi0, want.phi0), rel(got.phi1, want.phi1),
                   rel(got.phi2, want.phi2)});
}

} // namespace

TEST_CASE("matrix_phi of the zero matrix is I, I, I/2", "[integrator][matrix_phi]") {
  for (const auto &f :
       {matrix_phi(Matrix2x2{}), matrix_phi_by_squaring(Matrix2x2{})}) {
    REQUIRE(max_diff(f.phi0, {1.0, 0.0, 0.0, 1.0}) == 0.0);
    REQUIRE(max_diff(f.phi1, {1.0, 0.0, 0.0, 1.0}) == 0.0);
    REQUIRE(max_diff(f.phi2, {0.5, 0.0, 0.0, 0.5}) == 0.0);
  }
}

TEST_CASE("matrix_phi matches the block-exponential oracle on structured cases",
          "[integrator][matrix_phi]") {
  const Matrix2x2 cases[] = {
      {-1.0, 0.0, 0.0, -3.0},         // diagonal, distinct
      {-2.0, 0.0, 0.0, -2.0},         // repeated, diagonalizable
      {-2.0, 1.0, 0.0, -2.0},         // Jordan block
      {0.0, 1.0, 0.0, 0.0},           // nilpotent
      {-2.0, 1.0, 1e-12, -2.0},       // nearly defective
      {-2.0, 1.0, 0.0, -2.0 + 1e-9},  // nearly repeated
      {-300.0, 5.0, 2e-6, -300.0},    // stiff, nearly repeated
      {-1.0, 5.0, -5.0, -1.0},        // complex pair
      {-40.0, 60.0, -60.0, -40.0},    // stiff complex pair
      {-1e4, 3.0, 0.5, -2e3},         // stiff, coupled
      {-5000.0, 0.0, -2500.0, -50.0}, // stiff lower triangular
      {-80.0, 0.0, -40.0, -80.0},     // equal diagonal, off-diagonal coupling
      {-80.0, 0.0, -40.0, -80.0 + 1e-3},
      {1e-14, 0.0, 0.0, -1e-14}, // near zero
      {0.3, -0.2, 0.1, 0.25},    // small, unstable
      {2.0, 1.0, -0.5, 1.0},     // growing
  };
  for (const auto &m : cases) {
    const auto want = oracle(m);
    INFO("m = [" << m.m00 << ", " << m.m01 << "; " << m.m10 << ", " << m.m11 << "]");
    CHECK(phi_error(matrix_phi(m), want) < 1e-12);
    CHECK(phi_error(matrix_phi_by_squaring(m), want) < 1e-12);
  }
}

TEST_CASE("matrix_phi matches the oracle on random matrices over seven decades",
          "[integrator][matrix_phi]") {
  std::mt19937_64 rng(20261014);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_real_distribution<double> decade(-3.0, 4.0);
  double worst = 0.0;
  for (int trial = 0; trial < 2000; ++trial) {
    const double scale = std::pow(10.0, decade(rng));
    // Mostly dissipative spectra, as ETD symbols are, with random coupling.
    Matrix2x2 m{-std::abs(unit(rng)) * scale, unit(rng) * scale, unit(rng) * scale,
                -std::abs(unit(rng)) * scale};
    if (trial % 7 == 0) {
      m.m11 = m.m00 * (1.0 + 1e-6 * unit(rng)); // nearly repeated diagonal
      m.m01 = 0.0;
    }
    // Growth beyond e^30 is not an ETD symbol; it only tests overflow.
    const double mu = 0.5 * (m.m00 + m.m11);
    const double q = 0.25 * (m.m00 - m.m11) * (m.m00 - m.m11) + m.m01 * m.m10;
    if (mu + (q > 0.0 ? std::sqrt(q) : 0.0) > 30.0) {
      continue;
    }
    const auto want = oracle(m);
    const double err = phi_error(matrix_phi(m), want);
    worst = std::max(worst, err);
    INFO("trial " << trial << " m = [" << m.m00 << ", " << m.m01 << "; " << m.m10
                  << ", " << m.m11 << "]");
    REQUIRE(err < 1e-11);
  }
  INFO("worst relative error " << worst);
  CHECK(worst < 1e-11);
}

TEST_CASE("etd2_coefficients scale phi1 and phi2 by h", "[integrator][matrix_phi]") {
  const Matrix2x2 L{-30.0, 0.0, -15.0, -4.0};
  const double h = 0.07;
  const auto c = etd2_coefficients(L, h);
  const auto f = oracle(h * L);
  CHECK(max_diff(c.exp_hL, f.phi0) < 1e-14);
  CHECK(max_diff(c.h_phi1, h * f.phi1) < 1e-14 * h);
  CHECK(max_diff(c.h_phi2, h * f.phi2) < 1e-14 * h);
}
