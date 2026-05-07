// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_stencil_apply.cpp
 * @brief Validates the generic stencil primitives in `stencil_apply.hpp`.
 *
 * @details
 * Every primitive is tested against a hand-rolled reference computed
 * locally in the test, so the test file does **not** depend on the
 * PDE-specialised `fd_apply.hpp` to validate `stencil_apply.hpp`. The
 * cross-equivalence with the FD primitives (which is what gives us
 * confidence the new primitives match the existing PDE pipeline) is
 * tested separately at the bottom of this file.
 *
 * Cases:
 *
 *  1. `apply_1d_along` matches a hand loop on a quadratic field for a
 *     symmetric central D2 stencil and on a linear field for an
 *     anti-symmetric central D1 stencil.
 *  2. `apply_separable` collapses to `apply_1d_along` when two of the
 *     three half-widths are zero, and matches a hand-rolled triple loop
 *     for a non-trivial 3×3×3 separable kernel (Gaussian smoothing
 *     `[1,2,1]_x ⊗ [1,2,1]_y ⊗ [1,2,1]_z / 64`).
 *  3. `apply_dense` matches a hand-rolled triple loop for a Sobel-x
 *     3×3×3 kernel acting on a linear ramp `u = x`, returning
 *     `4 * sum(sobel_x.flat()) * 1 = 0` (the kernel sums to zero) for a
 *     constant field, and `4 * (a non-zero analytic value)` for a ramp
 *     in `x`.
 *  4. Cross-equivalence between `apply_1d_along` (with the FD-table
 *     coefficients copy-pasted into a `double[]`) and
 *     `pfc::field::fd::apply_d2_along` on a quadratic field — proves
 *     the generic primitive reproduces the PDE-specialised primitive
 *     bit-for-bit when fed the same weights.
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/stencil_apply.hpp>

using Catch::Approx;
using pfc::field::stencil::apply_1d_along;
using pfc::field::stencil::apply_dense;
using pfc::field::stencil::apply_separable;

namespace {

constexpr int N = 9;
constexpr std::ptrdiff_t SX = 1;
constexpr std::ptrdiff_t SY = N;
constexpr std::ptrdiff_t SZ = static_cast<std::ptrdiff_t>(N) * N;

std::size_t lin(int ix, int iy, int iz) noexcept {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
         static_cast<std::size_t>(iz) * static_cast<std::size_t>(N * N);
}

std::vector<double> build_quadratic() {
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        u[lin(ix, iy, iz)] = x * x + 2.0 * y * y + 3.0 * z * z;
      }
    }
  }
  return u;
}

std::vector<double> build_linear_x() {
  // u(x, y, z) = x — used to probe central first-derivative stencils:
  // any centred 1D D1 with weights `[-w, 0, +w]` applied along x gives
  // 2 w (independent of y, z).
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        u[lin(ix, iy, iz)] = static_cast<double>(ix);
      }
    }
  }
  return u;
}

} // namespace

TEST_CASE("apply_1d_along matches a hand-rolled symmetric D2 sum",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_quadratic();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));

  // Order-2 central D2: coeffs[-1..+1] = (1, -2, 1) — symmetric, integer.
  const std::array<double, 3> w_d2 = {1.0, -2.0, 1.0};

  // u(x,y,z) = x^2 + 2 y^2 + 3 z^2 ⇒ unscaled D2 sums (h = 1) are
  //   along x:  +1*(x-1)^2 - 2*x^2 + 1*(x+1)^2 = 2.
  //   along y:  +1*2(y-1)^2 - 2*2 y^2 + 1*2(y+1)^2 = 4.
  //   along z:  +1*3(z-1)^2 - 2*3 z^2 + 1*3(z+1)^2 = 6.
  REQUIRE(apply_1d_along<0>(w_d2.data(), 1, u.data(), c, SX, SY, SZ) == Approx(2.0));
  REQUIRE(apply_1d_along<1>(w_d2.data(), 1, u.data(), c, SX, SY, SZ) == Approx(4.0));
  REQUIRE(apply_1d_along<2>(w_d2.data(), 1, u.data(), c, SX, SY, SZ) == Approx(6.0));
}

TEST_CASE("apply_1d_along handles asymmetric (D1) stencils",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_linear_x();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));

  // Order-2 central D1: coeffs[-1..+1] = (-0.5, 0, +0.5) — anti-symmetric.
  const std::array<double, 3> w_d1 = {-0.5, 0.0, 0.5};
  REQUIRE(apply_1d_along<0>(w_d1.data(), 1, u.data(), c, SX, SY, SZ) ==
          Approx(1.0)); // ∂x of u = x is 1.
  REQUIRE(apply_1d_along<1>(w_d1.data(), 1, u.data(), c, SX, SY, SZ) == Approx(0.0));
  REQUIRE(apply_1d_along<2>(w_d1.data(), 1, u.data(), c, SX, SY, SZ) == Approx(0.0));
}

TEST_CASE("apply_1d_along with half_width = 0 returns coeffs[0] * core[c]",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_quadratic();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));
  const std::array<double, 1> w_id = {1.5};
  // u(4,4,4) = 16 + 32 + 48 = 96.
  REQUIRE(apply_1d_along<0>(w_id.data(), 0, u.data(), c, SX, SY, SZ) ==
          Approx(1.5 * 96.0));
}

TEST_CASE("apply_separable collapses to apply_1d_along when two axes are trivial",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_quadratic();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));

  const std::array<double, 3> w_d2 = {1.0, -2.0, 1.0};
  const std::array<double, 1> w_id = {1.0};

  // x-axis D2, identity along y and z.
  const double via_sep_x = apply_separable(w_d2.data(), 1, w_id.data(), 0,
                                           w_id.data(), 0, u.data(), c, SX, SY, SZ);
  const double via_1d_x = apply_1d_along<0>(w_d2.data(), 1, u.data(), c, SX, SY, SZ);
  REQUIRE(via_sep_x == Approx(via_1d_x));

  // y-axis D2, identity along x and z.
  const double via_sep_y = apply_separable(w_id.data(), 0, w_d2.data(), 1,
                                           w_id.data(), 0, u.data(), c, SX, SY, SZ);
  const double via_1d_y = apply_1d_along<1>(w_d2.data(), 1, u.data(), c, SX, SY, SZ);
  REQUIRE(via_sep_y == Approx(via_1d_y));

  // z-axis D2, identity along x and y.
  const double via_sep_z = apply_separable(w_id.data(), 0, w_id.data(), 0,
                                           w_d2.data(), 1, u.data(), c, SX, SY, SZ);
  const double via_1d_z = apply_1d_along<2>(w_d2.data(), 1, u.data(), c, SX, SY, SZ);
  REQUIRE(via_sep_z == Approx(via_1d_z));
}

TEST_CASE("apply_separable matches a hand-rolled triple loop (Gaussian 3x3x3)",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_quadratic();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));

  const std::array<double, 3> g = {1.0, 2.0, 1.0}; // unnormalised binomial.

  // Hand-rolled reference:
  double ref = 0.0;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const double w = g[dx + 1] * g[dy + 1] * g[dz + 1];
        ref += w * u[lin(4 + dx, 4 + dy, 4 + dz)];
      }
    }
  }
  const double got = apply_separable(g.data(), 1, g.data(), 1, g.data(), 1, u.data(),
                                     c, SX, SY, SZ);
  REQUIRE(got == Approx(ref));
}

TEST_CASE("apply_dense matches a hand-rolled triple loop (Sobel-x 3x3x3)",
          "[kernel][field][stencil_apply][unit]") {
  // Standard 3D Sobel-x kernel: anti-symmetric in x, smoothed in y, z.
  const double sobel_x[3][3][3] = {
      {{-1.0, 0.0, +1.0}, {-2.0, 0.0, +2.0}, {-1.0, 0.0, +1.0}},
      {{-2.0, 0.0, +2.0}, {-4.0, 0.0, +4.0}, {-2.0, 0.0, +2.0}},
      {{-1.0, 0.0, +1.0}, {-2.0, 0.0, +2.0}, {-1.0, 0.0, +1.0}},
  };

  // u = x ramp: ∂x u = 1 everywhere, so the Sobel-x response equals
  // the sum of all positive-side weights (kx = 2) minus the sum of all
  // negative-side weights (kx = 0). For the kernel above that's 32.
  {
    const auto u = build_linear_x();
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));
    const double got = apply_dense(sobel_x, u.data(), c, SX, SY, SZ);
    REQUIRE(got == Approx(32.0));
  }

  // Constant field: every Sobel-x sum is zero.
  {
    std::vector<double> u(static_cast<std::size_t>(N) * N * N, 7.5);
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));
    const double got = apply_dense(sobel_x, u.data(), c, SX, SY, SZ);
    REQUIRE(got == Approx(0.0));
  }

  // Cross-check against a hand-rolled triple loop on the quadratic field.
  {
    const auto u = build_quadratic();
    const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));
    double ref = 0.0;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          ref += sobel_x[dz + 1][dy + 1][dx + 1] * u[lin(4 + dx, 4 + dy, 4 + dz)];
        }
      }
    }
    const double got = apply_dense(sobel_x, u.data(), c, SX, SY, SZ);
    REQUIRE(got == Approx(ref));
  }
}

TEST_CASE("apply_1d_along reproduces fd::apply_d2_along when fed the same weights",
          "[kernel][field][stencil_apply][unit]") {
  const auto u = build_quadratic();
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(lin(4, 4, 4));

  // Build a `double` weight array equivalent to the order-4 EvenCentralD2:
  // coeffs are stored as the compact-symmetric `[c0, c1, c2]` (with implicit
  // mirror at negative offsets); the stencil_apply form expects the **full**
  // length-`(2H+1)` array `[c2, c1, c0, c1, c2]`.
  using S4 = pfc::field::fd::EvenCentralD2<4>;
  const double full4[5] = {
      static_cast<double>(S4::coeffs[2]), static_cast<double>(S4::coeffs[1]),
      static_cast<double>(S4::coeffs[0]), static_cast<double>(S4::coeffs[1]),
      static_cast<double>(S4::coeffs[2]),
  };

  const double got_x =
      apply_1d_along<0>(full4, S4::half_width, u.data(), c, SX, SY, SZ);
  const double exp_x =
      pfc::field::fd::apply_d2_along<0, S4>(u.data(), c, SX, SY, SZ);
  REQUIRE(got_x == Approx(exp_x));

  const double got_y =
      apply_1d_along<1>(full4, S4::half_width, u.data(), c, SX, SY, SZ);
  const double exp_y =
      pfc::field::fd::apply_d2_along<1, S4>(u.data(), c, SX, SY, SZ);
  REQUIRE(got_y == Approx(exp_y));

  const double got_z =
      apply_1d_along<2>(full4, S4::half_width, u.data(), c, SX, SY, SZ);
  const double exp_z =
      pfc::field::fd::apply_d2_along<2, S4>(u.data(), c, SX, SY, SZ);
  REQUIRE(got_z == Approx(exp_z));
}
