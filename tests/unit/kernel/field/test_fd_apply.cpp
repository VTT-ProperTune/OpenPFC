// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_apply.cpp
 * @brief Validates `apply_d2_along` (compile-time + runtime overloads).
 *
 * @details
 * Builds a 7^3 brick filled with `u(x,y,z) = x^2 + 2*y^2 + 3*z^2` (with
 * `dx = dy = dz = 1`, so `u_xx = 2`, `u_yy = 4`, `u_zz = 6`), then asks
 * `apply_d2_along` to return the **unscaled** central FD sum at the centre
 * cell `(3, 3, 3)` for orders 2, 4, and 6. Expected unscaled values are
 * `(true second derivative) * Stencil::denom * h^2`, which lets us verify
 * the integer arithmetic without floating-point spread.
 *
 * Both forms are exercised: the compile-time overload (`Stencil =
 * EvenCentralD2<Order>`) and the runtime overload (`view` populated by
 * `lookup_even_central_d2(order, ...)`).
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/field/fd_apply.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>

using Catch::Approx;
using pfc::field::fd::apply_d2_along;
using pfc::field::fd::EvenCentralD2;
using pfc::field::fd::EvenCentralD2View;
using pfc::field::fd::lookup_even_central_d2;

namespace {

constexpr int N = 7; // Big enough for half_width up to 3 around centre 3,3,3.
constexpr std::ptrdiff_t SX = 1;
constexpr std::ptrdiff_t SY = N;
constexpr std::ptrdiff_t SZ = static_cast<std::ptrdiff_t>(N) * N;

std::vector<double> build_field() {
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        const std::size_t c =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(N * N);
        u[c] = x * x + 2.0 * y * y + 3.0 * z * z;
      }
    }
  }
  return u;
}

} // namespace

TEST_CASE("apply_d2_along (compile-time stencil) returns unscaled FD sum",
          "[kernel][field][fd_apply][unit]") {
  const auto u = build_field();
  const std::ptrdiff_t c =
      3 + 3 * static_cast<std::ptrdiff_t>(N) + 3 * SZ; // centre cell

  SECTION("order 2 (denom = 1)") {
    REQUIRE(apply_d2_along<0, EvenCentralD2<2>>(u.data(), c, SX, SY, SZ) ==
            Approx(2.0)); // u_xx = 2 ⇒ 2 * 1
    REQUIRE(apply_d2_along<1, EvenCentralD2<2>>(u.data(), c, SX, SY, SZ) ==
            Approx(4.0)); // u_yy = 4
    REQUIRE(apply_d2_along<2, EvenCentralD2<2>>(u.data(), c, SX, SY, SZ) ==
            Approx(6.0)); // u_zz = 6
  }

  SECTION("order 4 (denom = 12)") {
    REQUIRE(apply_d2_along<0, EvenCentralD2<4>>(u.data(), c, SX, SY, SZ) ==
            Approx(24.0)); // 2 * 12
    REQUIRE(apply_d2_along<1, EvenCentralD2<4>>(u.data(), c, SX, SY, SZ) ==
            Approx(48.0)); // 4 * 12
    REQUIRE(apply_d2_along<2, EvenCentralD2<4>>(u.data(), c, SX, SY, SZ) ==
            Approx(72.0)); // 6 * 12
  }

  SECTION("order 6 (denom = 180)") {
    REQUIRE(apply_d2_along<0, EvenCentralD2<6>>(u.data(), c, SX, SY, SZ) ==
            Approx(360.0)); // 2 * 180
    REQUIRE(apply_d2_along<1, EvenCentralD2<6>>(u.data(), c, SX, SY, SZ) ==
            Approx(720.0)); // 4 * 180
    REQUIRE(apply_d2_along<2, EvenCentralD2<6>>(u.data(), c, SX, SY, SZ) ==
            Approx(1080.0)); // 6 * 180
  }
}

TEST_CASE("apply_d2_along (runtime view) returns unscaled FD sum",
          "[kernel][field][fd_apply][unit]") {
  const auto u = build_field();
  const std::ptrdiff_t c = 3 + 3 * static_cast<std::ptrdiff_t>(N) + 3 * SZ;

  EvenCentralD2View view{};
  for (int order : {2, 4, 6}) {
    REQUIRE(lookup_even_central_d2(order, &view));
    const double denom = static_cast<double>(view.denom);
    REQUIRE(apply_d2_along<0>(view, u.data(), c, SX, SY, SZ) == Approx(2.0 * denom));
    REQUIRE(apply_d2_along<1>(view, u.data(), c, SX, SY, SZ) == Approx(4.0 * denom));
    REQUIRE(apply_d2_along<2>(view, u.data(), c, SX, SY, SZ) == Approx(6.0 * denom));
  }
}

TEST_CASE("lookup_even_central_d2 rejects unsupported orders",
          "[kernel][field][fd_apply][unit]") {
  EvenCentralD2View view{};
  REQUIRE_FALSE(lookup_even_central_d2(0, &view));
  REQUIRE_FALSE(lookup_even_central_d2(1, &view));
  REQUIRE_FALSE(lookup_even_central_d2(3, &view));
  REQUIRE_FALSE(lookup_even_central_d2(22, &view));
  REQUIRE_FALSE(lookup_even_central_d2(-2, &view));
}
