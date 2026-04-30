// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fd_brick_laplacian.cpp
 * @brief Validates the templated brick Laplacian routines added on top of
 *        `apply_d2_along` (no MPI, no halo exchanger -- single-rank fixtures).
 *
 * @details
 * Three things are exercised here:
 *
 *  1. `laplacian_interior<Order>` (3D and the runtime-order overload) on a
 *     polynomial of degree `Order`, for which the central stencil is exact;
 *     compared to the analytic Laplacian.
 *  2. `laplacian2d_xy_interior<Order>` on the analogous 2D polynomial
 *     (`nz == 1`).
 *  3. `laplacian_periodic_separated<2>` and `laplacian2d_xy_periodic_separated<2>`
 *     on a periodic sin/cos field where the owned region is the full
 *     domain and synthetic single-rank face halos are filled by wrapping
 *     the owned data; the result is compared to the analytic Laplacian
 *     (modulo the standard 2nd-order truncation error).
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <openpfc/kernel/field/finite_difference.hpp>

using Catch::Approx;
using pfc::field::fd::laplacian2d_xy_interior;
using pfc::field::fd::laplacian2d_xy_periodic_separated;
using pfc::field::fd::laplacian_interior;
using pfc::field::fd::laplacian_periodic_separated;

namespace {

inline std::size_t lin(int ix, int iy, int iz, int nx, int ny) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
         static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) *
             static_cast<std::size_t>(ny);
}

} // namespace

TEST_CASE("laplacian_interior<2> is exact on a quadratic polynomial",
          "[kernel][field][fd][brick][unit]") {
  // u = x^2 + 2 y^2 + 3 z^2 with dx = dy = dz = 1 ⇒ Δu = 2 + 4 + 6 = 12.
  constexpr int N = 9;
  constexpr int hw = 1;
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  std::vector<double> lap(u.size(), 0.0);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        u[lin(ix, iy, iz, N, N)] = x * x + 2.0 * y * y + 3.0 * z * z;
      }
    }
  }
  laplacian_interior<2>(u.data(), lap.data(), N, N, N, 1.0, 1.0, 1.0, hw);
  for (int iz = hw; iz < N - hw; ++iz) {
    for (int iy = hw; iy < N - hw; ++iy) {
      for (int ix = hw; ix < N - hw; ++ix) {
        REQUIRE(lap[lin(ix, iy, iz, N, N)] == Approx(12.0));
      }
    }
  }
}

TEST_CASE("laplacian_interior<4> is exact on a quartic polynomial",
          "[kernel][field][fd][brick][unit]") {
  // u = x^4 + y^4 + z^4 with dx = dy = dz = 1
  // ⇒ Δu = 12 (x^2 + y^2 + z^2). The 4th-order central stencil is exact
  // for polynomials of degree ≤ 5 along each axis, so the discrete answer
  // matches the analytic one to round-off.
  constexpr int N = 11;
  constexpr int hw = 2;
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  std::vector<double> lap(u.size(), 0.0);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        u[lin(ix, iy, iz, N, N)] = x * x * x * x + y * y * y * y + z * z * z * z;
      }
    }
  }
  laplacian_interior<4>(u.data(), lap.data(), N, N, N, 1.0, 1.0, 1.0, hw);
  for (int iz = hw; iz < N - hw; ++iz) {
    for (int iy = hw; iy < N - hw; ++iy) {
      for (int ix = hw; ix < N - hw; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        const double expected = 12.0 * (x * x + y * y + z * z);
        REQUIRE(lap[lin(ix, iy, iz, N, N)] == Approx(expected));
      }
    }
  }
}

TEST_CASE("laplacian_interior runtime-order dispatcher matches templated form",
          "[kernel][field][fd][brick][unit]") {
  constexpr int N = 11;
  constexpr int hw = 2;
  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  std::vector<double> lap_tpl(u.size(), 0.0);
  std::vector<double> lap_run(u.size(), 0.0);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double x = static_cast<double>(ix);
        const double y = static_cast<double>(iy);
        const double z = static_cast<double>(iz);
        u[lin(ix, iy, iz, N, N)] = x * x * x + y * y * y + z * z * z;
      }
    }
  }
  laplacian_interior<4>(u.data(), lap_tpl.data(), N, N, N, 1.0, 1.0, 1.0, hw);
  laplacian_interior(4, u.data(), lap_run.data(), N, N, N, 1.0, 1.0, 1.0, hw);
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(lap_run[i] == Approx(lap_tpl[i]));
  }

  // An unsupported order is a no-op (matches the legacy contract).
  std::vector<double> lap_bad(u.size(), 7.0);
  laplacian_interior(3, u.data(), lap_bad.data(), N, N, N, 1.0, 1.0, 1.0, hw);
  for (double v : lap_bad) REQUIRE(v == Approx(7.0));
}

TEST_CASE("laplacian2d_xy_interior<2> is exact on a quadratic polynomial",
          "[kernel][field][fd][brick][unit]") {
  // u(x, y) = x^2 + 2 y^2, nz = 1, dx = dy = 1 ⇒ Δu = 6.
  constexpr int N = 9;
  constexpr int hw = 1;
  std::vector<double> u(static_cast<std::size_t>(N) * N);
  std::vector<double> lap(u.size(), 0.0);
  for (int iy = 0; iy < N; ++iy) {
    for (int ix = 0; ix < N; ++ix) {
      const double x = static_cast<double>(ix);
      const double y = static_cast<double>(iy);
      u[lin(ix, iy, 0, N, N)] = x * x + 2.0 * y * y;
    }
  }
  laplacian2d_xy_interior<2>(u.data(), lap.data(), N, N, 1, 1.0, 1.0, hw);
  for (int iy = hw; iy < N - hw; ++iy) {
    for (int ix = hw; ix < N - hw; ++ix) {
      REQUIRE(lap[lin(ix, iy, 0, N, N)] == Approx(6.0));
    }
  }
}

TEST_CASE("laplacian_periodic_separated<2> matches the analytic Laplacian "
          "(single-rank wrap)",
          "[kernel][field][fd][brick][unit]") {
  // Periodic 3D field on [0, 2π)^3 sampled on N^3 with a single rank.
  // u = sin(x) cos(y) sin(z) ⇒ Δu = -3 u.
  // We synthesise face halos by wrapping the owned data (single-rank
  // periodic), exactly as the SeparatedFaceHaloExchanger would after a
  // self-exchange on a 1-rank communicator.
  constexpr int N = 32;
  constexpr int hw = 1;
  const double dx = 2.0 * std::numbers::pi / static_cast<double>(N);
  const double inv_dx2 = 1.0 / (dx * dx);

  std::vector<double> u(static_cast<std::size_t>(N) * N * N);
  std::vector<double> lap(u.size(), 0.0);
  auto coord = [&](int i) { return static_cast<double>(i) * dx; };
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        u[lin(ix, iy, iz, N, N)] =
            std::sin(coord(ix)) * std::cos(coord(iy)) * std::sin(coord(iz));
      }
    }
  }

  // Build face halos (sizes = hw * other_two * remaining as in
  // halo_face_layout::face_halo_counts_analytic).
  std::vector<double> hpx(static_cast<std::size_t>(hw) * N * N);
  std::vector<double> hnx(hpx.size());
  std::vector<double> hpy(static_cast<std::size_t>(N) * hw * N);
  std::vector<double> hny(hpy.size());
  std::vector<double> hpz(static_cast<std::size_t>(N) * N * hw);
  std::vector<double> hnz(hpz.size());

  // X faces use layout [iz][iy][lx] with lx ∈ [0, hw); the +X face holds
  // the *first* hw columns of the +X neighbor's owned region (i.e. our
  // own first hw columns under periodic 1-rank wrap), and the -X face
  // holds the *last* hw columns of the -X neighbor.
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int lx = 0; lx < hw; ++lx) {
        const std::size_t hidx =
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(N * hw) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) + lx;
        hpx[hidx] = u[lin(lx, iy, iz, N, N)];
        hnx[hidx] = u[lin(N - hw + lx, iy, iz, N, N)];
      }
    }
  }
  // Y faces: [iz][ly][ix].
  for (int iz = 0; iz < N; ++iz) {
    for (int ly = 0; ly < hw; ++ly) {
      for (int ix = 0; ix < N; ++ix) {
        const std::size_t hidx =
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(N * hw) +
            static_cast<std::size_t>(ly) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(ix);
        hpy[hidx] = u[lin(ix, ly, iz, N, N)];
        hny[hidx] = u[lin(ix, N - hw + ly, iz, N, N)];
      }
    }
  }
  // Z faces: [lz][iy][ix].
  for (int lz = 0; lz < hw; ++lz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const std::size_t hidx =
            static_cast<std::size_t>(lz) * static_cast<std::size_t>(N * N) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(ix);
        hpz[hidx] = u[lin(ix, iy, lz, N, N)];
        hnz[hidx] = u[lin(ix, iy, N - hw + lz, N, N)];
      }
    }
  }

  std::array<const double *, 6> faces{hpx.data(), hnx.data(), hpy.data(),
                                      hny.data(), hpz.data(), hnz.data()};
  laplacian_periodic_separated<2>(u.data(), faces, lap.data(), N, N, N, inv_dx2,
                                  inv_dx2, inv_dx2, hw);

  // For 2nd-order central FD on a smooth periodic test, the relative
  // error scales like (dx)^2; we use a generous absolute tolerance.
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        const double expected = -3.0 * u[lin(ix, iy, iz, N, N)];
        REQUIRE(lap[lin(ix, iy, iz, N, N)] == Approx(expected).margin(0.05));
      }
    }
  }
}

TEST_CASE("laplacian2d_xy_periodic_separated<2> matches the analytic Laplacian "
          "(single-rank wrap)",
          "[kernel][field][fd][brick][unit]") {
  // u(x, y) = sin(x) cos(y) on [0, 2π)^2 ⇒ Δu = -2 u.
  constexpr int N = 32;
  constexpr int hw = 1;
  const double dx = 2.0 * std::numbers::pi / static_cast<double>(N);
  const double inv_dx2 = 1.0 / (dx * dx);

  std::vector<double> u(static_cast<std::size_t>(N) * N);
  std::vector<double> lap(u.size(), 0.0);
  auto coord = [&](int i) { return static_cast<double>(i) * dx; };
  for (int iy = 0; iy < N; ++iy) {
    for (int ix = 0; ix < N; ++ix) {
      u[lin(ix, iy, 0, N, N)] = std::sin(coord(ix)) * std::cos(coord(iy));
    }
  }

  std::vector<double> hpx(static_cast<std::size_t>(hw) * N);
  std::vector<double> hnx(hpx.size());
  std::vector<double> hpy(static_cast<std::size_t>(N) * hw);
  std::vector<double> hny(hpy.size());

  for (int iy = 0; iy < N; ++iy) {
    for (int lx = 0; lx < hw; ++lx) {
      const std::size_t hidx = static_cast<std::size_t>(iy) * hw + lx;
      hpx[hidx] = u[lin(lx, iy, 0, N, N)];
      hnx[hidx] = u[lin(N - hw + lx, iy, 0, N, N)];
    }
  }
  for (int ly = 0; ly < hw; ++ly) {
    for (int ix = 0; ix < N; ++ix) {
      const std::size_t hidx = static_cast<std::size_t>(ly) * N + ix;
      hpy[hidx] = u[lin(ix, ly, 0, N, N)];
      hny[hidx] = u[lin(ix, N - hw + ly, 0, N, N)];
    }
  }

  std::array<const double *, 6> faces{hpx.data(), hnx.data(), hpy.data(),
                                      hny.data(), nullptr,    nullptr};
  laplacian2d_xy_periodic_separated<2>(u.data(), faces, lap.data(), N, N, 1, inv_dx2,
                                       inv_dx2, hw);

  for (int iy = 0; iy < N; ++iy) {
    for (int ix = 0; ix < N; ++ix) {
      const double expected = -2.0 * u[lin(ix, iy, 0, N, N)];
      REQUIRE(lap[lin(ix, iy, 0, N, N)] == Approx(expected).margin(0.05));
    }
  }
}
