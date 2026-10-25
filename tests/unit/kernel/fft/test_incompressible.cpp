// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/fourier_vector.hpp>
#include <openpfc/solvers/incompressible/rotational.hpp>

using Catch::Matchers::WithinAbs;

namespace {

using Complex = pfc::field::Complex;

int world_size() {
  int n = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

struct Grid {
  int n{};
  std::array<int, 3> size{};
  std::array<double, 3> spacing{};
  pfc::fft::CPUFFT &fft;
  pfc::fft::Box3i outbox{};
  pfc::fft::Box3i inbox{};
  std::size_t nhat{};
  std::size_t nreal{};
};

/// `CPUFFT` owns a const plan, so the grid borrows it for the callback.
template <class F> void with_grid(int n, int nparts, F &&body) {
  const double h = pfc::two_pi / static_cast<double>(n);
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({h, h, h}));
  auto decomp = pfc::decomposition::create(domain, nparts);
  auto fft = pfc::fft::create(decomp);
  Grid grid{n,
            {n, n, n},
            {h, h, h},
            fft,
            fft.get_outbox_bounds(),
            fft.get_inbox_bounds(),
            fft.size_outbox(),
            fft.size_inbox()};
  REQUIRE(grid.nhat == static_cast<std::size_t>(grid.outbox.count()));
  REQUIRE(grid.nreal == static_cast<std::size_t>(grid.inbox.count()));
  body(grid);
}

std::size_t mode_at(const pfc::fft::Box3i &box, int i, int j, int k) {
  REQUIRE(box.contains({i, j, k}));
  return static_cast<std::size_t>(box.to_linear({i, j, k}));
}

double max_abs(const std::vector<Complex> &hat) {
  double peak = 0.0;
  for (const auto &z : hat) peak = std::max(peak, std::abs(z));
  return peak;
}

void require_relative(const std::vector<Complex> &got,
                      const std::vector<Complex> &ref, double tol) {
  REQUIRE(got.size() == ref.size());
  double diff = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  if (scale < 1.0) {
    REQUIRE(diff < tol);
    return;
  }
  REQUIRE(diff / scale < tol);
}

void fill_arbitrary(const Grid &grid, std::vector<Complex> &u,
                    std::vector<Complex> &v, std::vector<Complex> &w) {
  u.assign(grid.nhat, {});
  v.assign(grid.nhat, {});
  w.assign(grid.nhat, {});
  pfc::fft::kspace::for_each_kpoint(
      grid.outbox, grid.size, grid.spacing,
      [&](std::size_t idx, double, double, double, int i, int j, int k) {
        u[idx] = Complex(0.2 * i + 0.1, -0.3 * j);
        v[idx] = Complex(-0.4 * k + 0.2, 0.5 * i);
        w[idx] = Complex(0.7 * j, 0.1 * i - 0.2 * k);
      });
}

template <class Fx, class Fy, class Fz>
void sample_velocity(const Grid &grid, Fx &&fx, Fy &&fy, Fz &&fz,
                     std::vector<double> &rx, std::vector<double> &ry,
                     std::vector<double> &rz) {
  rx.assign(grid.nreal, 0.0);
  ry.assign(grid.nreal, 0.0);
  rz.assign(grid.nreal, 0.0);
  const auto &box = grid.inbox;
  for (int k = box.low[2]; k <= box.high[2]; ++k) {
    for (int j = box.low[1]; j <= box.high[1]; ++j) {
      for (int i = box.low[0]; i <= box.high[0]; ++i) {
        const auto idx = static_cast<std::size_t>(box.to_linear({i, j, k}));
        const double x = static_cast<double>(i) * grid.spacing[0];
        const double y = static_cast<double>(j) * grid.spacing[1];
        const double z = static_cast<double>(k) * grid.spacing[2];
        rx[idx] = fx(x, y, z);
        ry[idx] = fy(x, y, z);
        rz[idx] = fz(x, y, z);
      }
    }
  }
}

void forward_velocity(Grid &grid, const std::vector<double> &rx,
                      const std::vector<double> &ry, const std::vector<double> &rz,
                      std::vector<Complex> &u, std::vector<Complex> &v,
                      std::vector<Complex> &w) {
  u.resize(grid.nhat);
  v.resize(grid.nhat);
  w.resize(grid.nhat);
  grid.fft.forward(rx, u);
  grid.fft.forward(ry, v);
  grid.fft.forward(rz, w);
}

/// Project `-real` the same way `rotational_tendency` finishes a product.
void project_negated(Grid &grid, const std::vector<double> &rx,
                     const std::vector<double> &ry, const std::vector<double> &rz,
                     std::vector<Complex> &u, std::vector<Complex> &v,
                     std::vector<Complex> &w) {
  forward_velocity(grid, rx, ry, rz, u, v, w);
  for (std::size_t i = 0; i < grid.nhat; ++i) {
    u[i] = -u[i];
    v[i] = -v[i];
    w[i] = -w[i];
  }
  pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(), v.data(),
                            w.data(), grid.nhat);
  pfc::field::apply_two_thirds(grid.outbox, grid.size, grid.spacing, u.data(),
                               v.data(), w.data(), grid.nhat);
  pfc::field::zero_mean(grid.outbox, grid.size, grid.spacing, u.data(), v.data(),
                        w.data(), grid.nhat);
}

} // namespace

TEST_CASE("Leray projection removes the longitudinal part and is idempotent",
          "[field][leray]") {
  with_grid(16, 1, [](Grid &grid) {
    std::vector<Complex> u, v, w;
    fill_arbitrary(grid, u, v, w);
    const auto original_u = u;
    const auto original_v = v;
    const auto original_w = w;

    pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(),
                              v.data(), w.data(), grid.nhat);
    REQUIRE(pfc::field::max_modal_divergence(grid.outbox, grid.size, grid.spacing,
                                             u.data(), v.data(), w.data(),
                                             grid.nhat) < 1.0e-10);

    auto once_u = u;
    auto once_v = v;
    auto once_w = w;
    pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(),
                              v.data(), w.data(), grid.nhat);
    for (std::size_t i = 0; i < grid.nhat; ++i) {
      REQUIRE_THAT(std::abs(u[i] - once_u[i]), WithinAbs(0.0, 1.0e-12));
      REQUIRE_THAT(std::abs(v[i] - once_v[i]), WithinAbs(0.0, 1.0e-12));
      REQUIRE_THAT(std::abs(w[i] - once_w[i]), WithinAbs(0.0, 1.0e-12));
    }

    const std::size_t longitudinal = mode_at(grid.outbox, 1, 0, 0);
    u = original_u;
    v = original_v;
    w = original_w;
    std::fill(u.begin(), u.end(), Complex{});
    std::fill(v.begin(), v.end(), Complex{});
    std::fill(w.begin(), w.end(), Complex{});
    u[longitudinal] = Complex(9.0, 0.0);
    v[longitudinal] = Complex(1.0, 0.0);
    w[longitudinal] = Complex(-2.0, 0.0);
    pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(),
                              v.data(), w.data(), grid.nhat);
    REQUIRE_THAT(u[longitudinal].real(), WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(v[longitudinal].real(), WithinAbs(1.0, 1.0e-12));
    REQUIRE_THAT(w[longitudinal].real(), WithinAbs(-2.0, 1.0e-12));
  });
}

TEST_CASE("projection preserves a transverse mode, the mean, and the Nyquist mode",
          "[field][leray]") {
  constexpr int n = 16;
  with_grid(n, 1, [](Grid &grid) {
    std::vector<Complex> u(grid.nhat), v(grid.nhat), w(grid.nhat);
    const std::size_t transverse = mode_at(grid.outbox, 1, 0, 0);
    const std::size_t mean = mode_at(grid.outbox, 0, 0, 0);
    const std::size_t nyquist = mode_at(grid.outbox, n / 2, 0, 0);
    u[transverse] = Complex(0.0, 0.0);
    v[transverse] = Complex(3.0, -1.0);
    w[transverse] = Complex(-4.0, 0.5);
    u[mean] = Complex(1.5, 0.0);
    v[mean] = Complex(-2.5, 0.25);
    w[mean] = Complex(0.5, -0.5);
    u[nyquist] = Complex(4.0, 0.0);
    v[nyquist] = Complex(5.0, 0.0);
    w[nyquist] = Complex(6.0, 0.0);
    const auto before_u = u;
    const auto before_v = v;
    const auto before_w = w;

    pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(),
                              v.data(), w.data(), grid.nhat);
    for (std::size_t idx : {transverse, mean, nyquist}) {
      REQUIRE_THAT(std::abs(u[idx] - before_u[idx]), WithinAbs(0.0, 1.0e-12));
      REQUIRE_THAT(std::abs(v[idx] - before_v[idx]), WithinAbs(0.0, 1.0e-12));
      REQUIRE_THAT(std::abs(w[idx] - before_w[idx]), WithinAbs(0.0, 1.0e-12));
    }
    REQUIRE(pfc::field::max_modal_divergence(grid.outbox, grid.size, grid.spacing,
                                             u.data(), v.data(), w.data(),
                                             grid.nhat) < 1.0e-12);

    pfc::field::zero_mean(grid.outbox, grid.size, grid.spacing, u.data(), v.data(),
                          w.data(), grid.nhat);
    REQUIRE_THAT(std::abs(u[mean]), WithinAbs(0.0, 0.0));
    REQUIRE_THAT(std::abs(v[mean]), WithinAbs(0.0, 0.0));
    REQUIRE_THAT(std::abs(w[mean]), WithinAbs(0.0, 0.0));
    REQUIRE_THAT(std::abs(v[transverse] - before_v[transverse]),
                 WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(u[nyquist] - before_u[nyquist]), WithinAbs(0.0, 1.0e-12));
  });
}

TEST_CASE("spectral curl, gradient, and viscosity symbols match their definitions",
          "[field][leray]") {
  constexpr int n = 16;
  with_grid(n, 1, [](Grid &grid) {
    const std::size_t mode = mode_at(grid.outbox, 1, 2, 0);
    std::vector<Complex> phi(grid.nhat);
    phi[mode] = Complex(0.8, -0.3);
    std::vector<Complex> gx(grid.nhat), gy(grid.nhat), gz(grid.nhat);
    pfc::field::gradient_hat(grid.outbox, grid.size, grid.spacing, phi.data(),
                             gx.data(), gy.data(), gz.data(), grid.nhat);
    const auto wave = pfc::field::odd_wave(1, 2, 0, grid.size, grid.spacing);
    REQUIRE_THAT(std::abs(gx[mode] - Complex(0.0, wave.kx) * phi[mode]),
                 WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(gy[mode] - Complex(0.0, wave.ky) * phi[mode]),
                 WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(gz[mode] - Complex(0.0, wave.kz) * phi[mode]),
                 WithinAbs(0.0, 1.0e-12));

    std::vector<Complex> ox(grid.nhat), oy(grid.nhat), oz(grid.nhat);
    pfc::field::curl_hat(grid.outbox, grid.size, grid.spacing, gx.data(), gy.data(),
                         gz.data(), ox.data(), oy.data(), oz.data(), grid.nhat);
    REQUIRE(max_abs(ox) < 1.0e-12);
    REQUIRE(max_abs(oy) < 1.0e-12);
    REQUIRE(max_abs(oz) < 1.0e-12);

    std::vector<Complex> u, v, w;
    fill_arbitrary(grid, u, v, w);
    pfc::field::curl_hat(grid.outbox, grid.size, grid.spacing, u.data(), v.data(),
                         w.data(), ox.data(), oy.data(), oz.data(), grid.nhat);
    REQUIRE(pfc::field::max_modal_divergence(grid.outbox, grid.size, grid.spacing,
                                             ox.data(), oy.data(), oz.data(),
                                             grid.nhat) < 1.0e-10);

    const std::size_t decay_mode = mode_at(grid.outbox, 0, 1, 0);
    const auto decay_wave = pfc::field::odd_wave(0, 1, 0, grid.size, grid.spacing);
    REQUIRE_THAT(decay_wave.k2, WithinAbs(1.0, 1.0e-12));
    std::vector<Complex> lap(grid.nhat);
    lap[decay_mode] = Complex(1.25, -0.5);
    pfc::field::laplacian_hat(grid.outbox, grid.size, grid.spacing, lap.data(),
                              grid.nhat);
    REQUIRE_THAT(lap[decay_mode].real(), WithinAbs(-1.25, 1.0e-12));
    REQUIRE_THAT(lap[decay_mode].imag(), WithinAbs(0.5, 1.0e-12));

    std::vector<double> exp_dt, exp_half;
    pfc::incompressible::viscous_exponentials(grid.outbox, grid.size, grid.spacing,
                                              0.1, 0.05, exp_dt, exp_half);
    REQUIRE_THAT(exp_dt[decay_mode], WithinAbs(std::exp(-0.005), 1.0e-12));
    REQUIRE_THAT(exp_half[decay_mode], WithinAbs(std::exp(-0.0025), 1.0e-12));
    REQUIRE_THAT(exp_dt[mode_at(grid.outbox, 0, 0, 0)], WithinAbs(1.0, 1.0e-15));
    REQUIRE_THAT(exp_dt[mode_at(grid.outbox, n / 2, 0, 0)], WithinAbs(1.0, 1.0e-15));
  });
}

TEST_CASE("a 2/3 mask stops unresolved products from entering a retained mode",
          "[field][leray]") {
  // N=16, L=2π. Both waves sit outside the 2/3 ball (|k|=7 > 16/3). Their
  // quadratic product wraps to stored index (2, N-2, 0), wavevector (2, -2, 0),
  // which the mask would keep. Thresholds are fixed: the raw coefficient is
  // thousands, so deleting the mask fails `> 1`, and a live mask fails `< 1e-4`.
  constexpr int n = 16;
  with_grid(n, 1, [](Grid &grid) {
    std::vector<double> rx, ry, rz;
    sample_velocity(
        grid, [](double x, double y, double) { return std::sin(7.0 * x + y); },
        [](double x, double y, double) {
          return std::sin(7.0 * x) - 7.0 * std::sin(7.0 * x + y);
        },
        [](double, double, double) { return 0.0; }, rx, ry, rz);
    std::vector<Complex> u, v, w;
    forward_velocity(grid, rx, ry, rz, u, v, w);
    REQUIRE(std::abs(v[mode_at(grid.outbox, 7, 0, 0)]) > 1.0);

    const auto alias = pfc::field::odd_wave(2, n - 2, 0, grid.size, grid.spacing);
    REQUIRE_THAT(alias.kx, WithinAbs(2.0, 1.0e-12));
    REQUIRE_THAT(alias.ky, WithinAbs(-2.0, 1.0e-12));
    const std::size_t slot = mode_at(grid.outbox, 2, n - 2, 0);

    std::vector<Complex> raw_u, raw_v, raw_w, clean_u, clean_v, clean_w;
    pfc::incompressible::rotational_tendency(grid.fft, grid.size, grid.spacing, u, v,
                                             w, raw_u, raw_v, raw_w, false);
    pfc::incompressible::rotational_tendency(grid.fft, grid.size, grid.spacing, u, v,
                                             w, clean_u, clean_v, clean_w, true);
    REQUIRE(std::abs(raw_u[slot]) > 1.0);
    REQUIRE(std::abs(raw_v[slot]) > 1.0);
    REQUIRE(std::abs(clean_u[slot]) < 1.0e-4);
    REQUIRE(std::abs(clean_v[slot]) < 1.0e-4);
  });
}

TEST_CASE("rotational form matches advective and skew forms on a resolved field",
          "[field][leray]") {
  // Taylor-Green on N=32, L=2π. Modes ±1, products through mode 2, cutoff
  // index 32/3. The three forms are compared after Leray projection.
  constexpr int n = 32;
  with_grid(n, 1, [&](Grid &grid) {
    std::vector<double> rx, ry, rz;
    sample_velocity(
        grid,
        [](double x, double y, double z) {
          return std::sin(x) * std::cos(y) * std::cos(z);
        },
        [](double x, double y, double z) {
          return -std::cos(x) * std::sin(y) * std::cos(z);
        },
        [](double, double, double) { return 0.0; }, rx, ry, rz);
    std::vector<Complex> u, v, w;
    forward_velocity(grid, rx, ry, rz, u, v, w);
    pfc::field::apply_two_thirds(grid.outbox, grid.size, grid.spacing, u.data(),
                                 v.data(), w.data(), grid.nhat);

    auto gradient_real = [&](const std::vector<Complex> &hat) {
      std::vector<Complex> gx(grid.nhat), gy(grid.nhat), gz(grid.nhat);
      pfc::field::gradient_hat(grid.outbox, grid.size, grid.spacing, hat.data(),
                               gx.data(), gy.data(), gz.data(), grid.nhat);
      std::array<std::vector<double>, 3> real{std::vector<double>(grid.nreal),
                                              std::vector<double>(grid.nreal),
                                              std::vector<double>(grid.nreal)};
      grid.fft.backward(gx, real[0]);
      grid.fft.backward(gy, real[1]);
      grid.fft.backward(gz, real[2]);
      return real;
    };
    const auto gu = gradient_real(u);
    const auto gv = gradient_real(v);
    const auto gw = gradient_real(w);
    std::vector<double> ur(grid.nreal), vr(grid.nreal), wr(grid.nreal);
    grid.fft.backward(u, ur);
    grid.fft.backward(v, vr);
    grid.fft.backward(w, wr);

    std::vector<double> ax(grid.nreal), ay(grid.nreal), az(grid.nreal);
    std::vector<double> pxx(grid.nreal), pyy(grid.nreal), pzz(grid.nreal),
        pxy(grid.nreal), pxz(grid.nreal), pyz(grid.nreal);
    for (std::size_t i = 0; i < grid.nreal; ++i) {
      ax[i] = ur[i] * gu[0][i] + vr[i] * gu[1][i] + wr[i] * gu[2][i];
      ay[i] = ur[i] * gv[0][i] + vr[i] * gv[1][i] + wr[i] * gv[2][i];
      az[i] = ur[i] * gw[0][i] + vr[i] * gw[1][i] + wr[i] * gw[2][i];
      pxx[i] = ur[i] * ur[i];
      pyy[i] = vr[i] * vr[i];
      pzz[i] = wr[i] * wr[i];
      pxy[i] = ur[i] * vr[i];
      pxz[i] = ur[i] * wr[i];
      pyz[i] = vr[i] * wr[i];
    }
    auto derivative = [&](const std::vector<double> &field, int axis) {
      std::vector<Complex> hat(grid.nhat);
      grid.fft.forward(field, hat);
      return gradient_real(hat)[static_cast<std::size_t>(axis)];
    };
    std::vector<double> sx(grid.nreal), sy(grid.nreal), sz(grid.nreal);
    const auto dxx = derivative(pxx, 0);
    const auto dyx = derivative(pxy, 1);
    const auto dzx = derivative(pxz, 2);
    const auto dxy = derivative(pxy, 0);
    const auto dyy = derivative(pyy, 1);
    const auto dzy = derivative(pyz, 2);
    const auto dxz = derivative(pxz, 0);
    const auto dyz = derivative(pyz, 1);
    const auto dzz = derivative(pzz, 2);
    for (std::size_t i = 0; i < grid.nreal; ++i) {
      sx[i] = 0.5 * ax[i] + 0.5 * (dxx[i] + dyx[i] + dzx[i]);
      sy[i] = 0.5 * ay[i] + 0.5 * (dxy[i] + dyy[i] + dzy[i]);
      sz[i] = 0.5 * az[i] + 0.5 * (dxz[i] + dyz[i] + dzz[i]);
    }

    std::vector<Complex> rot_u, rot_v, rot_w, adv_u, adv_v, adv_w, skew_u, skew_v,
        skew_w;
    pfc::incompressible::rotational_tendency(grid.fft, grid.size, grid.spacing, u, v,
                                             w, rot_u, rot_v, rot_w, true);
    project_negated(grid, ax, ay, az, adv_u, adv_v, adv_w);
    project_negated(grid, sx, sy, sz, skew_u, skew_v, skew_w);
    require_relative(adv_u, rot_u, 1.0e-8);
    require_relative(adv_v, rot_v, 1.0e-8);
    require_relative(adv_w, rot_w, 1.0e-8);
    require_relative(skew_u, rot_u, 1.0e-8);
    require_relative(skew_v, rot_v, 1.0e-8);
    require_relative(skew_w, rot_w, 1.0e-8);

    std::vector<double> tu(grid.nreal), tv(grid.nreal), tw(grid.nreal);
    grid.fft.backward(rot_u, tu);
    grid.fft.backward(rot_v, tv);
    grid.fft.backward(rot_w, tw);
    double work = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < grid.nreal; ++i) {
      work += ur[i] * tu[i] + vr[i] * tv[i] + wr[i] * tw[i];
      scale += std::abs(ur[i] * tu[i]) + std::abs(vr[i] * tv[i]) +
               std::abs(wr[i] * tw[i]);
    }
    REQUIRE(scale > 1.0);
    REQUIRE(std::abs(work) / scale < 1.0e-8);
  });
}

TEST_CASE(
    "integrating-factor RK4 decays a transverse mode and drops a longitudinal force",
    "[field][leray]") {
  constexpr int n = 16;
  with_grid(n, 1, [&](Grid &grid) {
    const std::size_t mode = mode_at(grid.outbox, 0, 1, 0);
    const double nu = 0.1;
    const double dt = 0.05;
    const double expected = std::exp(-nu * dt);
    std::vector<double> exp_dt, exp_half;
    pfc::incompressible::viscous_exponentials(grid.outbox, grid.size, grid.spacing,
                                              nu, dt, exp_dt, exp_half);

    auto zero = [](const std::vector<Complex> &state, const std::vector<Complex> &,
                   const std::vector<Complex> &, std::vector<Complex> &tu,
                   std::vector<Complex> &tv, std::vector<Complex> &tw) {
      tu.assign(state.size(), {});
      tv.assign(state.size(), {});
      tw.assign(state.size(), {});
    };
    std::vector<Complex> u(grid.nhat), v(grid.nhat), w(grid.nhat);
    u[mode] = Complex(1.0, 0.0);
    pfc::incompressible::ifrk4_velocity(grid.outbox, grid.size, grid.spacing, u, v,
                                        w, exp_dt, exp_half, dt, zero, true);
    REQUIRE_THAT(u[mode].real(), WithinAbs(expected, 1.0e-12));
    REQUIRE_THAT(u[mode].imag(), WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(v[mode]), WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(w[mode]), WithinAbs(0.0, 1.0e-12));

    auto longitudinal = [&](const std::vector<Complex> &state,
                            const std::vector<Complex> &,
                            const std::vector<Complex> &, std::vector<Complex> &tu,
                            std::vector<Complex> &tv, std::vector<Complex> &tw) {
      tu.assign(state.size(), {});
      tv.assign(state.size(), {});
      tw.assign(state.size(), {});
      pfc::fft::kspace::for_each_kpoint(
          grid.outbox, grid.size, grid.spacing,
          [&](std::size_t idx, double, double, double, int i, int j, int k) {
            const auto wave = pfc::field::odd_wave(i, j, k, grid.size, grid.spacing);
            tu[idx] = Complex(wave.kx, 0.0);
            tv[idx] = Complex(wave.ky, 0.0);
            tw[idx] = Complex(wave.kz, 0.0);
          });
    };
    std::fill(u.begin(), u.end(), Complex{});
    std::fill(v.begin(), v.end(), Complex{});
    std::fill(w.begin(), w.end(), Complex{});
    u[mode] = Complex(1.0, 0.0);
    pfc::incompressible::ifrk4_velocity(grid.outbox, grid.size, grid.spacing, u, v,
                                        w, exp_dt, exp_half, dt, longitudinal, true);
    REQUIRE_THAT(u[mode].real(), WithinAbs(expected, 1.0e-12));
    REQUIRE_THAT(std::abs(v[mode]), WithinAbs(0.0, 1.0e-12));
    REQUIRE_THAT(std::abs(w[mode]), WithinAbs(0.0, 1.0e-12));
    REQUIRE(pfc::field::max_modal_divergence(grid.outbox, grid.size, grid.spacing,
                                             u.data(), v.data(), w.data(),
                                             grid.nhat) < 1.0e-12);
  });
}

TEST_CASE("Leray projection is divergence-free on each rank of a split",
          "[field][leray][MPI]") {
  if (world_size() != 2) SKIP("two-rank projection");
  constexpr int n = 16;
  with_grid(n, 2, [](Grid &grid) {
    std::vector<Complex> u, v, w;
    fill_arbitrary(grid, u, v, w);
    const auto u0 = u;
    const auto v0 = v;
    const auto w0 = w;
    pfc::field::leray_project(grid.outbox, grid.size, grid.spacing, u.data(),
                              v.data(), w.data(), grid.nhat);
    REQUIRE(pfc::field::max_modal_divergence(grid.outbox, grid.size, grid.spacing,
                                             u.data(), v.data(), w.data(),
                                             grid.nhat) < 1.0e-10);

    bool checked = false;
    pfc::fft::kspace::for_each_kpoint(
        grid.outbox, grid.size, grid.spacing,
        [&](std::size_t idx, double, double, double, int i, int j, int k) {
          if (checked) return;
          const auto wave = pfc::field::odd_wave(i, j, k, grid.size, grid.spacing);
          if (wave.k2 == 0.0) return;
          Complex cu = u0[idx];
          Complex cv = v0[idx];
          Complex cw = w0[idx];
          pfc::field::leray_mode(cu, cv, cw, wave);
          REQUIRE_THAT(std::abs(u[idx] - cu), WithinAbs(0.0, 1.0e-12));
          REQUIRE_THAT(std::abs(v[idx] - cv), WithinAbs(0.0, 1.0e-12));
          REQUIRE_THAT(std::abs(w[idx] - cw), WithinAbs(0.0, 1.0e-12));
          checked = true;
        });
    REQUIRE(checked);
  });
}
