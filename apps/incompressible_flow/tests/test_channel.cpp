// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_channel.cpp
 * @brief Plane Couette and plane Poiseuille on the Chebyshev wall.
 *
 * Both profiles are polynomials of degree at most 2, so every
 * degree from 2 upward is an exact grid. The check is the steady
 * balance: wall values, flow rate, wall shear, momentum, and a
 * zero acceleration. A viscous step holds both profiles and
 * advances sin(π z) at −ν π². The same step advances the
 * spanwise mode sin(π z) cos(2x) at −ν (4 + π²). Those
 * profiles have a zero convective product. The streamfunction
 * (1 − z²)² sin(x) matches a hand-derived product. One step
 * adds that acceleration to the viscous step. The projection
 * itself is checked here: a polynomial normal tendency
 * cancels, a streamwise mode stays impermeable, and a bad
 * grid is rejected. Turbulent channel statistics are not here.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <flow/channel.hpp>
#include <flow/wall.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Plane Couette and Poiseuille stay fixed on a channel wall", "[channel]") {
  const double nu = 0.3;
  const std::array<std::pair<int, int>, 3> grids{{{8, 6}, {8, 1}, {1, 6}}};
  const std::array<int, 3> degrees{{2, 8, 16}};
  for (const int degree : degrees) {
    for (const auto &grid : grids) {
      for (const auto profile :
           {flow::LaminarProfile::couette, flow::LaminarProfile::poiseuille}) {
        const auto report =
            flow::laminar_balance(profile, grid.first, grid.second, degree, nu);
        const bool couette = profile == flow::LaminarProfile::couette;
        REQUIRE(report.profile_error < 1e-10);
        REQUIRE(report.wall_error < 1e-10);
        REQUIRE_THAT(report.flow_rate, WithinAbs(couette ? 0.0 : 4.0 / 3.0, 1e-10));
        REQUIRE(report.flow_spread < 1e-10);
        REQUIRE_THAT(report.shear_plus, WithinAbs(couette ? 1.0 : -2.0, 1e-10));
        REQUIRE_THAT(report.shear_minus, WithinAbs(couette ? 1.0 : 2.0, 1e-10));
        REQUIRE(report.shear_spread < 1e-10);
        REQUIRE(std::abs(report.momentum) < 1e-10);
        REQUIRE(report.max_acceleration < 1e-8);
        REQUIRE(report.max_normal < 1e-9);
        REQUIRE(report.velocity_change < 1e-8);
        REQUIRE(report.step_drift < 1e-9);
      }
    }
  }
}

TEST_CASE("A wall-normal mode decays at the viscous rate", "[channel]") {
  const double nu = 0.3;
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const double step = 0.1;
  const std::array<std::pair<int, int>, 4> grids{{{1, 1}, {8, 1}, {1, 6}, {8, 6}}};

  const auto decay_error = [&](int nx, int ny, int degree, bool spanwise) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    const double wave = spanwise ? 4.0 : 0.0;
    const double rate = -nu * (wave + pi * pi);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    std::vector<double> initial(count, 0.0);
    for (int iz = 0; iz < nline; ++iz) {
      const double sine = std::sin(pi * z[static_cast<std::size_t>(iz)]);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const double sample = sine * (spanwise ? std::cos(2.0 * x) : 1.0);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          initial[index] = sample;
          if (spanwise) {
            velocity_y[index] = sample;
          } else {
            velocity_x[index] = sample;
          }
        }
      }
    }
    flow::viscous_advance(velocity_x, velocity_y, velocity_z, nx, ny, period, period,
                          nu, 0.0, step, 0.0, 0.0);
    const auto &advanced = spanwise ? velocity_y : velocity_x;
    const auto &quiet = spanwise ? velocity_x : velocity_y;
    double error = 0.0;
    double moved = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      const double expected = initial[i] * (1.0 + step * rate);
      error = std::max(error, std::abs(advanced[i] - expected));
      moved = std::max(moved, std::abs(advanced[i] - initial[i]));
      error = std::max(error, std::abs(quiet[i]));
      error = std::max(error, std::abs(velocity_z[i]));
    }
    REQUIRE(moved > 0.2);
    return error;
  };

  for (const auto &grid : grids) {
    const double coarse = decay_error(grid.first, grid.second, 16, false);
    const double fine = decay_error(grid.first, grid.second, 32, false);
    REQUIRE(coarse < 1e-7);
    REQUIRE(fine < 1e-9);
    REQUIRE(fine < coarse);
    if (grid.first >= 8) {
      const double mode_coarse = decay_error(grid.first, grid.second, 16, true);
      const double mode_fine = decay_error(grid.first, grid.second, 32, true);
      REQUIRE(mode_coarse < 1e-7);
      REQUIRE(mode_fine < 1e-9);
      REQUIRE(mode_fine < mode_coarse);
    }
  }
}

TEST_CASE("Parallel channel profiles have no convective term", "[channel]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const std::array<std::pair<int, int>, 3> grids{{{8, 6}, {8, 1}, {1, 6}}};

  const auto product_peak = [&](int nx, int ny, int degree, int kind) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    for (int iz = 0; iz < nline; ++iz) {
      const double node = z[static_cast<std::size_t>(iz)];
      const double sine = std::sin(pi * node);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          if (kind == 0) {
            velocity_x[index] = node;
          } else if (kind == 1) {
            velocity_x[index] = 1.0 - node * node;
          } else if (kind == 2) {
            velocity_x[index] = sine;
          } else {
            velocity_y[index] = sine * std::cos(2.0 * x);
          }
        }
      }
    }
    const auto term = flow::convective_tendency(velocity_x, velocity_y, velocity_z,
                                                nx, ny, period, period);
    double peak = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      peak = std::max(peak, std::abs(term.x[i]));
      peak = std::max(peak, std::abs(term.y[i]));
      peak = std::max(peak, std::abs(term.z[i]));
    }
    return peak;
  };

  for (const auto &grid : grids) {
    for (const int kind : {0, 1, 2}) {
      REQUIRE(product_peak(grid.first, grid.second, kind == 2 ? 16 : 2, kind) <
              1e-12);
    }
    if (grid.first >= 8) {
      REQUIRE(product_peak(grid.first, grid.second, 16, 3) < 1e-12);
    }
  }
}

TEST_CASE("A no-slip polynomial has a known convective term", "[channel]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto shape = [](double z) {
    const double factor = 1.0 - z * z;
    return factor * factor;
  };
  const auto slope = [](double z) { return 4.0 * z * (z * z - 1.0); };

  const auto residuals = [&](int nx, int ny, int degree) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    std::vector<double> expect_x(count, 0.0);
    std::vector<double> expect_y(count, 0.0);
    std::vector<double> expect_z(count, 0.0);
    for (int iz = 0; iz < nline; ++iz) {
      const double node = z[static_cast<std::size_t>(iz)];
      const double profile = shape(node);
      const double derivative = slope(node);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          velocity_x[index] = derivative * std::sin(x);
          velocity_z[index] = -profile * std::cos(x);
          expect_x[index] = -2.0 * profile * (1.0 + node * node) * std::sin(2.0 * x);
          expect_z[index] = -profile * derivative;
        }
      }
    }
    const auto term = flow::convective_tendency(velocity_x, velocity_y, velocity_z,
                                                nx, ny, period, period);
    double raw = 0.0;
    double product = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      raw = std::max(raw, std::abs(term.x[i] - expect_x[i]));
      raw = std::max(raw, std::abs(term.y[i] - expect_y[i]));
      raw = std::max(raw, std::abs(term.z[i] - expect_z[i]));
      product = std::max(product, std::abs(expect_x[i]));
      product = std::max(product, std::abs(expect_z[i]));
    }
    double projected_error = 0.0;
    double divergence = 0.0;
    double wall_normal = 0.0;
    if (degree >= 8) {
      const auto acceleration = flow::convective_acceleration(
          velocity_x, velocity_y, velocity_z, nx, ny, period, period);
      const auto projected = flow::impermeable_acceleration(
          expect_x, expect_y, expect_z, nx, ny, period, period);
      for (std::size_t i = 0; i < count; ++i) {
        projected_error =
            std::max(projected_error, std::abs(acceleration.x[i] - projected.x[i]));
        projected_error =
            std::max(projected_error, std::abs(acceleration.y[i] - projected.y[i]));
        projected_error =
            std::max(projected_error, std::abs(acceleration.z[i] - projected.z[i]));
      }
      const auto dx = pfc::fft::fourier_chebyshev_gradient(acceleration.x, nx, ny,
                                                           period, period);
      const auto dy = pfc::fft::fourier_chebyshev_gradient(acceleration.y, nx, ny,
                                                           period, period);
      const auto dz = pfc::fft::fourier_chebyshev_gradient(acceleration.z, nx, ny,
                                                           period, period);
      for (std::size_t i = 0; i < count; ++i) {
        divergence = std::max(divergence, std::abs(dx.x[i] + dy.y[i] + dz.z[i]));
      }
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          wall_normal = std::max(
              wall_normal,
              std::abs(acceleration.z[flow::channel_index(0, iy, ix, ny, nx)]));
          wall_normal = std::max(
              wall_normal,
              std::abs(
                  acceleration.z[flow::channel_index(nline - 1, iy, ix, ny, nx)]));
        }
      }
    }
    return std::array<double, 5>{
        {raw, projected_error, divergence, product, wall_normal}};
  };

  for (const auto grid : {std::pair<int, int>{8, 1}, std::pair<int, int>{8, 6}}) {
    const auto coarse = residuals(grid.first, grid.second, 4);
    const auto fine = residuals(grid.first, grid.second, 8);
    const auto finer = residuals(grid.first, grid.second, 16);
    INFO("degree-8 divergence=" << fine[2] << "; degree-16 divergence=" << finer[2]);
    // The velocity fits a degree-4 grid, so the sampled product is
    // exact from that degree. There is no coarser truncation gap.
    // Pressure contains a cosh(2z) mode even for polynomial velocity.
    // Differentiate on the enriched pressure grid before restriction.
    // The remaining degree-8 sampled-divergence truncation is below 1e-2;
    // degree 16 resolves this mode. No tolerance is relaxed.
    REQUIRE(coarse[3] > 0.5);
    REQUIRE(coarse[0] < 1e-10);
    REQUIRE(fine[0] < 1e-10);
    REQUIRE(finer[0] < 1e-10);
    REQUIRE(fine[1] < 1e-10);
    REQUIRE(finer[1] < 1e-10);
    REQUIRE(fine[2] < 1e-2);
    REQUIRE(finer[2] < 1e-9);
    REQUIRE(finer[2] < fine[2]);
    REQUIRE(fine[4] < 1e-9);
    REQUIRE(finer[4] < 1e-9);
  }
}

TEST_CASE("Channel pressure agrees with an independent hyperbolic solution",
          "[channel]") {
  const double period = 2 * std::acos(-1.0);
  // p''-4p = -4(1-z²)²(1+z²), p'(+/-1)=0. The polynomial particular
  // solution is 41/4 + 37z²/2 + 13z⁴/2 + z⁶. The even homogeneous
  // solution has amplitude -69/(2 sinh(2)). These formulas use no solver.
  const auto pressure = [](double z) {
    return 10.25 + z * z * (18.5 + z * z * (6.5 + z * z)) -
           69 * std::cosh(2 * z) / (2 * std::sinh(2.));
  };
  const auto slope = [](double z) {
    return z * (37 + z * z * (26 + 6 * z * z)) -
           69 * std::sinh(2 * z) / std::sinh(2.);
  };
  for (int degree : {8, 16, 32}) {
    const int nx = 8, ny = 3;
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> tx(nodes.size() * nx * ny), ty(tx.size()), tz(tx.size());
    for (int iz = 0; iz <= degree; ++iz)
      for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
          const double z = nodes[iz], x = period * ix / nx;
          tx[flow::channel_index(iz, iy, ix, ny, nx)] =
              -2 * (1 - z * z) * (1 - z * z) * (1 + z * z) * std::sin(2 * x);
        }
    const auto got =
        flow::impermeable_acceleration(tx, ty, tz, nx, ny, period, period);
    double error = 0, wall = 0;
    for (int iz = 0; iz <= degree; ++iz)
      for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
          const double z = nodes[iz], x = period * ix / nx;
          const auto i = flow::channel_index(iz, iy, ix, ny, nx);
          error = std::max(
              error, std::abs(got.x[i] - tx[i] - 2 * pressure(z) * std::sin(2 * x)));
          error = std::max(error, std::abs(got.z[i] + slope(z) * std::cos(2 * x)));
          error = std::max(error, std::abs(got.y[i]));
          if (iz == 0 || iz == degree) wall = std::max(wall, std::abs(got.z[i]));
        }
    INFO("degree=" << degree << " error=" << error << " wall=" << wall);
    REQUIRE(error < (degree == 8 ? 1e-5 : 1e-10));
    REQUIRE(wall < 1e-10);
  }
}

TEST_CASE("A parallel profile agrees with the viscous step", "[channel]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const double nu = 0.3;
  const double step = 0.1;
  const std::array<std::pair<int, int>, 3> grids{{{8, 6}, {8, 1}, {1, 6}}};

  const auto gap = [&](int nx, int ny, int degree, int kind) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    double wall_plus = 0.0;
    double wall_minus = 0.0;
    double force = 0.0;
    if (kind == 0) {
      wall_plus = 1.0;
      wall_minus = -1.0;
    } else if (kind == 1) {
      force = 2.0 * nu;
    }
    for (int iz = 0; iz < nline; ++iz) {
      const double node = z[static_cast<std::size_t>(iz)];
      const double sine = std::sin(pi * node);
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          if (kind == 0) {
            velocity_x[index] = node;
          } else if (kind == 1) {
            velocity_x[index] = 1.0 - node * node;
          } else if (kind == 2) {
            velocity_x[index] = sine;
          } else {
            velocity_y[index] = sine * std::cos(2.0 * x);
          }
        }
      }
    }
    auto linear_x = velocity_x;
    auto linear_y = velocity_y;
    auto linear_z = velocity_z;
    auto combined_x = velocity_x;
    auto combined_y = velocity_y;
    auto combined_z = velocity_z;
    flow::viscous_advance(linear_x, linear_y, linear_z, nx, ny, period, period, nu,
                          force, step, wall_plus, wall_minus);
    flow::channel_advance(combined_x, combined_y, combined_z, nx, ny, period, period,
                          nu, force, step, wall_plus, wall_minus);
    double difference = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      difference = std::max(difference, std::abs(combined_x[i] - linear_x[i]));
      difference = std::max(difference, std::abs(combined_y[i] - linear_y[i]));
      difference = std::max(difference, std::abs(combined_z[i] - linear_z[i]));
    }
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto top = flow::channel_index(0, iy, ix, ny, nx);
        const auto bottom = flow::channel_index(nline - 1, iy, ix, ny, nx);
        difference = std::max(difference, std::abs(combined_x[top] - wall_plus));
        difference = std::max(difference, std::abs(combined_x[bottom] - wall_minus));
        difference = std::max(difference, std::abs(combined_y[top]));
        difference = std::max(difference, std::abs(combined_y[bottom]));
        difference = std::max(difference, std::abs(combined_z[top]));
        difference = std::max(difference, std::abs(combined_z[bottom]));
      }
    }
    return difference;
  };

  for (const auto &grid : grids) {
    for (const int kind : {0, 1, 2}) {
      REQUIRE(gap(grid.first, grid.second, kind == 2 ? 16 : 2, kind) < 1e-10);
    }
    if (grid.first >= 8) {
      REQUIRE(gap(grid.first, grid.second, 16, 3) < 1e-10);
    }
  }
}

TEST_CASE("One channel step adds the convective acceleration", "[channel]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const double nu = 0.3;
  const double step = 0.1;
  const auto shape = [](double z) {
    const double factor = 1.0 - z * z;
    return factor * factor;
  };
  const auto slope = [](double z) { return 4.0 * z * (z * z - 1.0); };

  const auto check = [&](int nx, int ny, int degree) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    const int nline = static_cast<int>(z.size());
    const std::size_t count = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nline);
    std::vector<double> velocity_x(count, 0.0);
    std::vector<double> velocity_y(count, 0.0);
    std::vector<double> velocity_z(count, 0.0);
    for (int iz = 0; iz < nline; ++iz) {
      const double node = z[static_cast<std::size_t>(iz)];
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const double x =
              period * static_cast<double>(ix) / static_cast<double>(nx);
          const auto index = flow::channel_index(iz, iy, ix, ny, nx);
          velocity_x[index] = slope(node) * std::sin(x);
          velocity_z[index] = -shape(node) * std::cos(x);
        }
      }
    }
    const auto convection = flow::convective_acceleration(
        velocity_x, velocity_y, velocity_z, nx, ny, period, period);
    auto reference_x = velocity_x;
    auto reference_y = velocity_y;
    auto reference_z = velocity_z;
    flow::viscous_advance(reference_x, reference_y, reference_z, nx, ny, period,
                          period, nu, 0.0, step, 0.0, 0.0);
    auto viscous_x = reference_x;
    auto viscous_y = reference_y;
    auto viscous_z = reference_z;
    for (std::size_t i = 0; i < count; ++i) {
      reference_x[i] += step * convection.x[i];
      reference_y[i] += step * convection.y[i];
      reference_z[i] += step * convection.z[i];
    }
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto top = flow::channel_index(0, iy, ix, ny, nx);
        const auto bottom = flow::channel_index(nline - 1, iy, ix, ny, nx);
        reference_x[top] = 0.0;
        reference_x[bottom] = 0.0;
        reference_y[top] = 0.0;
        reference_y[bottom] = 0.0;
        reference_z[top] = 0.0;
        reference_z[bottom] = 0.0;
      }
    }
    auto combined_x = velocity_x;
    auto combined_y = velocity_y;
    auto combined_z = velocity_z;
    flow::channel_advance(combined_x, combined_y, combined_z, nx, ny, period, period,
                          nu, 0.0, step, 0.0, 0.0);
    double error = 0.0;
    double movement = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      error = std::max(error, std::abs(combined_x[i] - reference_x[i]));
      error = std::max(error, std::abs(combined_y[i] - reference_y[i]));
      error = std::max(error, std::abs(combined_z[i] - reference_z[i]));
      movement = std::max(movement, std::abs(combined_x[i] - viscous_x[i]));
      movement = std::max(movement, std::abs(combined_y[i] - viscous_y[i]));
      movement = std::max(movement, std::abs(combined_z[i] - viscous_z[i]));
    }
    return std::array<double, 2>{{error, movement}};
  };

  for (const auto grid : {std::pair<int, int>{8, 1}, std::pair<int, int>{8, 6}}) {
    const auto got = check(grid.first, grid.second, 16);
    // Step 0.1 times the projected convective acceleration moves
    // the interior by about 0.09. A dropped term would stay put.
    REQUIRE(got[0] < 1e-12);
    REQUIRE(got[1] > 0.05);
  }
}

namespace {

template <typename Sample>
[[nodiscard]] std::vector<double> volume_field(int nx, int ny, int degree,
                                               double period_x, double period_y,
                                               Sample sample) {
  const auto z = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) * z.size());
  for (int iz = 0; iz <= degree; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        values[(static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
                static_cast<std::size_t>(iy)) *
                   static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix)] =
            sample(x, y, z[static_cast<std::size_t>(iz)]);
      }
    }
  }
  return values;
}

[[nodiscard]] double max_abs_diff(const std::vector<double> &left,
                                  const std::vector<double> &right) {
  double peak = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    peak = std::max(peak, std::abs(left[i] - right[i]));
  }
  return peak;
}

[[nodiscard]] double max_abs(const std::vector<double> &values) {
  double peak = 0.0;
  for (double value : values) peak = std::max(peak, std::abs(value));
  return peak;
}

[[nodiscard]] double wall_normal(const flow::WallAcceleration &acceleration, int nx,
                                 int ny, int nline) {
  double peak = 0.0;
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const auto top = static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                       static_cast<std::size_t>(ix);
      const auto bottom =
          (static_cast<std::size_t>(nline - 1) * static_cast<std::size_t>(ny) +
           static_cast<std::size_t>(iy)) *
              static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(ix);
      peak = std::max(peak, std::abs(acceleration.z[top]));
      peak = std::max(peak, std::abs(acceleration.z[bottom]));
    }
  }
  return peak;
}

} // namespace

TEST_CASE("Channel wall cancels a polynomial normal tendency", "[channel]") {
  const double period = 2.0 * std::acos(-1.0);
  const auto zero = [](double, double, double) { return 0.0; };
  const std::array grids{std::pair{8, 6}, std::pair{8, 1}, std::pair{1, 6}};
  for (const int degree : {2, 4, 8, 16}) {
    for (const auto &grid : grids) {
      const int nx = grid.first;
      const int ny = grid.second;
      const auto quiescent = volume_field(nx, ny, degree, period, period, zero);
      for (const auto &sample : {std::function<double(double, double, double)>{
                                     [](double, double, double) { return 1.0; }},
                                 std::function<double(double, double, double)>{
                                     [](double, double, double z) { return z; }}}) {
        const auto normal = volume_field(nx, ny, degree, period, period, sample);
        const auto got = flow::impermeable_acceleration(quiescent, quiescent, normal,
                                                        nx, ny, period, period);
        REQUIRE(max_abs(got.x) < 1e-10);
        REQUIRE(max_abs(got.y) < 1e-10);
        REQUIRE(max_abs(got.z) < 1e-10);
        const auto again = flow::impermeable_acceleration(got.x, got.y, got.z, nx,
                                                          ny, period, period);
        REQUIRE(max_abs(again.x) < 1e-10);
        REQUIRE(max_abs(again.y) < 1e-10);
        REQUIRE(max_abs(again.z) < 1e-10);
      }

      if (nx >= 8) {
        const auto horizontal =
            volume_field(nx, ny, degree, period, period,
                         [](double x, double, double) { return std::cos(2.0 * x); });
        const auto removed = flow::impermeable_acceleration(
            horizontal, quiescent, quiescent, nx, ny, period, period);
        REQUIRE(max_abs(removed.x) < 1e-10);
        REQUIRE(max_abs(removed.y) < 1e-10);
        REQUIRE(max_abs(removed.z) < 1e-10);
      }
    }
  }
}

TEST_CASE("Channel wall keeps a streamwise mode impermeable", "[channel]") {
  const double period = 2.0 * std::acos(-1.0);
  const auto zero = [](double, double, double) { return 0.0; };
  const auto sample = [](double x, double, double z) {
    return z * std::cos(2.0 * x);
  };
  for (const auto &grid : {std::pair{8, 6}, std::pair{8, 1}}) {
    for (const int degree : {2, 8, 16}) {
      const int nx = grid.first;
      const int ny = grid.second;
      const auto tendency_x = volume_field(nx, ny, degree, period, period, sample);
      const auto quiescent = volume_field(nx, ny, degree, period, period, zero);
      const auto got = flow::impermeable_acceleration(
          tendency_x, quiescent, quiescent, nx, ny, period, period);
      REQUIRE(wall_normal(got, nx, ny, degree + 1) < 1e-9);
      // Degree 2 puts this divergence into the two tau modes, so the
      // pressure correction is zero there. A resolved degree sees it.
      if (degree >= 8) {
        REQUIRE(max_abs_diff(got.x, tendency_x) > 1e-3);
        REQUIRE(max_abs(got.z) > 1e-3);
      }
      if (degree == 16) {
        const auto again = flow::impermeable_acceleration(got.x, got.y, got.z, nx,
                                                          ny, period, period);
        REQUIRE(max_abs_diff(again.x, got.x) < 1e-8);
        REQUIRE(max_abs_diff(again.y, got.y) < 1e-8);
        REQUIRE(max_abs_diff(again.z, got.z) < 1e-8);
      }
    }
  }
}

TEST_CASE("Channel wall rejects a bad grid", "[channel]") {
  const std::vector<double> empty;
  const std::vector<double> plane{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> slab{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_field{0.0, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(
      flow::impermeable_acceleration(empty, empty, empty, 2, 2, 1.0, 1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(flow::impermeable_acceleration(slab, slab, slab, 0, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(flow::impermeable_acceleration(slab, slab, slab, 2, 0, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      flow::impermeable_acceleration(slab, short_field, slab, 2, 2, 1.0, 1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(flow::impermeable_acceleration(slab, slab, slab, 3, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(flow::impermeable_acceleration(slab, slab, slab, 2, 2, 0.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      flow::impermeable_acceleration(slab, slab, slab, 2, 2, 1.0, -1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      flow::impermeable_acceleration(plane, plane, plane, 2, 2, 1.0, 1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      flow::impermeable_acceleration(plane, plane, plane, 2, 2, -2.0, 1.0),
      std::invalid_argument);
}
