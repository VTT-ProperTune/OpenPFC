// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/field/level_set.hpp>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using pfc::field::level_set::advance_level_set;
using pfc::field::level_set::level_set_count;
using pfc::field::level_set::level_set_halo_width;
using pfc::field::level_set::level_set_negative_centroid;
using pfc::field::level_set::level_set_rate;
using pfc::field::level_set::level_set_rate_x_slab;
using pfc::field::level_set::level_set_stable_dt;
using pfc::field::level_set::level_set_zero_crossing_x;
using pfc::field::level_set::LevelSetCentroid;
using pfc::field::level_set::LevelSetGrid;
using pfc::field::level_set::LevelSetMode;
using pfc::field::level_set::LevelSetStep;
using pfc::field::level_set::redistance_level_set;

namespace {

void report(const std::string &label, double a, double b, double c, double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "LEVEL_SET " << label << " " << a << " " << b << " " << c << " " << d;
  std::cout << out.str() << '\n';
}

LevelSetGrid make_grid(int nx, int ny, int nz) {
  LevelSetGrid grid;
  grid.nx = nx;
  grid.ny = ny;
  grid.nz = nz;
  grid.hx = 1.0 / static_cast<double>(nx);
  grid.hy = 1.0 / static_cast<double>(ny);
  grid.hz = nz == 1 ? 1.0 : 1.0 / static_cast<double>(nz);
  return grid;
}

std::size_t index_of(const LevelSetGrid &grid, int i, int j, int k) {
  return static_cast<std::size_t>(i + grid.nx * (j + grid.ny * k));
}

int wrap_of(int i, int n) {
  const int m = i % n;
  return m < 0 ? m + n : m;
}

double cell_x(const LevelSetGrid &grid, int i) {
  return (static_cast<double>(i) + 0.5) * grid.hx;
}

double cell_y(const LevelSetGrid &grid, int j) {
  return (static_cast<double>(j) + 0.5) * grid.hy;
}

double cell_z(const LevelSetGrid &grid, int k) {
  return (static_cast<double>(k) + 0.5) * grid.hz;
}

void fill_ball(const LevelSetGrid &grid, std::vector<double> &phi, double cx,
               double cy, double cz, double radius) {
  phi.assign(level_set_count(grid), 0.0);
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const double dx = cell_x(grid, i) - cx;
        const double dy = cell_y(grid, j) - cy;
        const double dz = grid.nz == 1 ? 0.0 : cell_z(grid, k) - cz;
        phi[index_of(grid, i, j, k)] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
      }
    }
  }
}

double field_at(const LevelSetGrid &grid, const std::vector<double> &phi, int i,
                int j, int k) {
  return phi[index_of(grid, wrap_of(i, grid.nx), wrap_of(j, grid.ny),
                      wrap_of(k, grid.nz))];
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double gap = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    gap = std::max(gap, std::abs(a[i] - b[i]));
  return gap;
}

int nearest_cell(double x, double h, int n) {
  int i = static_cast<int>(std::floor(x / h));
  if (i < 0) i = 0;
  if (i >= n) i = n - 1;
  return i;
}

int interior_i(const LevelSetGrid &grid, const std::vector<double> &phi, double cx,
               double cy, int k) {
  const int j = nearest_cell(cy, grid.hy, grid.ny);
  const int i0 = nearest_cell(cx, grid.hx, grid.nx);
  if (phi[index_of(grid, i0, j, k)] < 0.0) return i0;
  for (int d = 1; d < grid.nx; ++d) {
    if (i0 - d >= 0 && phi[index_of(grid, i0 - d, j, k)] < 0.0) return i0 - d;
    if (i0 + d < grid.nx && phi[index_of(grid, i0 + d, j, k)] < 0.0) return i0 + d;
  }
  throw std::invalid_argument("level-set test found no negative cell");
}

struct Sample {
  double interface_error;
  double volume_error;
  double cx;
  double volume;
};

Sample compare_ball(const LevelSetGrid &grid, const std::vector<double> &phi,
                    double cx, double cy, double cz, double radius) {
  std::vector<double> exact;
  fill_ball(grid, exact, cx, cy, cz, radius);
  const LevelSetCentroid got = level_set_negative_centroid(grid, phi);
  const LevelSetCentroid ref = level_set_negative_centroid(grid, exact);
  const int j = nearest_cell(cy, grid.hy, grid.ny);
  const int k = grid.nz == 1 ? 0 : nearest_cell(cz, grid.hz, grid.nz);
  const int i_phi = interior_i(grid, phi, cx, cy, k);
  const int i_ref = interior_i(grid, exact, cx, cy, k);
  const double x_phi = level_set_zero_crossing_x(grid, phi, j, k, i_phi);
  const double x_ref = level_set_zero_crossing_x(grid, exact, j, k, i_ref);
  Sample sample;
  sample.interface_error = std::abs(x_phi - x_ref);
  sample.volume_error = std::abs(got.volume - ref.volume);
  sample.cx = got.x;
  sample.volume = got.volume;
  return sample;
}

int stable_steps(const LevelSetGrid &grid, double max_speed, double cfl,
                 double horizon, double &dt) {
  const double stable = level_set_stable_dt(grid, max_speed, cfl);
  int steps = static_cast<int>(std::ceil(horizon / stable));
  if (steps < 1) steps = 1;
  dt = horizon / static_cast<double>(steps);
  return steps;
}

std::vector<double> constant_field(std::size_t n, double value) {
  return std::vector<double>(n, value);
}

} // namespace

TEST_CASE("level-set upwind matches an independent one-step oracle",
          "[unit][field][level_set]") {
  const LevelSetGrid grid = make_grid(16, 8, 1);
  const std::size_t n = level_set_count(grid);
  std::vector<double> phi(n);
  std::vector<double> vx(n, 0.2);
  std::vector<double> vy(n, -0.15);
  std::vector<double> vz(n, 0.0);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const double x = cell_x(grid, i);
      const double y = cell_y(grid, j);
      phi[index_of(grid, i, j, 0)] = std::sin(2.0 * 3.141592653589793 * x) +
                                     0.3 * std::sin(2.0 * 3.141592653589793 * y);
    }
  }
  std::vector<double> rate(n);
  level_set_rate(grid, phi, LevelSetMode::Advection, vx, vy, vz, {}, {}, rate);

  const int i = 3;
  const int j = 1;
  const std::size_t p = index_of(grid, i, j, 0);
  const double here = field_at(grid, phi, i, j, 0);
  const double dmx = (here - field_at(grid, phi, i - 1, j, 0)) / grid.hx;
  const double dpx = (field_at(grid, phi, i + 1, j, 0) - here) / grid.hx;
  const double dmy = (here - field_at(grid, phi, i, j - 1, 0)) / grid.hy;
  const double dpy = (field_at(grid, phi, i, j + 1, 0) - here) / grid.hy;
  const double ux = vx[p] > 0.0 ? vx[p] * dmx : vx[p] * dpx;
  const double uy = vy[p] < 0.0 ? vy[p] * dpy : vy[p] * dmy;
  const double expected = -(ux + uy);
  report("oracle_cell", rate[p], expected, rate[p] - expected, 0.0);
  REQUIRE(std::abs(rate[p] - expected) < 1e-12);

  std::vector<double> oracle(n);
  for (int jj = 0; jj < grid.ny; ++jj) {
    for (int ii = 0; ii < grid.nx; ++ii) {
      const std::size_t q = index_of(grid, ii, jj, 0);
      const double c = field_at(grid, phi, ii, jj, 0);
      const double mx = (c - field_at(grid, phi, ii - 1, jj, 0)) / grid.hx;
      const double px = (field_at(grid, phi, ii + 1, jj, 0) - c) / grid.hx;
      const double my = (c - field_at(grid, phi, ii, jj - 1, 0)) / grid.hy;
      const double py = (field_at(grid, phi, ii, jj + 1, 0) - c) / grid.hy;
      const double ux_q =
          vx[q] > 0.0 ? vx[q] * mx : (vx[q] < 0.0 ? vx[q] * px : 0.0);
      const double uy_q =
          vy[q] > 0.0 ? vy[q] * my : (vy[q] < 0.0 ? vy[q] * py : 0.0);
      oracle[q] = -(ux_q + uy_q);
    }
  }
  const double rate_gap = max_abs_diff(rate, oracle);
  report("oracle_field", rate_gap, 0.0, 0.0, 0.0);
  REQUIRE(rate_gap < 1e-12);

  constexpr double dt = 0.01;
  std::vector<double> stage(n);
  std::vector<double> second(n);
  std::vector<double> rk(n);
  for (std::size_t q = 0; q < n; ++q) stage[q] = phi[q] + dt * oracle[q];
  for (int jj = 0; jj < grid.ny; ++jj) {
    for (int ii = 0; ii < grid.nx; ++ii) {
      const std::size_t q = index_of(grid, ii, jj, 0);
      const double c = field_at(grid, stage, ii, jj, 0);
      const double mx = (c - field_at(grid, stage, ii - 1, jj, 0)) / grid.hx;
      const double px = (field_at(grid, stage, ii + 1, jj, 0) - c) / grid.hx;
      const double my = (c - field_at(grid, stage, ii, jj - 1, 0)) / grid.hy;
      const double py = (field_at(grid, stage, ii, jj + 1, 0) - c) / grid.hy;
      const double ux_q =
          vx[q] > 0.0 ? vx[q] * mx : (vx[q] < 0.0 ? vx[q] * px : 0.0);
      const double uy_q =
          vy[q] > 0.0 ? vy[q] * my : (vy[q] < 0.0 ? vy[q] * py : 0.0);
      second[q] = -(ux_q + uy_q);
    }
  }
  for (std::size_t q = 0; q < n; ++q)
    rk[q] = 0.5 * phi[q] + 0.5 * (stage[q] + dt * second[q]);

  std::vector<double> stepped = phi;
  const LevelSetStep step = advance_level_set(grid, stepped, LevelSetMode::Advection,
                                              vx, vy, vz, {}, {}, dt);
  const double step_gap = max_abs_diff(stepped, rk);
  report("oracle_rk2", step_gap, step.dt, step.negative_volume, 0.0);
  REQUIRE(step_gap < 1e-12);
  REQUIRE(step.dt == dt);
}

TEST_CASE("level-set Godunov flux matches an independent stencil",
          "[unit][field][level_set]") {
  const LevelSetGrid grid = make_grid(12, 10, 1);
  const std::size_t n = level_set_count(grid);
  std::vector<double> phi;
  fill_ball(grid, phi, 0.45, 0.5, 0.0, 0.16);
  std::vector<double> speed(n, 0.05);
  speed[index_of(grid, 2, 2, 0)] = -0.07;
  std::vector<double> rate(n);
  level_set_rate(grid, phi, LevelSetMode::NormalSpeed, {}, {}, {}, speed, {}, rate);

  const int samples[2][2] = {{4, 5}, {2, 2}};
  for (const auto &sample : samples) {
    const int i = sample[0];
    const int j = sample[1];
    const std::size_t p = index_of(grid, i, j, 0);
    const double c = field_at(grid, phi, i, j, 0);
    const double dmx = (c - field_at(grid, phi, i - 1, j, 0)) / grid.hx;
    const double dpx = (field_at(grid, phi, i + 1, j, 0) - c) / grid.hx;
    const double dmy = (c - field_at(grid, phi, i, j - 1, 0)) / grid.hy;
    const double dpy = (field_at(grid, phi, i, j + 1, 0) - c) / grid.hy;
    const double vn = speed[p];
    auto square = [](double s, double backward, double forward) {
      const double a = s >= 0.0 ? (backward > 0.0 ? backward : 0.0)
                                : (backward < 0.0 ? backward : 0.0);
      const double b = s >= 0.0 ? (forward < 0.0 ? forward : 0.0)
                                : (forward > 0.0 ? forward : 0.0);
      return std::max(a * a, b * b);
    };
    const double expected =
        -vn * std::sqrt(square(vn, dmx, dpx) + square(vn, dmy, dpy));
    report("godunov_cell", rate[p], expected, rate[p] - expected, vn);
    REQUIRE(std::abs(rate[p] - expected) < 1e-12);
  }
}

TEST_CASE("a translating circle converges and a stationary field does not",
          "[unit][field][level_set]") {
  constexpr double cx0 = 0.35;
  constexpr double cy0 = 0.50;
  constexpr double radius = 0.15;
  constexpr double speed = 0.25;
  constexpr double horizon = 0.40;
  constexpr double cfl = 0.35;
  double coarse_interface = 0.0;
  double coarse_volume = 0.0;
  for (int n : {32, 64}) {
    const LevelSetGrid grid = make_grid(n, n, 1);
    std::vector<double> phi;
    fill_ball(grid, phi, cx0, cy0, 0.0, radius);
    const std::vector<double> initial = phi;
    const std::size_t count = phi.size();
    const auto vx = constant_field(count, speed);
    const auto vy = constant_field(count, 0.0);
    const auto vz = constant_field(count, 0.0);
    double dt = 0.0;
    const int steps = stable_steps(grid, speed, cfl, horizon, dt);
    REQUIRE(dt <= level_set_stable_dt(grid, speed, cfl));
    std::vector<double> interface_hist;
    std::vector<double> volume_hist;
    interface_hist.reserve(static_cast<std::size_t>(steps) + 1);
    volume_hist.reserve(static_cast<std::size_t>(steps) + 1);
    for (int s = 0; s <= steps; ++s) {
      const double time = dt * static_cast<double>(s);
      const Sample now =
          compare_ball(grid, phi, cx0 + speed * time, cy0, 0.0, radius);
      interface_hist.push_back(now.interface_error);
      volume_hist.push_back(now.volume_error);
      if (s == steps) break;
      const LevelSetStep step = advance_level_set(grid, phi, LevelSetMode::Advection,
                                                  vx, vy, vz, {}, {}, dt);
      REQUIRE(step.dt == dt);
    }
    const Sample stuck =
        compare_ball(grid, initial, cx0 + speed * horizon, cy0, 0.0, radius);
    const LevelSetCentroid moved = level_set_negative_centroid(grid, phi);
    const LevelSetCentroid stayed = level_set_negative_centroid(grid, initial);
    report("translate", static_cast<double>(n), interface_hist.back(),
           volume_hist.back(), std::abs(moved.x - stayed.x));
    REQUIRE(interface_hist.size() == static_cast<std::size_t>(steps) + 1);
    REQUIRE(volume_hist.size() == interface_hist.size());
    REQUIRE(std::isfinite(volume_hist.back()));
    REQUIRE(std::abs(moved.x - stayed.x) > 0.5 * speed * horizon);
    REQUIRE(interface_hist.back() < stuck.interface_error);
    REQUIRE(interface_hist.back() < 2e-3);
    if (n == 32) {
      coarse_interface = interface_hist.back();
      coarse_volume = volume_hist.back();
    } else {
      report("translate_ratio", interface_hist.back() / coarse_interface,
             volume_hist.back() / coarse_volume, coarse_interface, coarse_volume);
      REQUIRE(interface_hist.back() < coarse_interface);
    }
  }
}

TEST_CASE("a circle rotating off the pivot follows the manufactured centre",
          "[unit][field][level_set]") {
  constexpr double pivot_x = 0.50;
  constexpr double pivot_y = 0.50;
  constexpr double cx0 = 0.62;
  constexpr double cy0 = 0.50;
  constexpr double radius = 0.08;
  constexpr double omega = 1.0;
  constexpr double horizon = 1.5707963267948966;
  constexpr double cfl = 0.35;
  double coarse = 0.0;
  for (int n : {32, 64}) {
    const LevelSetGrid grid = make_grid(n, n, 1);
    const std::size_t count = level_set_count(grid);
    std::vector<double> vx(count);
    std::vector<double> vy(count);
    std::vector<double> vz(count, 0.0);
    double max_speed = 0.0;
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t p = index_of(grid, i, j, 0);
        vx[p] = -omega * (cell_y(grid, j) - pivot_y);
        vy[p] = omega * (cell_x(grid, i) - pivot_x);
        max_speed = std::max(max_speed, std::hypot(vx[p], vy[p]));
      }
    }
    std::vector<double> phi;
    fill_ball(grid, phi, cx0, cy0, 0.0, radius);
    const std::vector<double> initial = phi;
    double dt = 0.0;
    const int steps = stable_steps(grid, max_speed, cfl, horizon, dt);
    std::vector<double> center_hist;
    std::vector<double> volume_hist;
    for (int s = 0; s <= steps; ++s) {
      const double angle = omega * dt * static_cast<double>(s);
      const double rx = cx0 - pivot_x;
      const double ry = cy0 - pivot_y;
      const double cx = pivot_x + std::cos(angle) * rx - std::sin(angle) * ry;
      const double cy = pivot_y + std::sin(angle) * rx + std::cos(angle) * ry;
      const LevelSetCentroid got = level_set_negative_centroid(grid, phi);
      std::vector<double> exact;
      fill_ball(grid, exact, cx, cy, 0.0, radius);
      const LevelSetCentroid ref = level_set_negative_centroid(grid, exact);
      center_hist.push_back(std::hypot(got.x - ref.x, got.y - ref.y));
      volume_hist.push_back(std::abs(got.volume - ref.volume));
      if (s == steps) break;
      const LevelSetStep step = advance_level_set(grid, phi, LevelSetMode::Advection,
                                                  vx, vy, vz, {}, {}, dt);
      REQUIRE(step.dt == dt);
    }
    const double angle = omega * horizon;
    const double rx = cx0 - pivot_x;
    const double ry = cy0 - pivot_y;
    const double cx = pivot_x + std::cos(angle) * rx - std::sin(angle) * ry;
    const double cy = pivot_y + std::sin(angle) * rx + std::cos(angle) * ry;
    std::vector<double> exact_final;
    fill_ball(grid, exact_final, cx, cy, 0.0, radius);
    const LevelSetCentroid final_ref =
        level_set_negative_centroid(grid, exact_final);
    const LevelSetCentroid stayed = level_set_negative_centroid(grid, initial);
    const LevelSetCentroid moved = level_set_negative_centroid(grid, phi);
    const double stuck = std::hypot(stayed.x - final_ref.x, stayed.y - final_ref.y);
    const double travel = std::hypot(moved.x - stayed.x, moved.y - stayed.y);
    report("rotate", static_cast<double>(n), center_hist.back(), volume_hist.back(),
           travel);
    REQUIRE(center_hist.size() == static_cast<std::size_t>(steps) + 1);
    REQUIRE(volume_hist.size() == center_hist.size());
    REQUIRE(std::isfinite(volume_hist.back()));
    REQUIRE(travel > 0.5 * stuck);
    REQUIRE(center_hist.back() < stuck);
    REQUIRE(center_hist.back() < 0.03);
    if (n == 32) {
      coarse = center_hist.back();
    } else {
      report("rotate_ratio", center_hist.back() / coarse, coarse, stuck, travel);
      REQUIRE(center_hist.back() < coarse);
    }
  }
}

TEST_CASE("normal speed expands and contracts a circle",
          "[unit][field][level_set]") {
  constexpr double cfl = 0.35;
  constexpr double vn_mag = 0.05;
  constexpr double horizon = 0.20;
  double coarse_expand = 0.0;
  for (int n : {48, 96}) {
    const LevelSetGrid grid = make_grid(n, n, 1);
    const double target = 0.20 + vn_mag * horizon;
    std::vector<double> phi;
    fill_ball(grid, phi, 0.5, 0.5, 0.0, 0.20);
    const auto speed = constant_field(phi.size(), vn_mag);
    double dt = 0.0;
    const int steps = stable_steps(grid, vn_mag, cfl, horizon, dt);
    std::vector<double> radius_hist;
    std::vector<double> volume_hist;
    for (int s = 0; s <= steps; ++s) {
      const double radius = 0.20 + vn_mag * dt * static_cast<double>(s);
      const Sample now = compare_ball(grid, phi, 0.5, 0.5, 0.0, radius);
      const int i0 = interior_i(grid, phi, 0.5, 0.5, 0);
      const int j0 = nearest_cell(0.5, grid.hy, grid.ny);
      const double crossing = level_set_zero_crossing_x(grid, phi, j0, 0, i0);
      radius_hist.push_back(crossing - 0.5);
      volume_hist.push_back(now.volume_error);
      if (s == steps) break;
      const LevelSetStep step = advance_level_set(
          grid, phi, LevelSetMode::NormalSpeed, {}, {}, {}, speed, {}, dt);
      REQUIRE(step.dt == dt);
    }
    const double radius_error = std::abs(radius_hist.back() - target);
    const double stuck_error = std::abs(0.20 - target);
    report("expand", static_cast<double>(n), radius_hist.back(), radius_error,
           volume_hist.back());
    REQUIRE(radius_hist.size() == static_cast<std::size_t>(steps) + 1);
    REQUIRE(radius_hist.back() > radius_hist.front());
    REQUIRE(radius_error < stuck_error);
    REQUIRE(radius_error < 5e-4);
    REQUIRE(volume_hist.back() < volume_hist.front() + stuck_error);
    if (n == 48) {
      coarse_expand = radius_error;
    } else {
      report("expand_ratio", radius_error / coarse_expand, coarse_expand, target,
             radius_hist.back());
      REQUIRE(radius_error < coarse_expand);
    }
  }

  const LevelSetGrid grid = make_grid(64, 64, 1);
  const double start = 0.24;
  const double target = start - vn_mag * horizon;
  std::vector<double> phi;
  fill_ball(grid, phi, 0.5, 0.5, 0.0, start);
  const auto speed = constant_field(phi.size(), -vn_mag);
  double dt = 0.0;
  const int steps = stable_steps(grid, vn_mag, cfl, horizon, dt);
  std::vector<double> radius_hist;
  for (int s = 0; s <= steps; ++s) {
    const int i0 = interior_i(grid, phi, 0.5, 0.5, 0);
    const int j0 = nearest_cell(0.5, grid.hy, grid.ny);
    radius_hist.push_back(level_set_zero_crossing_x(grid, phi, j0, 0, i0) - 0.5);
    if (s == steps) break;
    const LevelSetStep step = advance_level_set(grid, phi, LevelSetMode::NormalSpeed,
                                                {}, {}, {}, speed, {}, dt);
    REQUIRE(step.dt == dt);
  }
  const double radius_error = std::abs(radius_hist.back() - target);
  report("contract", radius_hist.front(), radius_hist.back(), radius_error, target);
  REQUIRE(radius_hist.back() < radius_hist.front());
  REQUIRE(radius_error < std::abs(start - target));
  REQUIRE(radius_error < 5e-4);
}

TEST_CASE("redistance pulls a scaled circle toward unit slope",
          "[unit][field][level_set]") {
  const LevelSetGrid grid = make_grid(64, 64, 1);
  constexpr double radius = 0.20;
  constexpr double scale = 2.5;
  constexpr double cfl = 0.30;
  std::vector<double> phi;
  fill_ball(grid, phi, 0.5, 0.5, 0.0, radius);
  for (double &value : phi) value *= scale;
  const std::vector<double> sign = phi;
  const int j0 = nearest_cell(0.5, grid.hy, grid.ny);
  const int i_before = interior_i(grid, phi, 0.5, 0.5, 0);
  const double x_before = level_set_zero_crossing_x(grid, phi, j0, 0, i_before);

  auto slope_error = [&](const std::vector<double> &field) {
    double sum = 0.0;
    int count = 0;
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const double dx = cell_x(grid, i) - 0.5;
        const double dy = cell_y(grid, j) - 0.5;
        const double r = std::sqrt(dx * dx + dy * dy);
        const double band = std::abs(r - radius);
        if (band < 2.0 * grid.hx || band > 6.0 * grid.hx || r < 3.0 * grid.hx)
          continue;
        const double gx = (field_at(grid, field, i + 1, j, 0) -
                           field_at(grid, field, i - 1, j, 0)) /
                          (2.0 * grid.hx);
        const double gy = (field_at(grid, field, i, j + 1, 0) -
                           field_at(grid, field, i, j - 1, 0)) /
                          (2.0 * grid.hy);
        sum += std::abs(std::hypot(gx, gy) - 1.0);
        ++count;
      }
    }
    const double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
    return std::pair<double, int>{mean, count};
  };

  const auto before_slope = slope_error(phi);
  const double before = before_slope.first;
  REQUIRE(before_slope.second > 10);
  const LevelSetStep step = redistance_level_set(grid, phi, sign, 48, cfl);
  const auto after_slope = slope_error(phi);
  const double after = after_slope.first;
  REQUIRE(after_slope.second > 10);
  const int i_after = interior_i(grid, phi, 0.5, 0.5, 0);
  const double x_after = level_set_zero_crossing_x(grid, phi, j0, 0, i_after);
  report("redistance", before, after, std::abs(x_after - x_before), step.dt);
  REQUIRE(after < 0.02);
  REQUIRE(std::abs(x_after - x_before) < 4e-3);
  REQUIRE(step.dt == level_set_stable_dt(grid, 1.0, cfl));
}

TEST_CASE("a translating sphere moves and a rejected step does not write",
          "[unit][field][level_set]") {
  const LevelSetGrid grid = make_grid(24, 24, 24);
  constexpr double cx0 = 0.40;
  constexpr double radius = 0.16;
  constexpr double speed = 0.20;
  constexpr double horizon = 0.50;
  std::vector<double> phi;
  fill_ball(grid, phi, cx0, 0.5, 0.5, radius);
  const std::vector<double> initial = phi;
  const auto vx = constant_field(phi.size(), speed);
  const auto vy = constant_field(phi.size(), 0.0);
  const auto vz = constant_field(phi.size(), 0.0);
  double dt = 0.0;
  const int steps = stable_steps(grid, speed, 0.35, horizon, dt);
  std::vector<double> interface_hist;
  std::vector<double> volume_hist;
  for (int s = 0; s <= steps; ++s) {
    const Sample now = compare_ball(
        grid, phi, cx0 + speed * dt * static_cast<double>(s), 0.5, 0.5, radius);
    interface_hist.push_back(now.interface_error);
    volume_hist.push_back(now.volume_error);
    if (s == steps) break;
    const LevelSetStep step = advance_level_set(grid, phi, LevelSetMode::Advection,
                                                vx, vy, vz, {}, {}, dt);
    REQUIRE(step.dt == dt);
  }
  const Sample stuck =
      compare_ball(grid, initial, cx0 + speed * horizon, 0.5, 0.5, radius);
  const double moved = std::abs(level_set_negative_centroid(grid, phi).x -
                                level_set_negative_centroid(grid, initial).x);
  report("sphere", interface_hist.back(), volume_hist.back(), moved,
         stuck.interface_error);
  REQUIRE(interface_hist.size() > 2);
  REQUIRE(moved > 0.5 * speed * horizon);
  REQUIRE(interface_hist.back() < stuck.interface_error);
  REQUIRE(interface_hist.back() < 5e-3);
  REQUIRE(std::isfinite(volume_hist.back()));
  REQUIRE(volume_hist.back() < 5e-3);

  std::vector<double> kept = initial;
  REQUIRE_THROWS_AS(advance_level_set(grid, kept, LevelSetMode::Advection, vx, vy,
                                      vz, {}, {}, 0.0),
                    std::invalid_argument);
  REQUIRE(kept == initial);
  REQUIRE_THROWS_AS(advance_level_set(grid, kept, LevelSetMode::Advection, vx, vy,
                                      vz, {}, {}, -dt),
                    std::invalid_argument);
  REQUIRE(kept == initial);
  std::vector<double> bad_speed = vx;
  bad_speed[4] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(advance_level_set(grid, kept, LevelSetMode::Advection, bad_speed,
                                      vy, vz, {}, {}, dt),
                    std::domain_error);
  REQUIRE(kept == initial);
  std::vector<double> bad_field = initial;
  bad_field[6] = std::numeric_limits<double>::infinity();
  const std::vector<double> bad_copy = bad_field;
  REQUIRE_THROWS_AS(advance_level_set(grid, bad_field, LevelSetMode::Advection, vx,
                                      vy, vz, {}, {}, dt),
                    std::domain_error);
  REQUIRE(bad_field == bad_copy);
  LevelSetGrid broken = grid;
  broken.nx = 1;
  REQUIRE_THROWS_AS(advance_level_set(broken, kept, LevelSetMode::Advection, vx, vy,
                                      vz, {}, {}, dt),
                    std::invalid_argument);
  REQUIRE(level_set_halo_width == 1);
  REQUIRE_THROWS_AS(level_set_stable_dt(grid, speed, 0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(level_set_stable_dt(grid, 0.0, 0.5), std::invalid_argument);
  REQUIRE(level_set_stable_dt(grid, speed, 1.0) == grid.hx / speed);
}

TEST_CASE("an x-slab of width one matches the periodic rate",
          "[unit][field][level_set]") {
  const LevelSetGrid grid = make_grid(8, 6, 2);
  const std::size_t n = level_set_count(grid);
  const std::size_t plane =
      static_cast<std::size_t>(grid.ny) * static_cast<std::size_t>(grid.nz);
  std::vector<double> phi(n);
  std::vector<double> vx(n);
  std::vector<double> vy(n);
  std::vector<double> vz(n);
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t p = index_of(grid, i, j, k);
        phi[p] = std::sin(0.7 * cell_x(grid, i)) + 0.2 * cell_y(grid, j) -
                 0.1 * cell_z(grid, k);
        vx[p] = 0.2 + 0.05 * cell_y(grid, j);
        vy[p] = -0.1 * cell_x(grid, i);
        vz[p] = 0.03;
      }
    }
  }
  std::vector<double> periodic(n);
  level_set_rate(grid, phi, LevelSetMode::Advection, vx, vy, vz, {}, {}, periodic);
  std::vector<double> xm(plane);
  std::vector<double> xp(plane);
  for (int k = 0; k < grid.nz; ++k) {
    for (int j = 0; j < grid.ny; ++j) {
      const std::size_t s = static_cast<std::size_t>(j + grid.ny * k);
      xm[s] = phi[index_of(grid, grid.nx - 1, j, k)];
      xp[s] = phi[index_of(grid, 0, j, k)];
    }
  }
  std::vector<double> slab(n);
  level_set_rate_x_slab(grid, phi, xm, xp, LevelSetMode::Advection, vx, vy, vz, {},
                        {}, slab);
  const double gap = max_abs_diff(periodic, slab);
  report("halo", gap, static_cast<double>(level_set_halo_width),
         static_cast<double>(plane), 0.0);
  REQUIRE(gap < 1e-12);
  REQUIRE(xm.size() == plane);
  std::vector<double> short_halo(plane - 1, 0.0);
  REQUIRE_THROWS_AS(level_set_rate_x_slab(grid, phi, short_halo, xp,
                                          LevelSetMode::Advection, vx, vy, vz, {},
                                          {}, slab),
                    std::invalid_argument);
}

TEST_CASE("two ranks exchange an x-slab of width one", "[level_set][mpi]") {
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 2) SKIP("level-set slab exchange needs two ranks");

  constexpr int local_nx = 4;
  constexpr int ny = 8;
  constexpr int nz = 1;
  const LevelSetGrid global = make_grid(local_nx * 2, ny, nz);
  LevelSetGrid local = global;
  local.nx = local_nx;
  const std::size_t local_n = level_set_count(local);
  const std::size_t plane = static_cast<std::size_t>(ny * nz);
  const int x0 = rank * local_nx;
  std::vector<double> phi(local_n);
  std::vector<double> vx(local_n);
  std::vector<double> vy(local_n);
  std::vector<double> vz(local_n, 0.0);
  std::vector<double> full(level_set_count(global));
  std::vector<double> full_vx(full.size());
  std::vector<double> full_vy(full.size());
  std::vector<double> full_vz(full.size(), 0.0);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < global.nx; ++i) {
      const std::size_t g = index_of(global, i, j, 0);
      full[g] = std::sin(2.0 * 3.141592653589793 * cell_x(global, i)) +
                0.4 * cell_y(global, j);
      full_vx[g] = 0.25;
      full_vy[g] = -0.1;
      if (i >= x0 && i < x0 + local_nx) {
        const std::size_t p = index_of(local, i - x0, j, 0);
        phi[p] = full[g];
        vx[p] = full_vx[g];
        vy[p] = full_vy[g];
      }
    }
  }
  std::vector<double> left_edge(plane);
  std::vector<double> right_edge(plane);
  for (int j = 0; j < ny; ++j) {
    left_edge[static_cast<std::size_t>(j)] = phi[index_of(local, 0, j, 0)];
    right_edge[static_cast<std::size_t>(j)] =
        phi[index_of(local, local_nx - 1, j, 0)];
  }
  const int other = 1 - rank;
  std::vector<double> xm(plane);
  std::vector<double> xp(plane);
  MPI_Sendrecv(right_edge.data(), static_cast<int>(plane), MPI_DOUBLE, other, 11,
               xm.data(), static_cast<int>(plane), MPI_DOUBLE, other, 11,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(left_edge.data(), static_cast<int>(plane), MPI_DOUBLE, other, 12,
               xp.data(), static_cast<int>(plane), MPI_DOUBLE, other, 12,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<double> slab(local_n);
  level_set_rate_x_slab(local, phi, xm, xp, LevelSetMode::Advection, vx, vy, vz, {},
                        {}, slab);
  std::vector<double> serial(full.size());
  level_set_rate(global, full, LevelSetMode::Advection, full_vx, full_vy, full_vz,
                 {}, {}, serial);
  double gap = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < local_nx; ++i) {
      gap = std::max(gap, std::abs(slab[index_of(local, i, j, 0)] -
                                   serial[index_of(global, x0 + i, j, 0)]));
    }
  }
  double global_gap = 0.0;
  MPI_Allreduce(&gap, &global_gap, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
    report("mpi_slab", global_gap, static_cast<double>(level_set_halo_width),
           static_cast<double>(plane), 0.0);
  REQUIRE(global_gap < 1e-12);
}
