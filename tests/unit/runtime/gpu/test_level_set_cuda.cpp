// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * One-device level set against the host stepper. HIP is not run.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/field/level_set.hpp>
#include <openpfc/runtime/gpu/level_set_cuda.hpp>

#include <fixtures/session.hpp>

#include "test_helpers.hpp"

using pfc::field::level_set::advance_level_set;
using pfc::field::level_set::level_set_count;
using pfc::field::level_set::level_set_negative_centroid;
using pfc::field::level_set::level_set_stable_dt;
using pfc::field::level_set::level_set_zero_crossing_x;
using pfc::field::level_set::LevelSetCuda;
using pfc::field::level_set::LevelSetGrid;
using pfc::field::level_set::LevelSetMode;
using pfc::field::level_set::redistance_level_set;

namespace {

void report(const std::string &label, double a, double b, double c, double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "LEVEL_SET_CUDA " << label << " " << a << " " << b << " " << c << " " << d;
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

double max_abs(const std::vector<double> &a, const std::vector<double> &b) {
  double gap = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    gap = std::max(gap, std::abs(a[i] - b[i]));
  return gap;
}

void fill_disk(const LevelSetGrid &grid, std::vector<double> &phi, double cx,
               double cy, double radius) {
  phi.assign(level_set_count(grid), 0.0);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const double x = (static_cast<double>(i) + 0.5) * grid.hx - cx;
      const double y = (static_cast<double>(j) + 0.5) * grid.hy - cy;
      phi[index_of(grid, i, j, 0)] = std::sqrt(x * x + y * y) - radius;
    }
  }
}

} // namespace

TEST_CASE("CUDA level set matches the host step and records its own history",
          "[gpu][level_set]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("CUDA not available");

  const LevelSetGrid small = make_grid(8, 8, 1);
  LevelSetCuda device(small);
  const std::size_t n = level_set_count(small);
  std::vector<double> phi(n);
  std::vector<double> host = phi;
  std::vector<double> vx(n, 0.2);
  std::vector<double> vy(n, -0.1);
  std::vector<double> vz(n, 0.0);
  for (int j = 0; j < small.ny; ++j) {
    for (int i = 0; i < small.nx; ++i) {
      const double x = (static_cast<double>(i) + 0.5) * small.hx;
      const double y = (static_cast<double>(j) + 0.5) * small.hy;
      phi[index_of(small, i, j, 0)] = std::sin(2.0 * 3.141592653589793 * x) +
                                      0.25 * std::sin(2.0 * 3.141592653589793 * y);
    }
  }
  host = phi;
  constexpr double dt = 0.02;
  REQUIRE(
      advance_level_set(small, host, LevelSetMode::Advection, vx, vy, vz, {}, {}, dt)
          .dt == dt);
  REQUIRE(device.advance(small, phi, LevelSetMode::Advection, vx, vy, vz, {}, {}, dt)
              .dt == dt);
  const double wave_gap = max_abs(host, phi);
  report("wave", wave_gap, static_cast<double>(device.profile().kernel_launches),
         static_cast<double>(device.profile().host_to_device_bytes),
         static_cast<double>(device.profile().device_to_host_bytes));
  REQUIRE(wave_gap < 1e-10);
  REQUIRE(device.profile().kernel_launches == 4);
  REQUIRE(device.profile().host_to_device_bytes == 4 * n * sizeof(double));
  REQUIRE(device.profile().device_to_host_bytes == n * sizeof(double));

  std::vector<double> normal_host;
  std::vector<double> normal_device;
  fill_disk(small, normal_host, 0.45, 0.5, 0.18);
  normal_device = normal_host;
  const std::vector<double> vn(n, 0.04);
  REQUIRE(advance_level_set(small, normal_host, LevelSetMode::NormalSpeed, {}, {},
                            {}, vn, {}, dt)
              .dt == dt);
  REQUIRE(device
              .advance(small, normal_device, LevelSetMode::NormalSpeed, {}, {}, {},
                       vn, {}, dt)
              .dt == dt);
  const double normal_gap = max_abs(normal_host, normal_device);
  report("normal", normal_gap, static_cast<double>(device.profile().kernel_launches),
         static_cast<double>(device.profile().host_to_device_bytes),
         static_cast<double>(device.profile().device_to_host_bytes));
  REQUIRE(normal_gap < 1e-10);
  REQUIRE(device.profile().kernel_launches == 4);
  REQUIRE(device.profile().host_to_device_bytes == 2 * n * sizeof(double));
  REQUIRE(device.profile().device_to_host_bytes == n * sizeof(double));

  const std::vector<double> before_reject = normal_device;
  REQUIRE_THROWS_AS(device.advance(small, normal_device, LevelSetMode::Advection, vx,
                                   vy, vz, {}, {}, 0.0),
                    std::invalid_argument);
  REQUIRE(normal_device == before_reject);
  REQUIRE(device.profile().kernel_launches == 0);
  REQUIRE(device.profile().host_to_device_bytes == 0);
  REQUIRE(device.profile().device_to_host_bytes == 0);

  const LevelSetGrid grid = make_grid(32, 32, 1);
  LevelSetCuda mover(grid);
  constexpr double cx0 = 0.35;
  constexpr double speed = 0.25;
  constexpr double horizon = 0.40;
  constexpr double radius = 0.15;
  std::vector<double> moved;
  std::vector<double> host_moved;
  fill_disk(grid, moved, cx0, 0.5, radius);
  host_moved = moved;
  const auto move_vx = std::vector<double>(moved.size(), speed);
  const auto zeros = std::vector<double>(moved.size(), 0.0);
  const double stable = level_set_stable_dt(grid, speed, 0.35);
  int steps = static_cast<int>(std::ceil(horizon / stable));
  const double step_dt = horizon / static_cast<double>(steps);
  std::vector<double> interface_hist;
  std::vector<double> volume_hist;
  const int j_mid = static_cast<int>(std::floor(0.5 / grid.hy));
  for (int s = 0; s <= steps; ++s) {
    const double cx = cx0 + speed * step_dt * static_cast<double>(s);
    std::vector<double> exact;
    fill_disk(grid, exact, cx, 0.5, radius);
    const int i_mid = static_cast<int>(std::floor(cx / grid.hx));
    const double x_now = level_set_zero_crossing_x(grid, moved, j_mid, 0, i_mid);
    const double x_exact = level_set_zero_crossing_x(grid, exact, j_mid, 0, i_mid);
    interface_hist.push_back(std::abs(x_now - x_exact));
    volume_hist.push_back(std::abs(level_set_negative_centroid(grid, moved).volume -
                                   level_set_negative_centroid(grid, exact).volume));
    if (s == steps) break;
    REQUIRE(mover
                .advance(grid, moved, LevelSetMode::Advection, move_vx, zeros, zeros,
                         {}, {}, step_dt)
                .dt == step_dt);
    REQUIRE(advance_level_set(grid, host_moved, LevelSetMode::Advection, move_vx,
                              zeros, zeros, {}, {}, step_dt)
                .dt == step_dt);
  }
  const double pair = max_abs(moved, host_moved);
  const double travel = std::abs(level_set_negative_centroid(grid, moved).x - cx0);
  std::vector<double> stuck;
  fill_disk(grid, stuck, cx0, 0.5, radius);
  std::vector<double> exact_final;
  fill_disk(grid, exact_final, cx0 + speed * horizon, 0.5, radius);
  const int i_final =
      static_cast<int>(std::floor((cx0 + speed * horizon) / grid.hx));
  const int i_stuck = static_cast<int>(std::floor(cx0 / grid.hx));
  const double stuck_interface =
      std::abs(level_set_zero_crossing_x(grid, stuck, j_mid, 0, i_stuck) -
               level_set_zero_crossing_x(grid, exact_final, j_mid, 0, i_final));
  report("translate", pair, interface_hist.back(), volume_hist.back(), travel);
  REQUIRE(pair < 1e-10);
  REQUIRE(interface_hist.size() == static_cast<std::size_t>(steps) + 1);
  REQUIRE(volume_hist.size() == interface_hist.size());
  REQUIRE(travel > 0.5 * speed * horizon);
  REQUIRE(interface_hist.back() < stuck_interface);
  REQUIRE(interface_hist.back() < 2e-3);
  REQUIRE(mover.profile().kernel_launches == 4);
  REQUIRE(mover.profile().device_to_host_bytes == moved.size() * sizeof(double));
  REQUIRE(mover.profile().host_to_device_bytes == 4 * moved.size() * sizeof(double));

  std::vector<double> scaled;
  fill_disk(grid, scaled, 0.5, 0.5, 0.20);
  for (double &value : scaled) value *= 2.5;
  std::vector<double> host_scaled = scaled;
  const std::vector<double> sign = scaled;
  REQUIRE(redistance_level_set(grid, host_scaled, sign, 8, 0.30).dt ==
          level_set_stable_dt(grid, 1.0, 0.30));
  REQUIRE(mover.redistance(grid, scaled, sign, 8, 0.30).dt ==
          level_set_stable_dt(grid, 1.0, 0.30));
  const double redistance_gap = max_abs(scaled, host_scaled);
  report("redistance", redistance_gap,
         static_cast<double>(mover.profile().kernel_launches),
         static_cast<double>(mover.profile().host_to_device_bytes),
         static_cast<double>(mover.profile().device_to_host_bytes));
  REQUIRE(redistance_gap < 1e-10);
  REQUIRE(mover.profile().kernel_launches == 4 * 8);
  REQUIRE(mover.profile().host_to_device_bytes ==
          2 * scaled.size() * sizeof(double));
  REQUIRE(mover.profile().device_to_host_bytes == scaled.size() * sizeof(double));

  double slope_before = 0.0;
  double slope_after = 0.0;
  int band = 0;
  std::vector<double> slope_field;
  fill_disk(grid, slope_field, 0.5, 0.5, 0.20);
  for (double &value : slope_field) value *= 2.5;
  auto slope = [&](const std::vector<double> &field) {
    double sum = 0.0;
    int count = 0;
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const double x = (static_cast<double>(i) + 0.5) * grid.hx - 0.5;
        const double y = (static_cast<double>(j) + 0.5) * grid.hy - 0.5;
        const double r = std::sqrt(x * x + y * y);
        const double band_r = std::abs(r - 0.20);
        if (band_r < 2.0 * grid.hx || band_r > 6.0 * grid.hx) continue;
        const auto at = [&](int ii, int jj) {
          const int ix = (ii + grid.nx) % grid.nx;
          const int jx = (jj + grid.ny) % grid.ny;
          return field[index_of(grid, ix, jx, 0)];
        };
        const double gx = (at(i + 1, j) - at(i - 1, j)) / (2.0 * grid.hx);
        const double gy = (at(i, j + 1) - at(i, j - 1)) / (2.0 * grid.hy);
        sum += std::abs(std::hypot(gx, gy) - 1.0);
        ++count;
      }
    }
    band = count;
    return sum / static_cast<double>(count);
  };
  slope_before = slope(slope_field);
  const std::vector<double> slope_sign = slope_field;
  REQUIRE(mover.redistance(grid, slope_field, slope_sign, 48, 0.30).dt ==
          level_set_stable_dt(grid, 1.0, 0.30));
  slope_after = slope(slope_field);
  report("slope", slope_before, slope_after, static_cast<double>(band),
         static_cast<double>(mover.profile().kernel_launches));
  REQUIRE(band > 10);
  REQUIRE(slope_after < 0.5 * slope_before);
  REQUIRE(slope_after < 0.05);
  REQUIRE(mover.profile().kernel_launches == 4 * 48);
  REQUIRE(mover.profile().device_to_host_bytes ==
          slope_field.size() * sizeof(double));
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) MPI_Init(&argc, &argv);
  const int result = pfc::test::run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) MPI_Finalize();
  return result;
}
