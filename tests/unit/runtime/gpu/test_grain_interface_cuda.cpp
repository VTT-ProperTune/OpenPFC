// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * One-device compact grain interface. HIP is not run.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/field/grain_interface.hpp>
#include <openpfc/runtime/gpu/grain_interface_cuda.hpp>

#include <fixtures/session.hpp>

#include "test_helpers.hpp"

using pfc::field::grain_interface::build_grain_interface;
using pfc::field::grain_interface::dense_level_set_bytes;
using pfc::field::grain_interface::grain_cell_count;
using pfc::field::grain_interface::GrainGrid;
using pfc::field::grain_interface::GrainInterface;
using pfc::field::grain_interface::GrainInterfaceCuda;
using pfc::field::grain_interface::GrainSeed;
using pfc::field::grain_interface::resident_device_bytes;

namespace {

void report(const std::string &label, double a, double b, double c, double d) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "GRAIN_CUDA " << label << " " << a << " " << b << " " << c << " " << d
      << '\n';
  std::cout << out.str();
}

GrainGrid cube(int n) {
  GrainGrid grid;
  grid.nx = n;
  grid.ny = n;
  grid.nz = n;
  grid.hx = 1.0 / static_cast<double>(n);
  grid.hy = grid.hx;
  grid.hz = grid.hx;
  return grid;
}

std::vector<GrainSeed> irregular_seeds(const GrainGrid &grid, int count) {
  const double lx = grid.nx * grid.hx;
  const double ly = grid.ny * grid.hy;
  const double lz = grid.nz * grid.hz;
  std::vector<GrainSeed> seeds(static_cast<std::size_t>(count));
  for (int s = 0; s < count; ++s) {
    double x = std::fmod(0.17 + s * 0.6180339887498948, 1.0) * lx;
    double y = std::fmod(0.31 + s * 0.4142135623730950, 1.0) * ly;
    double z = std::fmod(0.47 + s * 0.7071067811865476, 1.0) * lz;
    if (!(x < lx)) x = 0.0;
    if (!(y < ly)) y = 0.0;
    if (!(z < lz)) z = 0.0;
    seeds[static_cast<std::size_t>(s)] = GrainSeed{x, y, z};
  }
  return seeds;
}

} // namespace

TEST_CASE("CUDA grain interface matches the host band and records its bytes",
          "[gpu][grain_interface]") {
  if (!pfc::gpu::test::is_cuda_available()) SKIP("CUDA not available");

  const GrainGrid grid = cube(32);
  const int grains = 64;
  const GrainInterface host =
      build_grain_interface(grid, irregular_seeds(grid, grains));
  const std::size_t n = grain_cell_count(grid);
  GrainInterfaceCuda device(host);
  GrainInterface bad = host;
  bad.grain_id[0] = -1;
  REQUIRE_THROWS_AS(device.measure(bad), std::invalid_argument);
  REQUIRE(device.profile().kernel_launches == 0);
  REQUIRE(device.profile().host_to_device_bytes == 0);
  REQUIRE(device.profile().device_to_host_bytes == 0);
  REQUIRE(device.profile().device_bytes == 0);

  const auto checked = device.measure(host);
  const std::size_t expect = resident_device_bytes(n, host.band.size(), grains);
  const std::size_t dense = dense_level_set_bytes(grains, n);
  report("memory", static_cast<double>(checked.profile.device_bytes),
         static_cast<double>(expect), static_cast<double>(dense),
         static_cast<double>(checked.mismatches));
  report("traffic", static_cast<double>(checked.profile.kernel_launches),
         static_cast<double>(checked.profile.host_to_device_bytes),
         static_cast<double>(checked.profile.device_to_host_bytes),
         static_cast<double>(host.band.size()));
  REQUIRE(checked.mismatches == 0);
  REQUIRE(checked.cell_count == host.cell_count);
  REQUIRE(checked.profile.device_bytes == expect);
  REQUIRE(checked.profile.device_bytes < dense);
  REQUIRE(checked.profile.kernel_launches == 2);
  REQUIRE(checked.profile.host_to_device_bytes ==
          n * sizeof(int) + host.band.size() * sizeof(host.band.front()));
  REQUIRE(checked.profile.device_to_host_bytes ==
          static_cast<std::size_t>(grains) * sizeof(int) + sizeof(int));
  REQUIRE(device.profile().device_bytes == expect);
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
