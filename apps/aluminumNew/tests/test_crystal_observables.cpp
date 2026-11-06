// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Aluminum consumes the same installed observables as tungsten. There is no
// second front or orientation implementation in this application.

#include <openpfc/frontend/io/crystal_diagnostics.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;

namespace {

double envelope(double x, double front) { return 0.5 + (x - front) * 0.4 / 31.0; }

std::vector<double> crystal_front(double front) {
  constexpr int nx = 32;
  constexpr int ny = 8;
  constexpr int nz = 2;
  std::vector<double> values(static_cast<std::size_t>(nx * ny * nz));
  const double scale = std::sqrt(2.0);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const double wave =
          std::cos(2.0 * std::numbers::pi * static_cast<double>(j) / ny);
      for (int i = 0; i < nx; ++i) {
        values[static_cast<std::size_t>(i + nx * (j + ny * k))] =
            envelope(static_cast<double>(i), front) * wave * scale;
      }
    }
  }
  return values;
}

pfc::observables::OwnedDensity own(const std::vector<double> &values) {
  pfc::observables::OwnedDensity density;
  density.values = values;
  density.global_size = {32, 8, 2};
  density.spacing = {1.0, 1.0, 1.0};
  density.origin = {0.0, 0.0, 0.0};
  density.box = pfc::Box3i::from_bounds({0, 0, 0}, {31, 7, 1});
  return density;
}

} // namespace

TEST_CASE("aluminum writes crystal observables through DiagnosticsSeries",
          "[aluminum][crystal_observables]") {
  const auto early_values = crystal_front(15.3);
  const auto later_values = crystal_front(15.7);
  const auto early = own(early_values);
  const auto later = own(later_values);
  const pfc::observables::PlanarFrontOptions options;
  const auto first =
      pfc::observables::planar_front_position(early, MPI_COMM_SELF, options);
  const std::array<std::array<double, 3>, 1> candidate{{{0.5, 0.0, 0.0}}};
  const pfc::observables::IndexWindow all{{0, 0, 0}, {31, 7, 1}};
  const auto path =
      std::filesystem::temp_directory_path() / "aluminum-crystal-observables.csv";
  std::filesystem::remove(path);
  {
    pfc::io::DiagnosticsSeries series(
        pfc::io::crystal_diagnostic_columns(candidate.size()),
        pfc::io::DiagnosticsSeriesOptions{
            .path = path, .comm = MPI_COMM_SELF, .overwrite = true});
    pfc::io::record_crystal_diagnostics(
        series, 4, 2.0, later, MPI_COMM_SELF, options, &first, 0.0,
        pfc::observables::CrystalSymmetry::none, all, candidate);
    series.close();
  }
  std::ifstream input(path);
  std::string header;
  std::string row;
  REQUIRE(std::getline(input, header));
  REQUIRE(std::getline(input, row));
  REQUIRE(header.find("front_position") != std::string::npos);
  REQUIRE(header.find("grain_0") != std::string::npos);
  std::stringstream stream(row);
  std::string item;
  std::vector<std::string> values;
  while (std::getline(stream, item, ',')) values.push_back(item);
  REQUIRE(std::stod(values[2]) == Approx(15.7).margin(1e-9));
  REQUIRE(std::stod(values[3]) == Approx(0.2).margin(1e-9));
  REQUIRE(std::stod(values[4]) == 1.0);
  std::filesystem::remove(path);

  // Integer modes are periodic, so the full-window stencil is the pure wave.
  constexpr int n = 24;
  constexpr int nz = 2;
  const double kx = 2.0 * std::numbers::pi * 2.0 / static_cast<double>(n);
  const double ky = 2.0 * std::numbers::pi / static_cast<double>(n);
  const double angle = std::atan2(ky, kx);
  std::vector<double> wave(static_cast<std::size_t>(n * n * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        wave[static_cast<std::size_t>(i + n * (j + n * k))] = std::cos(
            kx * static_cast<double>(i) + ky * static_cast<double>(j) - 0.8);
      }
    }
  }
  pfc::observables::OwnedDensity crystal;
  crystal.values = wave;
  crystal.global_size = {n, n, nz};
  crystal.spacing = {1.0, 1.0, 1.0};
  crystal.origin = {0.0, 0.0, 0.0};
  crystal.box = pfc::Box3i::from_bounds({0, 0, 0}, {n - 1, n - 1, nz - 1});
  const pfc::observables::IndexWindow window{{0, 0, 0}, {n - 1, n - 1, nz - 1}};
  const auto orientation = pfc::observables::region_orientation(
      crystal, MPI_COMM_SELF, window, pfc::observables::CrystalSymmetry::four_fold);
  REQUIRE(orientation.valid);
  REQUIRE(orientation.angle_xy == Approx(angle).margin(1e-8));
  REQUIRE(orientation.wavevector[0] == Approx(kx).margin(1e-8));
  REQUIRE(orientation.wavevector[1] == Approx(ky).margin(1e-8));
  const std::array<std::array<double, 3>, 2> candidates{
      {{kx, ky, 0.0}, {0.5 * kx, 0.0, 0.0}}};
  const auto grains = pfc::observables::label_grains(
      crystal, MPI_COMM_SELF, window, pfc::observables::CrystalSymmetry::four_fold,
      candidates);
  REQUIRE(grains.valid);
  REQUIRE(grains.voting_cells > 0);
  REQUIRE(grains.fraction[0] == 1.0);
  REQUIRE(grains.fraction[1] == 0.0);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int status = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return status;
}
