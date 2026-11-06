// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/io/crystal_diagnostics.hpp>
#include <openpfc/kernel/field/crystal_observables.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Approx;
using pfc::observables::CrystalSymmetry;
using pfc::observables::FrontValidity;
using pfc::observables::IndexWindow;
using pfc::observables::OwnedDensity;
using pfc::observables::PlanarFrontOptions;

namespace {

OwnedDensity full_density(const std::vector<double> &values,
                          const std::array<int, 3> &size, MPI_Comm comm) {
  (void)comm;
  OwnedDensity density;
  density.values = values;
  density.global_size = size;
  density.spacing = {1.0, 1.0, 1.0};
  density.origin = {0.0, 0.0, 0.0};
  density.box =
      pfc::Box3i::from_bounds({0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1});
  return density;
}

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
        const std::size_t id = static_cast<std::size_t>(i + nx * (j + ny * k));
        values[id] = envelope(static_cast<double>(i), front) * wave * scale;
      }
    }
  }
  return values;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream stream(line);
  std::string item;
  while (std::getline(stream, item, ',')) out.push_back(item);
  return out;
}

} // namespace

TEST_CASE("planar crystal front resolves a sub-cell translation",
          "[crystal_observables]") {
  constexpr std::array<int, 3> size{32, 8, 2};
  const auto early_values = crystal_front(15.3);
  const auto later_values = crystal_front(15.7);
  const OwnedDensity early = full_density(early_values, size, MPI_COMM_SELF);
  const OwnedDensity later = full_density(later_values, size, MPI_COMM_SELF);
  const PlanarFrontOptions options;
  const auto first =
      pfc::observables::planar_front_position(early, MPI_COMM_SELF, options);
  const auto second =
      pfc::observables::planar_front_position(later, MPI_COMM_SELF, options);
  REQUIRE(first.validity == FrontValidity::valid);
  REQUIRE(second.validity == FrontValidity::valid);
  REQUIRE(first.crossings == 1);
  REQUIRE(first.position == Approx(15.3).margin(1e-9));
  REQUIRE(second.position == Approx(15.7).margin(1e-9));
  REQUIRE(second.position - first.position == Approx(0.4).margin(1e-9));
  REQUIRE(std::abs(second.position - first.position) < 1.0);
  const auto speed = pfc::observables::front_velocity(first, 0.0, second, 2.0);
  REQUIRE(speed.valid);
  REQUIRE(speed.velocity == Approx(0.2).margin(1e-9));

  std::vector<double> flat(early_values.size(), 0.2);
  const OwnedDensity flat_density = full_density(flat, size, MPI_COMM_SELF);
  const auto missing =
      pfc::observables::planar_front_position(flat_density, MPI_COMM_SELF, options);
  REQUIRE(missing.validity == FrontValidity::no_crossing);
  REQUIRE_FALSE(pfc::observables::front_velocity(first, 0.0, missing, 2.0).valid);

  std::vector<double> slab(early_values.size(), 0.0);
  for (int k = 0; k < size[2]; ++k) {
    for (int j = 0; j < size[1]; ++j) {
      for (int i = 8; i <= 20; ++i) {
        slab[static_cast<std::size_t>(i + size[0] * (j + size[1] * k))] = 1.0;
      }
    }
  }
  const auto band = pfc::observables::planar_front_position(
      full_density(slab, size, MPI_COMM_SELF), MPI_COMM_SELF, options);
  REQUIRE(band.validity == FrontValidity::not_single_front);

  auto poisoned = early_values;
  poisoned[3] = std::numeric_limits<double>::quiet_NaN();
  const auto broken = pfc::observables::planar_front_position(
      full_density(poisoned, size, MPI_COMM_SELF), MPI_COMM_SELF, options);
  REQUIRE(broken.validity == FrontValidity::nonfinite);
}

TEST_CASE("planar front follows a caller-selected axis", "[crystal_observables]") {
  constexpr int nx = 4;
  constexpr int ny = 16;
  constexpr int nz = 2;
  std::vector<double> values(static_cast<std::size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const double sample = 0.5 + (static_cast<double>(j) - 7.3) * 0.4 / 15.0;
      for (int i = 0; i < nx; ++i) {
        values[static_cast<std::size_t>(i + nx * (j + ny * k))] = sample;
      }
    }
  }
  const OwnedDensity density =
      full_density(values, std::array<int, 3>{nx, ny, nz}, MPI_COMM_SELF);
  PlanarFrontOptions options;
  options.axis = 1;
  const auto front =
      pfc::observables::planar_front_position(density, MPI_COMM_SELF, options);
  REQUIRE(front.validity == FrontValidity::valid);
  REQUIRE(front.position == Approx(7.3).margin(1e-9));
}

TEST_CASE("a descending planar RMS front reports its real sub-cell position",
          "[crystal_observables]") {
  constexpr int nx = 32, ny = 8, nz = 2;
  const std::array<int, 3> size{nx, ny, nz};
  std::vector<double> values(static_cast<std::size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const double phase = 2.0 * std::numbers::pi * static_cast<double>(j) / ny;
      const double wave = std::sqrt(2.0) * std::cos(phase);
      for (int i = 0; i < nx; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i + nx * (j + ny * k));
        values[idx] = (1.0 - envelope(static_cast<double>(i), 15.3)) * wave;
      }
    }
  }
  auto input = full_density(values, size, MPI_COMM_SELF);
  input.origin[0] = 5.0;
  PlanarFrontOptions options;
  options.rising = false;
  const auto falling =
      pfc::observables::planar_front_position(input, MPI_COMM_SELF, options);
  REQUIRE(falling.validity == FrontValidity::valid);
  REQUIRE(falling.crossings == 1);
  REQUIRE(falling.position == Approx(20.3).margin(1e-9));
  const auto bad_time = pfc::observables::front_velocity(
      falling, -std::numeric_limits<double>::infinity(), falling, 2.0);
  REQUIRE_FALSE(bad_time.valid);
}

TEST_CASE("crystal symmetry folds equivalent wavevectors", "[crystal_observables]") {
  const double fifth = std::numbers::pi / 5.0;
  const std::array<double, 3> square{std::cos(fifth), std::sin(fifth), 0.2};
  const std::array<double, 3> turned{std::cos(fifth + std::numbers::pi / 2.0),
                                     std::sin(fifth + std::numbers::pi / 2.0), 0.2};
  const auto square_rep = pfc::observables::representative_wavevector(
      square, CrystalSymmetry::four_fold);
  const auto turned_rep = pfc::observables::representative_wavevector(
      turned, CrystalSymmetry::four_fold);
  REQUIRE(square_rep[0] == Approx(turned_rep[0]).margin(1e-12));
  REQUIRE(square_rep[1] == Approx(turned_rep[1]).margin(1e-12));
  REQUIRE(square_rep[2] == Approx(0.2).margin(1e-12));
  REQUIRE(std::atan2(square_rep[1], square_rep[0]) == Approx(fifth).margin(1e-12));

  const std::array<double, 3> along_y{0.0, 1.0, 0.0};
  const auto folded_y = pfc::observables::representative_wavevector(
      along_y, CrystalSymmetry::four_fold);
  REQUIRE(folded_y[0] == Approx(1.0).margin(1e-12));
  REQUIRE(folded_y[1] == Approx(0.0).margin(1e-12));

  // Shift by one six-fold step while keeping the first component positive, so
  // the cosine gauge does not flip the out-of-plane component.
  const double hex_angle = fifth - std::numbers::pi / 3.0;
  const std::array<double, 3> hex{std::cos(hex_angle), std::sin(hex_angle), -0.4};
  const auto hex_rep =
      pfc::observables::representative_wavevector(hex, CrystalSymmetry::six_fold);
  REQUIRE(std::atan2(hex_rep[1], hex_rep[0]) == Approx(fifth).margin(1e-12));
  REQUIRE(hex_rep[2] == Approx(-0.4).margin(1e-12));

  const std::array<double, 3> not_hex{std::cos(fifth + std::numbers::pi / 2.0),
                                      std::sin(fifth + std::numbers::pi / 2.0), 0.0};
  const auto distinct = pfc::observables::representative_wavevector(
      not_hex, CrystalSymmetry::six_fold);
  REQUIRE(std::abs(std::atan2(distinct[1], distinct[0]) - fifth) > 0.1);

  const std::array<double, 3> opposed{-std::cos(fifth), -std::sin(fifth), -0.3};
  const auto gauged =
      pfc::observables::representative_wavevector(opposed, CrystalSymmetry::none);
  REQUIRE(gauged[0] == Approx(std::cos(fifth)).margin(1e-12));
  REQUIRE(gauged[1] == Approx(std::sin(fifth)).margin(1e-12));
  REQUIRE(gauged[2] == Approx(0.3).margin(1e-12));

  const std::array<double, 3> cubic_in{0.1, 0.2, -0.3};
  const auto cubic =
      pfc::observables::representative_wavevector(cubic_in, CrystalSymmetry::cubic);
  REQUIRE(cubic[0] == Approx(0.1).margin(1e-12));
  REQUIRE(cubic[1] == Approx(0.2).margin(1e-12));
  REQUIRE(cubic[2] == Approx(0.3).margin(1e-12));
  const auto axis = pfc::observables::representative_wavevector(
      std::array<double, 3>{1.0, 0.0, 0.0}, CrystalSymmetry::cubic);
  REQUIRE(axis[0] == Approx(0.0).margin(1e-12));
  REQUIRE(axis[1] == Approx(0.0).margin(1e-12));
  REQUIRE(axis[2] == Approx(1.0).margin(1e-12));
}

double mode_wavenumber(int mode, int count) {
  return 2.0 * std::numbers::pi * static_cast<double>(mode) /
         static_cast<double>(count);
}

TEST_CASE("a manufactured density wave recovers its orientation",
          "[crystal_observables]") {
  constexpr int n = 32;
  constexpr int nz = 4;
  const std::array<int, 3> size{n, n, nz};
  auto periodic = [&](int mx, int my, int mz, double phase) {
    std::vector<double> values(static_cast<std::size_t>(n * n * nz));
    const double kx = mode_wavenumber(mx, n);
    const double ky = mode_wavenumber(my, n);
    const double kz = mode_wavenumber(mz, nz);
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          values[static_cast<std::size_t>(i + n * (j + n * k))] =
              std::cos(kx * i + ky * j + kz * k + phase);
        }
      }
    }
    return values;
  };
  const IndexWindow all{{0, 0, 0}, {n - 1, n - 1, nz - 1}};
  const double angle = std::atan2(1.0, 2.0);
  const auto base_values = periodic(2, 1, 0, 1.3);
  // A quarter turn about z takes mode (2, 1) to (-1, 2).
  const auto turned_values = periodic(-1, 2, 0, -0.7);
  const auto base = pfc::observables::region_orientation(
      full_density(base_values, size, MPI_COMM_SELF), MPI_COMM_SELF, all,
      CrystalSymmetry::four_fold);
  const auto turned = pfc::observables::region_orientation(
      full_density(turned_values, size, MPI_COMM_SELF), MPI_COMM_SELF, all,
      CrystalSymmetry::four_fold);
  REQUIRE(base.valid);
  REQUIRE(turned.valid);
  REQUIRE(base.angle_xy == Approx(angle).margin(1e-8));
  REQUIRE(turned.angle_xy == Approx(base.angle_xy).margin(1e-8));
  REQUIRE(base.wavevector[0] == Approx(mode_wavenumber(2, n)).margin(1e-8));
  REQUIRE(base.wavevector[1] == Approx(mode_wavenumber(1, n)).margin(1e-8));

  constexpr int m = 28;
  std::vector<double> hex_values(static_cast<std::size_t>(m * m * m));
  const double fifth = std::numbers::pi / 5.0;
  const double kappa = 0.4;
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        hex_values[static_cast<std::size_t>(i + m * (j + m * k))] =
            std::cos(kappa * std::cos(fifth + std::numbers::pi / 3.0) * i +
                     kappa * std::sin(fifth + std::numbers::pi / 3.0) * j + 0.2);
      }
    }
  }
  const std::array<int, 3> hex_size{m, m, m};
  const IndexWindow interior{{1, 1, 1}, {m - 2, m - 2, m - 2}};
  const auto hex = pfc::observables::region_orientation(
      full_density(hex_values, hex_size, MPI_COMM_SELF), MPI_COMM_SELF, interior,
      CrystalSymmetry::six_fold);
  REQUIRE(hex.valid);
  REQUIRE(hex.angle_xy == Approx(fifth).margin(1e-8));

  std::vector<double> cubic_values(hex_values.size());
  const std::array<double, 3> cubic_k{0.1, 0.2, -0.3};
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        cubic_values[static_cast<std::size_t>(i + m * (j + m * k))] =
            std::cos(cubic_k[0] * i + cubic_k[1] * j + cubic_k[2] * k + 0.4);
      }
    }
  }
  const auto cubic = pfc::observables::region_orientation(
      full_density(cubic_values, hex_size, MPI_COMM_SELF), MPI_COMM_SELF, interior,
      CrystalSymmetry::cubic);
  REQUIRE(cubic.valid);
  REQUIRE(cubic.wavevector[0] == Approx(0.1).margin(1e-7));
  REQUIRE(cubic.wavevector[1] == Approx(0.2).margin(1e-7));
  REQUIRE(cubic.wavevector[2] == Approx(0.3).margin(1e-7));
}

TEST_CASE("two simultaneous perpendicular waves are not one grain direction",
          "[crystal_observables]") {
  constexpr int nx = 64, ny = 64, nz = 2;
  const std::array<int, 3> size{nx, ny, nz};
  std::vector<double> values(static_cast<std::size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double x = 2.0 * std::numbers::pi * 4.0 * i / nx;
        const double y = 2.0 * std::numbers::pi * 4.0 * j / ny;
        const std::size_t idx = static_cast<std::size_t>(i + nx * (j + ny * k));
        values[idx] = std::cos(x) + std::cos(y);
      }
    }
  }
  const auto density = full_density(values, size, MPI_COMM_SELF);
  const IndexWindow window{{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
  const auto orientation = pfc::observables::region_orientation(
      density, MPI_COMM_SELF, window, CrystalSymmetry::four_fold);
  REQUIRE_FALSE(orientation.valid);
  const double k = 2.0 * std::numbers::pi * 4.0 / nx;
  std::array<std::array<double, 3>, 2> candidates{};
  candidates[0] = {k, 0.0, 0.0};
  candidates[1] = {0.0, k, 0.0};
  const auto grains = pfc::observables::label_grains(
      density, MPI_COMM_SELF, window, CrystalSymmetry::four_fold, candidates);
  REQUIRE_FALSE(grains.valid);
}

TEST_CASE("two orientations separated by a synthetic boundary",
          "[crystal_observables]") {
  constexpr int nx = 32;
  constexpr int ny = 24;
  constexpr int nz = 4;
  const std::array<double, 3> left{mode_wavenumber(2, nx), 0.0, 0.0};
  const std::array<double, 3> right{mode_wavenumber(2, nx), mode_wavenumber(1, ny),
                                    0.0};
  const double right_angle = std::atan2(right[1], right[0]);
  std::vector<double> values(static_cast<std::size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::array<double, 3> &wave = i < 16 ? left : right;
        const double phase = i < 16 ? 0.4 : -1.1;
        values[static_cast<std::size_t>(i + nx * (j + ny * k))] =
            std::cos(wave[0] * i + wave[1] * j + wave[2] * k + phase);
      }
    }
  }
  const std::array<int, 3> size{nx, ny, nz};
  const OwnedDensity density = full_density(values, size, MPI_COMM_SELF);
  const IndexWindow clean_left{{1, 0, 0}, {14, ny - 1, nz - 1}};
  const IndexWindow clean_right{{17, 0, 0}, {30, ny - 1, nz - 1}};
  const std::array<std::array<double, 3>, 2> candidates{left, right};
  const auto left_wave = pfc::observables::region_orientation(
      density, MPI_COMM_SELF, clean_left, CrystalSymmetry::none);
  const auto right_wave = pfc::observables::region_orientation(
      density, MPI_COMM_SELF, clean_right, CrystalSymmetry::none);
  REQUIRE(left_wave.valid);
  REQUIRE(right_wave.valid);
  REQUIRE(left_wave.angle_xy == Approx(0.0).margin(1e-8));
  REQUIRE(right_wave.angle_xy == Approx(right_angle).margin(1e-8));
  REQUIRE(right_wave.wavevector[0] == Approx(right[0]).margin(1e-7));
  REQUIRE(right_wave.wavevector[1] == Approx(right[1]).margin(1e-7));

  const auto left_grains = pfc::observables::label_grains(
      density, MPI_COMM_SELF, clean_left, CrystalSymmetry::none, candidates);
  const auto right_grains = pfc::observables::label_grains(
      density, MPI_COMM_SELF, clean_right, CrystalSymmetry::none, candidates);
  REQUIRE(left_grains.valid);
  REQUIRE(right_grains.valid);
  REQUIRE(left_grains.voting_cells > 0);
  REQUIRE(right_grains.voting_cells > 0);
  REQUIRE(left_grains.fraction[0] == 1.0);
  REQUIRE(left_grains.fraction[1] == 0.0);
  REQUIRE(right_grains.fraction[0] == 0.0);
  REQUIRE(right_grains.fraction[1] == 1.0);
}

TEST_CASE("diagnostics series records the crystal observables",
          "[crystal_observables]") {
  constexpr std::array<int, 3> size{32, 8, 2};
  const auto early_values = crystal_front(15.3);
  const auto later_values = crystal_front(15.7);
  const OwnedDensity early = full_density(early_values, size, MPI_COMM_SELF);
  const OwnedDensity later = full_density(later_values, size, MPI_COMM_SELF);
  const PlanarFrontOptions options;
  const auto first =
      pfc::observables::planar_front_position(early, MPI_COMM_SELF, options);
  const std::array<std::array<double, 3>, 1> unused{{{0.5, 0.0, 0.0}}};
  const IndexWindow all{{0, 0, 0}, {size[0] - 1, size[1] - 1, size[2] - 1}};
  const auto path =
      std::filesystem::temp_directory_path() / "openpfc-crystal-observables.csv";
  std::filesystem::remove(path);
  {
    pfc::io::DiagnosticsSeries series(
        pfc::io::crystal_diagnostic_columns(unused.size()),
        pfc::io::DiagnosticsSeriesOptions{
            .path = path, .comm = MPI_COMM_SELF, .overwrite = true});
    pfc::io::record_crystal_diagnostics(series, 2, 2.0, later, MPI_COMM_SELF,
                                        options, &first, 0.0, CrystalSymmetry::none,
                                        all, unused);
    series.close();
  }
  std::ifstream input(path);
  std::string header;
  std::string row;
  REQUIRE(std::getline(input, header));
  REQUIRE(std::getline(input, row));
  const auto columns = split_csv(header);
  const auto values = split_csv(row);
  REQUIRE(columns.size() == values.size());
  REQUIRE(columns[0] == "step");
  REQUIRE(columns[1] == "time");
  REQUIRE(columns[2] == "front_position");
  REQUIRE(columns[3] == "front_velocity");
  REQUIRE(columns[4] == "front_valid");
  REQUIRE(std::stod(values[2]) == Approx(15.7).margin(1e-9));
  REQUIRE(std::stod(values[3]) == Approx(0.2).margin(1e-9));
  REQUIRE(std::stod(values[4]) == 1.0);
  std::filesystem::remove(path);
}

TEST_CASE("split ranks agree with the one-rank crystal observables",
          "[MPI][crystal_observables]") {
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE((size == 1 || size == 2));
  constexpr int nx = 32;
  constexpr int ny = 16;
  constexpr int nz = 2;
  const std::array<int, 3> shape{nx, ny, nz};
  std::vector<double> complete(static_cast<std::size_t>(nx * ny * nz));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      const double weight = j < ny / 2 ? 1.0 : 3.0;
      for (int i = 0; i < nx; ++i) {
        complete[static_cast<std::size_t>(i + nx * (j + ny * k))] =
            envelope(static_cast<double>(i), 15.3) * weight;
      }
    }
  }
  const double transverse = std::sqrt(0.5 * (1.0 + 9.0));
  PlanarFrontOptions options;
  options.level = 0.5 * transverse;
  const OwnedDensity serial = full_density(complete, shape, MPI_COMM_SELF);
  const auto serial_front =
      pfc::observables::planar_front_position(serial, MPI_COMM_SELF, options);
  REQUIRE(serial_front.validity == FrontValidity::valid);
  REQUIRE(serial_front.position == Approx(15.3).margin(1e-9));

  const int y0 = rank * (ny / size);
  const int y1 = (rank + 1) * (ny / size) - 1;
  pfc::Box3i owned = pfc::Box3i::from_bounds({0, y0, 0}, {nx - 1, y1, nz - 1});
  std::vector<double> local(static_cast<std::size_t>(owned.count()));
  for (int k = 0; k < owned.size[2]; ++k) {
    for (int j = 0; j < owned.size[1]; ++j) {
      for (int i = 0; i < owned.size[0]; ++i) {
        const int gy = y0 + j;
        local[static_cast<std::size_t>(i +
                                       owned.size[0] * (j + owned.size[1] * k))] =
            complete[static_cast<std::size_t>(i + nx * (gy + ny * k))];
      }
    }
  }
  OwnedDensity part;
  part.values = local;
  part.global_size = shape;
  part.spacing = {1.0, 1.0, 1.0};
  part.origin = {0.0, 0.0, 0.0};
  part.box = owned;
  const auto split_front =
      pfc::observables::planar_front_position(part, MPI_COMM_WORLD, options);
  REQUIRE(split_front.validity == serial_front.validity);
  REQUIRE(split_front.position == Approx(serial_front.position).margin(1e-9));

  const std::array<double, 3> left_k{mode_wavenumber(2, nx), 0.0, 0.0};
  const std::array<double, 3> right_k{mode_wavenumber(2, nx), mode_wavenumber(1, ny),
                                      0.0};
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::array<double, 3> &wave = i < nx / 2 ? left_k : right_k;
        complete[static_cast<std::size_t>(i + nx * (j + ny * k))] =
            std::cos(wave[0] * i + wave[1] * j);
      }
    }
  }
  for (int k = 0; k < owned.size[2]; ++k) {
    for (int j = 0; j < owned.size[1]; ++j) {
      for (int i = 0; i < owned.size[0]; ++i) {
        const int gy = y0 + j;
        local[static_cast<std::size_t>(i +
                                       owned.size[0] * (j + owned.size[1] * k))] =
            complete[static_cast<std::size_t>(i + nx * (gy + ny * k))];
      }
    }
  }
  const OwnedDensity serial_wave = full_density(complete, shape, MPI_COMM_SELF);
  const IndexWindow clean_right{{nx / 2 + 1, 0, 0}, {nx - 2, ny - 1, nz - 1}};
  const auto serial_orientation = pfc::observables::region_orientation(
      serial_wave, MPI_COMM_SELF, clean_right, CrystalSymmetry::none);
  const auto split_orientation = pfc::observables::region_orientation(
      part, MPI_COMM_WORLD, clean_right, CrystalSymmetry::none);
  REQUIRE(serial_orientation.valid);
  REQUIRE(split_orientation.valid);
  REQUIRE(split_orientation.angle_xy ==
          Approx(serial_orientation.angle_xy).margin(1e-12));
  REQUIRE(split_orientation.wavevector[0] ==
          Approx(serial_orientation.wavevector[0]).margin(1e-12));
  REQUIRE(split_orientation.wavevector[1] ==
          Approx(serial_orientation.wavevector[1]).margin(1e-12));
  const std::array<std::array<double, 3>, 2> candidates{left_k, right_k};
  const auto serial_grains = pfc::observables::label_grains(
      serial_wave, MPI_COMM_SELF, clean_right, CrystalSymmetry::none, candidates);
  const auto split_grains = pfc::observables::label_grains(
      part, MPI_COMM_WORLD, clean_right, CrystalSymmetry::none, candidates);
  REQUIRE(serial_grains.fraction[1] == 1.0);
  REQUIRE(split_grains.counts == serial_grains.counts);
  REQUIRE(split_grains.voting_cells == serial_grains.voting_cells);

  if (size == 2) {
    // A rank-local malformed owned span must not strand its peer in an
    // MPI collective; both ranks take the same invalid-input path.
    OwnedDensity malformed = part;
    if (rank == 0) {
      const auto shortened = std::span<const double>(local.data(), local.size() - 1);
      malformed.values = shortened;
    }
    const auto invalid_front = [&] {
      return pfc::observables::planar_front_position(malformed, MPI_COMM_WORLD,
                                                     options);
    };
    REQUIRE_THROWS_AS(invalid_front(), std::invalid_argument);
  }
}
