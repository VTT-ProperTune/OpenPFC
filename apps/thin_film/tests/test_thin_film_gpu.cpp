// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_thin_film_gpu.cpp
 * @brief CPU vs HIP SpectralETDSession parity for thin-film lubrication.
 */

#include "test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <thin_film/nonlinear_driver.hpp>
#include <thin_film/thin_film_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {
      {"model", {{"name", "thin_film"}, {"params", {{"h0", 1.0}, {"A", 0.05}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.2}, {"dt", 0.1}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "h"},
         {"type", "cosine_mode"},
         {"h0", 1.0},
         {"amplitude", 0.01},
         {"nx", 2},
         {"ny", 0},
         {"nz", 0}}}}};
}

double max_abs_diff(const std::vector<double> &a, const double *b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

std::string tmp_dir() {
  const char *t = std::getenv("TMPDIR");
  return (t != nullptr && *t != '\0') ? std::string(t) : std::string("/tmp");
}

/// Last non-empty row of a `thin_film_nonlinear` diagnostics CSV, as
/// `{min_h, volume, hole_area_fraction, dominant_spacing}`.
std::array<double, 4> last_csv_row(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line, last;
  while (std::getline(in, line)) {
    if (!line.empty()) last = line;
  }
  REQUIRE(!last.empty());
  std::stringstream ss(last);
  std::string cell;
  std::vector<std::string> cells;
  while (std::getline(ss, cell, ',')) cells.push_back(cell);
  REQUIRE(cells.size() == 9);
  return {std::stod(cells[2]), std::stod(cells[5]), std::stod(cells[6]),
          std::stod(cells[7])};
}

nlohmann::json nonlinear_mini_settings(const std::string &csv_path) {
  return {{"domain", {{"Lx", 32}, {"Ly", 32}, {"Lz", 1}, {"dx", 1.0}}},
          {"model",
           {{"params",
             {{"h0", 1.0},
              {"gamma", 1.0},
              {"M0", 1.0},
              {"A", 0.05},
              {"h_star", 0.2}}}}},
          {"timestepping", {{"t1", 0.5}, {"dt", 0.01}, {"saveat", -1.0}}},
          {"initial_conditions", {{"amplitude", 0.02}, {"seed", 7}}},
          {"diagnostics", {{"csv", csv_path}}}};
}

} // namespace

TEST_CASE("ThinFilmHIPSession matches host within 1e-10",
          "[thin_film][hip][session]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  thin_film::register_catalog();
  const json settings = mini_settings();
  thin_film::ThinFilmSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();
  thin_film::ThinFilmHIPSession dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.run();

  const auto &h_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == h_h.size());
    REQUIRE(max_abs_diff(h_h, d, n) < 1e-10);
  });
}

TEST_CASE("thin_film_nonlinear HIP matches host dewetting observables",
          "[thin_film][hip][nonlinear]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const std::string dir = tmp_dir();
  const std::string json_path = dir + "/thin_film_nonlinear_hip_test.json";
  const std::string host_csv = dir + "/thin_film_nonlinear_hip_test_host.csv";
  const std::string dev_csv = dir + "/thin_film_nonlinear_hip_test_dev.csv";

  auto write_case = [&](const std::string &csv) {
    if (rank == 0) {
      std::ofstream(json_path) << nonlinear_mini_settings(csv).dump();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  };

  write_case(host_csv);
  const int rc_host =
      thin_film::run_thin_film_nonlinear<pfc::HostSpace,
                                         pfc::sim::stacks::SpectralCPUStack>(
          rank, nproc, MPI_COMM_WORLD, json_path);
  REQUIRE(rc_host == 0);

  write_case(dev_csv);
  const int rc_dev = thin_film::run_thin_film_nonlinear<
      pfc::HIPSpace, pfc::sim::stacks::GPUSpectralStack<pfc::HIPSpace>>(
      rank, nproc, MPI_COMM_WORLD, json_path);
  REQUIRE(rc_dev == 0);

  if (rank == 0) {
    const auto h = last_csv_row(host_csv);
    const auto d = last_csv_row(dev_csv);
    // rocFFT vs FFTW round differently, so this is a tolerance, not the
    // bit-identical guarantee the framework gives host-vs-host.
    for (std::size_t i = 0; i < h.size(); ++i) {
      const double scale = std::max(1.0, std::abs(h[i]));
      CHECK(std::abs(h[i] - d[i]) < 1.0e-6 * scale);
    }
  }
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  return Catch::Session().run(argc, argv);
}
