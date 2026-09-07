// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_ehd_film_gpu.cpp
 * @brief CPU vs HIP SpectralETDSession parity for the EHD film.
 */

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <ehd_film/ehd_film_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {
      {"model", {{"name", "ehd_film"}, {"params", {{"B", 1.0}, {"A", 0.0}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.05}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "h"},
         {"type", "cosine_mode"},
         {"h0", 1.0},
         {"amplitude", 0.05},
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

} // namespace

TEST_CASE("EhdFilmHIPSession matches host within 1e-10",
          "[ehd_film][hip][session]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  ehd_film::register_catalog();
  const json settings = mini_settings();
  ehd_film::EhdFilmSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();
  ehd_film::EhdFilmHIPSession dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.run();

  const auto &h_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == h_h.size());
    REQUIRE(max_abs_diff(h_h, d, n) < 1e-10);
  });
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  return Catch::Session().run(argc, argv);
}
