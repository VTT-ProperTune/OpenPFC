// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_cahn_hilliard_gpu.cpp
 * @brief CPU vs HIP SpectralETDSession parity for Cahn–Hilliard.
 */

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <cahn_hilliard/cahn_hilliard_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {
      {"model", {{"name", "cahn_hilliard"}, {"params", {{"c0", 0.32}}}}},
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
       {{{"target", "c"},
         {"type", "cosine_mode"},
         {"c0", 0.32},
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

} // namespace

TEST_CASE("CahnHilliardHIPSession matches host within 1e-10",
          "[cahn_hilliard][hip][session]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  cahn_hilliard::register_catalog();
  const json settings = mini_settings();
  cahn_hilliard::CahnHilliardSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();
  cahn_hilliard::CahnHilliardHIPSession dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.run();

  const auto &c_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == c_h.size());
    REQUIRE(max_abs_diff(c_h, d, n) < 1e-10);
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
