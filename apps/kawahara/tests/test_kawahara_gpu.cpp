// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_kawahara_gpu.cpp
 * @brief CPU vs HIP SpectralETDSession parity for Kawahara.
 */

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <kawahara/kawahara_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {
      {"model",
       {{"name", "kawahara"},
        {"params", {{"alpha", 1.0}, {"beta", 1.0}, {"gamma", -1.0}}}}},
      {"domain",
       {{"Lx", 32},
        {"Ly", 1},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.1}, {"dt", 0.05}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "u"},
         {"type", "cosine_mode"},
         {"u0", 0.0},
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

TEST_CASE("KawaharaHIPSession matches host within 1e-10",
          "[kawahara][hip][session]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  kawahara::register_catalog();
  const json settings = mini_settings();
  kawahara::KawaharaSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();
  kawahara::KawaharaHIPSession dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.run();

  const auto &u_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == u_h.size());
    REQUIRE(max_abs_diff(u_h, d, n) < 1e-10);
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
