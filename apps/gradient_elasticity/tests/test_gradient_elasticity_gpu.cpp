// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_gradient_elasticity_gpu.cpp
 * @brief CPU vs HIP one-shot session parity for gradient elasticity.
 */

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <gradient_elasticity/gradient_elasticity_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {
      {"model",
       {{"name", "gradient_elasticity"},
        {"params", {{"ell", 1.0}, {"eps0", 0.02}}}}},
      {"domain",
       {{"Lx", 16},
        {"Ly", 16},
        {"Lz", 1},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 1.0}, {"dt", 1.0}, {"saveat", -1.0}}},
      {"initial_conditions",
       {{{"target", "g"},
         {"type", "cosine_mode"},
         {"g0", 0.0},
         {"amplitude", 1.0},
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

TEST_CASE("GradientElasticityHIPSession matches host within 1e-10",
          "[gradient_elasticity][hip][session]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  gradient_elasticity::register_catalog();
  const json settings = mini_settings();
  gradient_elasticity::GradientElasticityCPUSession host(settings, rank, nproc,
                                                         MPI_COMM_WORLD);
  host.run();
  gradient_elasticity::GradientElasticityHIPSession dev(settings, rank, nproc,
                                                        MPI_COMM_WORLD);
  dev.run();

  const auto &ux_h = host.ux().vec();
  const auto &uy_h = host.uy().vec();
  dev.ux().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == ux_h.size());
    REQUIRE(max_abs_diff(ux_h, d, n) < 1e-10);
  });
  dev.uy().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == uy_h.size());
    REQUIRE(max_abs_diff(uy_h, d, n) < 1e-10);
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
