// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP) &&                                 \
    !defined(OPENPFC_TEST_TUNGSTEN_ETD_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <tungsten/tungsten_etd_gpu_session.hpp>
#include <tungsten/tungsten_etd_session.hpp>

using nlohmann::json;

namespace {

json mini_settings() {
  return {{"model",
           {{"name", "tungsten"},
            {"params",
             {{"n0", -0.10},
              {"n_sol", -0.047},
              {"n_vap", -0.464},
              {"T", 3300.0},
              {"T0", 156000.0},
              {"Bx", 0.8582},
              {"alpha", 0.50},
              {"alpha_farTol", 0.001},
              {"alpha_highOrd", 4},
              {"lambda", 0.22},
              {"stabP", 0.2},
              {"shift_u", 0.3341},
              {"shift_s", 0.1898},
              {"p2", 1.0},
              {"p3", -0.5},
              {"p4", 0.333333333},
              {"q20", -0.0037},
              {"q21", 1.0},
              {"q30", -12.4567},
              {"q31", 20.0},
              {"q40", 45.0}}}}},
          {"domain",
           {{"Lx", 8},
            {"Ly", 8},
            {"Lz", 8},
            {"dx", 1.0},
            {"dy", 1.0},
            {"dz", 1.0},
            {"origin", "corner"}}},
          {"timestepping",
           {{"t0", 0.0}, {"t1", 0.02}, {"dt", 0.01}, {"saveat", 0.01}}},
          {"initial_conditions",
           {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};
}

double max_abs_diff(const std::vector<double> &a, const double *b,
                    std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

} // namespace

TEST_CASE("TungstenETDGpuSession matches host TungstenETDSession within 1e-10",
          "[tungsten][gpu][session]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
#endif

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const json settings = mini_settings();
  tungsten::TungstenETDSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();

#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  tungsten::TungstenETDHipSession dev(settings, rank, nproc, MPI_COMM_WORLD);
#else
  tungsten::TungstenETDCudaSession dev(settings, rank, nproc, MPI_COMM_WORLD);
#endif
  dev.run();

  const auto &psi_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    REQUIRE(max_abs_diff(psi_h, d, n) < 1e-10);
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

#endif
