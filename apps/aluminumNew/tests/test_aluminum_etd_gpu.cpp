// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_ALUMINUM_ETD_HIP) &&                                      \
    !defined(OPENPFC_TEST_ALUMINUM_ETD_CUDA)

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

#include <aluminum/aluminum_etd_gpu_session.hpp>
#include <aluminum/aluminum_etd_session.hpp>

using nlohmann::json;

namespace {

json model_params() {
  return {{"n0", -0.0060},           {"alpha", 0.20},
          {"n_sol", -0.036},         {"n_vap", -1.297},
          {"T_const", 980.0},        {"T_min", 780.0},
          {"T_max", 1280.0},         {"T0", 89285.0},
          {"Bx", 0.817900686921996}, {"G_grid", 0.0},
          {"V_grid", 0.0},           {"x_initial", 130.0},
          {"alpha_farTol", 0.001},   {"alpha_highOrd", 0},
          {"lambda", 0.22},          {"stabP", 0.0},
          {"shift_u", 1.0},          {"shift_s", 0.0},
          {"p2_bar", 0.8286531831},  {"p3_bar", -0.04204863},
          {"p4_bar", 0.007533},      {"q20_bar", 0.016531729105214},
          {"q21_bar", 5.467},        {"q30_bar", 1.7152418049986},
          {"q31_bar", 0.45},         {"q40_bar", 0.787482}};
}

json mini_settings() {
  return {
      {"model", {{"name", "aluminum"}, {"params", model_params()}}},
      {"domain",
       {{"Lx", 8},
        {"Ly", 8},
        {"Lz", 8},
        {"dx", 1.0},
        {"dy", 1.0},
        {"dz", 1.0},
        {"origin", "corner"}}},
      {"timestepping", {{"t0", 0.0}, {"t1", 0.02}, {"dt", 0.01}, {"saveat", 0.01}}},
      {"initial_conditions",
       {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.0060}}}}};
}

double max_abs_diff(const std::vector<double> &a, const double *b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

template <class HostSession, class DeviceSession>
void run_host_vs_device(const json &settings) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  HostSession host(settings, rank, nproc, MPI_COMM_WORLD);
  host.run();

  DeviceSession dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.run();

  const auto &psi_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    REQUIRE(max_abs_diff(psi_h, d, n) < 1e-10);
  });
}

} // namespace

TEST_CASE("AluminumETDGPUSession matches host AluminumETDSession within 1e-10",
          "[aluminum][gpu][session]") {
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<aluminum::AluminumETDSession, aluminum::AluminumETDHIPSession>(
      mini_settings());
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<aluminum::AluminumETDSession, aluminum::AluminumETDCUDASession>(
      mini_settings());
#endif
}

TEST_CASE("AluminumETDGPUSession matches host with G_grid",
          "[aluminum][gpu][session][temperature]") {
  json s = mini_settings();
  s["model"]["params"]["G_grid"] = 0.5;
  s["model"]["params"]["V_grid"] = 0.1;
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<aluminum::AluminumETDSession, aluminum::AluminumETDHIPSession>(
      s);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<aluminum::AluminumETDSession, aluminum::AluminumETDCUDASession>(
      s);
#endif
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
