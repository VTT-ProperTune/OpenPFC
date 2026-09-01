// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP) &&                                      \
    !defined(OPENPFC_TEST_TUNGSTEN_ETD_CUDA)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <tungsten/tungsten_etd_gpu_session.hpp>
#include <tungsten/tungsten_etd_session.hpp>

using nlohmann::json;

namespace {

json model_params() {
  return {{"n0", -0.10},        {"n_sol", -0.047},
          {"n_vap", -0.464},    {"T", 3300.0},
          {"T0", 156000.0},     {"Bx", 0.8582},
          {"alpha", 0.50},      {"alpha_farTol", 0.001},
          {"alpha_highOrd", 4}, {"lambda", 0.22},
          {"stabP", 0.2},       {"shift_u", 0.3341},
          {"shift_s", 0.1898},  {"p2", 1.0},
          {"p3", -0.5},         {"p4", 0.333333333},
          {"q20", -0.0037},     {"q21", 1.0},
          {"q30", -12.4567},    {"q31", 20.0},
          {"q40", 45.0}};
}

json mini_settings() {
  return {
      {"model", {{"name", "tungsten"}, {"params", model_params()}}},
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
       {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};
}

/** Pre-M0 App-GPU-IC: JSON single_seed IC, ≥2 steps. */
json single_seed_settings() {
  json s = mini_settings();
  s["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}},
                   {{"target", "psi"},
                    {"type", "single_seed"},
                    {"amp_eq", 0.215936},
                    {"rho_seed", -0.047}}});
  return s;
}

double max_abs_diff(const std::vector<double> &a, const double *b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

template <class HostSession, class DeviceSession>
void run_host_vs_device(const json &settings, bool require_variation) {
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
  if (require_variation) {
    double span = 0.0;
    for (double x : psi_h) {
      span = std::max(span, std::abs(x - psi_h.front()));
    }
    REQUIRE(span > 1e-8);
  }

  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    REQUIRE(max_abs_diff(psi_h, d, n) < 1e-10);
  });
}

} // namespace

TEST_CASE("TungstenETDGPUSession matches host TungstenETDSession within 1e-10",
          "[tungsten][gpu][session]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<tungsten::TungstenETDSession, tungsten::TungstenETDHIPSession>(
      mini_settings(), false);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<tungsten::TungstenETDSession, tungsten::TungstenETDCUDASession>(
      mini_settings(), false);
#endif
}

TEST_CASE("TungstenETD 32^3/10-step sine IC host vs device within 1e-10",
          "[tungsten][gpu][session][parity32]") {
  json settings = mini_settings();
  settings["model"]["params"]["n0"] = -0.4;
  settings["model"]["params"]["T"] = 0.5;
  settings["domain"]["Lx"] = 32;
  settings["domain"]["Ly"] = 32;
  settings["domain"]["Lz"] = 32;
  settings["timestepping"] = {
      {"t0", 0.0}, {"t1", 0.1}, {"dt", 0.01}, {"saveat", 0.05}};
  settings["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}}});

#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  using Dev = tungsten::TungstenETDHIPSession;
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  using Dev = tungsten::TungstenETDCUDASession;
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

  auto fill_sine = [](double *d, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = -0.4 + 0.1 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                                   static_cast<double>(n));
    }
  };

  tungsten::TungstenETDSession host(settings, rank, nproc, MPI_COMM_WORLD);
  fill_sine(host.psi().vec().data(), host.psi().vec().size());
  host.run();

  Dev dev(settings, rank, nproc, MPI_COMM_WORLD);
  dev.psi().with_host_view([&](double *d, std::size_t n) { fill_sine(d, n); });
  dev.run();

  const auto &psi_h = host.psi().vec();
  dev.psi().with_host_view([&](double *d, std::size_t n) {
    REQUIRE(n == psi_h.size());
    REQUIRE(max_abs_diff(psi_h, d, n) < 1e-10);
  });
}

TEST_CASE("TungstenETDGPUSession single_seed IC matches host within 1e-10",
          "[tungsten][gpu][session][ic]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<tungsten::TungstenETDSession, tungsten::TungstenETDHIPSession>(
      single_seed_settings(), true);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<tungsten::TungstenETDSession, tungsten::TungstenETDCUDASession>(
      single_seed_settings(), true);
#endif
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = Catch::Session().run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) {
    MPI_Finalize();
  }
  return result;
}

#endif
