// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../../../tests/fixtures/session.hpp"

#if !defined(OPENPFC_TEST_ALUMINUM_ETD_HIP) &&                                      \
    !defined(OPENPFC_TEST_ALUMINUM_ETD_CUDA)

int main(int argc, char *argv[]) { return pfc::test::run(argc, argv); }

#else

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_session.hpp>

using nlohmann::json;

namespace {

/// Every field the schema declares, and nothing it does not. The schema is
/// enforced since 0.2, so an incomplete block here is a load failure.
json model_params() {
  return {{"alpha", 0.20},
          {"T_const", 980.0},
          {"T_min", 780.0},
          {"T_max", 1280.0},
          {"T0", 89285.0},
          {"Bx", 0.817900686921996},
          {"G_grid", 0.0},
          {"V_grid", 0.0},
          {"x_initial", 130.0},
          {"lambda", 0.22},
          {"stabP", 0.0},
          {"p2_bar", 0.8286531831},
          {"p3_bar", -0.04204863},
          {"p4_bar", 0.007533},
          {"q20_bar", 0.016531729105214},
          {"q21_bar", 5.467},
          {"q30_bar", 1.7152418049986},
          {"q31_bar", 0.45},
          {"q40_bar", 0.787482}};
}

json mini_settings() {
  return {
      {"model", {{"name", "aluminum"}, {"params", model_params()}}},
      {"domain",
       {{"size", {8, 8, 8}},
        {"lower", {0.0, 0.0, 0.0}},
        {"upper", {8.0, 8.0, 8.0}}}},
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

TEST_CASE("AluminumGPUSession matches host AluminumSession within 1e-10",
          "[aluminum][gpu][session]") {
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<aluminum::AluminumSession, aluminum::AluminumHIPSession>(
      mini_settings());
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<aluminum::AluminumSession, aluminum::AluminumCUDASession>(
      mini_settings());
#endif
}

TEST_CASE("AluminumGPUSession matches host with G_grid",
          "[aluminum][gpu][session][temperature]") {
  json s = mini_settings();
  s["model"]["params"]["G_grid"] = 0.5;
  s["model"]["params"]["V_grid"] = 0.1;
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<aluminum::AluminumSession, aluminum::AluminumHIPSession>(s);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<aluminum::AluminumSession, aluminum::AluminumCUDASession>(s);
#endif
}

// ADR 0006 tolerance baseline (tests/baselines/BASELINES.md): the float
// device session against the double host session, every saved state within
// 1e-5, on the isothermal seed_grid_fcc 32^3 case and on the 8^3 case with the
// temperature profile, over the HeFFTe and the single-device FFT.
TEST_CASE("AluminumETD float device vs double host within 1e-5",
          "[aluminum][gpu][session][float]") {
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  using Dev = aluminum::AluminumHIPFloatSession;
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  using Dev = aluminum::AluminumCUDAFloatSession;
#endif
  static_assert(std::is_same_v<typename Dev::Real, float>);
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  aluminum::register_catalog();

  json seeded = mini_settings();
  seeded["domain"]["size"] = {32, 32, 32};
  seeded["domain"]["upper"] = {32.0, 32.0, 32.0};
  seeded["timestepping"] = {
      {"t0", 0.0}, {"t1", 0.05}, {"dt", 0.01}, {"saveat", 0.01}};
  seeded["initial_conditions"].push_back({{"target", "psi"},
                                          {"type", "seed_grid_fcc"},
                                          {"X0", 8.0},
                                          {"Ny", 2},
                                          {"Nz", 2},
                                          {"radius", 4.0},
                                          {"amplitude", 0.4},
                                          {"rho", -0.036},
                                          {"rseed", 42}});
  json thermal = mini_settings();
  thermal["model"]["params"]["G_grid"] = 0.5;
  thermal["model"]["params"]["V_grid"] = 0.1;
  thermal["model"]["params"]["x_initial"] = 4.0;

  for (json settings : {seeded, thermal}) {
    aluminum::AluminumSession host(settings, rank, nproc, MPI_COMM_WORLD);
    std::vector<std::vector<double>> host_states;
    host.run([&] { host_states.push_back(host.psi().vec()); });
    REQUIRE_FALSE(host_states.empty());

    settings["model"]["precision"] = "float";
    for (const char *choice : {"heffte", "local"}) {
      if (nproc != 1 && std::string(choice) == "local") continue;
      INFO("fft = " << choice);
      settings["fft"] = choice;
      Dev dev(settings, rank, nproc, MPI_COMM_WORLD);
      std::size_t state = 0;
      dev.run([&] {
        REQUIRE(state < host_states.size());
        dev.psi().with_host_view([&](float *d, std::size_t n) {
          REQUIRE(n == host_states[state].size());
          double m = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(std::isfinite(d[i]));
            m = std::max(
                m, std::abs(host_states[state][i] - static_cast<double>(d[i])));
          }
          REQUIRE(m < 1e-5);
        });
        ++state;
      });
      REQUIRE(state == host_states.size());
    }
  }
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  return pfc::test::run(argc, argv);
}

#endif
