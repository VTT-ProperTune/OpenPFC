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
#include <aluminum/fixed_bc.hpp>
#include <aluminum/moving_bc.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>

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

// The aluminum fixed and moving boundaries on a device field change the same
// cells to the same values as the host path, in double and (ADR 0006, #872)
// in float; the float host path is the double scratch of apply_field_modifier.
TEST_CASE("Aluminum solidification BCs on a device field match the host path",
          "[aluminum][gpu][bc]") {
#if defined(OPENPFC_TEST_ALUMINUM_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) SKIP("HIP not available");
  using Space = pfc::HIPSpace;
#else
  if (!pfc::gpu::test::is_cuda_available()) SKIP("CUDA not available");
  using Space = pfc::CUDASpace;
#endif
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) MPI_Init(nullptr, nullptr);

  const pfc::Domain domain =
      pfc::domain::from_bounds({48, 6, 4}, {-20.0, 0.0, 0.0}, {33.0, 6.0, 4.0});
  const pfc::Box3i box = pfc::domain::index_box(domain);

  auto compare = [&]<class T>(pfc::FieldModifier &on_host,
                              pfc::FieldModifier &on_device) {
    REQUIRE(pfc::has_device_applier<T>(on_device));
    // Solid (-0.036) behind x = 5, vapour (-1.297) ahead: a front to track.
    auto fill = [&](T *d, std::size_t n) {
      for (std::size_t c = 0; c < n; ++c) {
        const int i = static_cast<int>(c % 48);
        const double x = pfc::domain::to_coords(domain, {i, 0, 0})[0];
        d[c] = static_cast<T>(x < 5.0 ? -0.036 : -1.297);
      }
    };
    pfc::data::Field<T> h(domain, box);
    pfc::data::Field<T, Space> d(domain, box);
    h.with_host_view(fill);
    d.with_host_view(fill);
    std::vector<T> before;
    h.with_host_view([&](T *p, std::size_t n) { before.assign(p, p + n); });
    for (double t : {0.0, 1.0}) {
      pfc::apply_field_modifier(on_host, h, t);
      pfc::apply_field_modifier(on_device, d, t);
    }
    REQUIRE(d.residency().device_valid());
    std::vector<T> hv, dv;
    h.with_host_view([&](T *p, std::size_t n) { hv.assign(p, p + n); });
    d.with_host_view([&](T *p, std::size_t n) { dv.assign(p, p + n); });
    const double tol = std::is_same_v<T, double> ? 1e-14 : 1e-6;
    int changed = 0;
    for (std::size_t c = 0; c < hv.size(); ++c) {
      REQUIRE((hv[c] != before[c]) == (dv[c] != before[c]));
      REQUIRE(std::abs(static_cast<double>(hv[c]) - static_cast<double>(dv[c])) <=
              tol);
      changed += hv[c] != before[c];
    }
    REQUIRE(changed > 0);
  };

  auto moving_pair = [](aluminum::MovingBC &a, aluminum::MovingBC &b) {
    for (auto *bc : {&a, &b}) {
      bc->set_threshold(-0.5);
      bc->set_xwidth(3.0);
      bc->set_disp(4.0);
      bc->set_xpos(-10.0);
    }
  };

  SECTION("fixed, double") {
    aluminum::FixedBC host(-1.297, -0.006), device(-1.297, -0.006);
    compare.template operator()<double>(host, device);
  }
  SECTION("fixed, float") {
    aluminum::FixedBC host(-1.297, -0.006), device(-1.297, -0.006);
    compare.template operator()<float>(host, device);
  }
  SECTION("moving, double") {
    aluminum::MovingBC host(-1.297, -0.006), device(-1.297, -0.006);
    moving_pair(host, device);
    compare.template operator()<double>(host, device);
    REQUIRE(host.get_xpos() == device.get_xpos());
  }
  SECTION("moving, float") {
    aluminum::MovingBC host(-1.297, -0.006), device(-1.297, -0.006);
    moving_pair(host, device);
    compare.template operator()<float>(host, device);
    REQUIRE(host.get_xpos() == device.get_xpos());
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
