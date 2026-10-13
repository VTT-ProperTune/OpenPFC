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
#include <string>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/device_field_modifier.hpp>
#include <tungsten/fixed_bc.hpp>
#include <tungsten/moving_bc.hpp>
#include <tungsten/tungsten_session.hpp>

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
       {{"size", {8, 8, 8}},
        {"lower", {0.0, 0.0, 0.0}},
        {"upper", {8.0, 8.0, 8.0}}}},
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
    REQUIRE(std::isfinite(a[i]));
    REQUIRE(std::isfinite(b[i]));
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

TEST_CASE("TungstenGPUSession matches host TungstenSession within 1e-10",
          "[tungsten][gpu][session]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<tungsten::TungstenSession, tungsten::TungstenHIPSession>(
      mini_settings(), false);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<tungsten::TungstenSession, tungsten::TungstenCUDASession>(
      mini_settings(), false);
#endif
}

TEST_CASE("TungstenETD 32^3/10-step sine IC host vs device within 1e-10",
          "[tungsten][gpu][session][parity32]") {
  json settings = mini_settings();
  settings["model"]["params"]["n0"] = -0.4;
  settings["model"]["params"]["T"] = 0.5;
  settings["domain"]["size"] = {32, 32, 32};
  settings["domain"]["upper"] = {32.0, 32.0, 32.0};
  settings["timestepping"] = {
      {"t0", 0.0}, {"t1", 0.1}, {"dt", 0.01}, {"saveat", 0.01}};
  settings["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}}});

#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  using Dev = tungsten::TungstenHIPSession;
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  using Dev = tungsten::TungstenCUDASession;
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

  tungsten::TungstenSession host(settings, rank, nproc, MPI_COMM_WORLD);
  fill_sine(host.psi().vec().data(), host.psi().vec().size());
  std::vector<std::vector<double>> host_states;
  host.run([&] { host_states.push_back(host.psi().vec()); });
  REQUIRE(host_states.size() == 11);

  std::vector<std::vector<double>> distributed;
  for (const char *choice : {"heffte", "local"}) {
    if (nproc != 1 && std::string(choice) == "local") continue;
    INFO("fft = " << choice);
    settings["fft"] = choice;
    Dev dev(settings, rank, nproc, MPI_COMM_WORLD);
    CHECK(dev.stack().fft_choice() ==
          (std::string(choice) == "local"
               ? pfc::sim::stacks::DeviceFFTChoice::single_device
               : pfc::sim::stacks::DeviceFFTChoice::heffte));
    dev.psi().with_host_view([&](double *d, std::size_t n) { fill_sine(d, n); });
    const bool reference = distributed.empty();
    std::size_t state = 0;
    dev.run([&] {
      REQUIRE(state < host_states.size());
      dev.psi().with_host_view([&](double *d, std::size_t n) {
        REQUIRE(n == host_states[state].size());
        REQUIRE(max_abs_diff(host_states[state], d, n) < 1e-10);
        if (reference)
          distributed.emplace_back(d, d + n);
        else
          REQUIRE(max_abs_diff(distributed[state], d, n) < 1e-10);
      });
      ++state;
    });
    REQUIRE(state == host_states.size());
  }
}

TEST_CASE("TungstenGPUSession single_seed IC matches host within 1e-10",
          "[tungsten][gpu][session][ic]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }
  run_host_vs_device<tungsten::TungstenSession, tungsten::TungstenHIPSession>(
      single_seed_settings(), true);
#else
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }
  run_host_vs_device<tungsten::TungstenSession, tungsten::TungstenCUDASession>(
      single_seed_settings(), true);
#endif
}

// Boundary conditions on a device field run on the device (no host round
// trip) and change the same cells to the same values as the host path. The
// box does not start at 0 and the moving front is tracked over two calls.
TEST_CASE("Solidification BCs on a device field match the host path",
          "[tungsten][gpu][bc]") {
#if defined(OPENPFC_TEST_TUNGSTEN_ETD_HIP)
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
  // Solid (-0.05) behind x = 5, vapour (-0.4) ahead: a front to track.
  auto fill = [&](double *d, std::size_t n) {
    for (std::size_t c = 0; c < n; ++c) {
      const int i = static_cast<int>(c % 48);
      const double x = pfc::domain::to_coords(domain, {i, 0, 0})[0];
      d[c] = x < 5.0 ? -0.05 : -0.4;
    }
  };

  auto compare = [&](pfc::FieldModifier &on_host, pfc::FieldModifier &on_device) {
    REQUIRE(pfc::has_device_applier(on_device));
    pfc::data::Field<double> h(domain, box);
    pfc::data::Field<double, Space> d(domain, box);
    h.with_host_view(fill);
    d.with_host_view(fill);
    std::vector<double> before;
    h.with_host_view([&](double *p, std::size_t n) { before.assign(p, p + n); });
    for (double t : {0.0, 1.0}) {
      pfc::apply_field_modifier(on_host, h, t);
      pfc::apply_field_modifier(on_device, d, t);
    }
    std::vector<double> hv, dv;
    h.with_host_view([&](double *p, std::size_t n) { hv.assign(p, p + n); });
    d.with_host_view([&](double *p, std::size_t n) { dv.assign(p, p + n); });
    int changed = 0;
    for (std::size_t c = 0; c < hv.size(); ++c) {
      REQUIRE((hv[c] != before[c]) == (dv[c] != before[c]));
      REQUIRE(std::abs(hv[c] - dv[c]) <= 1e-14);
      changed += hv[c] != before[c];
    }
    REQUIRE(changed > 0);
  };

  SECTION("fixed") {
    tungsten::FixedBC host(-0.464, -0.1), device(-0.464, -0.1);
    compare(host, device);
  }
  SECTION("moving") {
    tungsten::MovingBC host(-0.464, -0.1), device(-0.464, -0.1);
    for (auto *bc : {&host, &device}) {
      bc->set_threshold(-0.2);
      bc->set_xwidth(3.0);
      bc->set_disp(4.0);
      bc->set_xpos(-10.0);
    }
    compare(host, device);
    REQUIRE(host.get_xpos() == device.get_xpos());
  }
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
