// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <system_error>
#include <unistd.h>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mpi.h>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed_grid.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <tungsten/common/tungsten_spectral.hpp>
#include <tungsten/tungsten_session.hpp>
#include <tungsten/tungsten_physics.hpp>

using Catch::Approx;
using Catch::Matchers::WithinRel;
using nlohmann::json;

namespace {

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

json tungsten_params_json() {
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

json golden_settings(int n, double t1, double dt) {
  return {{"model", {{"name", "tungsten"}, {"params", tungsten_params_json()}}},
          {"domain",
           {{"Lx", n},
            {"Ly", n},
            {"Lz", n},
            {"dx", 1.0},
            {"dy", 1.0},
            {"dz", 1.0},
            {"origin", "corner"}}},
          {"timestepping", {{"t0", 0.0}, {"t1", t1}, {"dt", dt}, {"saveat", dt}}},
          {"initial_conditions",
           {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};
}

double sumsq(const std::vector<double> &v) {
  double s = 0.0;
  for (double x : v) {
    s += x * x;
  }
  return s;
}

} // namespace

TEST_CASE("TungstenPhysics schema round-trips JSON params",
          "[tungsten][physics][schema]") {
  auto schema = tungsten::TungstenPhysics<>::schema();
  json j = {{"n0", -0.12},        {"n_sol", -0.047},
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
  const auto vals = schema.parse(j);
  REQUIRE(vals.n0 == Approx(-0.12));
  TungstenParams p;
  tungsten::apply_schema_values(vals, p);
  REQUIRE(p.get_n0() == Approx(-0.12));
  REQUIRE(p.get_alpha_highOrd() == 4);
}

TEST_CASE("TungstenPhysics ETD weights: zero mode, near-zero, long-dt",
          "[tungsten][physics][spectral]") {
  constexpr double k_lap = -4.0;

  SECTION("zero mode") {
    tungsten::TungstenPhysics<> phys;
    auto op = tungsten::spectral::make_operator_params(phys.params);
    const auto mode = tungsten::spectral::physics_for_mode(k_lap, op);
    const double op_peak =
        op.stabP + op.p2_bar + op.q2_bar * mode.filterMF - mode.opCk;
    phys.params.set_stabP(op_peak - phys.params.get_p2_bar() -
                          phys.params.get_q2_bar() * mode.filterMF);
    op = tungsten::spectral::make_operator_params(phys.params);
    const double dt = 0.01;
    const auto legacy =
        tungsten::spectral::legacy_etd_weights_for_mode(k_lap, dt, op);
    const double L = phys.linear_symbol(k_lap);
    REQUIRE(std::abs(L) < 1e-12);
    const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
    REQUIRE(shared.exp_Ldt == Approx(legacy.opL).epsilon(1e-14));
    REQUIRE((k_lap * shared.phi1_L) == Approx(legacy.opN).epsilon(1e-14));
  }

  SECTION("near-zero opCk") {
    const double dt = 0.01;
    for (double target : {1e-15, 1e-14, 1e-13, 1e-12, 1e-11}) {
      tungsten::TungstenPhysics<> phys;
      auto op = tungsten::spectral::make_operator_params(phys.params);
      const auto mode = tungsten::spectral::physics_for_mode(k_lap, op);
      const double op_peak =
          op.stabP + op.p2_bar + op.q2_bar * mode.filterMF - mode.opCk;
      phys.params.set_stabP(target + op_peak - phys.params.get_p2_bar() -
                            phys.params.get_q2_bar() * mode.filterMF);
      op = tungsten::spectral::make_operator_params(phys.params);
      const auto legacy =
          tungsten::spectral::legacy_etd_weights_for_mode(k_lap, dt, op);
      const double L = phys.linear_symbol(k_lap);
      const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
      REQUIRE(shared.exp_Ldt == Approx(legacy.opL).epsilon(1e-12));
      REQUIRE((k_lap * shared.phi1_L) == Approx(legacy.opN).epsilon(1e-12));
    }
  }

  SECTION("long-dt") {
    tungsten::TungstenPhysics<> phys;
    const auto op = tungsten::spectral::make_operator_params(phys.params);
    for (double dt : {0.001, 0.01, 0.1, 1.0}) {
      const auto legacy =
          tungsten::spectral::legacy_etd_weights_for_mode(k_lap, dt, op);
      const double L = phys.linear_symbol(k_lap);
      const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
      REQUIRE(shared.exp_Ldt == Approx(legacy.opL).epsilon(1e-12));
      REQUIRE((k_lap * shared.phi1_L) == Approx(legacy.opN).epsilon(1e-12));
    }
  }
}

TEST_CASE("TungstenPhysics linear_symbol matches physics_for_mode",
          "[tungsten][physics][symbol]") {
  tungsten::TungstenPhysics<> phys;
  const auto op = tungsten::spectral::make_operator_params(phys.params);
  const double k_lap = -4.0;
  const auto mode = tungsten::spectral::physics_for_mode(k_lap, op);
  REQUIRE(phys.linear_symbol(k_lap) ==
          Approx(tungsten::spectral::linear_symbol(k_lap, mode.opCk)));
  REQUIRE(phys.filter_mf(k_lap) == Approx(mode.filterMF));
}

TEST_CASE("TungstenSession writes psi binary dumps on saveat",
          "[tungsten][physics][io]") {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("openpfc_tungsten_etd_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto pattern = (dir / "psi_%d.bin").string();
  json settings = {
      {"model",
       {{"name", "tungsten"},
        {"params", {{"n0", -0.10},        {"n_sol", -0.047},
                    {"n_vap", -0.464},    {"T", 3300.0},
                    {"T0", 156000.0},     {"Bx", 0.8582},
                    {"alpha", 0.50},      {"alpha_farTol", 0.001},
                    {"alpha_highOrd", 4}, {"lambda", 0.22},
                    {"stabP", 0.2},       {"shift_u", 0.3341},
                    {"shift_s", 0.1898},  {"p2", 1.0},
                    {"p3", -0.5},         {"p4", 0.333333333},
                    {"q20", -0.0037},     {"q21", 1.0},
                    {"q30", -12.4567},    {"q31", 20.0},
                    {"q40", 45.0}}}}},
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
       {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}},
      {"fields", {{{"name", "psi"}, {"data", pattern}}}}};

  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.dumps() == 3);
  REQUIRE(std::filesystem::exists(dir / "psi_0.bin"));
  REQUIRE(std::filesystem::exists(dir / "psi_1.bin"));
  REQUIRE(std::filesystem::exists(dir / "psi_2.bin"));
  REQUIRE(std::filesystem::file_size(dir / "psi_0.bin") ==
          8u * 8u * 8u * sizeof(double));
  std::filesystem::remove_all(dir);
}

TEST_CASE("TungstenSession checkpoint restart matches continuous run",
          "[tungsten][checkpoint][restart]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    return;
  }

  constexpr int N = 8;
  constexpr double dt = 0.01;
  json first = golden_settings(N, 0.02, dt);
  first["timestepping"]["integrator"] = json{{"method", "etd1"}};
  json full = golden_settings(N, 0.04, dt);
  full["timestepping"]["integrator"] = json{{"method", "etd1"}};

  tungsten::TungstenSession continuous(full, 0, 1, MPI_COMM_WORLD);
  continuous.run();

  const auto ckpt_root =
      std::filesystem::temp_directory_path() / "openpfc_tungsten_ckpt";
  std::error_code ec;
  std::filesystem::remove_all(ckpt_root, ec);
  std::filesystem::create_directories(ckpt_root);

  first["checkpoint"] = json{{"every", 2}, {"directory", ckpt_root.string()}};
  tungsten::TungstenSession head(first, 0, 1, MPI_COMM_WORLD);
  head.run();
  REQUIRE(head.time().get_increment() == 2);
  REQUIRE(std::filesystem::exists(ckpt_root / "step_2" / "metadata.json"));

  full["restart_from"] = (ckpt_root / "step_2").string();
  tungsten::TungstenSession tail(full, 0, 1, MPI_COMM_WORLD);
  REQUIRE(tail.time().get_increment() == 2);
  tail.run();

  REQUIRE(max_abs_diff(continuous.psi().vec(), tail.psi().vec()) < 1e-12);
  std::filesystem::remove_all(ckpt_root, ec);
}

TEST_CASE("TungstenSession 1-rank 100-step run", "[tungsten][golden]") {
  const json settings = golden_settings(8, 1.0, 0.01);
  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  const double s = sumsq(session.psi().vec());
  REQUIRE(std::isfinite(s));
  REQUIRE(s > 0.0);
}

TEST_CASE("TungstenSession 4-rank 16^3/20-step run", "[tungsten][golden][MPI]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 4) {
    SKIP("requires exactly 4 MPI ranks");
  }
  const json settings = golden_settings(16, 0.20, 0.01);
  tungsten::TungstenSession session(settings, rank, nproc, MPI_COMM_WORLD);
  session.run();
  double s = sumsq(session.psi().vec());
  double g = 0.0;
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::isfinite(g));
  REQUIRE(g > 0.0);
}

TEST_CASE("TungstenSession 32^3/10-step sine IC CPU checksum",
          "[tungsten][etd_cpu_golden][parity]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  json settings = golden_settings(32, 0.1, 0.01);
  settings["model"]["params"]["n0"] = -0.4;
  settings["model"]["params"]["T"] = 0.5;
  settings["timestepping"]["saveat"] = 0.05;
  settings["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}}});

  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  auto &psi = session.psi().vec();
  for (std::size_t i = 0; i < psi.size(); ++i) {
    psi[i] = -0.4 + 0.1 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                                   static_cast<double>(psi.size()));
  }
  session.run();

  double sum = 0.0;
  double sumsq_v = 0.0;
  for (double x : psi) {
    sum += x;
    sumsq_v += x * x;
  }
  std::cout << std::setprecision(17) << "CPU_GOLDEN tungsten_etd n=" << psi.size()
            << " sum=" << sum << " sumsq=" << sumsq_v << '\n';
  REQUIRE(psi.size() == 32768);
  REQUIRE(std::isfinite(sum));
  REQUIRE(std::isfinite(sumsq_v));
  // Tohtori g0005, gcc 15.2 Debug. Same 32³/10-step sine IC as Gen-1
  // tungsten-cpu-golden; checksums match that pin bitwise on this capture.
  REQUIRE_THAT(sum, WithinRel(-13107.200000000043, 1e-10));
  REQUIRE_THAT(sumsq_v, WithinRel(5406.3450894885682, 1e-10));
}

TEST_CASE("Tungsten seed_grid IC writes crystalline seeds",
          "[tungsten][etd][seed_grid]") {
  auto domain = pfc::domain::create(pfc::GridSize({16, 16, 16}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {15, 15, 15});
  pfc::data::Field<double> psi(domain, box, 0);
  pfc::SeedGrid ic(2, 2, 4.0, 3.0);
  ic.set_amplitude(0.2);
  ic.set_density(-0.047);
  pfc::apply_field_modifier(ic, psi, 0.0);
  double span = 0.0;
  for (double x : psi.vec()) {
    span = std::max(span, std::abs(x));
  }
  REQUIRE(span > 1e-6);
}

TEST_CASE("TungstenSession runs with moving BC JSON",
          "[tungsten][etd][moving_bc]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  tungsten::register_catalog();
  json settings = golden_settings(8, 0.02, 0.01);
  settings["boundary_conditions"] = json::array({{{"target", "psi"},
                                                  {"type", "moving"},
                                                  {"rho_low", -0.464},
                                                  {"rho_high", -0.10},
                                                  {"width", 2.0},
                                                  {"alpha", 1.0},
                                                  {"disp", 1.0},
                                                  {"xpos", 4.0}}});
  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.psi().vec().size() == 512);
}

TEST_CASE("TungstenSession writes profiling JSON", "[tungsten][etd][profiling]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  json settings = golden_settings(8, 0.02, 0.01);
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("tungsten_etd_prof_" + std::to_string(static_cast<long>(::getpid())));
  std::filesystem::create_directories(dir);
  const auto stem = (dir / "profile").string();
  settings["profiling"] = {{"enabled", true},
                           {"format", "json"},
                           {"output", stem},
                           {"print_report", false}};
  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(std::filesystem::exists(stem + ".json"));
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE("SpectralETDSession infers vtk from .vti path", "[tungsten][etd][vtk]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);
  json settings = golden_settings(8, 0.01, 0.01);
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("tungsten_etd_vtk_" + std::to_string(static_cast<long>(::getpid())));
  std::filesystem::create_directories(dir);
  settings["fields"] =
      json::array({{{"name", "psi"}, {"data", (dir / "psi_%04d.vti").string()}}});
  tungsten::TungstenSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.dumps() >= 1);
  bool found = false;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".vti") {
      found = true;
    }
  }
  REQUIRE(found);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
