// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <system_error>
#include <unistd.h>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mpi.h>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/model.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>
#include <tungsten/common/tungsten_spectral.hpp>
#include <tungsten/cpu/tungsten_model.hpp>
#include <tungsten/tungsten_etd_session.hpp>
#include <tungsten/tungsten_field_modifiers.hpp>
#include <tungsten/tungsten_physics.hpp>

using Catch::Approx;
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

void fill_ic(pfc::data::Field<double> &psi) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return -0.10 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
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

TEST_CASE("SpectralMeanFieldETDSystem matches Gen-1 Tungsten one step",
          "[tungsten][physics][etd]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Tungsten legacy(fft_legacy, domain);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_ic(ic);
  legacy.get_real_field("psi") = ic.vec();

  tungsten::TungstenPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  phys.params = legacy.params;
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::SpectralMeanFieldETDSystem<tungsten::TungstenPhysics<>> sys(
      phys, fft_new, state, dt);

  REQUIRE(sys.linear_symbol().size() == fft_new.size_outbox());
  REQUIRE(sys.filter_mf().size() == fft_new.size_outbox());

  legacy.step(0.0);
  sys.step(0.0);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("SpectralMeanFieldETDSystem matches Gen-1 Tungsten for 10 steps",
          "[tungsten][physics][etd][multistep]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Tungsten legacy(fft_legacy, domain);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_ic(ic);
  legacy.get_real_field("psi") = ic.vec();

  tungsten::TungstenPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  phys.params = legacy.params;
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::SpectralMeanFieldETDSystem<tungsten::TungstenPhysics<>> sys(
      phys, fft_new, state, dt);

  double t = 0.0;
  for (int step = 0; step < 10; ++step) {
    legacy.step(t);
    t = sys.step(t);
  }
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("TungstenETDSession JSON constant IC matches Gen-1 two steps",
          "[tungsten][physics][session]") {
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
       {{{"target", "psi"}, {"type", "constant"}, {"n0", -0.10}}}}};

  tungsten::TungstenETDSession session(settings, 0, 1, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Tungsten legacy(fft, domain);
  tungsten::apply_tungsten_json(settings["model"]["params"], legacy.params);
  pfc::initialize(legacy, 0.01);
  std::fill(legacy.get_real_field("psi").begin(), legacy.get_real_field("psi").end(),
            -0.10);

  session.run();
  legacy.step(0.0);
  legacy.step(0.01);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"), session.psi().vec()) < 1e-10);
}

TEST_CASE("TungstenETDSession JSON single_seed IC matches Gen-1 two steps",
          "[tungsten][physics][session][ic]") {
  json settings = golden_settings(8, 0.02, 0.01);
  settings["initial_conditions"] =
      json::array({{{"target", "psi"}, {"type", "constant"}, {"n0", -0.4}},
                   {{"target", "psi"},
                    {"type", "single_seed"},
                    {"amp_eq", 0.215936},
                    {"rho_seed", -0.047}}});

  tungsten::TungstenETDSession session(settings, 0, 1, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Tungsten legacy(fft, domain);
  tungsten::apply_tungsten_json(settings["model"]["params"], legacy.params);
  pfc::initialize(legacy, 0.01);
  auto &psi = legacy.get_real_field("psi");
  tungsten::apply_ics_from_json(settings, domain, fft.get_inbox_bounds(), psi.data(),
                                psi.size());

  session.run();
  legacy.step(0.0);
  legacy.step(0.01);
  REQUIRE(max_abs_diff(psi, session.psi().vec()) < 1e-10);
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

TEST_CASE("TungstenETDSession writes psi binary dumps on saveat",
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

  tungsten::TungstenETDSession session(settings, 0, 1, MPI_COMM_WORLD);
  session.run();
  REQUIRE(session.dumps() == 3);
  REQUIRE(std::filesystem::exists(dir / "psi_0.bin"));
  REQUIRE(std::filesystem::exists(dir / "psi_1.bin"));
  REQUIRE(std::filesystem::exists(dir / "psi_2.bin"));
  REQUIRE(std::filesystem::file_size(dir / "psi_0.bin") ==
          8u * 8u * 8u * sizeof(double));
  std::filesystem::remove_all(dir);
}

TEST_CASE("TungstenETDSession checkpoint restart matches continuous run",
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

  tungsten::TungstenETDSession continuous(full, 0, 1, MPI_COMM_WORLD);
  continuous.run();

  const auto ckpt_root =
      std::filesystem::temp_directory_path() / "openpfc_tungsten_ckpt";
  std::error_code ec;
  std::filesystem::remove_all(ckpt_root, ec);
  std::filesystem::create_directories(ckpt_root);

  first["checkpoint"] = json{{"every", 2}, {"directory", ckpt_root.string()}};
  tungsten::TungstenETDSession head(first, 0, 1, MPI_COMM_WORLD);
  head.run();
  REQUIRE(head.time().get_increment() == 2);
  REQUIRE(std::filesystem::exists(ckpt_root / "step_2" / "metadata.json"));

  full["restart_from"] = (ckpt_root / "step_2").string();
  tungsten::TungstenETDSession tail(full, 0, 1, MPI_COMM_WORLD);
  REQUIRE(tail.time().get_increment() == 2);
  tail.run();

  REQUIRE(max_abs_diff(continuous.psi().vec(), tail.psi().vec()) < 1e-12);
  std::filesystem::remove_all(ckpt_root, ec);
}

TEST_CASE("TungstenETDSession 1-rank 100-step golden vs Gen-1",
          "[tungsten][golden]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  constexpr double t1 = 1.0;
  constexpr int nsteps = 100;
  const json settings = golden_settings(N, t1, dt);
  tungsten::TungstenETDSession session(settings, 0, 1, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Tungsten legacy(fft, domain);
  tungsten::apply_tungsten_json(settings["model"]["params"], legacy.params);
  pfc::initialize(legacy, dt);
  std::fill(legacy.get_real_field("psi").begin(), legacy.get_real_field("psi").end(),
            -0.10);

  session.run();
  for (int i = 0; i < nsteps; ++i) {
    legacy.step(static_cast<double>(i) * dt);
  }
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"), session.psi().vec()) < 1e-10);
}

TEST_CASE("TungstenETDSession 4-rank golden vs Gen-1", "[tungsten][golden][MPI]") {
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 4) {
    SKIP("requires exactly 4 MPI ranks");
  }
  constexpr int N = 16;
  constexpr double dt = 0.01;
  constexpr double t1 = 0.20;
  constexpr int nsteps = 20;
  const json settings = golden_settings(N, t1, dt);
  tungsten::TungstenETDSession session(settings, rank, nproc, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, nproc);
  auto fft = pfc::fft::create(decomp, rank, MPI_COMM_WORLD);
  Tungsten legacy(fft, domain);
  tungsten::apply_tungsten_json(settings["model"]["params"], legacy.params);
  pfc::initialize(legacy, dt);
  std::fill(legacy.get_real_field("psi").begin(), legacy.get_real_field("psi").end(),
            -0.10);

  session.run();
  for (int i = 0; i < nsteps; ++i) {
    legacy.step(static_cast<double>(i) * dt);
  }
  const double local =
      max_abs_diff(legacy.get_real_field("psi"), session.psi().vec());
  double global_max = 0.0;
  MPI_Allreduce(&local, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(global_max < 1e-10);

  double s_new = sumsq(session.psi().vec());
  double s_old = sumsq(legacy.get_real_field("psi"));
  double g_new = 0.0;
  double g_old = 0.0;
  MPI_Allreduce(&s_new, &g_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&s_old, &g_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(g_new - g_old) < 1e-12 * (1.0 + std::abs(g_old)));
}
