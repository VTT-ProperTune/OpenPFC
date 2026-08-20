// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "Aluminum.hpp"
#include "SeedGridFCC.hpp"
#include <algorithm>
#include <aluminum/aluminum_etd_session.hpp>
#include <aluminum/aluminum_physics.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/moving_frame_mean_field_etd.hpp>
#include <openpfc/openpfc.hpp>

using namespace Catch::Matchers;

namespace {
// Keep MPI alive for the whole binary. The Gen-1 test constructs MPI_Worker
// which otherwise Finalize()s at the end of its SECTION and later FFT tests
// abort on MPI_Comm_size after MPI_FINALIZE.
pfc::MPI_Worker g_mpi(0, nullptr, MPI_COMM_WORLD, false);
} // namespace

/* Parameters from aluminumNew.json:
{
    "n0": -0.0060,
    "alpha": 0.20,
    "n_sol": -0.036,
    "n_vap": -1.297,
    "T_const": 980,
    "T_min": 780,
    "T_max": 1280,
    "T0": 89285.0,
    "Bx": 0.817900686921996,
    "G_grid": 0,
    "V_grid": 0,
    "x_initial": 130,
    "alpha_farTol": 0.001,
    "alpha_highOrd": 0,
    "lambda": 0.22,
    "stabP": 0.0,
    "shift_u": 1.0,
    "shift_s": 0.0,
    "p2_bar": 0.8286531831,
    "p3_bar": -0.04204863,
    "p4_bar": 0.007533,
    "q20_bar": 0.016531729105214,
    "q21_bar": 5.467,
    "q30_bar": 1.7152418049986,
    "q31_bar": 0.45,
    "q40_bar": 0.787482
}
*/

TEST_CASE("Aluminum functionality", "[Aluminum]") {
  SECTION("Step model and calculate norm of the result") {
    pfc::MPI_Worker worker(0, nullptr);
    // Domain at app boundary
    auto domain = pfc::domain::create({32, 32, 32});
    auto decomp = pfc::decomposition::create(domain, 1);
    auto fft = pfc::fft::create(decomp);

    // World wrap where Model requires it
    pfc::World world({0, 0, 0}, {31, 31, 31}, domain);
    Aluminum aluminum(fft, world);
    aluminum.set_n0(-0.0060);
    aluminum.set_alpha(0.20);
    aluminum.set_n_sol(-0.036);
    aluminum.set_n_vap(-1.297);
    aluminum.set_T_const(980);
    aluminum.set_T_min(780);
    aluminum.set_T_max(1280);
    aluminum.set_T0(89285.0);
    aluminum.set_Bx(0.817900686921996);
    aluminum.set_G_grid(0);
    aluminum.set_V_grid(0);
    aluminum.set_x_initial(130);
    aluminum.set_alpha_farTol(0.001);
    aluminum.set_alpha_highOrd(0);
    aluminum.set_lambda(0.22);
    aluminum.set_stabP(0.0);
    aluminum.set_shift_u(1.0);
    aluminum.set_shift_s(0.0);
    aluminum.set_p2_bar(0.8286531831);
    aluminum.set_p3_bar(-0.04204863);
    aluminum.set_p4_bar(0.007533);
    aluminum.set_q20_bar(0.016531729105214);
    aluminum.set_q21_bar(5.467);
    aluminum.set_q30_bar(1.7152418049986);
    aluminum.set_q31_bar(0.45);
    aluminum.set_q40_bar(0.787482);
    double dt = 1.0e-2;
    aluminum.initialize(dt);

    SeedGridFCC ic;
    ic.set_Nx(1);
    ic.set_Ny(2);
    ic.set_Nz(2);
    ic.set_X0(8.0);
    ic.set_radius(4.0);
    ic.set_amplitude(0.4);
    ic.set_rho(-0.036);
    ic.set_rseed(42);

    std::vector<double> &psi = aluminum.get_real_field("psi");
    std::fill(psi.begin(), psi.end(), -0.0060);
    ic.apply(aluminum, 0.0);

    std::array<double, 5> expected_norms{1297.08, 1250.21, 1209.28, 1173.19,
                                         1141.09};
    bool norms_match = true;
    for (int i = 0; i < 5; ++i) {
      double norm2 = 0.0;
      for (auto &x : psi) {
        norm2 += x * x;
      }
      std::cout << "norm: " << norm2 << '\n';
      norms_match &= std::abs(norm2 - expected_norms[i]) <= 0.1;
      aluminum.step(1.0);
    }
    REQUIRE(norms_match);
  }
}

namespace {

nlohmann::json aluminum_params_json() {
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

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void fill_cosine_ic(pfc::data::Field<double> &psi, double n0) {
  const auto n = pfc::domain::get_size(psi.domain());
  const auto dx = pfc::domain::get_spacing(psi.domain());
  const double lx = static_cast<double>(n[0]) * dx[0];
  psi.apply([&](double x, double, double) {
    return n0 + 0.01 * std::cos(2.0 * pfc::pi * x / lx);
  });
}

} // namespace

TEST_CASE("MovingFrameMeanFieldETDSystem matches Gen-1 Aluminum one step",
          "[aluminum][physics][etd]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Aluminum legacy(fft_legacy, domain);
  from_json(aluminum_params_json(), legacy);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_cosine_ic(ic, -0.0060);
  legacy.get_real_field("psi") = ic.vec();

  aluminum::AluminumPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::MovingFrameMeanFieldETDSystem<aluminum::AluminumPhysics<>> sys(
      phys, fft_new, state, dt);

  REQUIRE(sys.linear_symbol().size() == fft_new.size_outbox());
  REQUIRE(sys.correlation_kernel().size() == fft_new.size_outbox());

  legacy.step(0.0);
  sys.step(0.0);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("MovingFrameMeanFieldETDSystem matches Gen-1 Aluminum for 10 steps",
          "[aluminum][physics][etd][multistep]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Aluminum legacy(fft_legacy, domain);
  from_json(aluminum_params_json(), legacy);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_cosine_ic(ic, -0.0060);
  legacy.get_real_field("psi") = ic.vec();

  aluminum::AluminumPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::MovingFrameMeanFieldETDSystem<aluminum::AluminumPhysics<>> sys(
      phys, fft_new, state, dt);

  double t = 0.0;
  for (int step = 0; step < 10; ++step) {
    legacy.step(t);
    t = sys.step(t);
  }
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("MovingFrameMeanFieldETDSystem matches Gen-1 with G_grid",
          "[aluminum][physics][etd][temperature]") {
  constexpr int N = 8;
  constexpr double dt = 0.01;
  nlohmann::json j = aluminum_params_json();
  j["G_grid"] = 0.5;
  j["V_grid"] = 0.1;
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft_legacy = pfc::fft::create(decomp);
  auto fft_new = pfc::fft::create(decomp);

  Aluminum legacy(fft_legacy, domain);
  from_json(j, legacy);
  pfc::initialize(legacy, dt);
  pfc::data::Field<double> ic(domain, fft_legacy.get_inbox_bounds(), 0);
  fill_cosine_ic(ic, -0.0060);
  legacy.get_real_field("psi") = ic.vec();

  aluminum::AluminumPhysics<> phys;
  phys.domain = domain;
  phys.box = fft_new.get_inbox_bounds();
  aluminum::apply_aluminum_json(j, phys.params);
  pfc::SimulationState state;
  phys.declare_fields(state);
  state.get_field<double>("psi").vec() = ic.vec();
  pfc::sim::MovingFrameMeanFieldETDSystem<aluminum::AluminumPhysics<>> sys(
      phys, fft_new, state, dt);

  legacy.step(0.0);
  sys.step(0.0);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"),
                       state.get_field<double>("psi").vec()) < 1e-10);
}

TEST_CASE("AluminumETDSession JSON constant IC matches Gen-1 two steps",
          "[aluminum][physics][session]") {
  nlohmann::json settings = {
      {"model", {{"name", "aluminum"}, {"params", aluminum_params_json()}}},
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
  aluminum::AluminumETDSession session(settings, 0, 1, MPI_COMM_WORLD);

  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 8}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);
  auto fft = pfc::fft::create(decomp);
  Aluminum legacy(fft, domain);
  from_json(settings["model"]["params"], legacy);
  pfc::initialize(legacy, 0.01);
  std::fill(legacy.get_real_field("psi").begin(), legacy.get_real_field("psi").end(),
            -0.0060);

  session.run();
  legacy.step(0.0);
  legacy.step(0.01);
  REQUIRE(max_abs_diff(legacy.get_real_field("psi"), session.psi().vec()) < 1e-10);
}
