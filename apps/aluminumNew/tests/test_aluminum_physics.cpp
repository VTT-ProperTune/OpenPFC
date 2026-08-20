// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <aluminum/aluminum_physics.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

using Catch::Approx;
using nlohmann::json;

namespace {

json aluminum_params_json() {
  return {{"n0", -0.0060},
          {"alpha", 0.20},
          {"n_sol", -0.036},
          {"n_vap", -1.297},
          {"T_const", 980.0},
          {"T_min", 780.0},
          {"T_max", 1280.0},
          {"T0", 89285.0},
          {"Bx", 0.817900686921996},
          {"G_grid", 0.0},
          {"V_grid", 0.0},
          {"x_initial", 130.0},
          {"alpha_farTol", 0.001},
          {"alpha_highOrd", 0},
          {"lambda", 0.22},
          {"stabP", 0.0},
          {"shift_u", 1.0},
          {"shift_s", 0.0},
          {"p2_bar", 0.8286531831},
          {"p3_bar", -0.04204863},
          {"p4_bar", 0.007533},
          {"q20_bar", 0.016531729105214},
          {"q21_bar", 5.467},
          {"q30_bar", 1.7152418049986},
          {"q31_bar", 0.45},
          {"q40_bar", 0.787482}};
}

/** Gen-1 `Aluminum::prepare_operators` peak / opCk at one k (Aluminum.hpp). */
void gen1_operators(double k_lap, const aluminum::AluminumParams &p,
                    double &filter_mf, double &p_f, double &op_ck) {
  const double alpha2 = 2.0 * p.alpha * p.alpha;
  const double lambda2 = 2.0 * p.lambda * p.lambda;
  filter_mf = std::exp(k_lap / lambda2);
  const double k_wave = std::sqrt(-k_lap) - 1.0;
  const double kp = std::sqrt(-k_lap) - 2.0 / std::sqrt(3.0);
  const double g1 = std::exp(-(k_wave * k_wave) / alpha2);
  const double gp1 = std::exp(-(kp * kp) / alpha2);
  const double peak = (g1 > gp1) ? g1 : gp1;
  p_f = p.Bx * std::exp(-p.tau_const) * peak;
  op_ck = p.stabP + p.p2_bar - p_f + p.q2_bar_L * filter_mf;
}

} // namespace

TEST_CASE("AluminumPhysics schema round-trips JSON params",
          "[aluminum][physics][schema]") {
  auto schema = aluminum::AluminumPhysics<>::schema();
  json j = aluminum_params_json();
  j["n0"] = -0.01;
  const auto vals = schema.parse(j);
  REQUIRE(vals.n0 == Approx(-0.01));
  aluminum::AluminumParams p;
  aluminum::apply_schema_values(vals, p);
  REQUIRE(p.n0 == Approx(-0.01));
  REQUIRE(p.tau_const == Approx(980.0 / 89285.0));
  REQUIRE(p.q4_bar == Approx(p.q40_bar));
  REQUIRE(p.m_xpos == Approx(130.0));
}

TEST_CASE("AluminumPhysics schema rejects missing G_grid",
          "[aluminum][physics][schema]") {
  json j = aluminum_params_json();
  j.erase("G_grid");
  REQUIRE_THROWS_AS(aluminum::make_aluminum_schema().parse(j),
                    std::invalid_argument);
}

TEST_CASE("AluminumPhysics operators match Gen-1 prepare_operators formula",
          "[aluminum][physics][symbol]") {
  aluminum::AluminumPhysics<> phys;
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  const double k_lap = -4.0;
  double filter_mf = 0.0;
  double p_f = 0.0;
  double op_ck = 0.0;
  gen1_operators(k_lap, phys.params, filter_mf, p_f, op_ck);
  REQUIRE(phys.filter_mf(k_lap) == Approx(filter_mf).epsilon(1e-14));
  REQUIRE(phys.correlation_kernel(k_lap) == Approx(p_f).epsilon(1e-14));
  REQUIRE(phys.linear_symbol(k_lap) ==
          Approx(k_lap * op_ck).epsilon(1e-14));
}

TEST_CASE("AluminumPhysics N and free-energy match Gen-1 step formula",
          "[aluminum][physics][nonlinearity]") {
  aluminum::AluminumPhysics<> phys;
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  const double u = 0.1;
  const double v = 0.05;
  const double p_star = 0.2;

  SECTION("isothermal G_grid=0") {
    const double T_var = 0.0;
    const double q3 =
        phys.params.q31_bar * phys.params.T_const / phys.params.T0 +
        phys.params.q30_bar;
    const double expected_n = phys.params.p3_bar * u * u +
                              phys.params.p4_bar * u * u * u + q3 * v * v +
                              phys.params.q4_bar * v * v * v;
    REQUIRE(phys.nonlinearity(u, v, p_star, T_var) ==
            Approx(expected_n).epsilon(1e-14));
    REQUIRE(phys.free_energy_density(u, v, p_star, T_var) ==
            Approx(phys.params.p3_bar * u * u * u / 3.0 +
                   phys.params.p4_bar * u * u * u * u / 4.0 +
                   q3 * u * v * v / 3.0 +
                   phys.params.q4_bar * u * v * v * v / 4.0 -
                   u * p_star / 2.0 + phys.params.p2_bar * u * u / 2.0 +
                   phys.params.q2_bar * u * v / 2.0)
                .epsilon(1e-14));
  }

  SECTION("finite T_var") {
    const double T_var = 50.0;
    const double q2n = phys.params.q21_bar * T_var / phys.params.T0;
    const double q3n =
        phys.params.q31_bar * (phys.params.T_const + T_var) / phys.params.T0 +
        phys.params.q30_bar;
    const double kernel = -(1.0 - std::exp(-T_var / phys.params.T0)) * p_star;
    const double expected_n = phys.params.p3_bar * u * u +
                              phys.params.p4_bar * u * u * u + q2n * v +
                              q3n * v * v + phys.params.q4_bar * v * v * v -
                              kernel;
    REQUIRE(phys.nonlinearity(u, v, p_star, T_var) ==
            Approx(expected_n).epsilon(1e-14));
  }
}

TEST_CASE("AluminumPhysics temperature_variation is zero when G_grid is 0",
          "[aluminum][physics][temperature]") {
  aluminum::AluminumPhysics<> phys;
  aluminum::apply_aluminum_json(aluminum_params_json(), phys.params);
  phys.domain = pfc::domain::create({32, 32, 32});
  REQUIRE(phys.temperature_variation(10.0, 1.0) == Approx(0.0));

  phys.params.G_grid = 0.5;
  phys.params.V_grid = 0.1;
  const double T = phys.temperature_variation(10.0, 2.0);
  REQUIRE(T != Approx(0.0));
}

TEST_CASE("AluminumPhysics declares psi on SimulationState",
          "[aluminum][physics][fields]") {
  aluminum::AluminumPhysics<> phys;
  phys.domain = pfc::domain::create({8, 8, 8});
  phys.box = pfc::Box3i::from_bounds({0, 0, 0}, {7, 7, 7});
  pfc::SimulationState state;
  phys.declare_fields(state);
  REQUIRE(state.has_field("psi"));
  REQUIRE(state.get_field<double>("psi").size() == 8 * 8 * 8);
}
