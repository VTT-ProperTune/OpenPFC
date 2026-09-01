// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <stdexcept>
#include <tungsten/common/tungsten_input.hpp>
#include <tungsten/common/tungsten_params.hpp>
#include <tungsten/common/tungsten_spectral.hpp>

using namespace Catch::Matchers;
using json = nlohmann::json;

/* Parameters from tungsten_single_seed.json:
{
    "n0": -0.10,
    "alpha": 0.50,
    "n_sol": -0.047,
    "n_vap": -0.464,
    "T": 3300.0,
    "T0": 156000.0,
    "Bx": 0.8582,
    "alpha_farTol": 0.001,
    "alpha_highOrd": 4,
    "lambda": 0.22,
    "stabP": 0.2,
    "shift_u": 0.3341,
    "shift_s": 0.1898,
    "p2": 1.0,
    "p3": -0.5,
    "p4": 0.333333333,
    "q20": -0.0037,
    "q21": 1.0,
    "q30": -12.4567,
    "q31": 20.0,
    "q40": 45.0
}
*/

TEST_CASE("Tungsten JSON parsing", "[Tungsten][JSON]") {
  SECTION("Parse valid JSON configuration") {
    json j = {{"n0", -0.10},        {"n_sol", -0.047},
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

    TungstenParams params;
    from_json(j, params);

    REQUIRE_THAT(params.get_n0(), WithinAbs(-0.10, 1e-10));
    REQUIRE_THAT(params.get_n_sol(), WithinAbs(-0.047, 1e-10));
    REQUIRE_THAT(params.get_n_vap(), WithinAbs(-0.464, 1e-10));
    REQUIRE_THAT(params.get_T(), WithinAbs(3300.0, 1e-10));
    REQUIRE_THAT(params.get_T0(), WithinAbs(156000.0, 1e-10));
    REQUIRE_THAT(params.get_Bx(), WithinAbs(0.8582, 1e-10));
    REQUIRE_THAT(params.get_tau(), WithinAbs(3300.0 / 156000.0, 1e-10));
    REQUIRE(params.get_p2_bar() > 0.0);
    REQUIRE(params.get_q2_bar() != 0.0);
    REQUIRE(params.get_q3_bar() != 0.0);
  }

  SECTION("Reject invalid JSON - missing field") {
    json j = {{"n0", -0.10},
              {"n_sol", -0.047},
              // Missing n_vap
              {"T", 3300.0}};

    TungstenParams params;
    REQUIRE_THROWS_AS(from_json(j, params), std::invalid_argument);
  }

  SECTION("Reject invalid JSON - wrong type") {
    json j = {{"n0", "invalid"}, // Should be number
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
              {"q40", 45.0}};

    TungstenParams params;
    REQUIRE_THROWS_AS(from_json(j, params), std::invalid_argument);
  }
}

TEST_CASE("Tungsten parameter setters", "[Tungsten][Setters]") {
  TungstenParams params;

  SECTION("Set basic parameters") {
    params.set_n0(-0.10);
    params.set_n_sol(-0.047);
    params.set_n_vap(-0.464);
    params.set_T(3300.0);
    params.set_T0(156000.0);
    params.set_Bx(0.8582);

    REQUIRE_THAT(params.get_n0(), WithinAbs(-0.10, 1e-10));
    REQUIRE_THAT(params.get_n_sol(), WithinAbs(-0.047, 1e-10));
    REQUIRE_THAT(params.get_n_vap(), WithinAbs(-0.464, 1e-10));
    REQUIRE_THAT(params.get_T(), WithinAbs(3300.0, 1e-10));
    REQUIRE_THAT(params.get_T0(), WithinAbs(156000.0, 1e-10));
    REQUIRE_THAT(params.get_Bx(), WithinAbs(0.8582, 1e-10));
  }

  SECTION("Set parameters with derived values") {
    params.set_T(3300.0);
    params.set_T0(156000.0);
    REQUIRE_THAT(params.get_tau(), WithinAbs(3300.0 / 156000.0, 1e-10));

    params.set_shift_u(0.3341);
    params.set_shift_s(0.1898);
    params.set_p2(1.0);
    params.set_p3(-0.5);
    params.set_p4(0.333333333);

    double expected_p2_bar =
        1.0 + 2 * 0.1898 * (-0.5) + 3 * pow(0.1898, 2) * 0.333333333;
    REQUIRE_THAT(params.get_p2_bar(), WithinAbs(expected_p2_bar, 1e-8));
  }
}

// Helper to construct OperatorParams with representative values
tungsten::spectral::OperatorParams
make_test_params(double stabP, double p2_bar, double q2_bar, double T, double T0,
                 double Bx, double alpha2, double lambda2, double alpha_farTol,
                 int alpha_highOrd) {
  tungsten::spectral::OperatorParams p;
  p.stabP = stabP;
  p.p2_bar = p2_bar;
  p.q2_bar = q2_bar;
  p.T = T;
  p.T0 = T0;
  p.Bx = Bx;
  p.alpha2 = alpha2;
  p.lambda2 = lambda2;
  p.alpha_farTol = alpha_farTol;
  p.alpha_highOrd = alpha_highOrd;
  return p;
}

TEST_CASE("spectral_operators_exact_zero", "[tungsten][spectral]") {
  double k_laplacian = -4.0;
  double dt = 0.01;

  // Construct parameters such that opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar *
  // fMF = 0.0 By setting q2_bar = 0.0 and ensuringstabP + p2_bar - opPeak = 0.0
  auto p = make_test_params(1.0, 0.5, 0.0, 3300.0, 156000.0, 0.8582, 0.5, 0.0484,
                            0.001, 4);

  // Calculate what opPeak would be for k_laplacian = -4.0
  double k_val = std::sqrt(-k_laplacian) - 1.0;
  double k2 = k_val * k_val;
  double rTol = -p.alpha2 * std::log(p.alpha_farTol) - 1.0;
  double g1 = std::exp(-(k2 + rTol * std::pow(k_val, p.alpha_highOrd)) / p.alpha2);
  double g2 = 1.0 - 1.0 / p.alpha2 * k2;
  double gf = (k_val < 0.0) ? g1 : g2;
  double opPeak = p.Bx * std::exp(-p.T / p.T0) * gf;

  // Adjust stabP to make opCk = 0.0
  p.stabP = opPeak - p.p2_bar; // Then opCk = 0.0 + 0.5 - opPeak + 0.0*fMF = 0.0

  tungsten::spectral::ModeOperators out =
      tungsten::spectral::legacy_etd_weights_for_mode(k_laplacian, dt, p);

  // When opCk ≈ 0, expected opN = k_laplacian * dt from Taylor series
  double expected_opN = k_laplacian * dt;

  CHECK(out.opN == Catch::Approx(expected_opN).epsilon(1e-14));
  CHECK(out.opL == Catch::Approx(std::exp(k_laplacian * 0.0 * dt)).epsilon(1e-14));

  // Shared SpectralExpCoefficientCache mapping must match legacy weights
  const auto phys = tungsten::spectral::physics_for_mode(k_laplacian, p);
  const double L = tungsten::spectral::linear_symbol(k_laplacian, phys.opCk);
  const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
  const double shared_opN = k_laplacian * shared.phi1_L;
  CHECK(shared.exp_Ldt == Catch::Approx(out.opL).epsilon(1e-14));
  CHECK(shared_opN == Catch::Approx(out.opN).epsilon(1e-14));
}

TEST_CASE("spectral_operators_near_zero_no_cancellation",
          "[tungsten][spectral][numerical]") {
  double k_laplacian = -4.0;
  double dt = 0.01;

  auto p_base = make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582, 0.5,
                                 0.0484, 0.001, 4);

  std::vector<double> test_opCk_values = {1e-15, 1e-14, 1e-13, 1e-12, 1e-11};

  for (double target_opCk : test_opCk_values) {
    // Calculate opPeak to get the right starting point
    double k_val = std::sqrt(-k_laplacian) - 1.0;
    double k2 = k_val * k_val;
    double rTol = -p_base.alpha2 * std::log(p_base.alpha_farTol) - 1.0;
    double g1 = std::exp(-(k2 + rTol * std::pow(k_val, p_base.alpha_highOrd)) /
                         p_base.alpha2);
    double g2 = 1.0 - 1.0 / p_base.alpha2 * k2;
    double gf = (k_val < 0.0) ? g1 : g2;
    double opPeak = p_base.Bx * std::exp(-p_base.T / p_base.T0) * gf;

    // Adjust stabP to achieve target opCk
    // opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar * fMF
    // fMF = exp(k_laplacian / lambda2)
    double fMF = std::exp(k_laplacian / p_base.lambda2);
    p_base.stabP = target_opCk + opPeak - p_base.p2_bar - p_base.q2_bar * fMF;

    tungsten::spectral::ModeOperators out =
        tungsten::spectral::legacy_etd_weights_for_mode(k_laplacian, dt, p_base);

    // Reference: high-precision expm1 calculation
    double arg = k_laplacian * target_opCk * dt;
    double reference_opN = std::expm1(arg) / target_opCk;

    // Check within 10 ULPs of high-precision reference
    double relative_error =
        std::abs(out.opN - reference_opN) / std::abs(reference_opN);
    double max_relative_error = 10.0 * std::numeric_limits<double>::epsilon();
    CHECK(relative_error < max_relative_error);

    // Shared L + spectral_exp_coeffs vs legacy (near-zero may use |L| vs |opCk|)
    const auto phys = tungsten::spectral::physics_for_mode(k_laplacian, p_base);
    const double L = tungsten::spectral::linear_symbol(k_laplacian, phys.opCk);
    const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
    const double shared_opN = k_laplacian * shared.phi1_L;
    CHECK(shared.exp_Ldt == Catch::Approx(out.opL).epsilon(1e-12));
    CHECK(shared_opN == Catch::Approx(out.opN).epsilon(1e-12));
  }
}

TEST_CASE("spectral_operators_typical_values", "[tungsten][spectral]") {
  // Use representative parameter combinations from existing tests
  std::vector<std::tuple<double, double, tungsten::spectral::OperatorParams>>
      test_cases = {{-4.0, 0.01,
                     make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582, 0.5,
                                      0.0484, 0.001, 4)},
                    {-2.5, 0.005,
                     make_test_params(0.2, 0.3, 0.5, 3300.0, 156000.0, 0.8582, 0.5,
                                      0.0484, 0.001, 4)},
                    {-6.0, 0.001,
                     make_test_params(0.2, 0.7, 1.5, 3300.0, 156000.0, 0.8582, 0.5,
                                      0.0484, 0.001, 4)}};

  for (const auto &[k_laplacian, dt, p] : test_cases) {
    tungsten::spectral::ModeOperators out =
        tungsten::spectral::legacy_etd_weights_for_mode(k_laplacian, dt, p);

    // Calculate opCk for this case
    double k_val = std::sqrt(-k_laplacian) - 1.0;
    double k2 = k_val * k_val;
    double rTol = -p.alpha2 * std::log(p.alpha_farTol) - 1.0;
    double g1 = std::exp(-(k2 + rTol * std::pow(k_val, p.alpha_highOrd)) / p.alpha2);
    double g2 = 1.0 - 1.0 / p.alpha2 * k2;
    double gf = (k_val < 0.0) ? g1 : g2;
    double opPeak = p.Bx * std::exp(-p.T / p.T0) * gf;
    double fMF = std::exp(k_laplacian / p.lambda2);
    double opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar * fMF;

    double arg = k_laplacian * opCk * dt;
    double expected_opN = std::expm1(arg) / opCk;

    CHECK(out.opN == Catch::Approx(expected_opN).epsilon(1e-14));
    CHECK(out.opL == Catch::Approx(std::exp(arg)).epsilon(1e-14));

    const double L = tungsten::spectral::linear_symbol(k_laplacian, opCk);
    const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
    CHECK(shared.exp_Ldt == Catch::Approx(out.opL).epsilon(1e-14));
    CHECK((k_laplacian * shared.phi1_L) == Catch::Approx(out.opN).epsilon(1e-14));
  }
}

TEST_CASE("spectral_operators_stability_long_dt",
          "[tungsten][spectral][numerical]") {
  double k_laplacian = -4.0;

  auto p = make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582, 0.5, 0.0484,
                            0.001, 4);

  // Test a range of dt values where arg varies significantly
  std::vector<double> test_dt_values = {0.001, 0.01, 0.1, 1.0};

  for (double dt : test_dt_values) {
    tungsten::spectral::ModeOperators out =
        tungsten::spectral::legacy_etd_weights_for_mode(k_laplacian, dt, p);

    // Calculate opCk for this case
    double k_val = std::sqrt(-k_laplacian) - 1.0;
    double k2 = k_val * k_val;
    double rTol = -p.alpha2 * std::log(p.alpha_farTol) - 1.0;
    double g1 = std::exp(-(k2 + rTol * std::pow(k_val, p.alpha_highOrd)) / p.alpha2);
    double g2 = 1.0 - 1.0 / p.alpha2 * k2;
    double gf = (k_val < 0.0) ? g1 : g2;
    double opPeak = p.Bx * std::exp(-p.T / p.T0) * gf;
    double fMF = std::exp(k_laplacian / p.lambda2);
    double opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar * fMF;

    double arg = k_laplacian * opCk * dt;
    double expected_opN = std::expm1(arg) / opCk;
    double expected_opL = std::exp(arg);

    // Check both operators match mathematical definitions
    CHECK(out.opN == Catch::Approx(expected_opN).epsilon(1e-12));
    CHECK(out.opL == Catch::Approx(expected_opL).epsilon(1e-12));

    const double L = tungsten::spectral::linear_symbol(k_laplacian, opCk);
    const auto shared = pfc::integrator::spectral_exp_coeffs(L, dt);
    CHECK(shared.exp_Ldt == Catch::Approx(out.opL).epsilon(1e-12));
    CHECK((k_laplacian * shared.phi1_L) == Catch::Approx(out.opN).epsilon(1e-12));
  }
}

TEST_CASE("spectral_exp_cache_matches_legacy_etd_weights",
          "[tungsten][spectral][integrator]") {
  const double k_laplacian = -4.0;
  const double dt = 0.01;
  auto p = make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582, 0.5, 0.0484,
                            0.001, 4);

  const auto legacy =
      tungsten::spectral::legacy_etd_weights_for_mode(k_laplacian, dt, p);
  const auto phys = tungsten::spectral::physics_for_mode(k_laplacian, p);
  const double L = tungsten::spectral::linear_symbol(k_laplacian, phys.opCk);

  pfc::integrator::SpectralExpCoefficientCache<> cache;
  std::array<double, 1> L_arr{L};
  cache.ensure(L_arr, dt, pfc::integrator::SpectralExpOperatorId{.value = 1},
               pfc::integrator::SpectralExpDtId::from_bits(dt),
               pfc::integrator::SpectralExpConfigId{.value = 1});

  REQUIRE(cache.exp_Ldt().size() == 1);
  REQUIRE(cache.phi1_L().size() == 1);
  CHECK(cache.exp_Ldt()[0] == Catch::Approx(legacy.opL).epsilon(1e-14));
  CHECK((k_laplacian * cache.phi1_L()[0]) ==
        Catch::Approx(legacy.opN).epsilon(1e-14));
}

int main(int argc, char *argv[]) {
  // Initialize MPI once for all tests
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI initialization failed" << '\n';
    return 1;
  }

  // Run Catch2 tests
  int result = Catch::Session().run(argc, argv);

  // Finalize MPI
  MPI_Finalize();
  return result;
}
