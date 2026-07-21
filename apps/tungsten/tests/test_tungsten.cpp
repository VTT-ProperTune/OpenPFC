// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/openpfc.hpp>
#include <tungsten/common/tungsten_etd_workspace.hpp>
#include <tungsten/common/tungsten_spectral.hpp>
#include <tungsten/cpu/tungsten.hpp>

using namespace Catch::Matchers;

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

    pfc::MPI_Worker worker(0, nullptr);
    auto world = pfc::world::create({32, 32, 32});
    auto decomp = pfc::decomposition::create(world, 1);
    auto fft = pfc::fft::create(decomp);
    Tungsten tungsten(fft, world);
    from_json(j, tungsten);

    // Check basic parameters
    REQUIRE_THAT(tungsten.params.get_n0(), WithinAbs(-0.10, 1e-10));
    REQUIRE_THAT(tungsten.params.get_n_sol(), WithinAbs(-0.047, 1e-10));
    REQUIRE_THAT(tungsten.params.get_n_vap(), WithinAbs(-0.464, 1e-10));
    REQUIRE_THAT(tungsten.params.get_T(), WithinAbs(3300.0, 1e-10));
    REQUIRE_THAT(tungsten.params.get_T0(), WithinAbs(156000.0, 1e-10));
    REQUIRE_THAT(tungsten.params.get_Bx(), WithinAbs(0.8582, 1e-10));

    // Check derived parameters
    REQUIRE_THAT(tungsten.params.get_tau(), WithinAbs(3300.0 / 156000.0, 1e-10));
    REQUIRE(tungsten.params.get_p2_bar() > 0.0);
    REQUIRE(tungsten.params.get_q2_bar() != 0.0);
    REQUIRE(tungsten.params.get_q3_bar() != 0.0);
  }

  SECTION("Reject invalid JSON - missing field") {
    json j = {{"n0", -0.10},
              {"n_sol", -0.047},
              // Missing n_vap
              {"T", 3300.0}};

    pfc::MPI_Worker worker(0, nullptr);
    auto world = pfc::world::create({32, 32, 32});
    auto decomp = pfc::decomposition::create(world, 1);
    auto fft = pfc::fft::create(decomp);
    Tungsten tungsten(fft, world);

    REQUIRE_THROWS_AS(from_json(j, tungsten), std::invalid_argument);
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

    pfc::MPI_Worker worker(0, nullptr);
    auto world = pfc::world::create({32, 32, 32});
    auto decomp = pfc::decomposition::create(world, 1);
    auto fft = pfc::fft::create(decomp);
    Tungsten tungsten(fft, world);

    REQUIRE_THROWS_AS(from_json(j, tungsten), std::invalid_argument);
  }
}

TEST_CASE("Tungsten parameter setters", "[Tungsten][Setters]") {
  pfc::MPI_Worker worker(0, nullptr);
  auto world = pfc::world::create({32, 32, 32});
  auto decomp = pfc::decomposition::create(world, 1);
  auto fft = pfc::fft::create(decomp);
  Tungsten tungsten(fft, world);

  SECTION("Set basic parameters") {
    tungsten.params.set_n0(-0.10);
    tungsten.params.set_n_sol(-0.047);
    tungsten.params.set_n_vap(-0.464);
    tungsten.params.set_T(3300.0);
    tungsten.params.set_T0(156000.0);
    tungsten.params.set_Bx(0.8582);

    REQUIRE_THAT(tungsten.params.get_n0(), WithinAbs(-0.10, 1e-10));
    REQUIRE_THAT(tungsten.params.get_n_sol(), WithinAbs(-0.047, 1e-10));
    REQUIRE_THAT(tungsten.params.get_n_vap(), WithinAbs(-0.464, 1e-10));
    REQUIRE_THAT(tungsten.params.get_T(), WithinAbs(3300.0, 1e-10));
    REQUIRE_THAT(tungsten.params.get_T0(), WithinAbs(156000.0, 1e-10));
    REQUIRE_THAT(tungsten.params.get_Bx(), WithinAbs(0.8582, 1e-10));
  }

  SECTION("Set parameters with derived values") {
    tungsten.params.set_T(3300.0);
    tungsten.params.set_T0(156000.0);
    REQUIRE_THAT(tungsten.params.get_tau(), WithinAbs(3300.0 / 156000.0, 1e-10));

    tungsten.params.set_shift_u(0.3341);
    tungsten.params.set_shift_s(0.1898);
    tungsten.params.set_p2(1.0);
    tungsten.params.set_p3(-0.5);
    tungsten.params.set_p4(0.333333333);

    // Check that derived parameters are calculated
    double expected_p2_bar =
        1.0 + 2 * 0.1898 * (-0.5) + 3 * pow(0.1898, 2) * 0.333333333;
    REQUIRE_THAT(tungsten.params.get_p2_bar(), WithinAbs(expected_p2_bar, 1e-8));
  }
}

TEST_CASE("Tungsten functionality", "[Tungsten]") {
  SECTION("Step model and calculate norm of the result") {
    pfc::MPI_Worker worker(0, nullptr);
    // Create world with exact parameters from tungsten_single_seed.json
    // Grid: 32x32x32, spacing: 1.1107207345395915, origin: center
    // When origo="center", origin is at -0.5 * dx * Lx
    double grid_spacing = 1.1107207345395915;
    int Lx = 32;
    int Ly = 32;
    int Lz = 32;
    double x0 = -0.5 * grid_spacing * Lx;
    double y0 = -0.5 * grid_spacing * Ly;
    double z0 = -0.5 * grid_spacing * Lz;
    auto world = pfc::world::create(
        pfc::GridSize({Lx, Ly, Lz}), pfc::PhysicalOrigin({x0, y0, z0}),
        pfc::GridSpacing({grid_spacing, grid_spacing, grid_spacing}));
    auto decomp = pfc::decomposition::create(world, 1);
    auto fft = pfc::fft::create(decomp);

    Tungsten tungsten(fft, world);
    // Set parameters from tungsten_single_seed.json (exact values)
    // Order doesn't matter - derived parameters are calculated on-the-fly
    tungsten.params.set_n0(-0.10);
    tungsten.params.set_alpha(0.50);
    tungsten.params.set_n_sol(-0.047);
    tungsten.params.set_n_vap(-0.464);
    tungsten.params.set_T0(156000.0);
    tungsten.params.set_T(3300.0);
    tungsten.params.set_Bx(0.8582);
    tungsten.params.set_alpha_farTol(0.001);
    tungsten.params.set_alpha_highOrd(4);
    tungsten.params.set_lambda(0.22);
    tungsten.params.set_stabP(0.2);
    tungsten.params.set_shift_s(0.1898);
    tungsten.params.set_shift_u(0.3341);
    tungsten.params.set_p2(1.0);
    tungsten.params.set_p3(-0.5);
    tungsten.params.set_p4(0.333333333);
    tungsten.params.set_q20(-0.0037);
    tungsten.params.set_q21(1.0);
    tungsten.params.set_q30(-12.4567);
    tungsten.params.set_q31(20.0);
    tungsten.params.set_q40(45.0);
    double dt = 1.0;
    tungsten.initialize(dt);

    // Manually replicate the initial condition logic from the UI
    // This matches exactly what happens when initial conditions are applied
    std::vector<double> &psi = tungsten.get_real_field("psi");
    const pfc::World &w = pfc::get_world(tungsten);
    const auto &fft_ref = pfc::get_fft(tungsten);

    // 1. Constant initial condition: fill entire field with -0.4
    std::fill(psi.begin(), psi.end(), -0.4);

    // 2. Single seed initial condition: apply seed formula to points within
    // radius 64.0 Replicate the exact logic from SingleSeed::apply()
    pfc::types::Int3 low = pfc::fft::get_inbox(fft_ref).low;
    pfc::types::Int3 high = pfc::fft::get_inbox(fft_ref).high;

    auto spacing = pfc::world::get_spacing(w);
    auto origin = pfc::world::get_origin(w);
    double dx_ic = spacing[0];
    double dy_ic = spacing[1];
    double dz_ic = spacing[2];
    double x0_ic = origin[0];
    double y0_ic = origin[1];
    double z0_ic = origin[2];

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    double rho_seed = -0.047;
    double amp_eq = 0.215936;
    // Use seed radius of ~18.0 (about 50% of domain size, which is ~35.5 units)
    // Domain spans from -17.77 to +17.77, so radius 18 covers about half the domain
    double seed_radius = 18.0;
    double r2 = pow(seed_radius, 2);

    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0_ic + i * dx_ic;
          double y = y0_ic + j * dy_ic;
          double z = z0_ic + k * dz_ic;
          if (x * x + y * y + z * z < r2) {
            double u = rho_seed;
            for (int qi = 0; qi < 6; qi++) {
              u += 2.0 * amp_eq * cos(q[qi][0] * x + q[qi][1] * y + q[qi][2] * z);
            }
            psi[idx] = u;
          }
          idx += 1;
        }
      }
    }

    // Expected norms after each step (regression test values)
    // Grid: 32x32x32, spacing: 1.1107207345395915, dt=1.0
    // Initial conditions: constant(-0.4) + single_seed(amp_eq=0.215936,
    // rho_seed=-0.047, radius=18.0)
    // Updated after refactoring to use getters for derived parameters
    // (derived parameters now calculated on-the-fly, fixing parameter order bug)
    std::array<double, 11> expected_norms{
        11965.0889218507, // Initial state (after ICs applied)
        11259.9705028338, // After step 1
        11050.6245088282, // After step 2
        10903.2783913748, // After step 3
        10782.9029639299, // After step 4
        10678.0834708269, // After step 5
        10583.6114287826, // After step 6
        10496.5733807390, // After step 7
        10415.1258360383, // After step 8
        10338.0182669812, // After step 9
        10264.3672697202  // After step 10
    };

    // Verify initial norm (before any steps) - print for debugging
    double initial_norm = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    for (auto &x : psi) {
      initial_norm += x * x;
      min_val = std::min(min_val, x);
      max_val = std::max(max_val, x);
    }
    std::cout << "Initial norm: " << std::fixed << std::setprecision(10)
              << initial_norm << '\n';
    std::cout << "Initial field range: [" << min_val << ", " << max_val << "]"
              << '\n';
    std::cout << "Expected: constant -0.4 everywhere, then seed applied in center"
              << '\n';

    // Run 10 time steps and verify norms match expected values exactly
    // Expected norms are from actual simulation run with these exact parameters
    std::vector<double> actual_norms;
    for (int i = 0; i < 10; ++i) {
      tungsten.step(1.0);
      double norm2 = 0.0;
      for (auto &x : psi) {
        norm2 += x * x;
      }
      actual_norms.push_back(norm2);
      std::cout << "Step " << (i + 1) << " norm: " << std::fixed
                << std::setprecision(10) << norm2 << '\n';
    }

    // Print summary
    std::cout << "\nNorm changes:" << '\n';
    for (size_t i = 0; i < actual_norms.size(); ++i) {
      if (i == 0) {
        double change = actual_norms[i] - initial_norm;
        std::cout << "  Step " << (i + 1) << ": " << change << " (from initial)"
                  << '\n';
      } else {
        double change = actual_norms[i] - actual_norms[i - 1];
        std::cout << "  Step " << (i + 1) << ": " << change << " (from step " << i
                  << ")" << '\n';
      }
    }

    // Verify initial norm matches expected value
    REQUIRE_THAT(initial_norm, WithinAbs(expected_norms[0], 0.1));

    // Verify norms match expected values (tight tolerance for regression testing)
    REQUIRE(actual_norms.size() == 10);
    bool norms_match = true;
    for (int i = 0; i < 10; ++i) {
      norms_match &= std::abs(actual_norms[i] - expected_norms[i + 1]) <= 0.1;
    }
    REQUIRE(norms_match);
  }

  SECTION("Model initialization and allocation") {
    pfc::MPI_Worker worker(0, nullptr);
    auto world = pfc::world::create({32, 32, 32});
    auto decomp = pfc::decomposition::create(world, 1);
    auto fft = pfc::fft::create(decomp);

    Tungsten tungsten(fft, world);
    tungsten.params.set_n0(-0.10);
    tungsten.params.set_alpha(0.50);
    tungsten.params.set_T(3300.0);
    tungsten.params.set_T0(156000.0);
    tungsten.params.set_Bx(0.8582);
    tungsten.params.set_alpha_farTol(0.001);
    tungsten.params.set_alpha_highOrd(4);
    tungsten.params.set_lambda(0.22);
    tungsten.params.set_stabP(0.2);
    tungsten.params.set_shift_u(0.3341);
    tungsten.params.set_shift_s(0.1898);
    tungsten.params.set_p2(1.0);
    tungsten.params.set_p3(-0.5);
    tungsten.params.set_p4(0.333333333);
    tungsten.params.set_q20(-0.0037);
    tungsten.params.set_q21(1.0);
    tungsten.params.set_q30(-12.4567);
    tungsten.params.set_q31(20.0);
    tungsten.params.set_q40(45.0);

    double dt = 1.0;
    tungsten.initialize(dt);

    // Check that fields are allocated
    REQUIRE(tungsten.has_real_field("psi"));
    REQUIRE(tungsten.has_real_field("psiMF"));
    REQUIRE(tungsten.has_real_field("default")); // backward compatibility

    std::vector<double> &psi = tungsten.get_real_field("psi");
    REQUIRE(!psi.empty());
 }
}

// Helper to construct OperatorParams with representative values
tungsten::spectral::OperatorParams make_test_params(double stabP, double p2_bar,
                                                     double q2_bar, double T,
                                                     double T0, double Bx, double alpha2,
                                                     double lambda2, double alpha_farTol,
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

  // Construct parameters such that opCk = p.stabP + p.p2_bar - opPeak + p.q2_bar * fMF = 0.0
  // By setting q2_bar = 0.0 and ensuringstabP + p2_bar - opPeak = 0.0
  auto p = make_test_params(1.0, 0.5, 0.0, 3300.0, 156000.0, 0.8582,
                            0.5, 0.0484, 0.001, 4);

  // Calculate what opPeak would be for k_laplacian = -4.0
  double k_val = std::sqrt(-k_laplacian) - 1.0;
  double k2 = k_val * k_val;
  double rTol = -p.alpha2 * std::log(p.alpha_farTol) - 1.0;
  double g1 = std::exp(-(k2 + rTol * std::pow(k_val, p.alpha_highOrd)) / p.alpha2);
  double g2 = 1.0 - 1.0 / p.alpha2 * k2;
  double gf = (k_val < 0.0) ? g1 : g2;
  double opPeak = p.Bx * std::exp(-p.T / p.T0) * gf;

  // Adjust stabP to make opCk = 0.0
  p.stabP = opPeak - p.p2_bar;  // Then opCk = 0.0 + 0.5 - opPeak + 0.0*fMF = 0.0

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

  auto p_base = make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582,
                                 0.5, 0.0484, 0.001, 4);

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
    double relative_error = std::abs(out.opN - reference_opN) / std::abs(reference_opN);
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
      test_cases = {
          {-4.0, 0.01, make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582,
                                       0.5, 0.0484, 0.001, 4)},
          {-2.5, 0.005, make_test_params(0.2, 0.3, 0.5, 3300.0, 156000.0, 0.8582,
                                        0.5, 0.0484, 0.001, 4)},
          {-6.0, 0.001, make_test_params(0.2, 0.7, 1.5, 3300.0, 156000.0, 0.8582,
                                        0.5, 0.0484, 0.001, 4)}};

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

  auto p = make_test_params(0.2, 0.5, 1.0, 3300.0, 156000.0, 0.8582, 0.5,
                           0.0484, 0.001, 4);

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

  pfc::integrator::SpectralExpCoefficientCache cache;
  std::array<double, 1> L_arr{L};
  cache.ensure(L_arr, dt, pfc::integrator::SpectralExpOperatorId{.value = 1},
               pfc::integrator::SpectralExpDtId::from_bits(dt),
               tungsten::etd::k_tungsten_etd_config_id);

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
