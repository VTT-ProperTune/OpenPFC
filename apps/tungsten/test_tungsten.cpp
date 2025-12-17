// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include "tungsten.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <iomanip>
#include <mpi.h>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/openpfc.hpp>

using namespace pfc;
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

    MPI_Worker worker(0, nullptr);
    auto world = world::create({32, 32, 32});
    Tungsten tungsten(world);
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

    MPI_Worker worker(0, nullptr);
    auto world = world::create({32, 32, 32});
    Tungsten tungsten(world);

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

    MPI_Worker worker(0, nullptr);
    auto world = world::create({32, 32, 32});
    Tungsten tungsten(world);

    REQUIRE_THROWS_AS(from_json(j, tungsten), std::invalid_argument);
  }
}

TEST_CASE("Tungsten parameter setters", "[Tungsten][Setters]") {
  MPI_Worker worker(0, nullptr);
  auto world = world::create({32, 32, 32});
  Tungsten tungsten(world);

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
    MPI_Worker worker(0, nullptr);
    // Create world with exact parameters from tungsten_single_seed.json
    // Grid: 32x32x32, spacing: 1.1107207345395915, origin: center
    // When origo="center", origin is at -0.5 * dx * Lx
    double grid_spacing = 1.1107207345395915;
    int Lx = 32, Ly = 32, Lz = 32;
    double x0 = -0.5 * grid_spacing * Lx;
    double y0 = -0.5 * grid_spacing * Ly;
    double z0 = -0.5 * grid_spacing * Lz;
    auto world =
        world::create(GridSize({Lx, Ly, Lz}), PhysicalOrigin({x0, y0, z0}),
                      GridSpacing({grid_spacing, grid_spacing, grid_spacing}));
    auto decomp = decomposition::create(world, 1);
    auto fft = fft::create(decomp);

    Tungsten tungsten(world);
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
    tungsten.set_fft(fft);
    double dt = 1.0;
    tungsten.initialize(dt);

    // Manually replicate the initial condition logic from the UI
    // This matches exactly what happens when initial conditions are applied
    std::vector<double> &psi = tungsten.get_real_field("psi");
    const World &w = tungsten.get_world();
    const FFT &fft_ref = tungsten.get_fft();

    // 1. Constant initial condition: fill entire field with -0.4
    std::fill(psi.begin(), psi.end(), -0.4);

    // 2. Single seed initial condition: apply seed formula to points within
    // radius 64.0 Replicate the exact logic from SingleSeed::apply()
    types::Int3 low = fft::get_inbox(fft_ref).low;
    types::Int3 high = fft::get_inbox(fft_ref).high;

    auto spacing = world::get_spacing(w);
    auto origin = world::get_origin(w);
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
              << initial_norm << std::endl;
    std::cout << "Initial field range: [" << min_val << ", " << max_val << "]"
              << std::endl;
    std::cout << "Expected: constant -0.4 everywhere, then seed applied in center"
              << std::endl;

    // Run 10 time steps and verify norms match expected values exactly
    // Expected norms are from actual simulation run with these exact parameters
    std::vector<double> actual_norms;
    for (int i = 0; i < 10; ++i) {
      tungsten.step(1.0);
      double norm2 = 0.0;
      for (auto &x : psi) norm2 += x * x;
      actual_norms.push_back(norm2);
      std::cout << "Step " << (i + 1) << " norm: " << std::fixed
                << std::setprecision(10) << norm2 << std::endl;
    }

    // Print summary
    std::cout << "\nNorm changes:" << std::endl;
    for (size_t i = 0; i < actual_norms.size(); ++i) {
      if (i == 0) {
        double change = actual_norms[i] - initial_norm;
        std::cout << "  Step " << (i + 1) << ": " << change << " (from initial)"
                  << std::endl;
      } else {
        double change = actual_norms[i] - actual_norms[i - 1];
        std::cout << "  Step " << (i + 1) << ": " << change << " (from step " << i
                  << ")" << std::endl;
      }
    }

    // Verify initial norm matches expected value
    REQUIRE_THAT(initial_norm, WithinAbs(expected_norms[0], 0.1));

    // Verify norms match expected values (tight tolerance for regression testing)
    REQUIRE(actual_norms.size() == 10);
    for (int i = 0; i < 10; ++i) {
      REQUIRE_THAT(actual_norms[i], WithinAbs(expected_norms[i + 1], 0.1));
    }
  }

  SECTION("Model initialization and allocation") {
    MPI_Worker worker(0, nullptr);
    auto world = world::create({32, 32, 32});
    auto decomp = decomposition::create(world, 1);
    auto fft = fft::create(decomp);

    Tungsten tungsten(world);
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
    tungsten.set_fft(fft);

    double dt = 1.0;
    tungsten.initialize(dt);

    // Check that fields are allocated
    REQUIRE(tungsten.has_real_field("psi"));
    REQUIRE(tungsten.has_real_field("psiMF"));
    REQUIRE(tungsten.has_real_field("default")); // backward compatibility

    std::vector<double> &psi = tungsten.get_real_field("psi");
    REQUIRE(psi.size() > 0);
  }
}

int main(int argc, char *argv[]) {
  // Initialize MPI once for all tests
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI initialization failed" << std::endl;
    return 1;
  }

  // Run Catch2 tests
  int result = Catch::Session().run(argc, argv);

  // Finalize MPI
  MPI_Finalize();
  return result;
}
