// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 04_time_stepping.cpp
 * @brief Comprehensive demonstration of OpenPFC Time API
 *
 * This example showcases all aspects of the Time class:
 * - Basic time span setup and queries
 * - Time integration loops
 * - Save interval control
 * - Adaptive time stepping patterns
 * - Edge cases and error handling
 *
 * The Time class manages temporal evolution of simulations, tracking current
 * time, determining when to save output, and providing loop control.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <openpfc/time.hpp>
#include <vector>

using namespace pfc;

// ============================================================================
// Scenario 1: Basic Time Stepping
// ============================================================================

void scenario_basic_time_stepping() {
  std::cout << "\n=== Scenario 1: Basic Time Stepping ===\n\n";

  // Create time object: simulate from t=0 to t=1 with dt=0.1
  // saveat=0.0 disables automatic saving
  Time time({0.0, 1.0, 0.1}, 0.0);

  std::cout << "Time Configuration:\n";
  std::cout << "  Start time (t0):    " << time.get_t0() << "\n";
  std::cout << "  End time (t1):      " << time.get_t1() << "\n";
  std::cout << "  Time step (dt):     " << time.get_dt() << "\n";
  std::cout << "  Save interval:      " << time.get_saveat() << " (disabled)\n";
  std::cout << "  Current time:       " << time.get_current() << "\n";
  std::cout << "  Current increment:  " << time.get_increment() << "\n\n";

  // Manual stepping
  std::cout << "Manual Time Stepping:\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "  Step " << i << ": t = " << std::fixed << std::setprecision(2)
              << time.get_current() << ", increment = " << time.get_increment()
              << "\n";
    time.next();
  }

  std::cout << "  Step 5: t = " << std::fixed << std::setprecision(2)
            << time.get_current() << ", increment = " << time.get_increment()
            << "\n";
  std::cout << "  Is done? " << (time.done() ? "yes" : "no") << "\n";
}

// ============================================================================
// Scenario 2: Time Integration Loop
// ============================================================================

void scenario_integration_loop() {
  std::cout << "\n=== Scenario 2: Time Integration Loop ===\n\n";

  // Simulate from t=0 to t=5 with dt=0.01, save every 1.0 time units
  Time time({0.0, 5.0, 0.01}, 1.0);

  std::cout << "Running simulation from t=" << time.get_t0()
            << " to t=" << time.get_t1() << " with dt=" << time.get_dt() << "\n";
  std::cout << "Saving every " << time.get_saveat() << " time units\n\n";

  int step_count = 0;
  int save_count = 0;

  while (!time.done()) {
    // Simulate time step (placeholder - would call model.step() here)
    step_count++;

    // Check if we should save
    if (time.do_save()) {
      std::cout << "  [SAVE " << save_count++ << "] t = " << std::fixed
                << std::setprecision(2) << time.get_current() << " (step "
                << step_count << ")\n";
    }

    // Advance to next time step
    time.next();
  }

  std::cout << "\nSimulation Complete!\n";
  std::cout << "  Total steps: " << step_count << "\n";
  std::cout << "  Total saves: " << save_count << "\n";
  std::cout << "  Final time:  " << std::fixed << std::setprecision(2)
            << time.get_current() << "\n";
}

// ============================================================================
// Scenario 3: Save Interval Control
// ============================================================================

void scenario_save_intervals() {
  std::cout << "\n=== Scenario 3: Save Interval Control ===\n\n";

  struct Config {
    double t1;
    double dt;
    double saveat;
    const char *description;
  };

  std::vector<Config> configs = {
      {1.0, 0.1, 0.0, "No saving (saveat=0)"},
      {1.0, 0.1, 0.2, "Save every 0.2 time units"},
      {1.0, 0.1, 0.5, "Save every 0.5 time units"},
      {1.0, 0.1, 100.0, "Save only first/last (saveat > t1)"}};

  for (const auto &config : configs) {
    Time time({0.0, config.t1, config.dt}, config.saveat);

    std::cout << config.description << ":\n  Saves at: ";

    int saves = 0;
    while (!time.done()) {
      if (time.do_save()) {
        if (saves > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(1) << time.get_current();
        saves++;
      }
      time.next();
    }

    std::cout << " (" << saves << " total)\n";
  }
}

// ============================================================================
// Scenario 4: Non-Uniform Time Steps (Clamping Demo)
// ============================================================================

void scenario_clamping() {
  std::cout << "\n=== Scenario 4: Time Step Clamping ===\n\n";

  // dt=0.3 doesn't divide evenly into (t1-t0)=1.0
  Time time({0.0, 1.0, 0.3}, 0.0);

  std::cout << "Time span [0, 1] with dt=0.3 (doesn't divide evenly)\n";
  std::cout << "Time values:\n";

  while (!time.done()) {
    std::cout << "  t = " << std::fixed << std::setprecision(2) << time.get_current()
              << " (increment=" << time.get_increment() << ")\n";
    time.next();
  }

  std::cout << "  t = " << std::fixed << std::setprecision(2) << time.get_current()
            << " (increment=" << time.get_increment() << ")\n";
  std::cout
      << "\nNote: Final time is clamped to t1=1.0 (would be 1.2 without clamping)\n";
}

// ============================================================================
// Scenario 5: Adaptive Time Stepping Pattern
// ============================================================================

void scenario_adaptive_timestepping() {
  std::cout << "\n=== Scenario 5: Adaptive Time Stepping Pattern ===\n\n";

  Time time({0.0, 2.0, 0.1}, 0.5);

  std::cout << "Simulating with adaptive time stepping\n";
  std::cout << "Initial dt = " << time.get_dt() << "\n\n";

  // Simulate error-based adaptive time stepping
  // (This is a simplified demonstration - real adaptive stepping is more complex)
  int step = 0;
  while (!time.done() && step < 30) { // Limit iterations for demo
    double t = time.get_current();

    // Simulate error estimate (artificial for demo)
    double error = 0.01 * std::sin(10.0 * t) + 0.005;

    std::cout << "  Step " << step << ": t=" << std::fixed << std::setprecision(3)
              << t << ", error=" << std::scientific << std::setprecision(2) << error;

    // Check if error is acceptable
    if (error > 0.01) {
      std::cout << " → REJECT (reduce dt)\n";
      // In real code: reduce dt, rewind increment, retry step
      // For demo: just continue
    } else {
      std::cout << " → ACCEPT\n";
    }

    if (time.do_save()) {
      std::cout << "    [SAVE at t=" << std::fixed << std::setprecision(2)
                << time.get_current() << "]\n";
    }

    time.next();
    step++;
  }

  std::cout << "\nNote: Real adaptive time stepping requires manual dt management\n";
  std::cout << "      Use set_dt() and set_increment() to adjust step size\n";
}

// ============================================================================
// Scenario 6: Non-Zero Start Time (Restart Pattern)
// ============================================================================

void scenario_restart() {
  std::cout << "\n=== Scenario 6: Non-Zero Start Time (Restart) ===\n\n";

  // Simulate restarting from a checkpoint at t=5.0
  Time time({5.0, 15.0, 0.05}, 2.0);

  std::cout << "Restarting simulation from checkpoint\n";
  std::cout << "  Restart time (t0): " << time.get_t0() << "\n";
  std::cout << "  End time (t1):     " << time.get_t1() << "\n";
  std::cout << "  Time step (dt):    " << time.get_dt() << "\n";
  std::cout << "  Save interval:     " << time.get_saveat() << "\n\n";

  std::cout << "Resuming time integration:\n";

  int steps = 0;
  while (!time.done() && steps < 10) { // Show first 10 steps
    if (time.do_save()) {
      std::cout << "  [SAVE] t = " << std::fixed << std::setprecision(2)
                << time.get_current() << "\n";
    }
    time.next();
    steps++;
  }

  std::cout << "  ... (continuing to t=" << time.get_t1() << ")\n";
}

// ============================================================================
// Scenario 7: Error Handling and Validation
// ============================================================================

void scenario_error_handling() {
  std::cout << "\n=== Scenario 7: Error Handling ===\n\n";

  struct TestCase {
    std::array<double, 3> time_params;
    double saveat;
    const char *description;
  };

  std::vector<TestCase> invalid_cases = {
      {{-1.0, 10.0, 0.1}, 1.0, "Negative start time"},
      {{0.0, 10.0, 0.0}, 1.0, "Zero time step"},
      {{0.0, 10.0, -0.1}, 1.0, "Negative time step"},
      {{10.0, 5.0, 0.1}, 1.0, "End time before start time"},
      {{0.0, 10.0, 1e-12}, 1.0, "Time step too small"},
      {{0.0, 5.0, 0.1}, 10.0, "Save interval exceeds simulation time"}};

  for (const auto &test : invalid_cases) {
    std::cout << "Testing: " << test.description << "\n";
    try {
      Time time(test.time_params, test.saveat);
      std::cout << "  ✗ No exception thrown (unexpected!)\n";
    } catch (const std::invalid_argument &e) {
      std::cout << "  ✓ Caught exception: " << e.what() << "\n";
    }
  }

  std::cout << "\nValid configurations:\n";
  try {
    Time time1({0.0, 10.0, 0.1}, 1.0);
    std::cout << "  ✓ Standard time span [0, 10] with dt=0.1, saveat=1.0\n";

    Time time2({5.0, 15.0, 0.05}, 0.0);
    std::cout << "  ✓ Non-zero start time=5, no automatic saving\n";

    Time time3({0.0, 1.0, 1e-8}, 0.0);
    std::cout << "  ✓ Very small (but valid) time step dt=1e-8\n";
  } catch (const std::invalid_argument &e) {
    std::cout << "  ✗ Unexpected exception: " << e.what() << "\n";
  }
}

// ============================================================================
// Scenario 8: Integration with Physics Models
// ============================================================================

void scenario_physics_integration() {
  std::cout << "\n=== Scenario 8: Integration with Physics Models ===\n\n";

  // Simple exponential decay model: du/dt = -k*u
  // Analytical solution: u(t) = u0 * exp(-k*t)

  const double k = 0.5; // Decay constant
  double u = 1.0;       // Initial value
  const double u_analytical_final = std::exp(-k * 5.0);

  Time time({0.0, 5.0, 0.1}, 1.0);

  std::cout << "Simulating exponential decay: du/dt = -k*u, k=" << k << "\n";
  std::cout << "Using forward Euler time integration\n\n";

  while (!time.done()) {
    // Forward Euler: u_new = u_old + dt * f(u_old)
    double dt = time.get_dt();
    u = u + dt * (-k * u);

    if (time.do_save()) {
      double t = time.get_current();
      double u_exact = std::exp(-k * t);
      double error = std::abs(u - u_exact);

      std::cout << "  t=" << std::fixed << std::setprecision(2) << t
                << ": u_numerical=" << std::setprecision(6) << u
                << ", u_exact=" << u_exact << ", error=" << std::scientific
                << std::setprecision(2) << error << "\n";
    }

    time.next();
  }

  std::cout << "\nFinal Results:\n";
  std::cout << "  Numerical: " << std::fixed << std::setprecision(6) << u << "\n";
  std::cout << "  Analytical: " << u_analytical_final << "\n";
  std::cout << "  Error: " << std::scientific << std::setprecision(2)
            << std::abs(u - u_analytical_final) << "\n";
}

// ============================================================================
// Main: Run All Scenarios
// ============================================================================

int main() {
  std::cout << "╔═══════════════════════════════════════════════════════╗\n";
  std::cout << "║  OpenPFC Time API - Comprehensive Demonstration      ║\n";
  std::cout << "╚═══════════════════════════════════════════════════════╝\n";

  try {
    scenario_basic_time_stepping();
    scenario_integration_loop();
    scenario_save_intervals();
    scenario_clamping();
    scenario_adaptive_timestepping();
    scenario_restart();
    scenario_error_handling();
    scenario_physics_integration();

    std::cout << "\n╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║  All scenarios completed successfully!               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n❌ Error: " << e.what() << "\n";
    return 1;
  }
}
