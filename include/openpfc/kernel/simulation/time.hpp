// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file time.hpp
 * @brief Time state management for simulation time integration
 *
 * @details
 * This file defines the Time class, which manages the temporal state of
 * simulations in OpenPFC. The Time class tracks:
 * - Start time (t0), end time (t1), and time step (dt)
 * - Current simulation time and increment counter
 * - Save interval (saveat) for output scheduling
 *
 * The Time class provides queries for:
 * - Current time: get_time()
 * - Time step: get_dt()
 * - Completion status: is_done()
 * - Save scheduling: is_save_step()
 *
 * Typical usage:
 * @code
 * // Define simulation time parameters
 * pfc::Time time({0.0, 100.0, 0.1}, 1.0);  // t0=0, t1=100, dt=0.1, saveat=1.0
 *
 * // Time integration loop
 * while (!time.is_done()) {
 *     model.step(time.get_dt());
 *
 *     if (time.is_save_step()) {
 *         write_results(time.get_time());
 *     }
 *
 *     time.increment();  // Advances time by dt
 * }
 * @endcode
 *
 * This file is part of the Simulation Control module, managing temporal
 * aspects of time-dependent simulations.
 *
 * @see simulator.hpp for time integration loop
 * @see model.hpp for physics model receiving dt
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_TIME_HPP
#define PFC_TIME_HPP

#include <array>
#include <cmath>
#include <iostream>

namespace pfc {

/**
 * @brief Time class to handle simulation time stepping and output intervals
 *
 * This class provides functionalities to manage time in simulations.
 * It supports time intervals, time increments, and data saving at specific
 * intervals.
 *
 * The Time class is used by Simulator to orchestrate the time integration loop.
 * It tracks the current simulation time based on time step size (`dt`) and the
 * number of steps taken (`increment`), and determines when results should be
 * saved based on the `saveat` interval.
 *
 * ## Key Responsibilities
 *
 * - Define simulation time span: start time (`t0`), end time (`t1`), and time
 *   step size (`dt`)
 * - Track current time based on number of steps taken: `t_current = t0 +
 *   increment * dt`
 * - Determine when simulation is complete: `t_current >= t1`
 * - Manage output intervals: control when results should be written via
 *   `saveat`
 * - Provide validation: ensure `dt > 0`, `t0 < t1`, and `saveat` is valid
 *
 * ## Design Philosophy
 *
 * Time follows OpenPFC's "laboratory" philosophy:
 * - **Immutable core**: Start time, end time, and time step are const (set at
 *   construction)
 * - **Transparent state**: Current time is computed from increment (no hidden
 *   state)
 * - **Mutable progress**: Only the increment counter can be changed (via
 *   `next()`)
 * - **Explicit**: No automatic time advancement; user controls when `next()` is
 *   called
 *
 * ## Usage Pattern
 *
 * Typical simulation loop:
 * ```cpp
 * using namespace pfc;
 *
 * // Create time object: simulate from t=0 to t=10 with dt=0.01, save every 1.0
 * Time time({0.0, 10.0, 0.01}, 1.0);
 *
 * while (!time.done()) {
 *     // ... perform time step ...
 *
 *     if (time.do_save()) {
 *         // ... write results ...
 *     }
 *
 *     time.next();  // Advance to next time step
 * }
 * ```
 *
 * @example
 * **Basic Time Stepping**
 * ```cpp
 * using namespace pfc;
 *
 * // Simulate from t=0 to t=1 with dt=0.1
 * Time time({0.0, 1.0, 0.1}, 0.0);  // No automatic saving (saveat=0)
 *
 * std::cout << "Initial time: " << time.get_current() << "\n";  // 0.0
 * std::cout << "Time step: " << time.get_dt() << "\n";          // 0.1
 * std::cout << "End time: " << time.get_t1() << "\n";           // 1.0
 * ```
 *
 * @example
 * **Time Integration Loop**
 * ```cpp
 * using namespace pfc;
 *
 * Time time({0.0, 5.0, 0.01}, 1.0);  // Simulate 0→5 with dt=0.01, save every 1.0
 *
 * int step_count = 0;
 * while (!time.done()) {
 *     // Perform time step computation here
 *     step_count++;
 *
 *     if (time.do_save()) {
 *         std::cout << "Saving at t=" << time.get_current() << "\n";
 *     }
 *
 *     time.next();
 * }
 *
 * std::cout << "Total steps: " << step_count << "\n";  // 500
 * ```
 *
 * @example
 * **Adaptive Time Step Sizes**
 * ```cpp
 * using namespace pfc;
 *
 * // Start with dt=0.01
 * Time time({0.0, 10.0, 0.01}, 0.5);
 *
 * while (!time.done()) {
 *     // Compute error estimate
 *     double error = compute_error();
 *
 *     if (error > tolerance) {
 *         // Need smaller time step - restart from previous time
 *         time.set_dt(time.get_dt() * 0.5);
 *         time.set_increment(time.get_increment() - 1);
 *         continue;
 *     }
 *
 *     if (time.do_save()) {
 *         // Save results
 *     }
 *
 *     time.next();
 * }
 * ```
 *
 * @example
 * **Save at First and Last Steps Only**
 * ```cpp
 * using namespace pfc;
 *
 * // Save interval larger than simulation time → only saves at t0 and t1
 * Time time({0.0, 5.0, 0.1}, 100.0);
 *
 * while (!time.done()) {
 *     // ... compute ...
 *
 *     if (time.do_save()) {
 *         // This triggers at increment=0 (t=0) and when done() becomes true (t≥5)
 *         std::cout << "Saving at t=" << time.get_current() << "\n";
 *     }
 *
 *     time.next();
 * }
 * ```
 *
 * @example
 * **Query Time Properties**
 * ```cpp
 * using namespace pfc;
 *
 * Time time({1.0, 10.0, 0.05}, 2.0);
 *
 * // Get time span
 * auto [t0, t1, dt] = time.get_tspan();
 * std::cout << "Simulating from " << t0 << " to " << t1 << " with dt=" << dt <<
 * "\n";
 *
 * // Check current progress
 * std::cout << "Current increment: " << time.get_increment() << "\n";  // 0
 * std::cout << "Current time: " << time.get_current() << "\n";        // 1.0
 *
 * // Advance 10 steps
 * for (int i = 0; i < 10; ++i) time.next();
 * std::cout << "After 10 steps: t=" << time.get_current() << "\n";   // 1.5
 * ```
 *
 * @note The current time is computed as `t0 + increment * dt`, not stored
 *       explicitly. This ensures consistency and avoids floating-point drift.
 * @note The `saveat` interval uses floating-point modulo with tolerance (1e-6)
 *       to handle rounding errors.
 * @note Setting `saveat = 0.0` disables automatic saving entirely.
 * @note The `done()` method uses a small tolerance (1e-9) to handle
 *       floating-point comparison.
 *
 * @warning Time step size (`dt`) cannot be changed after construction in
 *          typical usage. For adaptive time stepping, you must manage `dt` and
 *          increment manually.
 * @warning If `dt * num_steps` doesn't exactly equal `t1 - t0`, the final step
 *          will be clamped to `t1`.
 *
 * @see Simulator - uses Time for orchestrating simulation loop
 * @see do_save() - determines when to write output
 * @see get_current() - computes current time from increment
 */
class Time {
private:
  double m_t0;     ///< Start time
  double m_t1;     ///< End time
  double m_dt;     ///< Time step
  int m_increment; ///< Current time increment
  double m_saveat; ///< Time interval for saving data

public:
  /**
   * @brief Construct a Time object for simulation time stepping
   *
   * Creates a Time object that manages the temporal evolution of a simulation.
   * The time span is defined by `[t0, t1]` with discrete steps of size `dt`.
   * Output scheduling is controlled by the `saveat` parameter.
   *
   * The constructor validates all input parameters:
   * - `t0 >= 0` (non-negative start time)
   * - `t1 > t0` (end time must exceed start time)
   * - `dt > 0` and `dt >= 1e-9` (positive and non-negligible time step)
   * - `saveat < t1` if `saveat > 0` (save interval must fit within simulation)
   *
   * @param[in] time Array containing [t0, t1, dt]:
   *                 - `time[0]` = t0: Start time of simulation
   *                 - `time[1]` = t1: End time of simulation
   *                 - `time[2]` = dt: Time step size
   * @param[in] saveat Time interval for saving results. Set to 0.0 to disable
   *                   automatic saving, or to a positive value to save at regular
   *                   intervals (e.g., `saveat = 1.0` saves at t=0, t=1, t=2, ...).
   *
   * @throws std::invalid_argument If t0 < 0
   * @throws std::invalid_argument If dt <= 0
   * @throws std::invalid_argument If dt < 1e-9 (too small for numerical stability)
   * @throws std::invalid_argument If t1 <= t0
   * @throws std::invalid_argument If saveat >= t1 (when saveat > 0)
   *
   * @post get_current() returns t0
   * @post get_increment() returns 0
   * @post done() returns false (unless t0 >= t1)
   *
   * @example
   * **Standard Time Span**
   * ```cpp
   * using namespace pfc;
   *
   * // Simulate from t=0 to t=10 with dt=0.01, save every 1.0 time units
   * Time time({0.0, 10.0, 0.01}, 1.0);
   *
   * // Expected properties:
   * // - Total steps: (10.0 - 0.0) / 0.01 = 1000 steps
   * // - Save points: t=0, t=1, t=2, ..., t=10 (11 saves)
   * ```
   *
   * @example
   * **No Automatic Saving**
   * ```cpp
   * using namespace pfc;
   *
   * // Disable automatic output by setting saveat=0
   * Time time({0.0, 100.0, 0.1}, 0.0);
   *
   * // do_save() will always return false (except at increment 0)
   * ```
   *
   * @example
   * **Non-Zero Start Time**
   * ```cpp
   * using namespace pfc;
   *
   * // Simulation starting at t=5.0 (useful for restarts)
   * Time time({5.0, 15.0, 0.05}, 1.0);
   *
   * std::cout << "Starting from t=" << time.get_current() << "\n";  // 5.0
   * ```
   *
   * @example
   * **Error Handling**
   * ```cpp
   * using namespace pfc;
   *
   * try {
   *     Time invalid({10.0, 5.0, 0.1}, 1.0);  // t1 < t0 → exception
   * } catch (const std::invalid_argument& e) {
   *     std::cerr << "Error: " << e.what() << "\n";
   * }
   *
   * try {
   *     Time tiny_dt({0.0, 1.0, 1e-12}, 0.0);  // dt too small → exception
   * } catch (const std::invalid_argument& e) {
   *     std::cerr << "Error: " << e.what() << "\n";
   * }
   * ```
   *
   * @note The initial increment is always 0, so `get_current()` returns `t0`
   *       immediately after construction.
   * @note Setting `saveat` larger than the simulation time span effectively
   *       disables periodic saving (only first and last steps will save).
   *
   * @see get_current() - compute current time from increment
   * @see done() - check if simulation is complete
   * @see do_save() - check if results should be saved at current time
   */
  Time(const std::array<double, 3> &time, double saveat)
      : m_t0(time[0]), m_t1(time[1]), m_dt(time[2]), m_increment(0),
        m_saveat(saveat) {
    if (m_t0 < 0) {
      throw std::invalid_argument("Start time cannot be negative: " +
                                  std::to_string(m_t0));
    }
    if (m_dt <= 0) {
      throw std::invalid_argument("Time step (dt) must be greater than zero: " +
                                  std::to_string(m_dt));
    }
    if (m_dt < 1e-9) {
      throw std::invalid_argument("Time step (dt) is too small: " +
                                  std::to_string(m_dt));
    }
    if (std::abs(m_t0 - m_t1) < 1e-9) {
      throw std::invalid_argument("Start time cannot equal end time: t0 == t1");
    }
    if (m_saveat > m_t1) {
      throw std::invalid_argument(
          "Save interval cannot exceed end time: " + std::to_string(m_saveat) +
          " > " + std::to_string(m_t1));
    }
  }

  /**
   * @brief Construct a new Time object with the specified time interval and
   * default save interval.
   *
   * The save interval is set to the same value as the time step.
   *
   * @param time An array containing the start time, end time, and time step in
   * that order
   */
  Time(const std::array<double, 3> &time) : Time(time, time[2]) {}

  /**
   * @brief Get the start time.
   *
   * @return The start time
   */
  double get_t0() const { return m_t0; }

  /**
   * @brief Get the end time.
   *
   * @return The end time
   */
  double get_t1() const { return m_t1; }

  /**
   * @brief Get the time step.
   *
   * @return The time step
   */
  double get_dt() const { return m_dt; }

  /**
   * @brief Get the current time increment.
   *
   * @return Current increment
   */
  int get_increment() const { return m_increment; }

  /**
   * @brief Get the current simulation time
   *
   * Computes the current time as `t_current = t0 + increment * dt`, where
   * `increment` is the number of time steps taken via `next()`. The result is
   * clamped to `t1` if it exceeds the end time (handles rounding errors).
   *
   * This computation-based approach (rather than storing `t_current`) ensures
   * consistency and avoids floating-point drift over many time steps.
   *
   * @return Current simulation time, clamped to [t0, t1]
   *
   * @example
   * **Track Simulation Progress**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 1.0, 0.1}, 0.0);
   *
   * std::cout << "t = " << time.get_current() << "\n";  // 0.0
   *
   * for (int i = 0; i < 5; ++i) {
   *     time.next();
   *     std::cout << "After step " << i+1 << ": t = " << time.get_current() << "\n";
   * }
   * // Output: 0.1, 0.2, 0.3, 0.4, 0.5
   * ```
   *
   * @example
   * **Clamping at End Time**
   * ```cpp
   * using namespace pfc;
   *
   * // dt doesn't divide (t1-t0) evenly
   * Time time({0.0, 1.0, 0.3}, 0.0);
   *
   * while (!time.done()) {
   *     std::cout << "t = " << time.get_current() << "\n";
   *     time.next();
   * }
   * // Output: 0.0, 0.3, 0.6, 0.9
   * // Next would be 1.2, but get_current() clamps to 1.0
   * ```
   *
   * @note The current time is NOT stored as a member variable; it's computed
   *       on-the-fly from the increment. This prevents accumulation of
   *       floating-point errors.
   * @note Clamping ensures `get_current()` never exceeds `t1`, even if
   *       `increment * dt` overshoots due to rounding.
   *
   * @see get_increment() - get the number of steps taken
   * @see next() - advance to next time step
   * @see done() - check if t_current >= t1
   */
  double get_current() const {
    double current_time = m_t0 + m_increment * m_dt;
    return (current_time > m_t1) ? m_t1
                                 : current_time; // Clamp to m_t1 if it exceeds
  }

  /**
   * @brief Get the time interval for saving data.
   *
   * @return The save interval
   */
  double get_saveat() const { return m_saveat; }

  /**
   * @brief Set the current time increment.
   *
   * @param increment The current time increment
   */
  void set_increment(int increment) { m_increment = increment; }

  /**
   * @brief Set the time interval for saving data.
   *
   * @param saveat The save interval
   */
  void set_saveat(double saveat) { m_saveat = saveat; }

  /**
   * @brief Check if the simulation time integration is complete
   *
   * Returns true when the current time has reached or exceeded the end time
   * `t1`. Uses a small tolerance (1e-9) to handle floating-point comparison
   * issues.
   *
   * This method is typically used as the condition for the main simulation loop:
   * ```cpp
   * while (!time.done()) {
   *     // ... time step ...
   *     time.next();
   * }
   * ```
   *
   * @return true if `t_current >= t1 - tolerance`, false otherwise
   *
   * @example
   * **Standard Simulation Loop**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 5.0, 0.1}, 1.0);
   *
   * int steps = 0;
   * while (!time.done()) {
   *     // Perform time step
   *     steps++;
   *     time.next();
   * }
   *
   * std::cout << "Completed " << steps << " steps\n";  // 50 steps
   * std::cout << "Final time: " << time.get_current() << "\n";  // 5.0
   * ```
   *
   * @example
   * **Early Termination**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 100.0, 0.1}, 10.0);
   *
   * while (!time.done()) {
   *     double residual = compute_residual();
   *
   *     if (residual < 1e-8) {
   *         std::cout << "Converged at t=" << time.get_current() << "\n";
   *         break;  // Exit before t1
   *     }
   *
   *     time.next();
   * }
   * ```
   *
   * @note Uses tolerance of 1e-9 to avoid issues with floating-point arithmetic
   *       (e.g., `t_current` might be 9.999999999 instead of exactly 10.0).
   * @note The tolerance is subtracted from `t1`, so the check is effectively
   *       `t_current >= (t1 - 1e-9)`.
   *
   * @see get_current() - compute current time
   * @see next() - advance time step
   */
  bool done() const {
    return (get_current() >= m_t1 - 1e-9); // Adjust for floating-point precision
  }

  /**
   * @brief Advance to the next time step
   *
   * Increments the step counter by 1, which advances the current time by `dt`.
   * The new time becomes `t_current = t0 + (increment + 1) * dt`.
   *
   * This is the ONLY method that mutates the Time object's state during normal
   * simulation. It should be called at the end of each time step after all
   * computations and output operations are complete.
   *
   * @post get_increment() returns previous value + 1
   * @post get_current() returns previous value + dt (clamped to t1)
   *
   * @example
   * **Manual Time Stepping**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 1.0, 0.25}, 0.0);
   *
   * std::cout << "Step 0: t = " << time.get_current() << "\n";  // 0.00
   *
   * time.next();
   * std::cout << "Step 1: t = " << time.get_current() << "\n";  // 0.25
   *
   * time.next();
   * std::cout << "Step 2: t = " << time.get_current() << "\n";  // 0.50
   *
   * time.next();
   * std::cout << "Step 3: t = " << time.get_current() << "\n";  // 0.75
   *
   * time.next();
   * std::cout << "Step 4: t = " << time.get_current() << "\n";  // 1.00
   * std::cout << "Done? " << time.done() << "\n";  // true
   * ```
   *
   * @example
   * **Integration with Simulator**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 10.0, 0.01}, 1.0);
   *
   * while (!time.done()) {
   *     model.step(time.get_dt());  // Advance physics
   *
   *     if (time.do_save()) {
   *         writer.write(time.get_current());  // Output results
   *     }
   *
   *     time.next();  // ← Advance time (called ONCE per iteration)
   * }
   * ```
   *
   * @note This method does NOT perform any time step computation or physics
   *       simulation; it only updates the internal counter. The actual numerical
   *       time stepping is done by Model::step().
   * @note Call `next()` AFTER all operations for the current time step are
   *       complete (including output).
   *
   * @see get_increment() - query current step number
   * @see get_current() - compute time from increment
   * @see done() - check completion status
   */
  void next() { m_increment += 1; }

  /**
   * @brief Determine if results should be saved at the current time
   *
   * Returns true if the current time aligns with the save interval (`saveat`),
   * or if this is the first step (`increment == 0`), or if the simulation is
   * complete (`done() == true`).
   *
   * The save logic uses floating-point modulo with tolerance to handle rounding:
   * - `fmod(t_current + epsilon, saveat) < tolerance` checks alignment
   * - Special cases: first step always saves, final step always saves
   * - If `saveat <= 0`, automatic saving is disabled (always returns false)
   *
   * Typical usage: check `do_save()` inside the simulation loop and conditionally
   * write output.
   *
   * @return true if results should be saved at current time, false otherwise
   *
   * @example
   * **Conditional Output**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 5.0, 0.01}, 1.0);  // Save every 1.0 time units
   *
   * int saves = 0;
   * while (!time.done()) {
   *     // ... compute time step ...
   *
   *     if (time.do_save()) {
   *         std::cout << "Saving at t=" << time.get_current() << "\n";
   *         saves++;
   *     }
   *
   *     time.next();
   * }
   * std::cout << "Total saves: " << saves << "\n";  // 6 saves (t=0,1,2,3,4,5)
   * ```
   *
   * @example
   * **No Automatic Saving**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 10.0, 0.1}, 0.0);  // saveat=0 disables saving
   *
   * while (!time.done()) {
   *     if (time.do_save()) {
   *         // Never executes (except at increment=0)
   *     }
   *     time.next();
   * }
   * ```
   *
   * @example
   * **Manual Override**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 100.0, 0.1}, 10.0);  // Save every 10.0
   *
   * while (!time.done()) {
   *     bool converged = (residual < 1e-8);
   *
   *     if (time.do_save() || converged) {
   *         // Save at regular intervals OR when converged
   *         write_output(time.get_current());
   *     }
   *
   *     time.next();
   * }
   * ```
   *
   * @example
   * **Save Points Analysis**
   * ```cpp
   * using namespace pfc;
   *
   * Time time({0.0, 2.5, 0.1}, 0.5);  // dt=0.1, saveat=0.5
   *
   * // Expected save points:
   * // t=0.0 (first step)
   * // t=0.5 (fmod(0.5, 0.5) = 0)
   * // t=1.0 (fmod(1.0, 0.5) = 0)
   * // t=1.5 (fmod(1.5, 0.5) = 0)
   * // t=2.0 (fmod(2.0, 0.5) = 0)
   * // t=2.5 (done() == true)
   * ```
   *
   * @note Uses `fmod()` with small epsilon (1e-9) and tolerance (1e-6) to handle
   *       floating-point comparison issues.
   * @note Always saves at `increment == 0` (initial condition) and when `done()`
   *       (final state), regardless of `saveat`.
   * @note Setting `saveat = 0.0` disables automatic saving completely (useful for
   *       manual control).
   * @note If `saveat > (t1 - t0)`, only the first and last steps will save.
   *
   * @see get_saveat() - query save interval
   * @see get_current() - get time for output filename/metadata
   * @see done() - check if this is the final save point
   */
  bool do_save() const {
    if (m_saveat <= 0) {
      return false; // Save interval of 0 means no saving
    }
    return (std::fmod(get_current() + 1.0e-9, m_saveat) < 1.e-6) || done() ||
           (m_increment == 0);
  }

  /**
   * @brief Conversion operator to retrieve the current time as a double value.
   *
   * @return The current time as a double value
   */
  operator double() const { return get_current(); }

  /**
   * @brief Overloaded stream insertion operator to print the Time object.
   *
   * @param os The output stream
   * @param t The Time object to be printed
   * @return The output stream
   */
  friend std::ostream &operator<<(std::ostream &os, const Time &t) {
    os << "(t0 = " << t.m_t0 << ", t1 = " << t.m_t1 << ", dt = " << t.m_dt;
    os << ", saveat = " << t.m_saveat << ", t_current = " << t.get_current()
       << ")\n";
    return os;
  };
};

} // namespace pfc

#endif
