// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pfc {

/**
 * @brief Time class to handle simulation time stepping and output intervals
 *
 * This class provides functionalities to manage time in simulations.
 * It supports time intervals, time increments, and data saving at specific
 * intervals.
 *
 * The Time class is used by Simulator to orchestrate the time integration loop.
 * It stores an explicit **accepted** simulation time (`m_accepted_time`) that
 * advances only on `next()` or `commit_attempt()`, and determines when results
 * should be saved based on the `saveat` interval keyed off accepted time.
 *
 * ## Key Responsibilities
 *
 * - Define simulation time span: start time (`t0`), end time (`t1`), and time
 *   step size (`dt`)
 * - Store accepted simulation time independently of the next candidate `dt`
 *   (so `set_dt` cannot rewrite past accepted time)
 * - Determine when simulation is complete: `accepted_time >= t1`
 * - Manage output intervals: control when results should be written via
 *   `saveat` (queries use accepted time only)
 * - Driver-owned attempt transactions: `clip_attempt_dt` / `begin_attempt` /
 *   `commit_attempt` / `reject_attempt` for adaptive control
 * - Track accepted/rejected adaptive step attempts via
 *   `increment_step_success()` / `increment_step_rejection()` (separate from
 *   `get_step_count()`, which only counts committed advances via `next()` /
 *   `commit_attempt`)
 * - Provide validation: ensure `dt > 0`, `t0 < t1`, and `saveat` is valid
 *
 * ## Design Philosophy
 *
 * Time follows OpenPFC's "laboratory" philosophy:
 * - **Immutable core**: Start time and end time are fixed at construction
 * - **Accepted clock**: Simulation time is stored as `m_accepted_time` and is
 *   unchanged for the duration of an active attempt
 * - **Mutable progress**: Accepted time advances via `next()` (fixed `dt`) or
 *   `commit_attempt()` (clipped attempted interval); `set_dt` only updates the
 *   recommended step size
 * - **Explicit**: No automatic time advancement; user controls when `next()` /
 *   commit is called
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
 * **Adaptive attempt transaction (preferred)**
 * ```cpp
 * using namespace pfc;
 *
 * Time time({0.0, 10.0, 0.01}, 0.5);
 * double candidate_dt = time.get_dt();
 *
 * while (!time.done()) {
 *     time.begin_attempt(candidate_dt);  // clips vs t1 and saveat
 *     const double attempted = time.get_attempted_dt();
 *     // Integrate with attempted (not get_dt()); accepted time stays fixed
 *     double error = attempt_step(time.get_accepted_time(), attempted);
 *
 *     if (error > tolerance) {
 *         time.reject_attempt();  // accepted time unchanged
 *         candidate_dt = attempted * 0.5;
 *         time.set_dt(candidate_dt);  // policy only — does not rewrite clock
 *         continue;
 *     }
 *
 *     time.commit_attempt();  // accepted += attempted
 *     time.increment_step_success();
 *     if (time.do_save()) {
 *         write_results(time.get_accepted_time());
 *     }
 *     candidate_dt = recommend_next_dt(...);
 *     time.set_dt(candidate_dt);
 * }
 * ```
 *
 * @warning Do not adapt by set_dt plus decrementing set_increment: that
 *          fixed-dt reconstruction helper rewrites accepted time as
 *          t0 + n * dt and is not the adaptive commit path.
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
 * @note Accepted time is stored explicitly (`m_accepted_time`) and advances on
 *       `next()` / `commit_attempt()`. `set_dt` does not rewrite accepted time.
 * @note The `saveat` interval uses floating-point modulo with tolerance (1e-6)
 *       to handle rounding errors.
 * @note Setting `saveat = 0.0` disables automatic saving entirely.
 * @note The `done()` method uses a small tolerance (1e-9) to handle
 *       floating-point comparison.
 *
 * @warning Prefer `begin_attempt` / `commit_attempt` / `reject_attempt` for
 *          adaptive control. `set_increment` is a fixed-`dt` reconstruction
 *          helper and will set accepted time to `min(t0 + n * dt, t1)`.
 * @warning If `dt * num_steps` doesn't exactly equal `t1 - t0`, the final step
 *          will be clamped to `t1`.
 *
 * @see Simulator - uses Time for orchestrating simulation loop
 * @see do_save() - determines when to write output
 * @see get_current() - returns accepted simulation time
 * @see clip_attempt_dt() - clip candidate dt before an attempt
 */

enum class IntegratorMethod { euler, rk2_heun };

class Time {
private:
  double m_t0;     ///< Start time
  double m_t1;     ///< End time
  double m_dt;     ///< Time step (policy / recommended next step)
  int m_increment; ///< Current time increment (committed step count)
  double m_accepted_time; ///< Accepted simulation time (immutable during attempt)
  double m_saveat; ///< Time interval for saving data
  IntegratorMethod m_method{IntegratorMethod::euler};
  int m_stage{0};       ///< Current stage index within this time step (0-based)
  int m_stage_count{1}; ///< Total number of stages in the current time step
  int accepted_steps_{0}; ///< Accepted adaptive step attempts
  int rejected_steps_{0}; ///< Rejected adaptive step attempts
  bool m_attempt_active{false}; ///< True while begin_attempt is open
  double m_attempted_dt{0.0};   ///< Clipped interval for the active attempt

  friend class TimeStateGuard;

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
  Time(const std::array<double, 3> &time, double saveat, IntegratorMethod method)
      : m_t0(time[0]), m_t1(time[1]), m_dt(time[2]), m_increment(0),
        m_accepted_time(time[0]), m_saveat(saveat), m_method(method) {
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
    if (!(m_t1 > m_t0)) {
      throw std::invalid_argument(
          "End time must be greater than start time (t1 > t0). Got t0=" +
          std::to_string(m_t0) + ", t1=" + std::to_string(m_t1));
    }
    if ((m_t1 - m_t0) < 1e-9) {
      throw std::invalid_argument(
          "Time span (t1 - t0) is too small; must be at least 1e-9. Got span " +
          std::to_string(m_t1 - m_t0));
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
  Time(const std::array<double, 3> &time)
      : Time(time, time[2], IntegratorMethod::euler) {}

  /**
   * @brief Construct a new Time object with the specified time interval and
   * default save interval (euler integrator).
   *
   * The save interval is set to the same value as the time step.
   *
   * @param time An array containing the start time, end time, and time step in
   * that order
   */
  Time(const std::array<double, 3> &time, double saveat)
      : Time(time, saveat, IntegratorMethod::euler) {}

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
   * @brief Set the time step.
   *
   * Sets a new time step size for adaptive time-stepping. The time step must
   * be positive to ensure meaningful time integration.
   *
   * This method enables runtime adjustment of the recommended / policy `dt`.
   * It does **not** change accepted simulation time; adaptive drivers must
   * advance time via @ref commit_attempt (or fixed-step @ref next).
   *
   * @param[in] dt The new time step size (must be > 0)
   * @throws std::invalid_argument If dt <= 0
   *
   * @post get_dt() returns the new dt value
   * @post get_accepted_time() is unchanged
   *
   * @note Changing dt does not rewrite past accepted time. Use
   *       @ref begin_attempt / @ref commit_attempt for clipped attempts.
   *
   * @see get_dt() - query the current time step
   * @see set_increment() - fixed-dt reconstruction helper (not adaptive commit)
   */
  void set_dt(double dt) {
    if (dt <= 0.0) {
      throw std::invalid_argument("Time step must be positive: " +
                                  std::to_string(dt));
    }
    m_dt = dt;
  }

  /**
   * @brief Get the current time increment.
   *
   * @return Current increment
   */
  int get_increment() const { return m_increment; }

  /**
   * @brief Get the number of steps completed so far.
   *
   * Self-documenting alternative to @ref get_increment() for integrator
   * authors querying progress (checkpoint/restart, adaptive step rejection
   * logic, output scheduling): "increment" reads as a small delta, not a
   * count, whereas "step count" states the intent directly. Returns the
   * same underlying value as get_increment().
   *
   * @return Current step count
   */
  int get_step_count() const { return m_increment; }

  /**
   * @brief Record a successful adaptive step attempt.
   *
   * Increments the accepted-attempt counter. Unlike @ref next(), this does
   * not advance simulation time; it only updates attempt statistics for
   * adaptive time-stepping algorithms.
   *
   * @note When used with @ref TimeStateGuard, call this before @c commit()
   *       (or outside an uncommitted guard); otherwise the guard restores
   *       the counter on destruction.
   *
   * @post get_accepted_steps() returns previous value + 1
   */
  void increment_step_success() { accepted_steps_++; }

  /**
   * @brief Record a rejected adaptive step attempt.
   *
   * Increments the rejected-attempt counter. Unlike @ref next(), this does
   * not advance simulation time; it only updates attempt statistics.
   *
   * @note @ref TimeStateGuard also restores accepted/rejected counters unless
   *       @c commit() was called. Call this after an uncommitted guard's
   *       scope ends (or outside the guard) so the rejection count is kept.
   *
   * @post get_rejected_steps() returns previous value + 1
   */
  void increment_step_rejection() { rejected_steps_++; }

  /**
   * @brief Get the number of accepted adaptive step attempts.
   *
   * @return Accepted step count (starts at 0)
   */
  int get_accepted_steps() const { return accepted_steps_; }

  /**
   * @brief Get the number of rejected adaptive step attempts.
   *
   * @return Rejected step count (starts at 0)
   */
  int get_rejected_steps() const { return rejected_steps_; }

  /**
   * @brief Get the current stage index within this time step.
   *
   * Stage tracking is independent of @ref get_increment(): stages range from
   * `0` to `get_stage_count() - 1` and describe progress within a single
   * multi-stage step (e.g. RK2/RK4), while the increment counts completed
   * time steps.
   *
   * @return Current stage index (0-based)
   */
  int get_stage() const { return m_stage; }

  /**
   * @brief Set the current stage index within this time step.
   *
   * Multi-stage steppers should call this before computing each stage.
   *
   * @param[in] stage New stage index (must satisfy `0 <= stage < get_stage_count()`)
   * @throws std::invalid_argument If stage is out of range
   */
  void set_stage(int stage) {
    if (stage < 0 || stage >= m_stage_count) {
      throw std::invalid_argument(
          "Stage must satisfy 0 <= stage < stage_count (stage_count=" +
          std::to_string(m_stage_count) + "): " + std::to_string(stage));
    }
    m_stage = stage;
  }

  /**
   * @brief Get the total number of stages in the current time step.
   *
   * @return Stage count (always >= 1)
   */
  int get_stage_count() const { return m_stage_count; }

  /**
   * @brief Set the total number of stages for this time step.
   *
   * Multi-stage steppers should call this once during initialization to
   * configure how many stages each step has (e.g. 2 for RK2, 4 for RK4).
   *
   * @param[in] stage_count New stage count (must be >= 1)
   * @throws std::invalid_argument If stage_count < 1
   */
  void set_stage_count(int stage_count) {
    if (stage_count < 1) {
      throw std::invalid_argument("Stage count must be >= 1: " +
                                  std::to_string(stage_count));
    }
    m_stage_count = stage_count;
  }

  /**
   * @brief Get the accepted simulation time (read-only)
   *
   * Returns the stored accepted clock. Unchanged for the duration of an
   * active attempt (`begin_attempt` … `commit_attempt` / `reject_attempt`).
   *
   * @return Accepted simulation time
   */
  [[nodiscard]] double get_accepted_time() const noexcept {
    return m_accepted_time;
  }

  /**
   * @brief Get the current (accepted) simulation time
   *
   * Returns the stored accepted time (`m_accepted_time`), clamped to `t1` if
   * needed. Prefer @ref get_accepted_time for adaptive drivers; this alias
   * preserves the historical name used by Simulator and writers.
   *
   * @return Current simulation time, clamped to [t0, t1]
   *
   * @note Accepted time advances on @ref next or @ref commit_attempt only.
   *       @ref set_dt does not rewrite this value.
   *
   * @see get_accepted_time() - explicit accepted-time accessor
   * @see get_increment() - get the number of steps taken
   * @see next() - advance to next time step (fixed dt)
   * @see done() - check if t_current >= t1
   */
  double get_current() const {
    return (m_accepted_time > m_t1) ? m_t1 : m_accepted_time;
  }

  /**
   * @brief Get the time interval for saving data.
   *
   * @return The save interval
   */
  double get_saveat() const { return m_saveat; }

  /**
   * @brief Get the integrator method.
   *
   * @return The integrator method
   */
  IntegratorMethod method() const noexcept { return m_method; }

  /**
   * @brief Set the current time increment (fixed-`dt` reconstruction helper).
   *
   * Sets the committed step count and reconstructs accepted time as
   * `min(t0 + increment * dt, t1)`. This preserves restart / rewind helpers
   * that assume a constant `dt`. It is **not** the adaptive commit path —
   * use @ref commit_attempt after a clipped attempt instead.
   *
   * The increment must be non-negative so that `get_current()` stays in
   * `[t0, t1]` and `done()` / `do_save()` remain meaningful.
   *
   * @param increment The current time increment (number of completed steps
   *                  from `t0`, i.e. same convention as after `next()`).
   * @throws std::invalid_argument if `increment < 0`
   *
   * @post get_increment() == increment
   * @post get_accepted_time() == min(t0 + increment * dt, t1)
   */
  void set_increment(int increment) {
    if (increment < 0) {
      throw std::invalid_argument("Time increment cannot be negative: " +
                                  std::to_string(increment));
    }
    m_increment = increment;
    m_accepted_time = std::min(m_t0 + static_cast<double>(increment) * m_dt, m_t1);
  }

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
   * @brief Advance to the next time step (fixed `dt`)
   *
   * Advances accepted time by `dt` (clamped to `t1`) and increments the step
   * counter by 1. For adaptive clipped intervals, use @ref begin_attempt /
   * @ref commit_attempt instead.
   *
   * @post get_increment() returns previous value + 1
   * @post get_accepted_time() returns previous value + dt (clamped to t1)
   *
   * @see get_increment() - query current step number
   * @see get_current() - accepted simulation time
   * @see done() - check completion status
   * @see commit_attempt() - advance by a clipped attempted interval
   */
  void next() {
    m_accepted_time = std::min(m_accepted_time + m_dt, m_t1);
    m_increment += 1;
  }

  /**
   * @brief Clip a candidate step interval against terminal and output bounds
   *
   * Returns an attempted `dt` such that `accepted_time + attempted_dt` does
   * not exceed `t1`. When `saveat > 0`, also does not pass the next
   * output-alignment time without landing on it.
   *
   * Alignment uses the same tolerance family as @ref do_save (`1e-9`):
   * `next_save = ceil((accepted + 1e-9) / saveat) * saveat`. When both the
   * terminal bound and saveat constrain the step, the **most restrictive**
   * (minimum) wins. If `next_save >= t1`, only the terminal bound matters.
   * When `saveat <= 0`, output-alignment clipping is skipped.
   *
   * @param[in] candidate_dt Proposed step size (must be > 0)
   * @return Clipped attempted interval
   * @throws std::invalid_argument If candidate_dt <= 0
   */
  [[nodiscard]] double clip_attempt_dt(double candidate_dt) const {
    if (candidate_dt <= 0.0) {
      throw std::invalid_argument(
          "Attempt interval (candidate_dt) must be greater than zero: " +
          std::to_string(candidate_dt));
    }
    const double remaining_t1 = m_t1 - m_accepted_time;
    double clipped = std::min(candidate_dt, remaining_t1);
    if (m_saveat > 0.0) {
      const double next_save =
          std::ceil((m_accepted_time + 1.0e-9) / m_saveat) * m_saveat;
      if (next_save > m_accepted_time) {
        clipped = std::min(clipped, next_save - m_accepted_time);
      }
    }
    return clipped;
  }

  /**
   * @brief Whether an attempt transaction is currently open
   */
  [[nodiscard]] bool attempt_active() const noexcept { return m_attempt_active; }

  /**
   * @brief Clipped interval for the active attempt
   *
   * @throws std::logic_error If no attempt is active
   */
  [[nodiscard]] double get_attempted_dt() const {
    if (!m_attempt_active) {
      throw std::logic_error(
          "get_attempted_dt() requires an active attempt (call begin_attempt)");
    }
    return m_attempted_dt;
  }

  /**
   * @brief Begin an attempt: clip candidate_dt and leave accepted time unchanged
   *
   * @param[in] candidate_dt Proposed step size (must be > 0)
   * @throws std::logic_error If an attempt is already active
   * @throws std::invalid_argument If candidate_dt <= 0 (via clip_attempt_dt)
   *
   * @post attempt_active() is true
   * @post get_accepted_time() is unchanged
   * @post get_attempted_dt() == clip_attempt_dt(candidate_dt)
   */
  void begin_attempt(double candidate_dt) {
    if (m_attempt_active) {
      throw std::logic_error(
          "begin_attempt() called while an attempt is already active");
    }
    m_attempted_dt = clip_attempt_dt(candidate_dt);
    m_attempt_active = true;
  }

  /**
   * @brief Commit the active attempt: advance accepted time by attempted_dt
   *
   * Does **not** auto-call @ref increment_step_success (counters stay
   * caller-owned).
   *
   * @throws std::logic_error If no attempt is active
   *
   * @post get_accepted_time() advanced by the attempted interval (clamped to t1)
   * @post get_increment() increased by 1
   * @post attempt_active() is false
   */
  void commit_attempt() {
    if (!m_attempt_active) {
      throw std::logic_error(
          "commit_attempt() requires an active attempt (call begin_attempt)");
    }
    m_accepted_time = std::min(m_accepted_time + m_attempted_dt, m_t1);
    m_increment += 1;
    m_attempt_active = false;
    m_attempted_dt = 0.0;
  }

  /**
   * @brief Reject the active attempt without changing accepted time
   *
   * Does **not** auto-call @ref increment_step_rejection (counters stay
   * caller-owned).
   *
   * @throws std::logic_error If no attempt is active
   *
   * @post get_accepted_time() and get_increment() unchanged
   * @post attempt_active() is false
   */
  void reject_attempt() {
    if (!m_attempt_active) {
      throw std::logic_error(
          "reject_attempt() requires an active attempt (call begin_attempt)");
    }
    m_attempt_active = false;
    m_attempted_dt = 0.0;
  }

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
    os << ", saveat = " << t.m_saveat << ", t_current = " << t.get_current();
    os << ", stage = " << t.m_stage << "/" << t.m_stage_count << ")\n";
    return os;
  };
};

/**
 * @brief RAII guard for temporal state rollback in adaptive time-stepping
 *
 * Captures dt, increment, accepted time, attempt flags, and accepted/rejected
 * step counters from a Time object on construction and restores them on
 * destruction unless commit() is called.
 *
 * Uncommitted restore order (destructor and move-assignment):
 * 1. set_dt(saved_dt)
 * 2. set_increment(saved_increment)  // may temporarily rewrite accepted via
 *    fixed-dt reconstruction
 * 3. Friend-restore m_accepted_time, then attempt flags (after setters)
 * 4. Friend-restore accepted/rejected counters
 *
 * Adaptive drivers should prefer begin_attempt / commit_attempt /
 * reject_attempt over rewriting set_dt + decrementing increment.
 *
 * Typical usage:
 * ```cpp
 * while (!time.done()) {
 *   bool accepted = false;
 *   {
 *     TimeStateGuard guard(time);
 *
 *     time.begin_attempt(candidate_dt);
 *     double error = attempt_step(time.get_accepted_time(),
 *                                 time.get_attempted_dt());
 *     if (error <= tolerance) {
 *       time.commit_attempt();
 *       time.increment_step_success();
 *       guard.commit();
 *       accepted = true;
 *     } else {
 *       time.reject_attempt();
 *     }
 *   }
 *   if (!accepted) {
 *     time.increment_step_rejection();
 *   }
 * }
 * ```
 *
 * @note Moving a guard after the referenced Time has been modified may
 *       restore stale state. Moved-from guards are in a valid but unspecified
 *       state and should not be used.
 */
class TimeStateGuard {
public:
  /**
   * @brief Construct a guard capturing the current temporal state
   * @param time Reference to the Time object to guard
   */
  explicit TimeStateGuard(Time &time)
      : m_time(time), m_saved_dt(time.get_dt()),
        m_saved_increment(time.get_increment()),
        m_saved_accepted_time(time.get_accepted_time()),
        m_saved_attempt_active(time.attempt_active()),
        m_saved_attempted_dt(time.m_attempted_dt),
        m_saved_accepted_steps(time.get_accepted_steps()),
        m_saved_rejected_steps(time.get_rejected_steps()), m_committed(false) {}

  /**
   * @brief Restore captured temporal state unless committed
   *
   * Applies set_dt / set_increment first, then friend-restores accepted time
   * and attempt flags so adaptive commits with attempted_dt != dt are not
   * clobbered by fixed-dt reconstruction.
   */
  ~TimeStateGuard() {
    if (!m_committed) {
      restore_uncommitted();
    }
  }

  // Copy operations deleted: only one guard should own rollback state
  TimeStateGuard(const TimeStateGuard &) = delete;
  TimeStateGuard &operator=(const TimeStateGuard &) = delete;

  /**
   * @brief Move constructor transfers guard ownership
   * @param other Source guard to move from
   *
   * The moved-from guard is left in a committed state (no restoration on
   * destruction) to prevent accidental rollback.
   */
  TimeStateGuard(TimeStateGuard &&other) noexcept
      : m_time(other.m_time), m_saved_dt(other.m_saved_dt),
        m_saved_increment(other.m_saved_increment),
        m_saved_accepted_time(other.m_saved_accepted_time),
        m_saved_attempt_active(other.m_saved_attempt_active),
        m_saved_attempted_dt(other.m_saved_attempted_dt),
        m_saved_accepted_steps(other.m_saved_accepted_steps),
        m_saved_rejected_steps(other.m_saved_rejected_steps),
        m_committed(other.m_committed) {
    other.m_committed = true;
  }

  /**
   * @brief Move assignment transfers guard ownership
   * @param other Source guard to move from
   * @return Reference to this guard
   *
   * If this guard was uncommitted, its saved values are restored before
   * taking ownership of other's state. The moved-from guard is left in
   * a committed state.
   */
  TimeStateGuard &operator=(TimeStateGuard &&other) noexcept {
    if (this != &other) {
      if (!m_committed) {
        restore_uncommitted();
      }
      m_time = other.m_time;
      m_saved_dt = other.m_saved_dt;
      m_saved_increment = other.m_saved_increment;
      m_saved_accepted_time = other.m_saved_accepted_time;
      m_saved_attempt_active = other.m_saved_attempt_active;
      m_saved_attempted_dt = other.m_saved_attempted_dt;
      m_saved_accepted_steps = other.m_saved_accepted_steps;
      m_saved_rejected_steps = other.m_saved_rejected_steps;
      m_committed = other.m_committed;
      other.m_committed = true;
    }
    return *this;
  }

  /**
   * @brief Mark the step as accepted, disabling restoration on destruction
   *
   * Call this when the adaptive step succeeds and the temporal state should
   * be preserved. After commit(), the destructor becomes a no-op.
   */
  void commit() noexcept { m_committed = true; }

  /**
   * @brief Check if the guard has been committed
   * @return true if commit() was called, false otherwise
   */
  [[nodiscard]] bool committed() const noexcept { return m_committed; }

private:
  void restore_uncommitted() noexcept {
    m_time.set_dt(m_saved_dt);
    m_time.set_increment(m_saved_increment);
    m_time.m_accepted_time = m_saved_accepted_time;
    m_time.m_attempt_active = m_saved_attempt_active;
    m_time.m_attempted_dt = m_saved_attempted_dt;
    m_time.accepted_steps_ = m_saved_accepted_steps;
    m_time.rejected_steps_ = m_saved_rejected_steps;
  }

  Time &m_time;          ///< Reference to guarded Time object
  double m_saved_dt;     ///< Captured time step value
  int m_saved_increment; ///< Captured increment counter
  double m_saved_accepted_time; ///< Captured accepted simulation time
  bool m_saved_attempt_active;  ///< Captured attempt-active flag
  double m_saved_attempted_dt;  ///< Captured attempted interval
  int m_saved_accepted_steps; ///< Captured accepted-attempt counter
  int m_saved_rejected_steps; ///< Captured rejected-attempt counter
  bool m_committed;      ///< Flag to disable restoration
};

/**
 * @brief Non-member spellings mirroring @ref Time (consistent with `pfc::get_time`,
 *        `pfc::get_world`, …).
 */
namespace time {

[[nodiscard]] inline double current(const Time &t) noexcept {
  return t.get_current();
}

[[nodiscard]] inline double dt(const Time &t) noexcept { return t.get_dt(); }

[[nodiscard]] inline bool done(const Time &t) noexcept { return t.done(); }

inline void next(Time &t) noexcept { t.next(); }

[[nodiscard]] inline bool do_save(const Time &t) noexcept { return t.do_save(); }

[[nodiscard]] inline int increment(const Time &t) noexcept {
  return t.get_increment();
}

[[nodiscard]] inline double t0(const Time &t) noexcept { return t.get_t0(); }

[[nodiscard]] inline double t1(const Time &t) noexcept { return t.get_t1(); }

[[nodiscard]] inline double saveat(const Time &t) noexcept { return t.get_saveat(); }

inline void set_increment(Time &t, int inc) { t.set_increment(inc); }

inline void set_saveat(Time &t, double s) { t.set_saveat(s); }
inline void set_dt(Time &t, double d) { t.set_dt(d); }

[[nodiscard]] inline int stage(const Time &t) noexcept { return t.get_stage(); }

[[nodiscard]] inline int stage_count(const Time &t) noexcept {
  return t.get_stage_count();
}

inline void set_stage(Time &t, int stage) { t.set_stage(stage); }

inline void set_stage_count(Time &t, int stage_count) {
  t.set_stage_count(stage_count);
}

[[nodiscard]] inline double accepted_time(const Time &t) noexcept {
  return t.get_accepted_time();
}

[[nodiscard]] inline double clip_attempt_dt(const Time &t,
                                            double candidate_dt) {
  return t.clip_attempt_dt(candidate_dt);
}

inline void begin_attempt(Time &t, double candidate_dt) {
  t.begin_attempt(candidate_dt);
}

inline void commit_attempt(Time &t) { t.commit_attempt(); }

inline void reject_attempt(Time &t) { t.reject_attempt(); }

[[nodiscard]] inline double attempted_dt(const Time &t) {
  return t.get_attempted_dt();
}

[[nodiscard]] inline bool attempt_active(const Time &t) noexcept {
  return t.attempt_active();
}

} // namespace time

} // namespace pfc

#endif
