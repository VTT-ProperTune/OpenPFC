// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_TIME_HPP
#define PFC_TIME_HPP

#include <array>
#include <cmath>
#include <iostream>

namespace pfc {

/**
 * @brief The Time class represents a time interval for simulations in PFC.
 *
 * The Time class encapsulates the start time, end time, time step, and other
 * parameters related to time increments in simulations. It provides methods to
 * query and update the current time and check if the time interval is
 * completed.
 */
class Time {
private:
  double m_t0;     ///< Start time
  double m_t1;     ///< End time
  double m_dt;     ///< Time step
  int m_increment; ///< Current time increment
  double m_saveat; ///< Time interval for saving data

public:
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
   * @brief Get the current time.
   *
   * @return The current time
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
   * @brief Check if the time interval is completed.
   *
   * @return True if the current time is greater than or equal to the end time,
   * False otherwise
   */
  bool done() const {
    return (get_current() >= m_t1 - 1e-9); // Adjust for floating-point precision
  }

  /**
   * @brief Move to the next time increment.
   */
  void next() { m_increment += 1; }

  /**
   * @brief Check if data should be saved at the current time.
   *
   * Data should be saved if the current time is within the save interval,
   * or if the time interval is completed, or if it is the first increment.
   *
   * @return True if data should be saved, False otherwise
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
