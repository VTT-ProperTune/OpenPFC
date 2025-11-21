// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file timeleft.hpp
 * @brief Estimated time remaining display
 *
 * @details
 * This header provides the TimeLeft class for converting elapsed time
 * in seconds to a human-readable format (days, hours, minutes, seconds).
 *
 * The class automatically breaks down time into appropriate units and
 * provides formatted output for displaying estimated time remaining
 * in long-running simulations.
 *
 * @code
 * #include <openpfc/utils/timeleft.hpp>
 *
 * double seconds_remaining = 3665.0;
 * pfc::utils::TimeLeft time(seconds_remaining);
 * std::cout << time << std::endl;  // Prints "1h 1m 5s"
 * @endcode
 *
 * @see mpi/timer.hpp for MPI-based timing
 * @see utils.hpp for other utility functions
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_UTILS_TIMELEFT_HPP
#define PFC_UTILS_TIMELEFT_HPP

#include <iostream>

namespace pfc {
namespace utils {

class TimeLeft {
private:
  int m_seconds = 0, m_minutes = 0, m_hours = 0, m_days = 0;

public:
  explicit TimeLeft(double t) : m_seconds(t) {
    if (m_seconds > 60) {
      m_minutes = m_seconds / 60;
      m_seconds -= m_minutes * 60;
    }
    if (m_minutes > 60) {
      m_hours = m_minutes / 60;
      m_minutes -= m_hours * 60;
    }
    if (m_hours > 24) {
      m_days = m_hours / 24;
      m_hours -= m_days * 24;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const TimeLeft &e) {
    if (e.m_days > 0) {
      os << e.m_days << "d ";
    }
    if (e.m_hours > 0) {
      os << e.m_hours << "h ";
    }
    if (e.m_minutes > 0) {
      os << e.m_minutes << "m ";
    }
    os << e.m_seconds << "s";
    return os;
  }
};
} // namespace utils
} // namespace pfc

#endif // PFC_UTILS_TIMELEFT_HPP
