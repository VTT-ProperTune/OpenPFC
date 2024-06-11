/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

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
