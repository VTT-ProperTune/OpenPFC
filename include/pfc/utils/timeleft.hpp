#pragma once

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
