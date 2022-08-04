#pragma once

#include <array>
#include <cmath>

namespace pfc {
class Time {
private:
  double m_t0, m_t1, m_dt;
  int m_increment;
  double m_saveat;

public:
  Time(const std::array<double, 3> &time, double saveat)
      : m_t0(time[0]), m_t1(time[1]), m_dt(time[2]), m_increment(0),
        m_saveat(saveat) {}

  Time(const std::array<double, 3> &time) : Time(time, time[2]) {}

  double get_t0() const { return m_t0; }
  double get_t1() const { return m_t1; }
  double get_dt() const { return m_dt; }
  int get_increment() const { return m_increment; }
  double get_current() const { return m_t0 + m_increment * m_dt; }
  double get_saveat() const { return m_saveat; }

  void set_increment(int increment) { m_increment = increment; }

  void set_saveat(double saveat) { m_saveat = saveat; }

  bool done() const { return get_current() >= m_t1; }
  void next() { m_increment += 1; }
  bool do_save() const {
    return (std::fmod(get_current() + 1.0e-9, m_saveat) < 1.e-6) || done() ||
           (m_increment == 0);
  }

  friend std::ostream &operator<<(std::ostream &os, const Time &t) {
    os << "(t0 = " << t.m_t0 << ", t1 = " << t.m_t1 << ", dt = " << t.m_dt;
    os << ", saveat = " << t.m_saveat << ", t_current = " << t.get_current()
       << ")\n";
    return os;
  };
};
} // namespace pfc
