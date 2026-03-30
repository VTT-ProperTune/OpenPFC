// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file timer.hpp
 * @brief MPI-based wall-clock timer
 *
 * @details
 * This header provides the timer class for measuring wall-clock time
 * using MPI_Wtime(), which provides consistent timing across MPI ranks.
 *
 * The timer class provides:
 * - tic(): Start a timing lap (required before each toc())
 * - toc(): Stop the current lap and return elapsed wall time since tic()
 * - duration(): Get total accumulated duration across completed laps
 * - reset(): Reset accumulated duration and clear any in-progress lap
 * - description(): Set/get timer description for logging
 *
 * Calling toc() without a preceding tic() (or calling toc() twice without an
 * intervening tic()) throws std::logic_error.
 *
 * @code
 * #include <openpfc/kernel/mpi/timer.hpp>
 *
 * pfc::mpi::timer t;
 * t.description("FFT computation");
 * t.tic();
 * // ... perform FFT ...
 * double elapsed = t.toc();
 * std::cout << t << std::endl;  // Print timer with description
 * @endcode
 *
 * @see mpi/environment.hpp for MPI initialization
 * @see mpi.hpp for top-level MPI utilities
 * @see openpfc/kernel/profiling/profiling.hpp for MPI-wide stats and step timing I/O
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_TIMER_HPP
#define PFC_MPI_TIMER_HPP

#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>

namespace pfc {
namespace mpi {

class timer {
  double tic_ = 0.0;
  double toc_ = 0.0;
  double duration_ = 0.0;
  std::string description_;
  bool lap_started_ = false;

public:
  void tic();
  double toc();
  double duration() const;
  void reset();
  void description(const std::string &);
  std::string description() const;
  friend std::ostream &operator<<(std::ostream &os, const timer &t);
};

inline void timer::reset() {
  duration_ = 0.0;
  lap_started_ = false;
}

inline std::string timer::description() const { return description_; }

inline void timer::description(const std::string &description) {
  description_ = description;
}

inline std::ostream &operator<<(std::ostream &os, const timer &t) {
  os << t.description();
  return os;
}

inline void timer::tic() {
  tic_ = MPI_Wtime();
  lap_started_ = true;
}

inline double timer::toc() {
  if (!lap_started_) {
    throw std::logic_error(
        "pfc::mpi::timer::toc() called without a matching tic()");
  }
  toc_ = MPI_Wtime();
  const double delta = toc_ - tic_;
  duration_ += delta;
  lap_started_ = false;
  return delta;
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_TIMER_HPP
