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
 * - tic(): Start timing
 * - toc(): Stop timing and return elapsed time
 * - duration(): Get total accumulated duration
 * - reset(): Reset accumulated duration
 * - description(): Set/get timer description for logging
 *
 * @code
 * #include <openpfc/mpi/timer.hpp>
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
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_TIMER_HPP
#define PFC_MPI_TIMER_HPP

#include <iostream>
#include <mpi.h>
#include <string>

namespace pfc {
namespace mpi {

class timer {
  double tic_;
  double toc_;
  double duration_ = 0.0;
  std::string description_;

public:
  void tic();
  double toc();
  double duration() const;
  void reset();
  void description(const std::string &);
  std::string description() const;
  friend std::ostream &operator<<(std::ostream &os, const timer &t);
};

inline void timer::reset() { duration_ = 0.0; }

inline std::string timer::description() const { return description_; }

inline void timer::description(const std::string &description) {
  description_ = description;
}

inline std::ostream &operator<<(std::ostream &os, const timer &t) {
  os << t.description();
  return os;
}

inline void timer::tic() { tic_ = MPI_Wtime(); }

inline double timer::toc() {
  toc_ = MPI_Wtime();
  duration_ += toc_ - tic_;
  return toc_ - tic_;
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_TIMER_HPP
