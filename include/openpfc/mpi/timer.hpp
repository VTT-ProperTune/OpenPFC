// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
