#pragma once

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

void timer::tic() {
  tic_ = MPI_Wtime();
}

double timer::toc() {
  toc_ = MPI_Wtime();
  double d = toc_ - tic_;
  duration_ += d;
  return d;
}

double timer::duration() const {
  return duration_;
}

std::string timer::description() const {
  return description_;
}

void timer::reset() {
  duration_ = 0.0;
}

void timer::description(const std::string &description) {
  description_ = description;
}

std::ostream &operator<<(std::ostream &os, const timer &t) {
  os << t.description() << ": " << t.duration() << " s";
  return os;
}

} // namespace mpi
} // namespace pfc
