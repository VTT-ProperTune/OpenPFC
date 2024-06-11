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
