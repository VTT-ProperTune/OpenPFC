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

#include <memory>
#include <mpi.h>

namespace pfc {
namespace mpi {

class communicator {

public:
  communicator();
  operator MPI_Comm() const;
  int rank() const;
  int size() const;

protected:
  std::shared_ptr<MPI_Comm> comm_ptr;
};

communicator::communicator() {
  comm_ptr.reset(new MPI_Comm(MPI_COMM_WORLD));
}

int communicator::size() const {
  int size_;
  MPI_Comm_size(MPI_Comm(*this), &size_);
  return size_;
}

communicator::operator MPI_Comm() const {
  if (comm_ptr)
    return *comm_ptr;
  else
    return MPI_COMM_NULL;
}

int communicator::rank() const {
  int rank_;
  MPI_Comm_rank(MPI_Comm(*this), &rank_);
  return rank_;
}

} // namespace mpi
} // namespace pfc
