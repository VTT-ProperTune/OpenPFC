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
