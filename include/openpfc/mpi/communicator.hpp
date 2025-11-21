// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file communicator.hpp
 * @brief MPI communicator wrapper class
 *
 * @details
 * This header provides the communicator class, which wraps MPI_Comm
 * for safe, RAII-style MPI communicator management.
 *
 * The communicator class:
 * - Wraps MPI_Comm in a std::shared_ptr for automatic cleanup
 * - Provides rank() and size() convenience methods
 * - Supports implicit conversion to MPI_Comm for use with MPI functions
 *
 * @code
 * #include <openpfc/mpi/communicator.hpp>
 *
 * pfc::mpi::communicator comm;
 * int rank = comm.rank();
 * int size = comm.size();
 * MPI_Barrier(comm);  // Implicit conversion
 * @endcode
 *
 * @see mpi/environment.hpp for MPI initialization
 * @see mpi.hpp for top-level MPI utilities
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_COMMUNICATOR_HPP
#define PFC_MPI_COMMUNICATOR_HPP

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

inline communicator::communicator() { comm_ptr.reset(new MPI_Comm(MPI_COMM_WORLD)); }

inline communicator::operator MPI_Comm() const {
  if (comm_ptr)
    return *comm_ptr;
  else
    return MPI_COMM_NULL;
}

inline int communicator::size() const {
  int size_;
  MPI_Comm_size(MPI_Comm(*this), &size_);
  return size_;
}

inline int communicator::rank() const {
  int rank_;
  MPI_Comm_rank(MPI_Comm(*this), &rank_);
  return rank_;
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_COMMUNICATOR_HPP
