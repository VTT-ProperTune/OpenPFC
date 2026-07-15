// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * #include <openpfc/kernel/mpi/communicator.hpp>
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

#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc::mpi {

class communicator {

private:
  static constexpr auto comm_deleter = [](MPI_Comm* comm) {
    if (comm && *comm != MPI_COMM_WORLD &&
        *comm != MPI_COMM_SELF &&
        *comm != MPI_COMM_NULL) {
      MPI_Comm_free(comm);
    }
    delete comm;
  };

public:
  communicator()
      : comm_ptr(std::shared_ptr<MPI_Comm>(new MPI_Comm(MPI_COMM_WORLD), comm_deleter)) {}

  /** @brief Wrap an existing communicator (e.g. application or sub-communicator). */
  explicit communicator(MPI_Comm c)
      : comm_ptr(std::shared_ptr<MPI_Comm>(new MPI_Comm(c), comm_deleter)) {}

  operator MPI_Comm() const;
  int rank() const;
  int size() const;

protected:
  std::shared_ptr<MPI_Comm> comm_ptr;
};

inline communicator::operator MPI_Comm() const {
  if (comm_ptr) {
    return *comm_ptr;
  }
  return MPI_COMM_NULL;
}

inline int communicator::size() const {
  int size_;
  int err = MPI_Comm_size(MPI_Comm(*this), &size_);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_size");
  return size_;
}

inline int communicator::rank() const {
  int rank_;
  int err = MPI_Comm_rank(MPI_Comm(*this), &rank_);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Comm_rank");
  return rank_;
}

} // namespace pfc::mpi

#endif // PFC_MPI_COMMUNICATOR_HPP
