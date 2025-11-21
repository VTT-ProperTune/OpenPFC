// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file mpi.hpp
 * @brief MPI utilities and wrappers
 *
 * @details
 * This header provides convenient MPI utility functions and aggregates
 * all MPI-related components (communicator, environment, timer, worker).
 *
 * The namespace pfc::mpi contains:
 * - get_rank(): Get current MPI rank
 * - get_size(): Get total number of MPI processes
 * - Communicator: MPI communicator wrapper
 * - Environment: MPI initialization/finalization management
 * - Timer: MPI timing utilities
 * - Worker: MPI task distribution helpers
 *
 * @code
 * #include <openpfc/mpi.hpp>
 *
 * int main(int argc, char** argv) {
 *     pfc::mpi::Environment env(argc, argv);
 *     int rank = pfc::mpi::get_rank();
 *     int size = pfc::mpi::get_size();
 *     std::cout << "Rank " << rank << " of " << size << std::endl;
 *     return 0;
 * }
 * @endcode
 *
 * @see mpi/communicator.hpp for MPI communicator wrapper
 * @see mpi/environment.hpp for MPI initialization
 * @see mpi/timer.hpp for MPI timing
 * @see mpi/worker.hpp for task distribution
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_HPP
#define PFC_MPI_HPP

#include "mpi/communicator.hpp"
#include "mpi/environment.hpp"
#include "mpi/timer.hpp"
#include "mpi/worker.hpp"
#include <mpi.h>

namespace pfc {
namespace mpi {

inline int get_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

inline int get_size() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_HPP
