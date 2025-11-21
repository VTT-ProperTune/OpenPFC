// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file worker.hpp
 * @brief MPI worker process management
 *
 * @details
 * This header provides the MPI_Worker class for managing individual MPI
 * worker processes with automatic initialization and output management.
 *
 * The MPI_Worker class:
 * - Automatically initializes and finalizes MPI
 * - Mutes stdout on non-root ranks to avoid duplicate output
 * - Provides rank and process count queries
 * - Manages MPI communicator
 *
 * This is particularly useful for applications where only rank 0 should
 * produce console output.
 *
 * @code
 * #include <openpfc/mpi/worker.hpp>
 *
 * int main(int argc, char** argv) {
 *     pfc::MPI_Worker worker(argc, argv, MPI_COMM_WORLD);
 *     // Only rank 0 prints to stdout
 *     std::cout << "Rank: " << worker.rank() << std::endl;
 *     return 0;
 * }
 * @endcode
 *
 * @see mpi/environment.hpp for alternative RAII MPI management
 * @see mpi/communicator.hpp for communicator wrapper
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_MPI_WORKER_HPP
#define PFC_MPI_WORKER_HPP

#include <mpi.h>

#include <iostream>

namespace pfc {

/**
 * @brief An MPI worker class that wraps MPI_Init and MPI_Finalize
 *
 * This class represents a single worker process in a MPI application. It wraps
 * the MPI_Init and MPI_Finalize calls and provides helper functions to get the
 * rank and number of processes in the MPI communicator.
 */
class MPI_Worker {
  MPI_Comm m_comm; ///< MPI communicator for this worker
  int m_rank;      ///< Rank of this worker process in the MPI communicator
  int m_num_procs; ///< Number of processes in the MPI communicator

public:
  /**
   * @brief Constructs an MPI worker instance and initializes MPI.
   *
   * This constructor initializes MPI and retrieves the rank and number of
   * processes in the given MPI communicator. If the rank is not zero, the
   * standard output is muted to avoid duplicate output.
   *
   * @param argc Pointer to the number of command-line arguments
   * @param argv Pointer to an array of command-line arguments
   * @param comm MPI communicator to use
   */
  MPI_Worker(int argc, char *argv[], MPI_Comm comm = MPI_COMM_WORLD) : m_comm(comm) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_num_procs);
    if (m_rank != 0) {
      mute();
    }
    std::cout << "MPI_Init(): initialized " << m_num_procs << " processes"
              << std::endl;
  }
  /**
   * @brief Destroys the MPI worker instance and finalizes MPI.
   *
   * This destructor finalizes MPI.
   */
  ~MPI_Worker() { MPI_Finalize(); }

  /**
   * @brief Returns the rank of this worker process in the MPI communicator.
   *
   * @return The rank of this worker process
   */
  int get_rank() const { return m_rank; }

  /**
   * @brief Returns the number of processes in the MPI communicator.
   *
   * @return The number of processes in the MPI communicator
   */
  int get_num_ranks() const { return m_num_procs; }

  /**
   * @brief Mutes the standard output.
   *
   * This function sets the failbit of the standard output stream, effectively
   * muting it.
   */
  void mute() { std::cout.setstate(std::ios::failbit); }

  /**
   * @brief Unmutes the standard output.
   *
   * This function clears the failbit of the standard output stream, effectively
   * unmuting it.
   */
  void unmute() { std::cout.clear(); }
};

} // namespace pfc

#endif // PFC_MPI_WORKER_HPP
