#pragma once

#include <iostream>
#include <mpi.h>

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
  MPI_Worker(int argc, char *argv[], MPI_Comm comm) : m_comm(comm) {
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
