// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_context.hpp
 * @brief Execution context passed to field modifiers with the Model
 *
 * @details
 * Carries information that is not naturally part of `Model` but must stay
 * consistent with the simulation (e.g. the MPI communicator used for
 * collectives such as MPI-IO, and whether this rank is rank 0 in that
 * communicator). The simulator builds this object when applying initial and
 * boundary conditions.
 */

#ifndef PFC_SIMULATION_CONTEXT_HPP
#define PFC_SIMULATION_CONTEXT_HPP

#include <mpi.h>

namespace pfc {

/**
 * @brief Read-only bundle for modifier application alongside Model
 */
class SimulationContext {
public:
  SimulationContext()
      : m_mpi_comm(MPI_COMM_WORLD), m_is_rank0(compute_is_rank0(MPI_COMM_WORLD)) {}

  explicit SimulationContext(MPI_Comm mpi_comm) noexcept
      : m_mpi_comm(mpi_comm), m_is_rank0(compute_is_rank0(mpi_comm)) {}

  /** @brief Communicator for MPI collectives (MPI-IO, barriers, reductions) */
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }

  /**
   * @brief True if this process is rank 0 in mpi_comm()
   *
   * Prefer this over Model::is_rank0() inside modifiers that use mpi_comm()
   * for I/O, so logging and single-writer behavior match the same communicator
   * the simulator passed in.
   */
  [[nodiscard]] bool is_rank0() const noexcept { return m_is_rank0; }

private:
  static bool compute_is_rank0(MPI_Comm comm) noexcept {
    if (comm == MPI_COMM_NULL) {
      return false;
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    return rank == 0;
  }

  MPI_Comm m_mpi_comm;
  bool m_is_rank0;
};

} // namespace pfc

#endif // PFC_SIMULATION_CONTEXT_HPP
