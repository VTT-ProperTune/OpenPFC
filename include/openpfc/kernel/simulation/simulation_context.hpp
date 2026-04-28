// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file simulation_context.hpp
 * @brief Execution context passed to field modifiers with the Model
 *
 * @details
 * Carries information that is not naturally part of `Model` but must stay
 * consistent with the simulation (e.g. the MPI communicator used for
 * collectives such as MPI-IO). The simulator builds this object when applying
 * initial and boundary conditions.
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
  SimulationContext() = default;

  explicit SimulationContext(MPI_Comm mpi_comm) noexcept : m_mpi_comm(mpi_comm) {}

  /** @brief Communicator for MPI collectives (MPI-IO, barriers, reductions) */
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }

private:
  MPI_Comm m_mpi_comm{MPI_COMM_WORLD};
};

} // namespace pfc

#endif // PFC_SIMULATION_CONTEXT_HPP
