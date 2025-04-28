// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_DECOMPOSITION_FACTORY_HPP
#define PFC_DECOMPOSITION_FACTORY_HPP

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include <mpi.h>

namespace pfc {

/**
 * @brief Factory function to create a Decomposition from MPI communicator.
 *
 * @param world The World object.
 * @param comm The MPI communicator (defaults to MPI_COMM_WORLD).
 * @return Decomposition
 */
Decomposition make_decomposition(const World &world, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace pfc

#endif // PFC_DECOMPOSITION_FACTORY_HPP
