// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file decomposition_factory.hpp
 * @brief Factory functions for creating domain decompositions
 *
 * @details
 * This header provides factory functions to create Decomposition objects
 * for distributing computational domains across MPI processes.
 *
 * Two factory functions are provided:
 * - make_decomposition(world, rank, num_domains): Explicit rank/size
 * - make_decomposition(world, comm): Use MPI communicator
 *
 * The factory encapsulates the logic for choosing an optimal decomposition
 * strategy based on domain geometry and number of processes.
 *
 * @code
 * #include <openpfc/factory/decomposition_factory.hpp>
 * #include <openpfc/core/world.hpp>
 *
 * pfc::World world({64, 64, 64}, {1.0, 1.0, 1.0});
 * auto decomp = pfc::make_decomposition(world, MPI_COMM_WORLD);
 * @endcode
 *
 * @see core/decomposition.hpp for Decomposition class
 * @see core/world.hpp for World definition
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#ifndef PFC_DECOMPOSITION_FACTORY_HPP
#define PFC_DECOMPOSITION_FACTORY_HPP

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include <mpi.h>

namespace pfc {

/**
 * @brief Factory function to create a Decomposition object.
 *
 * @param world The World object.
 * @param rank The rank of the current process (defaults to 0).
 * @param num_domains The total number of domains (defaults to 1).
 * @return Decomposition
 */
Decomposition make_decomposition(const World &world, int rank, int num_domains);

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
