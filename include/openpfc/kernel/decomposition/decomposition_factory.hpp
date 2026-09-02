// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file decomposition_factory.hpp
 * @brief Factory functions for creating domain decompositions
 */

#ifndef PFC_DECOMPOSITION_FACTORY_HPP
#define PFC_DECOMPOSITION_FACTORY_HPP

#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc {

[[nodiscard]] Decomposition make_decomposition(const Domain &domain, int rank,
                                               int num_domains);

[[nodiscard]] Decomposition make_decomposition(const Domain &domain,
                                               MPI_Comm comm = MPI_COMM_WORLD);

} // namespace pfc

#endif // PFC_DECOMPOSITION_FACTORY_HPP
