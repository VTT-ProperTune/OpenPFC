// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

namespace pfc {

[[nodiscard]] Decomposition make_decomposition(const Domain &domain, int rank,
                                               int num_domains) {
  (void)rank;
  return decomposition::create(domain, num_domains);
}

[[nodiscard]] Decomposition make_decomposition(const Domain &domain, MPI_Comm comm) {
  const int size = pfc::mpi::get_comm_size(comm);
  return decomposition::create(domain, size);
}

} // namespace pfc
