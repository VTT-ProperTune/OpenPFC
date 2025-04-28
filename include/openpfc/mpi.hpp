// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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
