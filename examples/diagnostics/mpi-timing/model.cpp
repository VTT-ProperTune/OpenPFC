// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <openpfc/kernel/mpi/timer.hpp>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  constexpr int n = 4096;
  pfc::mpi::timer timer;
  timer.description("Cosine sum, local work only");
  double checksum = 0;
  // start snippet laps
  for (int lap = 0; lap < 3; ++lap) {
    double local = 0, global = 0;
    timer.tic();
    for (int j = comm.rank(); j < n; j += comm.size())
      local += std::cos(2 * std::numbers::pi * j / n);
    timer.toc();
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    checksum = std::max(checksum, std::abs(global));
  }
  const double local_seconds = timer.duration();
  double maximum = 0, total = 0;
  MPI_Allreduce(&local_seconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_seconds, &total, 1, MPI_DOUBLE, MPI_SUM, comm);
  // end snippet laps
  if (comm.rank() == 0)
    std::cout << std::setprecision(17)
              << "ranks,laps,checksum,max_seconds,mean_seconds\n"
              << comm.size() << ",3," << checksum << ',' << maximum << ','
              << total / comm.size() << '\n';
  return checksum < 1e-9 ? 0 : 1;
}
