// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <iostream>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/communicator.hpp>
#include <openpfc/kernel/mpi/environment.hpp>

int main() {
  pfc::mpi::environment environment;
  pfc::mpi::communicator comm;
  try {
    // start snippet initialize
    const auto domain = pfc::domain::create({16, 8, 2});
    const auto partition = pfc::decomposition::create(domain, comm.size());
    auto u = pfc::data::field_from_subdomain<double>(partition, comm.rank(), 0);
    u.apply([](double x, double y, double) { return 1 + x + y * y; });
    // end snippet initialize

    // start snippet probe
    for (const auto point : {pfc::Int3{4, 2, 0}, pfc::Int3{12, 6, 0}}) {
      double local_value = 0.0;
      int local_hits = 0;
      u.for_each_owned([&](int i, int j, int k) {
        const auto low = u.box().low;
        if (pfc::Int3{i + low[0], j + low[1], k + low[2]} == point) {
          local_value = u(i, j, k);
          local_hits = 1;
        }
      });
      double value = 0.0;
      int hits = 0;
      MPI_Allreduce(&local_value, &value, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&local_hits, &hits, 1, MPI_INT, MPI_SUM, comm);
      if (hits != 1 || value != 1 + point[0] + point[1] * point[1]) return 1;
      if (comm.rank() == 0) {
        std::cout << "u(" << point[0] << ',' << point[1] << ",0)=" << value
                  << " owners=" << hits << '\n';
      }
    }
    // end snippet probe
  } catch (const std::exception &error) {
    if (comm.rank() == 0) std::cerr << error.what() << '\n';
    MPI_Abort(comm, 1);
    return 1;
  }
}
