// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "11_write_results.hpp"
#include <iostream>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/mpi/worker.hpp>
#include <vector>

using namespace pfc;
using pfc::data::field_from_subdomain_unpadded;
using pfc::data::field_from_subdomain;

// Write a Field to VTK via the thin VtkWriter wrapper (M2: no DiscreteField).
int main(int argc, char **argv) {
  MPI_Worker worker(argc, argv);
  auto domain = domain::create({4, 3, 2});
  auto decomp = decomposition::create(domain, 1);
  const int rank = worker.get_rank();

  auto field = field_from_subdomain_unpadded<double>(decomp, rank);
  for (std::size_t i = 0; i < field.size(); ++i) {
    field.data()[i] = static_cast<double>(i);
  }

  const auto local = field.local_size();
  const auto offset = field.lower_global();

  VtkWriter<double> writer;
  writer.set_uri("results.vti");
  writer.set_field_name("density");
  writer.set_domain(domain::get_size(domain), local, offset);
  writer.set_origin(domain::get_origin(domain));
  writer.set_spacing(domain::get_spacing(domain));
  std::cout << "Writing results to file: " << writer.get_uri() << "\n";
  writer.initialize();
  writer.write(std::vector<double>(field.data(), field.data() + field.size()));
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
