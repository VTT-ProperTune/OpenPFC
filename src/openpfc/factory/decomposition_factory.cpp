// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/factory/decomposition_factory.hpp"

namespace pfc {

Decomposition make_decomposition(const World &world, int rank, int num_domains) {
  using Box3D = Decomposition::Box3D;
  if (rank < 0 || rank >= num_domains) {
    throw std::logic_error("Cannot construct domain decomposition: !(rank < nprocs)");
  }
  if (num_domains <= 0) {
    throw std::logic_error("Cannot construct domain decomposition: !(nprocs > 0)");
  }
  int Lx = world.size()[0];
  int Ly = world.size()[1];
  int Lz = world.size()[2];
  int Lx_c = floor(world.size()[0] / 2) + 1;
  int Ly_c = world.size()[1];
  int Lz_c = world.size()[2];
  const Box3D real_indexes({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1});
  const Box3D complex_indexes({0, 0, 0}, {Lx_c - 1, Ly_c - 1, Lz_c - 1});
  const std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(real_indexes, num_domains);
  const std::vector<Box3D> real_boxes = heffte::split_world(real_indexes, proc_grid);
  const std::vector<Box3D> complex_boxes = heffte::split_world(complex_indexes, proc_grid);
  const Box3D inbox = real_boxes[rank];
  const Box3D outbox = complex_boxes[rank];
  const int r2c_direction = 0; ///< Real-to-complex symmetry direction. TODO: make this dynamic
  if (inbox.r2c(r2c_direction) != complex_indexes) {
    throw std::logic_error("Cannot construct domain decomposition: inbox.r2c != complex_indexes");
  }
  return Decomposition(world, inbox, outbox);
}

Decomposition make_decomposition(const World &world, MPI_Comm comm) {
  int rank;
  int size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  return make_decomposition(world, rank, size);
}

} // namespace pfc
