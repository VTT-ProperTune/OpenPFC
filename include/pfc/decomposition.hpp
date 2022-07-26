#pragma once

#include "constants.hpp"
#include <heffte.h>
#include <iostream>

namespace PFC {

class Decomposition {
private:
  const int Lx, Lx_c, Ly, Ly_c, Lz, Lz_c, id, tot;
  const heffte::box3d<int> real_indexes, complex_indexes;
  const std::array<int, 3> proc_grid;
  const std::vector<heffte::box3d<int>> real_boxes, complex_boxes;

  static int get_comm_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  static int get_comm_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

public:
  const heffte::box3d<int> inbox, outbox;
  Decomposition(const std::array<int, 3> &dims, int id, int tot)
      : Lx(dims[0]), Lx_c(floor(Lx / 2) + 1), Ly(dims[1]), Ly_c(Ly),
        Lz(dims[2]), Lz_c(Lz), id(id), tot(tot),
        real_indexes({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1}),
        complex_indexes({0, 0, 0}, {Lx_c - 1, Ly_c - 1, Lz_c - 1}),
        proc_grid(heffte::proc_setup_min_surface(real_indexes, tot)),
        real_boxes(heffte::split_world(real_indexes, proc_grid)),
        complex_boxes(heffte::split_world(complex_indexes, proc_grid)),
        inbox(heffte::box3d<int>(real_boxes[id])),
        outbox(heffte::box3d<int>(complex_boxes[id])) {
    assert(real_indexes.r2c(constants::r2c_direction) == complex_indexes);
  };

  Decomposition(const std::array<int, 3> &dims, MPI_Comm comm = MPI_COMM_WORLD)
      : Decomposition(dims, get_comm_rank(comm), get_comm_size(comm)) {}

  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
    os << "***** DOMAIN DECOMPOSITION STATUS *****\n";
    os << "Real-to-complex symmetry is used (r2c direction = "
       << (char)('x' + constants::r2c_direction) << ")\n";
    os << "Domain is split into " << d.tot << " parts ";
    os << "(minimum surface processor grid: [" << d.proc_grid[0] << ", "
       << d.proc_grid[1] << ", " << d.proc_grid[2] << "])\n";
    os << "Domain in real space: [" << d.Lx << ", " << d.Ly << ", " << d.Lz
       << "] (" << d.real_indexes.count() << " indexes)\n";
    os << "Domain in complex space: [" << d.Lx_c << ", " << d.Ly_c << ", "
       << d.Lz_c << "] (" << d.complex_indexes.count() << " indexes)\n";
    for (int i = 0; i < d.tot; i++) {
      auto in = d.real_boxes[i];
      auto out = d.complex_boxes[i];
      os << "Domain " << i + 1 << "/" << d.tot << ": ";
      os << "[" << in.low[0] << ", " << in.low[1] << ", " << in.low[2]
         << "] x ";
      os << "[" << in.high[0] << ", " << in.high[1] << ", " << in.high[2]
         << "] (" << in.count() << " indexes) => ";
      os << "[" << out.low[0] << ", " << out.low[1] << ", " << out.low[2]
         << "] x ";
      os << "[" << out.high[0] << ", " << out.high[1] << ", " << out.high[2]
         << "] (" << out.count() << " indexes)\n";
    }
    return os;
  }
};
} // namespace PFC
