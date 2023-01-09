#pragma once

#include "constants.hpp"
#include "utils.hpp"
#include "world.hpp"
#include <heffte.h>
#include <iostream>

namespace pfc {

class Decomposition {
private:
  const World &m_world;
  const int Lx_c, Ly_c, Lz_c, id, tot;
  const heffte::box3d<int> real_indexes, complex_indexes;
  const std::array<int, 3> proc_grid;
  const std::vector<heffte::box3d<int>> real_boxes, complex_boxes;

public:
  const heffte::box3d<int> world, inbox, outbox;
  const int r2c_direction = constants::r2c_direction;

  Decomposition(const World &world, int id, int tot)
      : m_world(world), Lx_c(floor(m_world.Lx / 2) + 1), Ly_c(m_world.Ly),
        Lz_c(m_world.Lz), id(id), tot(tot), real_indexes(m_world),
        complex_indexes({0, 0, 0}, {Lx_c - 1, Ly_c - 1, Lz_c - 1}),
        proc_grid(heffte::proc_setup_min_surface(real_indexes, tot)),
        real_boxes(heffte::split_world(real_indexes, proc_grid)),
        complex_boxes(heffte::split_world(complex_indexes, proc_grid)),
        world(heffte::box3d<int>({0, 0, 0}, {Lx - 1, Ly - 1, Lz - 1})),
        inbox(heffte::box3d<int>(real_boxes[id])),
        outbox(heffte::box3d<int>(complex_boxes[id])) {
    assert(real_indexes.r2c(r2c_direction) == complex_indexes);
  };

  Decomposition(const World &world, MPI_Comm comm)
      : Decomposition(world, mpi::get_comm_rank(comm),
                      mpi::get_comm_size(comm)) {}

  Decomposition(const Decomposition &) = delete;
  Decomposition &operator=(Decomposition &) = delete;

  const World &get_world() const { return m_world; }

  int get_id() const { return id; }

  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
    const World &w = d.get_world();
    os << "***** DOMAIN DECOMPOSITION STATUS *****\n";
    os << "Real-to-complex symmetry is used (r2c direction = "
       << (char)('x' + constants::r2c_direction) << ")\n";
    os << "Domain is split into " << d.tot << " parts ";
    os << "(minimum surface processor grid: [" << d.proc_grid[0] << ", "
       << d.proc_grid[1] << ", " << d.proc_grid[2] << "])\n";
    os << "Domain in real space: [" << w.Lx << ", " << w.Ly << ", " << w.Lz
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
} // namespace pfc
