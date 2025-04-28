// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/decomposition.hpp"
#include "openpfc/backends/heffte_adapter.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pfc {

Decomposition::Decomposition(const World &world, int rank, int num_domains)
    : m_world(world),
      m_rank(rank < num_domains ? rank
                                : throw std::logic_error("Cannot construct domain decomposition: !(rank < nprocs)")),
      m_num_domains(num_domains), Lx_c(floor(m_world.size()[0] / 2) + 1), Ly_c(m_world.size()[1]),
      Lz_c(m_world.size()[2]), real_indexes(to_heffte_box(world)),
      complex_indexes({0, 0, 0}, {Lx_c - 1, Ly_c - 1, Lz_c - 1}),
      proc_grid(heffte::proc_setup_min_surface(real_indexes, num_domains)),
      real_boxes(heffte::split_world(real_indexes, proc_grid)),
      complex_boxes(heffte::split_world(complex_indexes, proc_grid)), inbox(real_boxes[rank]),
      outbox(complex_boxes[rank]) {
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);
}

const std::array<int, 3> &Decomposition::get_inbox_size() const {
  return inbox.size;
}

const std::array<int, 3> &Decomposition::get_inbox_offset() const {
  return inbox.low;
}

const std::array<int, 3> &Decomposition::get_outbox_size() const {
  return outbox.size;
}

const std::array<int, 3> &Decomposition::get_outbox_offset() const {
  return outbox.low;
}

const World &Decomposition::get_world() const {
  return m_world;
}
int Decomposition::get_rank() const {
  return m_rank;
}
int Decomposition::get_num_domains() const {
  return m_num_domains;
}

std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
  const World &w = d.get_world();
  os << "***** DOMAIN DECOMPOSITION STATUS *****\n";
  os << "Real-to-complex symmetry is used (r2c direction = " << (char)('x' + d.r2c_direction) << ")\n";
  os << "Domain is split into " << d.get_num_domains() << " parts ";
  os << "(minimum surface processor grid: [" << d.proc_grid[0] << ", " << d.proc_grid[1] << ", " << d.proc_grid[2]
     << "])\n";
  os << "Domain in real space: [" << w.size()[0] << ", " << w.size()[1] << ", " << w.size()[2] << "] ("
     << d.real_indexes.count() << " indexes)\n";
  os << "Domain in complex space: [" << d.Lx_c << ", " << d.Ly_c << ", " << d.Lz_c << "] (" << d.complex_indexes.count()
     << " indexes)\n";
  for (int i = 0; i < d.get_num_domains(); i++) {
    const auto &in = d.real_boxes[i];
    const auto &out = d.complex_boxes[i];
    os << "Domain " << i + 1 << "/" << d.get_num_domains() << ": ";
    os << "[" << in.low[0] << ", " << in.low[1] << ", " << in.low[2] << "] x ";
    os << "[" << in.high[0] << ", " << in.high[1] << ", " << in.high[2] << "] (" << in.count() << " indexes) => ";
    os << "[" << out.low[0] << ", " << out.low[1] << ", " << out.low[2] << "] x ";
    os << "[" << out.high[0] << ", " << out.high[1] << ", " << out.high[2] << "] (" << out.count() << " indexes)\n";
  }
  return os;
}

} // namespace pfc
