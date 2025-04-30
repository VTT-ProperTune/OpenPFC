// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/decomposition.hpp"
#include "openpfc/backends/heffte_adapter.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace pfc {

Decomposition::Decomposition(const World &world, const heffte::box3d<int> &inbox, const heffte::box3d<int> &outbox)
    : m_world(world),
      m_inbox(inbox),
      m_outbox(outbox) {}

const Decomposition::Box3D &Decomposition::get_inbox() const noexcept { return m_inbox; }

const Decomposition::Box3D &Decomposition::get_outbox() const noexcept { return m_outbox; }

const Decomposition::Int3 &Decomposition::get_inbox_size() const noexcept { return get_inbox().size; }

const Decomposition::Int3 &Decomposition::get_inbox_offset() const noexcept { return get_inbox().low; }

const Decomposition::Int3 &Decomposition::get_outbox_size() const noexcept { return get_outbox().size; }

const Decomposition::Int3 &Decomposition::get_outbox_offset() const noexcept { return get_outbox().low; }

const World &Decomposition::get_world() const noexcept { return m_world; }

/*
int Decomposition::get_rank() const {
  return m_rank;
}

int Decomposition::get_num_domains() const {
  return m_num_domains;
}
*/

std::ostream &operator<<(std::ostream &os, const Decomposition &) {
  // const World &w = d.get_world();
  os << "***** DOMAIN DECOMPOSITION STATUS *****\n";
  int r2c_direction = 0; // TODO: make this dynamic
  os << "Real-to-complex symmetry is used (r2c direction = " << (char)('x' + r2c_direction) << ")\n";
  // :D
  /*
  os << "Domain is split into " << d.get_num_domains() << " parts ";
  os << "(minimum surface processor grid: [" << d.proc_grid[0] << ", " << d.proc_grid[1] << ", " << d.proc_grid[2]
     << "])\n";
  */
  // os << "Domain in real space: [" << w.size()[0] << ", " << w.size()[1] << ", " << w.size()[2] << "]\n";
  // os << "Domain in complex space: [" << d.Lx_c << ", " << d.Ly_c << ", " << d.Lz_c << "]\n";
  /*
  for (int i = 0; i < d.get_num_domains(); i++) {
    const auto &in = d.real_boxes[i];
    const auto &out = d.complex_boxes[i];
    os << "Domain " << i + 1 << "/" << d.get_num_domains() << ": ";
    os << "[" << in.low[0] << ", " << in.low[1] << ", " << in.low[2] << "] x ";
    os << "[" << in.high[0] << ", " << in.high[1] << ", " << in.high[2] << "] (" << in.count() << " indexes) => ";
    os << "[" << out.low[0] << ", " << out.low[1] << ", " << out.low[2] << "] x ";
    os << "[" << out.high[0] << ", " << out.high[1] << ", " << out.high[2] << "] (" << out.count() << " indexes)\n";
  }
 */
  return os;
}

} // namespace pfc
