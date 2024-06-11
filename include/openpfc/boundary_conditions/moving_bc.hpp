/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#ifndef PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP
#define PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP

#include <cmath>
#include <limits>
#include <mpi.h>

#include "../field_modifier.hpp"
#include "../utils.hpp"

namespace pfc {

class MovingBC : public FieldModifier {

private:
  double m_rho_low, m_rho_high;
  double m_xwidth = 15.0;
  double m_alpha = 1.0;
  double m_xpos = 0.0;
  int m_idx = 0;
  double m_disp = 40.0;
  bool m_first = true;
  std::vector<double> xline, global_xline;
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = mpi::get_comm_rank(comm);
  int size = mpi::get_comm_size(comm);

public:
  MovingBC() = default;
  MovingBC(double rho_low, double rho_high) : m_rho_low(rho_low), m_rho_high(rho_high) {}

  void set_rho_low(double rho_low) { m_rho_low = rho_low; }
  void set_rho_high(double rho_high) { m_rho_high = rho_high; }

  void set_xpos(double xpos) { m_xpos = xpos; }

  void set_xwidth(double xwidth) { m_xwidth = xwidth; }
  double get_xwidth() const { return m_xwidth; }

  void set_alpha(double alpha) { m_alpha = alpha; }

  void set_disp(double disp) { m_disp = disp; }

  void apply(Model &m, double) override {
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    const World &w = m.get_world();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    auto Lx = w.Lx;

    if (m_first) {
      xline.resize(Lx);
      if (rank == 0) global_xline.resize(Lx);
    }

    fill(xline.begin(), xline.end(), std::numeric_limits<double>::min());

    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++)
      for (int j = low[1]; j <= high[1]; j++)
        for (int i = low[0]; i <= high[0]; i++) xline[i] = std::max(xline[i], field[idx++]);

    MPI_Reduce(xline.data(), global_xline.data(), xline.size(), MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
      if (m_first) {
        for (int i = global_xline.size() - 1; i >= 0; i--) {
          if (global_xline[i] > 0.1) {
            m_idx = i;
            break;
          }
        }
      } else {
        while (global_xline[m_idx % w.Lx] > 0.1) {
          m_idx += 1;
        }
      }
    }

    double new_xpos = w.x0 + m_idx * w.dx + m_disp;
    if (new_xpos > m_xpos) {
      m_xpos = new_xpos;
    }
    MPI_Bcast(&m_xpos, 1, MPI_DOUBLE, 0, comm);

    if (m_first) {
      m_first = false;
    }

    fill_bc(m);
  }

  void fill_bc(Model &m) {
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_real_field(get_field_name());
    const World &w = m.get_world();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;
    double l = w.Lx * w.dx;
    double xpos = fmod(m_xpos, l);
    double xwidth = m_xwidth;
    double alpha = m_alpha;
    long int idx = 0;

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          double dist = x - xpos;
          if (std::abs(dist) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * dist));
            field[idx] = m_rho_low * S + m_rho_high * (1.0 - S);
          }
          if (xpos < xwidth && std::abs(dist - l) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * (dist - l)));
            field[idx] = m_rho_low * S + m_rho_high * (1.0 - S);
          }
          if (xpos > l - xwidth && std::abs(dist + l) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * (dist + l)));
            field[idx] = m_rho_low * S + m_rho_high * (1.0 - S);
          }
          idx += 1;
        }
      }
    }
  }
};

} // namespace pfc

#endif // PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP
