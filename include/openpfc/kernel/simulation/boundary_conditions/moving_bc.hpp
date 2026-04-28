// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file moving_bc.hpp
 * @brief Moving boundary condition that tracks solidification front
 *
 * @details
 * This file defines the MovingBC class, which enforces a boundary condition
 * that moves based on the solidification front position. The boundary:
 * - Automatically detects the solid-liquid interface
 * - Moves the boundary region to follow the interface
 * - Maintains fixed values ahead of and behind the front
 *
 * Useful for:
 * - Directional solidification simulations
 * - Moving frame of reference
 * - Efficient domain usage (only simulate near interface)
 *
 * Usage:
 * @code
 * auto bc = std::make_unique<pfc::MovingBC>();
 * bc->set_rho_low(0.0);
 * bc->set_rho_high(1.0);
 * bc->set_field_name("density");
 * simulator.add_boundary_condition(std::move(bc));
 * @endcode
 *
 * @see field_modifier.hpp for base class
 * @see fixed_bc.hpp for stationary boundary condition
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP
#define PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <sstream>

#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace pfc {

class MovingBC : public FieldModifier {

private:
  double m_rho_low, m_rho_high;
  double m_xwidth = 15.0;
  double m_alpha = 1.0;
  double m_xpos = 0.0;
  double m_threshold = 0.1;
  int m_idx = 0;
  double m_disp = 40.0;
  bool m_first = true;
  std::vector<double> xline, global_xline;
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = mpi::get_comm_rank(comm);
  int size = mpi::get_comm_size(comm);
  std::string m_name = "MovingBC";

public:
  MovingBC() = default;
  MovingBC(double rho_low, double rho_high)
      : m_rho_low(rho_low), m_rho_high(rho_high) {}

  void set_rho_low(double rho_low) { m_rho_low = rho_low; }
  void set_rho_high(double rho_high) { m_rho_high = rho_high; }

  void set_xpos(double xpos) { m_xpos = xpos; }
  double get_xpos() const { return m_xpos; }

  void set_xwidth(double xwidth) { m_xwidth = xwidth; }
  double get_xwidth() const { return m_xwidth; }

  void set_alpha(double alpha) { m_alpha = alpha; }

  void set_disp(double disp) { m_disp = disp; }

  void set_threshold(double threshold) { m_threshold = threshold; }
  double get_threshold() const { return m_threshold; }

  const std::string &get_modifier_name() const override { return m_name; }

  void set_mpi_comm(MPI_Comm c) noexcept override {
    comm = c;
    rank = mpi::get_comm_rank(comm);
    size = mpi::get_comm_size(comm);
  }

  void apply(Model &m, double time) override {
    (void)time;
    const fft::IFFT &fft = get_fft(m);
    Field &field = get_real_field(m, get_field_name());
    const World &w = get_world(m);
    Int3 low = get_inbox(fft).low;
    Int3 high = get_inbox(fft).high;

    auto Lx = get_size(w, 0);
    auto dx = get_spacing(w, 0);
    auto x0 = get_origin(w, 0);

    if (m_first) {
      xline.resize(Lx);
      // Receive buffer is ignored on non-root ranks, but a valid length-Lx buffer
      // avoids undefined behavior from empty-vector data() on some MPI stacks.
      global_xline.resize(Lx);
    }

    fill(xline.begin(), xline.end(), std::numeric_limits<double>::min());

    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          xline[i] = std::max(xline[i], field[idx++]);
        }
      }
    }

    MPI_Reduce(xline.data(), global_xline.data(), static_cast<int>(xline.size()),
               MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
      if (m_first) {
        for (int i = static_cast<int>(global_xline.size()) - 1; i >= 0; i--) {
          if (global_xline[i] > m_threshold) {
            m_idx = i;
            break;
          }
        }
      } else {
        // Advance at most one domain period; if every column stays above the
        // threshold (e.g. uniform supersaturated field), an unbounded loop would
        // hang the whole simulation.
        int scanned = 0;
        while (global_xline[m_idx % Lx] > m_threshold && scanned < Lx) {
          m_idx += 1;
          scanned += 1;
        }
      }
    }

    double new_xpos = x0 + (m_idx * dx) + m_disp;
    m_xpos = std::max(new_xpos, m_xpos);
    MPI_Bcast(&m_xpos, 1, MPI_DOUBLE, 0, comm);

    if (m_first) {
      m_first = false;
    }

    if (rank == 0) {
      const Logger lg{LogLevel::Debug, 0};
      std::ostringstream oss;
      oss << "Boundary position: " << m_xpos;
      log_debug(lg, oss.str());
    }

    fill_bc(m);
  }

  void fill_bc(Model &m) {
    const World &w = get_world(m);
    const double Lx = get_size(w, 0);
    const double dx = get_spacing(w, 0);
    const double l = Lx * dx;
    const double xpos = std::fmod(m_xpos, l);
    const double xwidth = m_xwidth;
    const double alpha = m_alpha;

    pfc::field::apply_inplace(
        m, get_field_name(), [=](const pfc::Real3 &X, double current) {
          const double x = X[0];
          const double dist = x - xpos;
          auto blend = [&](double d) {
            const double S = 1.0 / (1.0 + std::exp(-alpha * d));
            return (m_rho_low * S) + (m_rho_high * (1.0 - S));
          };

          if (std::abs(dist) < xwidth) {
            return blend(dist);
          }
          if (xpos < xwidth && std::abs(dist - l) < xwidth) {
            return blend(dist - l);
          }
          if (xpos > l - xwidth && std::abs(dist + l) < xwidth) {
            return blend(dist + l);
          }
          return current; // outside transition bands, keep value
        });
  }
};

} // namespace pfc

#endif // PFC_BOUNDARY_CONDITIONS_MOVING_BC_HPP
