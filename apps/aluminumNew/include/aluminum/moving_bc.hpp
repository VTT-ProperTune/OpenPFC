// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file aluminum/moving_bc.hpp
 * @brief Directional-solidification moving BC (tungsten / aluminum JSON App path)
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
 * auto bc = std::make_unique<aluminum::MovingBC>();
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

#pragma once
// Deliberate app-owned copy: these are solidification-model rules, not a
// physics-agnostic OpenPFC API. Independently buildable apps cannot include a
// sibling app. Share them only after extraction to their model's software owner;
// until then retain parity with the other solidification app's boundary rules.

#include <algorithm>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <aluminum/solidification_bc_rules.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace aluminum {

class MovingBC : public pfc::FieldModifier {

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
  int rank = pfc::mpi::get_comm_rank(comm);
  int size = pfc::mpi::get_comm_size(comm);
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

  /// Unwrapped front position/index and first-detection flag (no scratch buffers).
  [[nodiscard]] nlohmann::json checkpoint_state() const override {
    return {{"xpos", m_xpos}, {"idx", m_idx}, {"first", m_first}};
  }

  /// Missing or malformed front state cannot reproduce a moving-BC restart.
  void restore_checkpoint_state(const nlohmann::json &state) override {
    if (!state.is_object() || !state.contains("xpos") ||
        !state["xpos"].is_number() || !state.contains("idx") ||
        !state["idx"].is_number_integer() || state["idx"] < 0 ||
        state["idx"] > std::numeric_limits<int>::max() || !state.contains("first") ||
        !state["first"].is_boolean()) {
      throw std::invalid_argument(
          "MovingBC checkpoint: missing or invalid front state (xpos, idx, first)");
    }
    const double xpos = state["xpos"].get<double>();
    if (!std::isfinite(xpos)) {
      throw std::invalid_argument("MovingBC checkpoint: xpos must be finite");
    }
    m_xpos = xpos;
    m_idx = state["idx"].get<int>();
    m_first = state["first"].get<bool>();
  }

  void set_mpi_comm(MPI_Comm c) noexcept override {
    comm = c;
    rank = pfc::mpi::get_comm_rank(comm);
    size = pfc::mpi::get_comm_size(comm);
  }

  /**
   * @brief Advance the front from the local column maxima of the field.
   * @param local_max max over y and z of each local x column, box.low[0]
   *        first (what the host loop and the device reduction both produce)
   */
  void track_front(const std::vector<double> &local_max, const pfc::Domain &domain,
                   const pfc::Box3i &box) {
    const auto Lx = pfc::domain::get_size(domain, 0);
    const auto dx = pfc::domain::get_spacing(domain, 0);
    const auto x0 = pfc::domain::get_origin(domain, 0);

    if (xline.size() != static_cast<std::size_t>(Lx)) {
      xline.resize(Lx);
      // Receive buffer is ignored on non-root ranks, but a valid length-Lx buffer
      // avoids undefined behavior from empty-vector data() on some MPI stacks.
      global_xline.resize(Lx);
    }
    fill(xline.begin(), xline.end(), std::numeric_limits<double>::lowest());
    for (std::size_t i = 0; i < local_max.size(); ++i) {
      xline[box.low[0] + i] = local_max[i];
    }

    if (xline.empty() ||
        xline.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MovingBC::apply: MPI_Reduce count must be in (0, INT_MAX]");
    }

    // Fail closed before any root m_idx mutation from global_xline.
    pfc::mpi::throw_on_mpi_error(MPI_Reduce(xline.data(), global_xline.data(),
                                            static_cast<int>(xline.size()),
                                            MPI_DOUBLE, MPI_MAX, 0, comm),
                                 "MPI_Reduce");

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

    // Keep the unwrapped index identical on every rank for restart capture.
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&m_idx, 1, MPI_INT, 0, comm),
                                 "MPI_Bcast MovingBC index");
    double new_xpos = x0 + (m_idx * dx) + m_disp;
    m_xpos = std::max(new_xpos, m_xpos);
    // Fail closed before fill_bc so ranks never apply a divergent m_xpos.
    pfc::mpi::throw_on_mpi_error(MPI_Bcast(&m_xpos, 1, MPI_DOUBLE, 0, comm),
                                 "MPI_Bcast");

    if (m_first) {
      m_first = false;
    }

    if (rank == 0) {
      const pfc::Logger lg{pfc::LogLevel::Debug, 0};
      std::ostringstream oss;
      oss << "Boundary position: " << m_xpos;
      pfc::log_debug(lg, oss.str());
    }
  }

  /// The band around the current front, with its periodic images.
  [[nodiscard]] MovingBandRule band_rule(const pfc::Domain &domain) const {
    // Wrap in coordinates relative to the lower x limit, so the periodic
    // images of the front are found for any box placement.
    const double x0 = pfc::domain::lower(domain)[0];
    const double l = pfc::domain::extent(domain)[0];
    double xpos = std::fmod(m_xpos - x0, l);
    if (xpos < 0.0) xpos += l;
    return {
        .blend = {.alpha = m_alpha, .rho_low = m_rho_low, .rho_high = m_rho_high},
        .x0 = x0,
        .l = l,
        .xpos = xpos,
        .xwidth = m_xwidth};
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double time = 0.0) override {
    (void)time;
    const int nx = box.size[0];
    std::vector<double> local_max(nx, std::numeric_limits<double>::lowest());
    long int idx = 0;
    for (int k = box.low[2]; k <= box.high[2]; k++) {
      for (int j = box.low[1]; j <= box.high[1]; j++) {
        for (int i = 0; i < nx; i++) {
          local_max[i] = std::max(local_max[i], field[idx++]);
        }
      }
    }
    track_front(local_max, domain, box);
    fill_bc(field, domain, box);
  }

  void fill_bc(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
               const pfc::Box3i &box) {
    const MovingBandRule r = band_rule(domain);
    pfc::field::apply_inplace(field, domain, box,
                              [r](const pfc::Real3 &X, double current) {
                                return r(X[0], X[1], X[2], current);
                              });
  }
};

} // namespace aluminum
