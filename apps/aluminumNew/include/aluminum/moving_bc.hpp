// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file aluminum/moving_bc.hpp
 * @brief Directional-solidification moving BC (tungsten / aluminum JSON App path)
 *
 * @details
 * This file defines the MovingBC class, which enforces a boundary condition
 * that moves based on the solidification front position. The boundary:
 * - Automatically detects the solid-liquid interface from a per-column
 *   statistic of the target field (see FrontTracker)
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <aluminum/solidification_bc_rules.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/field/operations.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace aluminum {

/**
 * @brief Which per-column statistic of the target field marks a column solid.
 *
 * Each x column (all y, z of one x) is reduced to one number and compared
 * with the tracker threshold; the front is the last solid column.
 * - `amplitude` (default): column maximum minus column mean, zero in a
 *   uniform melt and positive in a crystal, whatever the melt density level.
 *   The threshold is a fraction of the largest column amplitude in the
 *   domain (default 0.5), so the reservoir band, where the blend damps the
 *   crystal, and weak nuclei ahead of the front do not count as solid while
 *   a low-amplitude crystal still does. For density fields.
 * - `mean_field`: the column mean, for models whose solid and liquid differ
 *   in mean density or for a smooth field such as a concentration. Default
 *   threshold: the midpoint of `rho_high` and `rho_low`.
 * - `cutoff`: the column maximum above an absolute level, the historical
 *   rule with threshold 0.1; it breaks when the melt level shifts.
 */
enum class FrontTracker { cutoff, mean_field, amplitude };

inline const char *to_string(FrontTracker tracker) noexcept {
  switch (tracker) {
  case FrontTracker::cutoff: return "cutoff";
  case FrontTracker::mean_field: return "mean_field";
  case FrontTracker::amplitude: return "amplitude";
  }
  return "amplitude";
}

inline FrontTracker front_tracker_from_string(std::string_view name) {
  if (name == "cutoff") return FrontTracker::cutoff;
  if (name == "mean_field") return FrontTracker::mean_field;
  if (name == "amplitude") return FrontTracker::amplitude;
  throw std::invalid_argument("MovingBC: unknown front tracker mode '" +
                              std::string(name) +
                              "' (expected cutoff, mean_field or amplitude)");
}

class MovingBC : public pfc::FieldModifier {

private:
  double m_rho_low = 0.0, m_rho_high = 0.0;
  double m_xwidth = 15.0;
  double m_alpha = 1.0;
  double m_xpos = 0.0;
  FrontTracker m_tracker = FrontTracker::amplitude;
  std::optional<double> m_threshold;
  int m_idx = 0;
  double m_disp = 40.0;
  bool m_first = true;
  std::vector<double> xline, global_xline, xsum, global_xsum;
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

  void set_tracker(FrontTracker tracker) { m_tracker = tracker; }
  FrontTracker get_tracker() const { return m_tracker; }

  /// An explicit threshold replaces the tracker's default.
  void set_threshold(double threshold) { m_threshold = threshold; }
  bool has_explicit_threshold() const { return m_threshold.has_value(); }

  /**
   * @brief The threshold the tracker compares against.
   *
   * Explicit value if set; otherwise 0.5 for `amplitude` (a fraction of the
   * largest column amplitude), the midpoint of `rho_low` and `rho_high` for
   * `mean_field`, and the historical absolute 0.1 for `cutoff`.
   */
  double get_threshold() const {
    if (m_threshold) return *m_threshold;
    switch (m_tracker) {
    case FrontTracker::cutoff: return 0.1;
    case FrontTracker::amplitude: return 0.5;
    case FrontTracker::mean_field: return 0.5 * (m_rho_low + m_rho_high);
    }
    return 0.1;
  }

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
   * @brief Advance the front from the local column maxima and sums of the field.
   * @param local_max max over y and z of each local x column, box.low[0]
   *        first (what the host loop and the device reduction both produce)
   * @param local_sum sum over the same columns, same layout
   */
  void track_front(const std::vector<double> &local_max,
                   const std::vector<double> &local_sum, const pfc::Domain &domain,
                   const pfc::Box3i &box) {
    const auto Lx = pfc::domain::get_size(domain, 0);
    const auto dx = pfc::domain::get_spacing(domain, 0);
    const auto x0 = pfc::domain::get_origin(domain, 0);
    if (local_sum.size() != local_max.size()) {
      throw std::invalid_argument(
          "MovingBC::track_front: column max and sum lengths differ");
    }
    const double threshold = get_threshold();

    if (xline.size() != static_cast<std::size_t>(Lx)) {
      xline.resize(Lx);
      xsum.resize(Lx);
      // Receive buffers are ignored on non-root ranks, but a valid length-Lx
      // buffer avoids undefined behavior from empty-vector data() on some MPI
      // stacks.
      global_xline.resize(Lx);
      global_xsum.resize(Lx);
    }
    fill(xline.begin(), xline.end(), std::numeric_limits<double>::lowest());
    fill(xsum.begin(), xsum.end(), 0.0);
    for (std::size_t i = 0; i < local_max.size(); ++i) {
      xline[box.low[0] + i] = local_max[i];
      xsum[box.low[0] + i] = local_sum[i];
    }

    if (xline.empty() ||
        xline.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "MovingBC::apply: MPI_Reduce count must be in (0, INT_MAX]");
    }

    // Fail closed before any root m_idx mutation from the reduced lines.
    pfc::mpi::throw_on_mpi_error(MPI_Reduce(xline.data(), global_xline.data(),
                                            static_cast<int>(xline.size()),
                                            MPI_DOUBLE, MPI_MAX, 0, comm),
                                 "MPI_Reduce");
    pfc::mpi::throw_on_mpi_error(MPI_Reduce(xsum.data(), global_xsum.data(),
                                            static_cast<int>(xsum.size()),
                                            MPI_DOUBLE, MPI_SUM, 0, comm),
                                 "MPI_Reduce");

    if (rank == 0) {
      // Cells per global x column: the mean over y and z.
      const double column_cells =
          static_cast<double>(pfc::domain::get_size(domain, 1)) *
          static_cast<double>(pfc::domain::get_size(domain, 2));
      // Amplitude is judged relative to the strongest column, so the band
      // (a damped crystal) and weak nuclei ahead stay liquid.
      double largest_amplitude = 0.0;
      for (int i = 0; i < Lx; ++i) {
        largest_amplitude = std::max(
            largest_amplitude, global_xline[i] - global_xsum[i] / column_cells);
      }
      // A uniform field differs from its mean only by rounding; that is no
      // crystal.
      const double amplitude_floor =
          1e-9 * std::max({1.0, std::abs(m_rho_low), std::abs(m_rho_high)});
      const auto solid = [&](int i) {
        const double max = global_xline[i];
        const double mean = global_xsum[i] / column_cells;
        switch (m_tracker) {
        case FrontTracker::cutoff: return max > threshold;
        case FrontTracker::mean_field: return mean > threshold;
        case FrontTracker::amplitude:
          return largest_amplitude > amplitude_floor &&
                 max - mean > threshold * largest_amplitude;
        }
        return false;
      };
      if (m_first) {
        for (int i = static_cast<int>(global_xline.size()) - 1; i >= 0; i--) {
          if (solid(i)) {
            m_idx = i;
            break;
          }
        }
      } else {
        // Advance at most one domain period; if every column stays solid
        // (e.g. uniform supersaturated field), an unbounded loop would hang
        // the whole simulation.
        int scanned = 0;
        while (solid(m_idx % Lx) && scanned < Lx) {
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
    std::vector<double> local_sum(nx, 0.0);
    long int idx = 0;
    for (int k = box.low[2]; k <= box.high[2]; k++) {
      for (int j = box.low[1]; j <= box.high[1]; j++) {
        for (int i = 0; i < nx; i++) {
          const double v = field[idx++];
          local_max[i] = std::max(local_max[i], v);
          local_sum[i] += v;
        }
      }
    }
    track_front(local_max, local_sum, domain, box);
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
