// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file adaptive_controller.hpp
 * @brief AdaptiveTimeController: error evidence → accept/reject + next dt.
 *
 * @details
 * Closes the adaptive chain: `ErrorEvidence` (or an embedded pair) →
 * `normalize_error_evidence` → `AdaptiveControlConfig` → `Time` attempt
 * transactions. Does not own the stepper or the field.
 *
 * Normalized metric, for each field or component,
 * `e / (atol + rtol * solution_scale)`. `rtol` is not taken relative to 1
 * unless the evidence says `SolutionScale::Unit`. The embedded helper uses
 * `solution_scale = max(|y_accepted|, |y_candidate|)` on each component and
 * reduces the component-wise maximum with `MPI_MAX`. Field-norm evidence
 * instead reduces each field's norm and scale with `MPI_MAX`, which is
 * `||e||_inf / (atol + rtol * ||y||_inf)` for that field.
 *
 * Memoryless controller (`StepController::memoryless`, the default):
 * \verbatim
 * factor = safety_factor * metric^(-1/error_order)
 * \endverbatim
 * `error_order` is the exponent in that local-error model. For an embedded
 * pair it is the order of the difference (3 for Bogacki–Shampine 3(2)).
 *
 * PI controller (`StepController::pi`), used only when this step will be
 * accepted and a previous accepted metric exists:
 * \verbatim
 * factor = safety_factor
 *          * metric^(-(pi_k_i + pi_k_p) / error_order)
 *          * previous^(pi_k_p / error_order)
 * \endverbatim
 * Defaults `(pi_k_i, pi_k_p) = (0.3, 0.4)` are the PI.3.4 gains. When the
 * two metrics are equal the same law reduces to
 * \verbatim
 * factor = safety_factor * metric^(-pi_k_i/error_order)
 * \endverbatim
 * which is a milder integral controller, not the memoryless law. The first
 * acceptance and every rejection use the memoryless law. Gustafsson's PI.3.4
 * and Söderlind's equation (4.21) (2001, Automatic control and adaptive
 * time-stepping) write an extra factor `theta` to the power
 * (`pi_k_i` / `error_order`), with `theta = 0.8`. That factor is not applied
 * here. `safety_factor` is the only safety multiplier, and it sits outside
 * the powers, as in the memoryless law.
 *
 * `factor` is then clamped to `[shrink_max, growth_max]` and
 * `next_dt = clamp(attempted_dt * factor, min_dt, max_dt)`.
 * A non-positive finite metric uses `growth_max`. A non-finite metric uses
 * `shrink_max`. Invalid evidence is a rejection that proposes `dt` from a
 * synthetic metric of 2 and does not advance accepted time.
 *
 * PI history changes only in `apply`. An accepted PI step with a positive
 * finite metric stores that metric. An accepted metric of 0 clears it,
 * because the logarithm of the metric is undefined. A rejection leaves the
 * stored metric unchanged. Fixed mode neither reads nor writes it. The
 * sequential-rejection streak is separate: an acceptance clears it, and
 * reaching `max_sequential_rejections` throws. `capture_state` /
 * `restore_state` persist the metric and the streak because both affect a
 * later `dt`. `reset_history` clears only the metric.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/integrator/error_evidence.hpp>
#include <openpfc/kernel/simulation/adaptive_control_config.hpp>
#include <openpfc/kernel/simulation/time.hpp>

namespace pfc::sim {

/**
 * @brief Outcome of one controller decision (no Time mutation).
 */
struct AdaptiveDecision {
  bool accepted{false};
  double next_dt{0.0};
  double metric{0.0};
  bool decision_available{true};
};

/**
 * @brief Controller memory that changes a later `dt`.
 *
 * The previous accepted metric is the PI state. The rejection streak feeds
 * the sequential-rejection cap. The two counters are diagnostic.
 */
struct AdaptiveControllerState {
  bool has_previous_metric{false};
  double previous_accepted_metric{0.0};
  int sequential_rejections{0};
  int accepted_count{0};
  int rejected_count{0};
};

/**
 * @brief Policy object that turns error evidence into a Time commit/reject.
 */
class AdaptiveTimeController {
public:
  /**
   * @param cfg          Validated adaptive-control policy.
   * @param error_order  Positive order `k` in `metric^(-1/k)`.
   * @throws std::invalid_argument if `cfg` fails `validate` or order < 1.
   */
  explicit AdaptiveTimeController(AdaptiveControlConfig cfg, int error_order = 3)
      : m_cfg(std::move(cfg)), m_error_order(error_order) {
    const auto result = validate(m_cfg);
    if (!result.ok()) {
      throw std::invalid_argument(result.format());
    }
    if (m_error_order < 1) {
      throw std::invalid_argument(
          "AdaptiveTimeController: error_order must be >= 1");
    }
  }

  [[nodiscard]] const AdaptiveControlConfig &config() const noexcept {
    return m_cfg;
  }

  [[nodiscard]] int error_order() const noexcept { return m_error_order; }

  [[nodiscard]] int accepted_count() const noexcept { return m_accepted; }

  [[nodiscard]] int rejected_count() const noexcept { return m_rejected; }

  [[nodiscard]] int sequential_rejections() const noexcept {
    return m_sequential_rejections;
  }

  [[nodiscard]] bool has_previous_accepted_metric() const noexcept {
    return m_has_previous;
  }

  [[nodiscard]] double previous_accepted_metric() const noexcept {
    return m_previous_metric;
  }

  /**
   * @brief Drop the previous accepted metric.
   *
   * The rejection streak and the diagnostic counters stay. The next PI
   * decision uses the memoryless law until another step is accepted.
   */
  void reset_history() noexcept {
    m_has_previous = false;
    m_previous_metric = 0.0;
  }

  [[nodiscard]] AdaptiveControllerState capture_state() const noexcept {
    return AdaptiveControllerState{
        .has_previous_metric = m_has_previous,
        .previous_accepted_metric = m_previous_metric,
        .sequential_rejections = m_sequential_rejections,
        .accepted_count = m_accepted,
        .rejected_count = m_rejected,
    };
  }

  /**
   * @brief Replace controller memory.
   *
   * @throws std::invalid_argument if the streak or counters are negative, or
   *         a stored metric is missing, non-finite, or not strictly positive.
   */
  void restore_state(const AdaptiveControllerState &state) {
    if (state.sequential_rejections < 0 || state.accepted_count < 0 ||
        state.rejected_count < 0) {
      throw std::invalid_argument(
          "AdaptiveTimeController::restore_state: counts must be non-negative");
    }
    if (state.has_previous_metric &&
        (!std::isfinite(state.previous_accepted_metric) ||
         state.previous_accepted_metric <= 0.0)) {
      throw std::invalid_argument(
          "AdaptiveTimeController::restore_state: previous metric must be "
          "finite and > 0");
    }
    m_has_previous = state.has_previous_metric;
    m_previous_metric =
        state.has_previous_metric ? state.previous_accepted_metric : 0.0;
    m_sequential_rejections = state.sequential_rejections;
    m_accepted = state.accepted_count;
    m_rejected = state.rejected_count;
  }

  /**
   * @brief Decide accept/reject and next dt from method-independent evidence.
   *
   * Reduces rank-local evidence, then normalizes. `NoDecision` is a reject
   * that shrinks dt. Does not read PI history in fixed mode, and does not
   * write history in any mode: `apply` does that.
   */
  [[nodiscard]] AdaptiveDecision decide(double attempted_dt,
                                        const pfc::integrator::ErrorEvidence &ev,
                                        MPI_Comm comm = MPI_COMM_WORLD) const {
    if (attempted_dt <= 0.0) {
      throw std::invalid_argument(
          "AdaptiveTimeController::decide: attempted_dt must be > 0");
    }
    if (m_cfg.mode == AdaptiveControlMode::fixed) {
      AdaptiveDecision d;
      d.accepted = true;
      d.next_dt = attempted_dt;
      d.metric = 0.0;
      d.decision_available = true;
      return d;
    }

    const auto reduced = pfc::integrator::reduce_error_evidence(ev, comm);
    pfc::integrator::ErrorTolerances tol;
    tol.absolute = m_cfg.atol;
    tol.relative = m_cfg.rtol;
    if (!m_cfg.atol_per_field.empty()) {
      tol.absolute_per_field = m_cfg.atol_per_field;
    }
    if (!m_cfg.rtol_per_field.empty()) {
      tol.relative_per_field = m_cfg.rtol_per_field;
    }
    const auto n = pfc::integrator::normalize_error_evidence(reduced, tol);
    return finish(attempted_dt, n.metric, n.decision_available);
  }

  /**
   * @brief Decide from an embedded pair, scaled by the solution.
   *
   * For each component, `scale_i = max(|accepted_i|, |candidate_i|)` and
   * `ratio_i = |error_i| / (atol_i + rtol_i * scale_i)`. The metric is the
   * maximum ratio. Ranks reduce that scalar with `MPI_MAX`, so every rank
   * decides from the same metric and the same attempted `dt`. A relative
   * tolerance is never taken against the constant 1 on this path.
   *
   * Per-field tolerance vectors, when set, must have the same length as
   * `error`. A length mismatch, a non-finite entry, or unequal span lengths
   * fails closed.
   */
  [[nodiscard]] AdaptiveDecision
  decide_from_embedded_error(double attempted_dt, std::span<const double> error,
                             std::span<const double> accepted,
                             std::span<const double> candidate,
                             MPI_Comm comm = MPI_COMM_WORLD) const {
    if (attempted_dt <= 0.0) {
      throw std::invalid_argument(
          "AdaptiveTimeController::decide_from_embedded_error: attempted_dt "
          "must be > 0");
    }
    if (m_cfg.mode == AdaptiveControlMode::fixed) {
      AdaptiveDecision d;
      d.accepted = true;
      d.next_dt = attempted_dt;
      d.metric = 0.0;
      d.decision_available = true;
      return d;
    }

    double local_metric = 0.0;
    int local_ok = 1;
    int local_inf = 0;
    // An empty span is a rank with no local components: metric 0, still valid.
    // Scalar tolerances apply to every component. A per-field vector must
    // match this span; decomposed DOFs should use the scalar tolerances and
    // leave the per-field vectors for `decide` on one norm per field.
    const bool lengths_ok =
        error.size() == accepted.size() && error.size() == candidate.size();
    const bool atol_ok =
        m_cfg.atol_per_field.empty() || m_cfg.atol_per_field.size() == error.size();
    const bool rtol_ok =
        m_cfg.rtol_per_field.empty() || m_cfg.rtol_per_field.size() == error.size();
    if (!lengths_ok || !atol_ok || !rtol_ok) {
      local_ok = 0;
    } else {
      for (std::size_t i = 0; i < error.size(); ++i) {
        const double err = error[i];
        const double ya = accepted[i];
        const double yc = candidate[i];
        if (!std::isfinite(err) || !std::isfinite(ya) || !std::isfinite(yc)) {
          local_ok = 0;
          break;
        }
        const double atol =
            m_cfg.atol_per_field.empty() ? m_cfg.atol : m_cfg.atol_per_field[i];
        const double rtol =
            m_cfg.rtol_per_field.empty() ? m_cfg.rtol : m_cfg.rtol_per_field[i];
        if (!std::isfinite(atol) || atol < 0.0 || !std::isfinite(rtol) ||
            rtol < 0.0) {
          local_ok = 0;
          break;
        }
        const double scale = std::max(std::abs(ya), std::abs(yc));
        const double den = atol + rtol * scale;
        if (!(den > 0.0)) {
          local_inf = 1;
          continue;
        }
        local_metric = std::max(local_metric, std::abs(err) / den);
      }
    }

    int size = 1;
    int mpi_err = MPI_Comm_size(comm, &size);
    pfc::mpi::throw_on_mpi_error(mpi_err,
                                 "MPI_Comm_size in decide_from_embedded_error");
    if (size > 1) {
      int ok_all = 0;
      int inf_any = 0;
      double metric_all = 0.0;
      mpi_err = MPI_Allreduce(&local_ok, &ok_all, 1, MPI_INT, MPI_LAND, comm);
      pfc::mpi::throw_on_mpi_error(mpi_err, "MPI_Allreduce for embedded validity");
      mpi_err = MPI_Allreduce(&local_inf, &inf_any, 1, MPI_INT, MPI_LOR, comm);
      pfc::mpi::throw_on_mpi_error(mpi_err,
                                   "MPI_Allreduce for embedded infinite error");
      mpi_err =
          MPI_Allreduce(&local_metric, &metric_all, 1, MPI_DOUBLE, MPI_MAX, comm);
      pfc::mpi::throw_on_mpi_error(mpi_err, "MPI_Allreduce for embedded metric");
      local_ok = ok_all;
      local_inf = inf_any;
      local_metric = metric_all;
    }

    if (local_ok == 0) {
      return finish(attempted_dt, std::numeric_limits<double>::quiet_NaN(), false);
    }
    if (local_inf != 0) {
      return finish(attempted_dt, std::numeric_limits<double>::infinity(), true);
    }
    return finish(attempted_dt, local_metric, true);
  }

  /**
   * @brief Commit or reject the open Time attempt and install `next_dt`.
   *
   * Updates PI history only for an accepted adaptive PI step. Fixed mode
   * does not read or write that metric. A rejection does not replace it.
   *
   * @throws std::logic_error if no Time attempt is active.
   * @throws std::runtime_error if sequential rejections hit the configured cap.
   */
  void apply(Time &time, const AdaptiveDecision &d) {
    if (!time.attempt_active()) {
      throw std::logic_error(
          "AdaptiveTimeController::apply: Time has no active attempt");
    }
    if (d.accepted && d.decision_available) {
      time.commit_attempt();
      time.increment_step_success();
      time.set_dt(d.next_dt);
      ++m_accepted;
      m_sequential_rejections = 0;
      note_accepted_metric(d.metric);
      return;
    }
    time.reject_attempt();
    time.increment_step_rejection();
    time.set_dt(d.next_dt);
    ++m_rejected;
    ++m_sequential_rejections;
    if (m_sequential_rejections >= m_cfg.max_sequential_rejections) {
      throw std::runtime_error(
          "AdaptiveTimeController: sequential rejections reached " +
          std::to_string(m_cfg.max_sequential_rejections));
    }
  }

private:
  [[nodiscard]] AdaptiveDecision finish(double attempted_dt, double metric,
                                        bool available) const {
    AdaptiveDecision d;
    d.metric = metric;
    d.decision_available = available;
    if (!available) {
      d.accepted = false;
      d.next_dt = propose_dt(attempted_dt, /*metric=*/2.0, /*accepted_step=*/false);
      return d;
    }
    d.accepted = std::isfinite(metric) && metric <= 1.0;
    d.next_dt = propose_dt(attempted_dt, metric, d.accepted);
    return d;
  }

  [[nodiscard]] double propose_dt(double attempted_dt, double metric,
                                  bool accepted_step) const {
    double factor = m_cfg.growth_max;
    if (std::isfinite(metric) && metric > 0.0) {
      const double order = static_cast<double>(m_error_order);
      const bool use_pi = accepted_step && m_cfg.controller == StepController::pi &&
                          m_has_previous && std::isfinite(m_previous_metric) &&
                          m_previous_metric > 0.0;
      if (use_pi) {
        factor = m_cfg.safety_factor *
                 std::pow(metric, -(m_cfg.pi_k_i + m_cfg.pi_k_p) / order) *
                 std::pow(m_previous_metric, m_cfg.pi_k_p / order);
      } else {
        factor = m_cfg.safety_factor * std::pow(metric, -1.0 / order);
      }
    } else if (!std::isfinite(metric)) {
      factor = m_cfg.shrink_max;
    }
    factor = std::min(m_cfg.growth_max, std::max(m_cfg.shrink_max, factor));
    const double next = attempted_dt * factor;
    return std::min(m_cfg.max_dt, std::max(m_cfg.min_dt, next));
  }

  void note_accepted_metric(double metric) noexcept {
    if (m_cfg.mode == AdaptiveControlMode::fixed ||
        m_cfg.controller != StepController::pi) {
      return;
    }
    if (std::isfinite(metric) && metric > 0.0) {
      m_has_previous = true;
      m_previous_metric = metric;
      return;
    }
    if (std::isfinite(metric) && metric == 0.0) {
      m_has_previous = false;
      m_previous_metric = 0.0;
    }
  }

  AdaptiveControlConfig m_cfg{};
  int m_error_order{3};
  int m_accepted{0};
  int m_rejected{0};
  int m_sequential_rejections{0};
  bool m_has_previous{false};
  double m_previous_metric{0.0};
};

} // namespace pfc::sim
