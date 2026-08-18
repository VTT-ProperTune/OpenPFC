// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file adaptive_controller.hpp
 * @brief AdaptiveTimeController: error evidence → accept/reject + next dt.
 *
 * @details
 * Closes the M6 adaptive chain: `ErrorEvidence` (or an embedded-error vector)
 * → `normalize_error_evidence` → `AdaptiveControlConfig` scale/bounds →
 * `Time` attempt transactions. Does **not** own the stepper or the field.
 *
 * Next-`dt` proposal (adaptive mode):
 * `factor = clamp(safety * metric^(-1/order), shrink_max, growth_max)`,
 * then `next_dt = clamp(attempted_dt * factor, min_dt, max_dt)`.
 * Fixed mode always accepts and keeps `attempted_dt`.
 *
 * `apply` commits or rejects the open `Time` attempt, updates `Time::dt`,
 * and bumps Time plus controller accept/reject counters. Sequential
 * rejections at `max_sequential_rejections` throw.
 */

#include <algorithm>
#include <cmath>
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
 * @brief Policy object that turns error evidence into a Time commit/reject.
 */
class AdaptiveTimeController {
public:
  /**
   * @param cfg          Validated adaptive-control policy.
   * @param error_order  Positive order used in `metric^(-1/order)` (e.g. 3
   *                     for Bogacki–Shampine 3(2)).
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

  /**
   * @brief Decide accept/reject and next dt from method-independent evidence.
   *
   * Reduces rank-local evidence, then normalizes. `NoDecision` is treated as
   * a reject that shrinks dt (time must not advance).
   */
  [[nodiscard]] AdaptiveDecision decide(double attempted_dt,
                                        const pfc::integrator::ErrorEvidence &ev,
                                        MPI_Comm comm = MPI_COMM_WORLD) const {
    if (attempted_dt <= 0.0) {
      throw std::invalid_argument(
          "AdaptiveTimeController::decide: attempted_dt must be > 0");
    }
    AdaptiveDecision d;
    if (m_cfg.mode == AdaptiveControlMode::fixed) {
      d.accepted = true;
      d.next_dt = attempted_dt;
      d.metric = 0.0;
      d.decision_available = true;
      return d;
    }

    const auto reduced = pfc::integrator::reduce_error_evidence(ev, comm);
    const pfc::integrator::ErrorTolerances tol{m_cfg.atol, m_cfg.rtol};
    const auto n = pfc::integrator::normalize_error_evidence(reduced, tol);
    d.metric = n.metric;
    d.decision_available = n.decision_available;
    if (!n.decision_available) {
      d.accepted = false;
      d.next_dt = propose_dt(attempted_dt, /*metric=*/2.0);
      return d;
    }
    d.accepted =
        (n.verdict == pfc::integrator::StepAttemptVerdict::Accept);
    d.next_dt = propose_dt(attempted_dt, n.metric);
    return d;
  }

  /**
   * @brief Decide from an embedded-error vector (`|u_high - u_low|`).
   *
   * Uses the max-abs entry as the single field norm.
   */
  [[nodiscard]] AdaptiveDecision
  decide_from_embedded_error(double attempted_dt,
                             std::span<const double> error,
                             MPI_Comm comm = MPI_COMM_WORLD) const {
    double max_abs = 0.0;
    for (double e : error) {
      max_abs = std::max(max_abs, std::abs(e));
    }
    const double norms[1] = {max_abs};
    const auto ev = pfc::integrator::make_embedded_pair_evidence(
        norms, pfc::integrator::AggregationScope::AlreadyReduced,
        m_error_order);
    return decide(attempted_dt, ev, comm);
  }

  /**
   * @brief Commit or reject the open Time attempt and install `next_dt`.
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
  [[nodiscard]] double propose_dt(double attempted_dt, double metric) const {
    double factor = m_cfg.growth_max;
    if (std::isfinite(metric) && metric > 0.0) {
      factor = m_cfg.safety_factor *
               std::pow(metric, -1.0 / static_cast<double>(m_error_order));
    } else if (!std::isfinite(metric) || metric <= 0.0) {
      factor = (metric <= 0.0 && std::isfinite(metric)) ? m_cfg.growth_max
                                                        : m_cfg.shrink_max;
    }
    factor = std::min(m_cfg.growth_max, std::max(m_cfg.shrink_max, factor));
    const double next = attempted_dt * factor;
    return std::min(m_cfg.max_dt, std::max(m_cfg.min_dt, next));
  }

  AdaptiveControlConfig m_cfg{};
  int m_error_order{3};
  int m_accepted{0};
  int m_rejected{0};
  int m_sequential_rejections{0};
};

} // namespace pfc::sim
