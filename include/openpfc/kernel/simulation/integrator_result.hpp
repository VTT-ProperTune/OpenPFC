// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file integrator_result.hpp
 * @brief Value-type contract for a single time-integration step outcome.
 *
 * @details
 * `IntegratorResult` carries accept/reject status, the candidate new time,
 * an optional error estimate, and a rejection reason. Factories always leave
 * `last_step_complete` false; callers that own an end time may set
 * `result.last_step_complete = (result.new_time >= t_end)` themselves.
 *
 * Factory names are `make_accepted` / `make_rejected` (not `accepted` /
 * `rejected`) because a static member function cannot share a name with the
 * `bool accepted` data member under GCC 11 / C++20.
 */

#include <string>

namespace pfc::sim::steppers {

/**
 * @brief Outcome of one time-integration attempt (accept/reject contract).
 */
struct IntegratorResult {
  bool accepted{false};
  bool last_step_complete{false};
  double new_time{0.0};
  double error_estimate{0.0};
  std::string rejection_reason{};

  /** @brief Accepted step at @p new_time; `last_step_complete` remains false. */
  static IntegratorResult make_accepted(double new_time) {
    IntegratorResult r;
    r.accepted = true;
    r.last_step_complete = false;
    r.new_time = new_time;
    r.error_estimate = 0.0;
    r.rejection_reason.clear();
    return r;
  }

  /**
   * @brief Rejected step; @p current_time is stored in `new_time` and
   *        @p reason in `rejection_reason`.
   */
  static IntegratorResult make_rejected(double current_time,
                                        const std::string &reason) {
    IntegratorResult r;
    r.accepted = false;
    r.last_step_complete = false;
    r.new_time = current_time;
    r.error_estimate = 0.0;
    r.rejection_reason = reason;
    return r;
  }
};

} // namespace pfc::sim::steppers
