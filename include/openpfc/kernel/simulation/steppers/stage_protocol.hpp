// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_protocol.hpp
 * @brief C++20 concepts constraining a stepper's Rhs template parameter.
 *
 * @details
 * `StageFunction` and `MultiStageFunction` capture the callable signature
 * every stepper's `Rhs` template parameter is actually invoked with:
 * `rhs(t, u, du)` filling `du` in place. Verified directly against the
 * real call sites, not assumed: `EulerStepper::step()` calls
 * `m_rhs(t, u, m_du)` (euler.hpp), `ExplicitRKStepper::step()` calls
 * `m_rhs(stage_time, u_temp, m_du)` (explicit_rk.hpp), and
 * `RK2HeunStepper::step()` calls `m_rhs(t, u, m_du)` /
 * `m_rhs(t + m_dt, m_predictor, m_rhs_predictor)` (rk2_heun.hpp) -- in
 * every one of these, `u` is passed as a plain (non-const)
 * `std::vector<double>&`, so the concept's own test parameter must be
 * non-const too: a `const std::vector<double>&` test parameter cannot
 * bind to a callable whose `operator()` demands non-const `u`, which is
 * exactly what broke the build the first time this concept was applied to
 * the real stepper classes (constraint failures against every existing
 * Rhs type in test_euler_stepper.cpp/test_steppers.cpp, all of which take
 * non-const `u`). A callable that itself declares `u` as `const
 * std::vector<double>&` (e.g. the factory lambdas in euler.hpp/
 * explicit_rk.hpp) still satisfies this concept -- a non-const lvalue
 * argument binds to either a const or non-const reference parameter, so
 * testing with non-const `u` is strictly more permissive, not less.
 *
 * @see euler.hpp, explicit_rk.hpp, rk2_heun.hpp, rk3_heun.hpp for the
 *      stepper classes constrained by these concepts.
 */

#include <tuple>
#include <vector>

namespace pfc::sim::steppers {

/**
 * @brief Satisfied by any single-field stage-evaluation callable:
 *        `rhs(t, u, du)` filling `du` in place.
 */
template <class Rhs>
concept StageFunction = requires(Rhs rhs, double t, std::vector<double> &u,
                                 std::vector<double> &du) { rhs(t, u, du); };

/**
 * @brief Satisfied by a two-field stage-evaluation callable:
 *        `rhs(t, u_pack, du_pack)` filling every field in `du_pack`.
 */
template <class Rhs>
concept MultiStageFunction =
    requires(Rhs rhs, double t,
             std::tuple<std::vector<double> &, std::vector<double> &> u_pack,
             std::tuple<std::vector<double> &, std::vector<double> &> du_pack) {
      rhs(t, u_pack, du_pack);
    };

} // namespace pfc::sim::steppers
