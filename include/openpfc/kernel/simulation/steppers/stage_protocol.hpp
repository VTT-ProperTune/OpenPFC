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
 * @see euler.hpp, explicit_rk.hpp, rk2_heun.hpp, rk3_heun.hpp,
 *      imex_euler.hpp for the stepper classes constrained by these concepts
 *      (IMEX Euler uses StageFunction / MultiStageFunction for the explicit
 *      half E).
 */

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pfc::sim::steppers {

namespace detail {

template <std::size_t N, typename T,
          typename Seq = std::make_index_sequence<N>>
struct n_ref_tuple;

template <std::size_t N, typename T, std::size_t... I>
struct n_ref_tuple<N, T, std::index_sequence<I...>> {
  using type = std::tuple<std::conditional_t<true, T &, decltype(I)>...>;
};

} // namespace detail

/**
 * @brief Satisfied by a single-field stage-evaluation callable on `Scalar`
 *        buffers: `rhs(t, u, du)` filling `du` in place.
 */
template <class Rhs, class Scalar>
concept StageFunctionFor = requires(Rhs rhs, double t, std::vector<Scalar> &u,
                                    std::vector<Scalar> &du) {
  rhs(t, u, du);
};

/**
 * @brief Real (`double`) stage-evaluation callable. Existing steppers use this.
 */
template <class Rhs>
concept StageFunction = StageFunctionFor<Rhs, double>;

/**
 * @brief Satisfied by an N-field stage-evaluation callable on `Scalar`
 *        buffers: `rhs(t, u_pack, du_pack)` filling every field in `du_pack`.
 *
 * Default `N == 2` and `Scalar == double` keep existing
 * `MultiStageFunction<Rhs>` call sites.
 */
template <class Rhs, std::size_t N = 2, class Scalar = double>
concept MultiStageFunction = requires(
    Rhs rhs, double t,
    typename detail::n_ref_tuple<N, std::vector<Scalar>>::type u_pack,
    typename detail::n_ref_tuple<N, std::vector<Scalar>>::type du_pack) {
  rhs(t, u_pack, du_pack);
};

} // namespace pfc::sim::steppers
