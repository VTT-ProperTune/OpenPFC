// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stepper_concept.hpp
 * @brief C++20 concepts validating the stepper interface at compile time.
 *
 * @details
 * `SingleFieldStepper` and `MultiFieldStepper` capture the interface every
 * stepper class in this directory already implements (`EulerStepper`,
 * `RK2HeunStepper`, `ExplicitRKStepper`, `MultiEulerStepper`,
 * `MultiExplicitRKStepper`): a `step()` method that advances the state and
 * returns the new time, and a `dt() const noexcept` accessor. This header
 * adds no new type and no runtime behavior -- it is a compile-time check
 * only, consumed via `static_assert` (see test_stepper_concept.cpp), the
 * concept-constrained-template extension pattern established in
 * ADR-0003 contract 7 (docs/adr/0003-time-integrator-interface.md).
 *
 * @see euler.hpp, rk2_heun.hpp, explicit_rk.hpp for the concrete stepper
 *      classes this concept validates.
 */

#include <concepts>
#include <vector>

namespace pfc::sim::steppers {

/**
 * @brief Satisfied by any single-field stepper: `step(t, u) -> double` and
 *        `dt() -> double`.
 */
template <class T>
concept SingleFieldStepper = requires(T stepper, double t, std::vector<double> &u) {
  { stepper.step(t, u) } -> std::same_as<double>;
  { stepper.dt() } -> std::same_as<double>;
};

/**
 * @brief Satisfied by a multi-field stepper: `step(t, u1, u2) -> double`,
 *        `dt() -> double`, and a `field_count` static member reporting an
 *        integral count of at least 1 field.
 */
template <class T>
concept MultiFieldStepper =
    requires(T stepper, double t, std::vector<double> &u1, std::vector<double> &u2) {
      { stepper.step(t, u1, u2) } -> std::same_as<double>;
      { stepper.dt() } -> std::same_as<double>;
      requires std::integral<decltype(T::field_count)>;
      requires T::field_count >= 1;
    };

} // namespace pfc::sim::steppers
