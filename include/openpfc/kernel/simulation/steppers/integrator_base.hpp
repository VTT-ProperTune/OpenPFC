// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file integrator_base.hpp
 * @brief Runtime-polymorphic integrator interface and type-erased wrappers.
 *
 * @details
 * `IntegratorBase` is a pure virtual contract for replaceable time integration.
 * Template steppers (`EulerStepper`, `RK2HeunStepper`) remain the performance
 * path; `EulerIntegrator` / `RK2HeunIntegrator` wrap them with
 * `std::function` type erasure so callers can swap methods via
 * `std::unique_ptr<IntegratorBase>` without rebuilding model physics.
 *
 * Rollback restores ODE time `t` and field `u` from wrapper-owned checkpoints.
 * That is complementary to `TimeStateGuard` in `time.hpp`, which only restores
 * `Time::dt` / `Time::increment` on a `Time&` — applications that combine both
 * must use both mechanisms.
 */

#include <cstddef>
#include <functional>
#include <vector>

#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/rk2_heun.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Type-erased RHS: `rhs(t, u, du)` filling `du` in place.
 */
using RhsFunction = std::function<void(
    double t, const std::vector<double> &u, std::vector<double> &du)>;

/**
 * @brief Pure virtual runtime contract for a single-field time integrator.
 */
class IntegratorBase {
public:
  virtual ~IntegratorBase() = default;

  /** Advance `u` by one step; returns the new ODE time. */
  virtual double step(double t, std::vector<double> &u) = 0;

  /** Classical order of the method (Euler=1, RK2 Heun=2). */
  virtual int get_order() const = 0;

  virtual std::size_t get_accepted_steps() const = 0;
  virtual std::size_t get_rejected_steps() const = 0;

  /**
   * @brief Restore ODE time `t` and field `u` from the last pre-step
   *        checkpoint.
   *
   * Complementary to `TimeStateGuard`, which only restores `Time::dt` /
   * `Time::increment` on a `Time&` (see `time.hpp`).
   */
  virtual void rollback(double &t, std::vector<double> &u) = 0;

  IntegratorBase(const IntegratorBase &) = delete;
  IntegratorBase &operator=(const IntegratorBase &) = delete;
  IntegratorBase(IntegratorBase &&) = delete;
  IntegratorBase &operator=(IntegratorBase &&) = delete;

protected:
  IntegratorBase() = default;
};

/**
 * @brief Type-erased forward-Euler integrator implementing `IntegratorBase`.
 */
class EulerIntegrator final : public IntegratorBase {
public:
  /** Matches `EulerStepper(double dt, std::size_t local_size, Rhs rhs)`. */
  explicit EulerIntegrator(RhsFunction rhs, double dt, std::size_t local_size)
      : m_stepper(dt, local_size, std::move(rhs)) {}

  double step(double t, std::vector<double> &u) override {
    m_u_checkpoint = u;
    m_t_checkpoint = t;
    m_has_checkpoint = true;
    const double t_new = m_stepper.step(t, u);
    ++m_accepted;
    return t_new;
  }

  int get_order() const override { return 1; }
  std::size_t get_accepted_steps() const override { return m_accepted; }
  std::size_t get_rejected_steps() const override { return m_rejected; }

  void rollback(double &t, std::vector<double> &u) override {
    if (m_has_checkpoint) {
      u = m_u_checkpoint;
      t = m_t_checkpoint;
      m_has_checkpoint = false;
    }
    ++m_rejected;
  }

  EulerIntegrator(const EulerIntegrator &) = delete;
  EulerIntegrator &operator=(const EulerIntegrator &) = delete;
  EulerIntegrator(EulerIntegrator &&) = delete;
  EulerIntegrator &operator=(EulerIntegrator &&) = delete;

private:
  EulerStepper<RhsFunction> m_stepper;
  std::vector<double> m_u_checkpoint;
  double m_t_checkpoint{0.0};
  bool m_has_checkpoint{false};
  std::size_t m_accepted{0};
  std::size_t m_rejected{0};
};

/**
 * @brief Type-erased RK2 Heun integrator implementing `IntegratorBase`.
 *
 * Owns its own field/time checkpoints because `RK2HeunStepper` has no
 * `save_state` / `restore_state` API.
 */
class RK2HeunIntegrator final : public IntegratorBase {
public:
  /** Matches `RK2HeunStepper(double dt, std::size_t local_size, Rhs rhs)`. */
  explicit RK2HeunIntegrator(RhsFunction rhs, double dt,
                             std::size_t local_size)
      : m_stepper(dt, local_size, std::move(rhs)) {}

  double step(double t, std::vector<double> &u) override {
    m_u_checkpoint = u;
    m_t_checkpoint = t;
    m_has_checkpoint = true;
    const double t_new = m_stepper.step(t, u);
    ++m_accepted;
    return t_new;
  }

  int get_order() const override { return 2; }
  std::size_t get_accepted_steps() const override { return m_accepted; }
  std::size_t get_rejected_steps() const override { return m_rejected; }

  void rollback(double &t, std::vector<double> &u) override {
    if (m_has_checkpoint) {
      u = m_u_checkpoint;
      t = m_t_checkpoint;
      m_has_checkpoint = false;
    }
    ++m_rejected;
  }

  RK2HeunIntegrator(const RK2HeunIntegrator &) = delete;
  RK2HeunIntegrator &operator=(const RK2HeunIntegrator &) = delete;
  RK2HeunIntegrator(RK2HeunIntegrator &&) = delete;
  RK2HeunIntegrator &operator=(RK2HeunIntegrator &&) = delete;

private:
  RK2HeunStepper<RhsFunction> m_stepper;
  std::vector<double> m_u_checkpoint;
  double m_t_checkpoint{0.0};
  bool m_has_checkpoint{false};
  std::size_t m_accepted{0};
  std::size_t m_rejected{0};
};

} // namespace pfc::sim::steppers
