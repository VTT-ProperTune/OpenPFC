// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file euler.hpp
 * @brief Explicit forward-Euler stepper for arbitrary point-wise RHS callables.
 *
 * @details
 * `EulerStepper` is a **pure ODE integrator** that applies one forward-Euler
 * step in place,
 *
 *     u += dt * rhs(t, u)
 *
 * It owns nothing more than `dt`, an internal scratch `du` buffer, and a
 * user-supplied `Rhs` callable. The callable is the only thing that knows
 * about the spatial discretization (FD, spectral, custom, ...); the stepper
 * itself is agnostic.
 *
 * `Rhs` must be invocable as
 *
 *     rhs(double t, std::vector<Scalar>& u, std::vector<Scalar>& du)
 *
 * and is expected to **fill** `du` (sized `local_size` by the constructor).
 * `u` is passed read-only by convention; the stepper performs the
 * `u += dt * du` accumulation itself. Cells that `rhs` leaves untouched keep
 * their previous `du` value (the buffer is value-initialized once at
 * construction; subsequent steps overwrite whatever the RHS chooses to
 * overwrite). The stepper does not perform halo exchange or any other
 * backend pre-processing — that is the application's responsibility (FD
 * needs a halo exchange before each step; spectral does not).
 *
 * Most applications do not construct `EulerStepper` directly. Use one of
 * the `pfc::sim::steppers::create` factories at the bottom of this file to
 * bind a model + gradient evaluator to the canonical
 * `for_each_interior(model, eval, du, t)` RHS. They follow the
 * `domain::create` and `pfc::data::field_from_subdomain` pattern
 * used throughout OpenPFC for creating domains and fields.
 *
 * Higher-order explicit methods (RK2, RK4) live in sibling files in this
 * folder under `pfc::sim::steppers::`. First-order IMEX Euler with an
 * explicit–implicit split is in `imex_euler.hpp`.
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop the `create` factories wrap
 * @see openpfc/kernel/field/grad_concepts.hpp for the per-member detection
 *      concepts that drive backend pruning
 * @see openpfc/kernel/data/grid_field.hpp for the typed field bundle
 *      that the `Field` overload derives `local_size` from
 * @see imex_euler.hpp for first-order IMEX Euler (`ImexEulerStepper`)
 */

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/simulation/steppers/stage_protocol.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>
#include <openpfc/kernel/simulation/steppers/stepper_validation.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Pure forward-Euler ODE stepper: `u += dt * rhs(t, u)`.
 *
 * @tparam Rhs    Any callable invocable as
 *                `rhs(double t, std::vector<Scalar>& u, std::vector<Scalar>& du)`.
 *                It must fill `du`; the stepper adds `dt * du` to `u`.
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 */
template <class Rhs, class Scalar = double>
  requires StageFunctionFor<Rhs, Scalar>
class EulerStepper {
public:
  using scalar_type = Scalar;
  using Attempt = StepAttempt<Scalar>;

  EulerStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, Scalar{}), m_candidate(local_size, Scalar{}),
        m_u_checkpoint(local_size, Scalar{}), m_rhs(std::move(rhs)) {}

  /**
   * @brief Isolate `u + dt * rhs(t, u)` without writing `u`.
   *
   * @return Successful `StepAttempt<Scalar>` whose candidate is method-owned.
   */
  [[nodiscard]] Attempt attempt(double t, const std::vector<Scalar> &u) {
    m_rhs(t, const_cast<std::vector<Scalar> &>(u), m_du);
    const std::size_t n = u.size();
    for (std::size_t i = 0; i < n; ++i) {
      m_candidate[i] = u[i] + Scalar(m_dt) * m_du[i];
    }
    return Attempt(t, m_dt, t + m_dt, /*success=*/true, m_candidate);
  }

  /** Advance `u` by one explicit-Euler step; commit of `attempt`. */
  double step(double t, std::vector<Scalar> &u) {
    const Attempt r = attempt(t, u);
    commit_step_attempt(u, r);
    return r.t1;
  }

  /** Isolate a candidate from host field state (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar> F>
  [[nodiscard]] Attempt attempt(double t, const F &u) {
    return attempt(t, u.vec());
  }

  /** Advance host field state by one explicit-Euler step. */
  template <pfc::field::HostFieldState<Scalar> F>
  double step(double t, F &u) {
    return step(t, u.vec());
  }

  double dt() const noexcept { return m_dt; }

  /**
   * @brief Save current field state for potential rollback.
   * @param u Current field buffer to checkpoint.
   *
   * This stores a deep copy of u into internal m_u_checkpoint.
   * The checkpoint can be restored via restore_state() to revert
   * the field to its pre-step state, enabling adaptive time-stepping
   * with step rejection.
   *
   * @note This is part of the duck-typed checkpoint protocol. Future
   * steppers must implement save_state(), restore_state(), and can_rollback()
   * with matching signatures to support adaptive error control.
   */
  void save_state(const std::vector<Scalar> &u) { m_u_checkpoint = u; }

  /**
   * @brief Restore field state to last checkpointed state.
   * @param u Field buffer to restore into.
   *
   * This copies m_u_checkpoint back into u, reverting the field to its
   * pre-step state. Used when an adaptive step is rejected due to error
   * estimates exceeding tolerance.
   *
   * @note This is part of the duck-typed checkpoint protocol. Must be
   * called after save_state() to have valid checkpoint data.
   */
  void restore_state(std::vector<Scalar> &u) { u = m_u_checkpoint; }

  /**
   * @brief Check whether this stepper supports rollback.
   * @return Always true for EulerStepper.
   *
   * This enables duck-typed protocol checking without RTTI. Application
   * code can use `if (stepper.can_rollback())` to conditionally enable
   * adaptive error control patterns.
   *
   * @note This is part of the duck-typed checkpoint protocol. Future
   * steppers should return true if they implement save_state() and
   * restore_state().
   */
  [[nodiscard]] bool can_rollback() const noexcept { return true; }

private:
  double m_dt{0.0};
  std::vector<Scalar> m_du;
  std::vector<Scalar> m_candidate;
  std::vector<Scalar> m_u_checkpoint;
  Rhs m_rhs;
};

/**
 * @brief Multi-field forward-Euler ODE stepper.
 *
 * Owns one `du` buffer per field (still SoA: each buffer is a contiguous
 * `std::vector<Scalar>` matching its field's local size) and accumulates
 * `u_k += dt * du_k` per field. `Rhs` is invocable as
 *
 *     rhs(double t,
 *         std::tuple<std::vector<Scalar>&, ...> u_pack,
 *         std::tuple<std::vector<Scalar>&, ...> du_pack)
 *
 * and must fill the `du` tuple element-by-element. See the
 * `pfc::sim::steppers::create(std::tuple<...>, ...)` factory below for
 * the canonical wiring against `for_each_interior`.
 *
 * @tparam Rhs    Multi-field RHS callable as described above.
 * @tparam N      Number of fields.
 * @tparam Scalar Field element type (`double` or `std::complex<double>`).
 */
template <class Rhs, std::size_t N, class Scalar = double>
class MultiEulerStepper {
public:
  using RhsType = Rhs;
  using scalar_type = Scalar;
  static constexpr std::size_t field_count = N;
  MultiEulerStepper(double dt, std::array<std::size_t, N> local_sizes, Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    for (std::size_t i = 0; i < N; ++i) {
      m_du[i].assign(local_sizes[i], Scalar{});
      m_u_work[i].assign(local_sizes[i], Scalar{});
      m_candidate[i].assign(local_sizes[i], Scalar{});
      m_u_checkpoint[i].assign(local_sizes[i], Scalar{});
    }
  }

  /**
   * @brief Isolate one Euler update per field without writing accepted buffers.
   */
  template <class... U>
  [[nodiscard]] MultiStepAttemptResult<N, Scalar>
  attempt(double t, const std::vector<U> &...u_accepted) {
    static_assert(sizeof...(U) == N,
                  "MultiEulerStepper::attempt: buffer count must match N");
    static_assert((std::is_same_v<U, Scalar> && ...),
                  "MultiEulerStepper requires std::vector<Scalar>");
    copy_accepted_to_work(std::index_sequence_for<U...>{}, u_accepted...);
    auto u_pack = make_work_tuple(std::index_sequence_for<U...>{});
    auto du_pack = make_du_tuple(std::index_sequence_for<U...>{});
    m_rhs(t, u_pack, du_pack);
    form_candidates(std::index_sequence_for<U...>{}, u_accepted...);
    return MultiStepAttemptResult<N, Scalar>(t, m_dt, t + m_dt,
                                             /*success=*/true, candidate_ptrs());
  }

  /** Isolate one Euler update per host field (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar>... Fs>
    requires(sizeof...(Fs) == N)
  [[nodiscard]] MultiStepAttemptResult<N, Scalar>
  attempt(double t, const Fs &...u_accepted) {
    return attempt(t, u_accepted.vec()...);
  }

  /** Advance every field by one explicit-Euler step; commit of `attempt`. */
  template <class... U> double step(double t, std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N,
                  "MultiEulerStepper::step: number of u buffers must match N.");
    const auto r = attempt(t, u_buffers...);
    commit_candidates(std::index_sequence_for<U...>{}, u_buffers...);
    return r.t1;
  }

  /** Advance every host field by one explicit-Euler step (via `vec()`). */
  template <pfc::field::HostFieldState<Scalar>... Fs>
    requires(sizeof...(Fs) == N)
  double step(double t, Fs &...u_buffers) {
    return step(t, u_buffers.vec()...);
  }

  double dt() const noexcept { return m_dt; }

  /**
   * @brief Save current field states for potential rollback.
   * @param u_buffers Field buffers to checkpoint (must be std::vector<double>).
   *
   * This stores deep copies of all field buffers into internal m_u_checkpoint array.
   * The checkpoint can be restored via restore_state() to revert all fields to their
   * pre-step state, enabling adaptive time-stepping with step rejection.
   *
   * @note The field order must match the order used in step().
   * @note This is part of the duck-typed checkpoint protocol. Future
   * steppers must implement save_state(), restore_state(), and can_rollback()
   * with matching signatures to support adaptive error control.
   */
  template <class... U> void save_state(const std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N,
                  "Number of fields must match template parameter N");
    static_assert((std::is_same_v<U, double> && ...),
                  "MultiEulerStepper checkpoint requires std::vector<Scalar>");
    std::size_t i = 0;
    ((m_u_checkpoint[i++] = u_buffers), ...);
  }

  /**
   * @brief Restore field states to last checkpointed states.
   * @param u_buffers Field buffers to restore into (must be std::vector<double>).
   *
   * This copies m_u_checkpoint back into all field buffers, reverting them to
   * their pre-step state. Used when an adaptive step is rejected due to error
   * estimates exceeding tolerance.
   *
   * @note The field order must match the order used in step().
   * @note This is part of the duck-typed checkpoint protocol. Must be
   * called after save_state() to have valid checkpoint data.
   */
  template <class... U> void restore_state(std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N,
                  "Number of fields must match template parameter N");
    static_assert((std::is_same_v<U, double> && ...),
                  "MultiEulerStepper checkpoint requires std::vector<Scalar>");
    std::size_t i = 0;
    ((u_buffers = m_u_checkpoint[i++]), ...);
  }

  /**
   * @brief Check whether this stepper supports rollback.
   * @return Always true for MultiEulerStepper.
   *
   * This enables duck-typed protocol checking without RTTI. Application
   * code can use `if (stepper.can_rollback())` to conditionally enable
   * adaptive error control patterns.
   *
   * @note This is part of the duck-typed checkpoint protocol. Future
   * steppers should return true if they implement save_state() and
   * restore_state().
   */
  [[nodiscard]] bool can_rollback() const noexcept { return true; }

private:
  template <std::size_t... I> auto make_du_tuple(std::index_sequence<I...>) {
    return std::tie(m_du[I]...);
  }

  template <std::size_t... I> auto make_work_tuple(std::index_sequence<I...>) {
    return std::tie(m_u_work[I]...);
  }

  template <std::size_t... I, class... U>
  void copy_accepted_to_work(std::index_sequence<I...>,
                             const std::vector<U> &...u_accepted) {
    ((m_u_work[I] = u_accepted), ...);
  }

  template <std::size_t... I, class... U>
  void form_candidates(std::index_sequence<I...>,
                       const std::vector<U> &...u_accepted) {
    auto one = [this](std::vector<Scalar> &cand, const std::vector<Scalar> &u,
                      const std::vector<Scalar> &du) {
      for (std::size_t i = 0; i < u.size(); ++i) {
        cand[i] = u[i] + Scalar(m_dt) * du[i];
      }
    };
    (one(m_candidate[I], u_accepted, m_du[I]), ...);
  }

  template <std::size_t... I, class... U>
  void commit_candidates(std::index_sequence<I...>,
                         std::vector<U> &...u_accepted) const {
    ((u_accepted = m_candidate[I]), ...);
  }

  [[nodiscard]] std::array<const std::vector<Scalar> *, N>
  candidate_ptrs() const {
    return candidate_ptrs_impl(std::make_index_sequence<N>{});
  }

  template <std::size_t... I>
  [[nodiscard]] std::array<const std::vector<Scalar> *, N>
  candidate_ptrs_impl(std::index_sequence<I...>) const {
    return {&m_candidate[I]...};
  }

  double m_dt{0.0};
  std::array<std::vector<Scalar>, N> m_du;
  std::array<std::vector<Scalar>, N> m_u_work;
  std::array<std::vector<Scalar>, N> m_candidate;
  std::array<std::vector<Scalar>, N> m_u_checkpoint;
  Rhs m_rhs;
};

// -----------------------------------------------------------------------------
// `create` free-function factories.
//
// They build an `EulerStepper` (single-field) or `MultiEulerStepper`
// (multi-field) whose RHS is the canonical point-wise loop
//
//     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
//
// over the interior cells exposed by `eval`. The stepper itself remains
// agnostic of the (Eval, Model) types — the wiring lives entirely inside the
// captured lambda below.
// -----------------------------------------------------------------------------

/**
 * @brief Build an `EulerStepper` for the canonical point-wise RHS, given the
 *        local buffer size explicitly.
 *
 * Prefer the `Field` overload when you have one — it derives
 * `local_size` from `u.size()`.
 *
 * @param eval        Per-point gradient evaluator (e.g.
 *                    `pfc::field::FdGradient<G>`,
 *                    `pfc::field::SpectralGradient<G>`). Captured by
 *                    reference; must outlive the returned stepper.
 * @param model       Physics model with a method
 *                    `rhs(double t, const G&) -> double`.
 *                    Captured by reference; must outlive the returned
 *                    stepper.
 * @param dt          Time-step size.
 * @param local_size  Number of cells in the rank-local field buffer
 *                    (typically `u.size()`).
 */
template <class Eval, class Model>
[[nodiscard]] auto create(Eval &eval, const Model &model, double dt,
                          std::size_t local_size) {
  pfc::sim::steppers::validate_rhs_signature<Model, Eval>();
  pfc::sim::steppers::validate_spatial_compatibility<Eval>();
  auto rhs = [&eval, &model](double t, const std::vector<double> & /*u*/,
                             std::vector<double> &du) {
    pfc::sim::for_each_interior(model, eval, du.data(), t);
  };
  return EulerStepper<decltype(rhs)>(dt, local_size, std::move(rhs));
}

/**
 * @brief Build an `EulerStepper` for the canonical point-wise RHS, deriving
 *        the local buffer size from the field bundle.
 *
 * Mirrors the `domain::create` and `pfc::data::field_from_subdomain`
 * pattern used elsewhere in OpenPFC.
 *
 * @param u      Local field whose `size()` defines the internal `du` buffer
 *               (and which the application owns). Not stored by the stepper.
 * @param eval   Per-point gradient evaluator. Captured by reference.
 * @param model  Physics model. Captured by reference.
 * @param dt     Time-step size.
 */
template <class T, class Eval, class Model>
[[nodiscard]] auto create(const pfc::data::Field<T> &u, Eval &eval,
                          const Model &model, double dt) {
  return create(eval, model, dt, u.size());
}


/**
 * @brief Multi-field overload: build a `MultiEulerStepper` from a tuple of
 *        `Field` references, a composite evaluator, and a model whose
 *        `rhs` returns a tuple-protocol bundle of increments.
 *
 * The composite evaluator (typically `pfc::field::CompositeGradient<...>`)
 * is responsible for returning a per-point bundle the model can read. The
 * model's `rhs(t, g)` must return a tuple-protocol-compatible bundle (a
 * `std::tuple` or a struct exposing `as_tuple()`); the stepper scatters
 * the elements into the per-field `du` buffers in order.
 *
 * @param fields  Tuple of `Field` references whose `size()` defines
 *                each per-field internal `du` buffer. The fields themselves
 *                are not stored by the stepper.
 * @param eval    Composite per-point evaluator. Captured by reference.
 * @param model   Multi-field physics model. Captured by reference.
 * @param dt      Time-step size.
 */
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::data::Field<Ts> &...> fields,
                          Eval &eval, const Model &model, double dt) {
  constexpr std::size_t N = sizeof...(Ts);
  std::array<std::size_t, N> sizes{};
  std::apply(
      [&](auto &...f) {
        std::size_t i = 0;
        ((sizes[i++] = f.size()), ...);
      },
      fields);

  auto rhs = [&eval, &model](double t, auto & /*u_tuple*/, auto &du_tuple) {
    auto du_ptrs = std::apply(
        [](auto &...vs) { return std::make_tuple(vs.data()...); }, du_tuple);
    pfc::sim::for_each_interior(model, eval, du_ptrs, t);
  };
  return MultiEulerStepper<decltype(rhs), N>(dt, sizes, std::move(rhs));
}


} // namespace pfc::sim::steppers
