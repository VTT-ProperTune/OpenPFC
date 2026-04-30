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
 *     rhs(double t, std::vector<double>& u, std::vector<double>& du)
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
 * `for_each_interior(model, eval, du, t)` RHS. They mirror the
 * `world::create`, `decomposition::create`, `fft::create`, `field::create`
 * convention used throughout OpenPFC.
 *
 * Further methods (RK2, RK4, IMEX) belong in sibling files in this folder
 * under `pfc::sim::steppers::`.
 *
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the canonical
 *      point-wise driver loop the `create` factories wrap
 * @see openpfc/kernel/field/grad_concepts.hpp for the per-member detection
 *      concepts that drive backend pruning
 * @see openpfc/kernel/field/local_field.hpp for the typed field bundle
 *      that the `LocalField` overload derives `local_size` from
 */

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Pure forward-Euler ODE stepper: `u += dt * rhs(t, u)`.
 *
 * @tparam Rhs Any callable invocable as
 *             `rhs(double t, std::vector<double>& u, std::vector<double>& du)`.
 *             It must fill `du`; the stepper adds `dt * du` to `u`.
 */
template <class Rhs> class EulerStepper {
public:
  EulerStepper(double dt, std::size_t local_size, Rhs rhs)
      : m_dt(dt), m_du(local_size, 0.0), m_rhs(std::move(rhs)) {}

  /** Advance `u` by one explicit-Euler step in place; returns the new time. */
  double step(double t, std::vector<double> &u) {
    m_rhs(t, u, m_du);
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * m_du[static_cast<std::size_t>(li)];
    }
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  double m_dt{0.0};
  std::vector<double> m_du;
  Rhs m_rhs;
};

/**
 * @brief Multi-field forward-Euler ODE stepper.
 *
 * Owns one `du` buffer per field (still SoA: each buffer is a contiguous
 * `std::vector<double>` matching its field's local size) and accumulates
 * `u_k += dt * du_k` per field. `Rhs` is invocable as
 *
 *     rhs(double t,
 *         std::tuple<std::vector<double>&, ...> u_pack,
 *         std::tuple<std::vector<double>&, ...> du_pack)
 *
 * and must fill the `du` tuple element-by-element. See the
 * `pfc::sim::steppers::create(std::tuple<...>, ...)` factory below for
 * the canonical wiring against `for_each_interior`.
 *
 * @tparam Rhs  Multi-field RHS callable as described above.
 * @tparam N    Number of fields.
 */
template <class Rhs, std::size_t N> class MultiEulerStepper {
public:
  MultiEulerStepper(double dt, std::array<std::size_t, N> local_sizes, Rhs rhs)
      : m_dt(dt), m_rhs(std::move(rhs)) {
    for (std::size_t i = 0; i < N; ++i) m_du[i].assign(local_sizes[i], 0.0);
  }

  /** Advance every field by one explicit-Euler step in place. */
  template <class... U> double step(double t, std::vector<U> &...u_buffers) {
    static_assert(sizeof...(U) == N,
                  "MultiEulerStepper::step: number of u buffers must match N.");
    auto u_pack = std::tie(u_buffers...);
    auto du_pack = make_du_tuple(std::index_sequence_for<U...>{});
    m_rhs(t, u_pack, du_pack);
    accumulate(u_pack, du_pack, std::index_sequence_for<U...>{});
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  template <std::size_t... I> auto make_du_tuple(std::index_sequence<I...>) {
    return std::tie(m_du[I]...);
  }

  template <class UPack, class DuPack, std::size_t... I>
  void accumulate(UPack &u, DuPack &du, std::index_sequence<I...>) {
    (apply_one(std::get<I>(u), std::get<I>(du)), ...);
  }

  void apply_one(std::vector<double> &u, std::vector<double> &du) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(u.size());
    for (std::ptrdiff_t li = 0; li < n; ++li) {
      u[static_cast<std::size_t>(li)] += m_dt * du[static_cast<std::size_t>(li)];
    }
  }

  double m_dt{0.0};
  std::array<std::vector<double>, N> m_du;
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
 * Prefer the `LocalField` overload when you have one — it derives
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
 * Mirrors the `world::create`, `decomposition::create`, `fft::create`,
 * `field::create` family used elsewhere in OpenPFC.
 *
 * @param u      Local field whose `size()` defines the internal `du` buffer
 *               (and which the application owns). Not stored by the stepper.
 * @param eval   Per-point gradient evaluator. Captured by reference.
 * @param model  Physics model. Captured by reference.
 * @param dt     Time-step size.
 */
template <class T, class Eval, class Model>
[[nodiscard]] auto create(const pfc::field::LocalField<T> &u, Eval &eval,
                          const Model &model, double dt) {
  return create(eval, model, dt, u.size());
}

/**
 * @brief Multi-field overload: build a `MultiEulerStepper` from a tuple of
 *        `LocalField` references, a composite evaluator, and a model whose
 *        `rhs` returns a tuple-protocol bundle of increments.
 *
 * The composite evaluator (typically `pfc::field::CompositeGradient<...>`)
 * is responsible for returning a per-point bundle the model can read. The
 * model's `rhs(t, g)` must return a tuple-protocol-compatible bundle (a
 * `std::tuple` or a struct exposing `as_tuple()`); the stepper scatters
 * the elements into the per-field `du` buffers in order.
 *
 * @param fields  Tuple of `LocalField` references whose `size()` defines
 *                each per-field internal `du` buffer. The fields themselves
 *                are not stored by the stepper.
 * @param eval    Composite per-point evaluator. Captured by reference.
 * @param model   Multi-field physics model. Captured by reference.
 * @param dt      Time-step size.
 */
template <class... Ts, class Eval, class Model>
[[nodiscard]] auto create(std::tuple<pfc::field::LocalField<Ts> &...> fields,
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
