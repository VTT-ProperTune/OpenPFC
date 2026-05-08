// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file du_field.hpp
 * @brief Stack-bound residual field: a `du` buffer bundled with an
 *        evaluator and a "prepare parent" hook.
 *
 * @details
 * `pfc::sim::DuField<G, Eval>` is the **stack-friendly** entry point for
 * compact time loops that work on a `LocalField<double>` (FFT-safe core
 * + sparse face halos for FD, or pure spectral inboxes):
 *
 *     du.apply([](const G& g) { return ...physics... });   // hot path
 *     u += dt * du;                                         // explicit Euler
 *     t += dt;
 *
 * The class hides three pieces of plumbing that the compact heat3d
 * spectral driver and `pfc::sim::stacks::FdCpuStack`-based callers do
 * not want to see:
 *
 *  1. The **per-point evaluator** (`pfc::gradient::FDGradient<G>`,
 *     `pfc::field::SpectralGradient<G>`) that materialises the `G`
 *     aggregate from the parent field.
 *  2. The **once-per-call backend prep** the parent needs before its
 *     derivatives are valid: a sparse halo exchange for FD, or nothing
 *     for spectral (whose own `Eval::prepare()` runs the FFT inside the
 *     loop driver). The "prepare parent" step is supplied by the stack
 *     as a `std::function<void()>` so the user never sees the exchange.
 *  3. The **scratch `du` buffer** used by the explicit-Euler write-back.
 *     `apply(...)` writes only to interior cells; halo cells stay zero
 *     and are still a valid contribution to `u += dt * du` (`+= 0`).
 *
 * **Decoupled vs bundled.** For the new lab-style FD driver
 * (`apps/heat3d/src/cpu/heat3d_fd.cpp`) the separate primitives
 * `pfc::PaddedHaloExchanger` + `pfc::start_exchange` /
 * `pfc::finish_exchange`, `pfc::gradient::FDGradient<G>` +
 * `pfc::gradient::evaluate(grad, idx)`, and `pfc::field::for_each(brick,
 * fn)` are the recommended path — they keep halo, gradient, and
 * iteration as three visible concerns. `DuField` is preserved for the
 * spectral and `FdCpuStack` paths where the stack still wants to bundle
 * those concerns into a single `du.apply(...)` call.
 *
 * This is the teaching counterpart to `pfc::sim::steppers::EulerStepper`,
 * which packages the same machinery for non-trivial multi-field models
 * (`apps/kobayashi`, `apps/pfc-1`). For single-field explicit-Euler
 * problems on a `LocalField`, the laboratory form here keeps the Euler
 * line on the page.
 *
 * Lifetime: a `DuField` is constructed by the stack
 * (`stack.du<G>()`) and **must not outlive its parent stack** — the
 * captured "prepare parent" lambda holds a pointer into the stack and
 * the evaluator references the parent field's storage.
 *
 * @see openpfc/kernel/field/local_field.hpp and
 *      openpfc/kernel/field/padded_brick.hpp for `+=` /
 *      `operator*(double, …)` axpy targets (`LocalField`, `PaddedBrick`).
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the inner
 *      driver loop that `apply()` dispatches to.
 * @see openpfc/kernel/simulation/steppers/euler.hpp for the multi-field
 *      stepper that solves the same kind of problem at a different layer
 *      of abstraction.
 * @see apps/heat3d/src/cpu/heat3d_fd.cpp for the decoupled lab-style FD
 *      driver that uses the explicit primitives instead of `DuField`.
 */

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/field/scaled_field.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>

namespace pfc::sim {

/**
 * @brief Residual field bound to a parent `u` field plus an evaluator.
 *
 * @tparam G    The model-owned per-point grads aggregate the user's RHS
 *              lambda accepts (e.g. `heat3d::HeatGrads`).
 * @tparam Eval The per-point evaluator type (e.g.
 *              `pfc::field::FdGradient<G>`,
 *              `pfc::field::SpectralGradient<G>`). The evaluator must
 *              expose `prepare()`, `(i,j,k) -> G`, and the
 *              `imin/imax/jmin/jmax/kmin/kmax/idx` interior-iteration
 *              surface required by `pfc::sim::for_each_interior`.
 */
template <class G, class Eval> class DuField {
public:
  /**
   * @param local_size       Number of cells in the parent field's
   *                         `LocalField<double>` storage. The internal
   *                         `du` buffer is sized to match so layouts
   *                         agree for the `u += dt * du` axpy.
   * @param eval             Per-point evaluator (moved in). The evaluator
   *                         is expected to hold a pointer into the parent
   *                         field's contiguous buffer; the parent must
   *                         outlive this object.
   * @param prepare_parent   Callable invoked once at the top of every
   *                         `apply(...)` to refresh the parent's
   *                         dependencies (e.g. an MPI halo exchange for
   *                         FD, or a no-op for spectral). Stored as a
   *                         `std::function` so the stack can wire in the
   *                         appropriate prep without templating
   *                         `DuField` on the prepare type.
   */
  template <class PrepareFn>
  DuField(std::size_t local_size, Eval eval, PrepareFn &&prepare_parent)
      : m_data(local_size, 0.0), m_eval(std::move(eval)),
        m_prepare(std::forward<PrepareFn>(prepare_parent)) {}

  DuField(const DuField &) = delete;
  DuField &operator=(const DuField &) = delete;
  DuField(DuField &&) noexcept = default;
  DuField &operator=(DuField &&) noexcept = default;

  /**
   * @brief Compute `du = rhs(g)` (or `rhs(t, g)`) per interior cell.
   *
   * Auto-detects the lambda signature:
   *  - `(const G&) -> double`           — autonomous RHS
   *  - `(double t, const G&) -> double` — non-autonomous RHS
   *
   * Internally:
   *  1. Calls the captured `prepare_parent` (FD halo exchange / spectral
   *     no-op).
   *  2. Calls `eval.prepare()` (FD no-op / spectral forward FFT and
   *     spectral multiplies).
   *  3. Iterates every interior cell and writes
   *     `du[idx(i,j,k)] = rhs(t, eval(i,j,k))`.
   *
   * Halo cells of the internal `du` buffer are not touched by the loop;
   * they remain at the zero they were constructed with, which is the
   * additive identity for the subsequent `u += dt * du`.
   *
   * @tparam RhsFn  Callable, see signatures above.
   * @param rhs_fn  The RHS lambda.
   * @param t       Simulation time forwarded to the lambda when its
   *                signature includes `t`. Defaults to `0.0` for
   *                autonomous problems.
   */
  template <class RhsFn> void apply(RhsFn &&rhs_fn, double t = 0.0) {
    m_prepare();
    LambdaModel<std::decay_t<RhsFn>> model{std::forward<RhsFn>(rhs_fn)};
    pfc::sim::for_each_interior(model, m_eval, m_data.data(), t);
  }

  /** Read-only view of the underlying buffer. */
  [[nodiscard]] const double *data() const noexcept { return m_data.data(); }

  /** Number of cells in the underlying buffer (matches the parent). */
  [[nodiscard]] std::size_t size() const noexcept { return m_data.size(); }

  /**
   * @brief Build a `ScaledField` proxy from a scalar and this `DuField`.
   *
   * Enables `u += dt * du;` at the call site by routing through
   * `LocalField::operator+=(ScaledField)`.
   */
  friend pfc::field::ScaledField operator*(double alpha,
                                           const DuField &du) noexcept {
    return pfc::field::ScaledField{alpha, du.m_data.data(), du.m_data.size()};
  }

private:
  /**
   * @brief Tiny adapter that satisfies `for_each_interior`'s
   *        `model.rhs(t, g)` contract from a free `(g)` or `(t, g)`
   *        lambda.
   */
  template <class Fn> struct LambdaModel {
    Fn fn;
    template <class GG> [[nodiscard]] auto rhs(double t, const GG &g) const {
      if constexpr (std::is_invocable_v<const Fn &, double, const GG &>) {
        return fn(t, g);
      } else {
        return fn(g);
      }
    }
  };

  std::vector<double> m_data;
  Eval m_eval;
  std::function<void()> m_prepare;
};

} // namespace pfc::sim
