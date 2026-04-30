// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file for_each_interior.hpp
 * @brief Apply a per-point RHS over the interior of a discretization.
 *
 * @details
 * `for_each_interior` is the canonical driver loop for the point-wise
 * abstraction
 *
 *     du[{i,j,k}] = model.rhs(t, eval(i,j,k))
 *
 * It depends on its template parameters only through duck typing:
 *
 *  - `Model` must expose `rhs(double t, const G&)` returning either a
 *    scalar (single-field models) or a model-defined "increments"
 *    aggregate (multi-field models). `G` is whatever the evaluator
 *    returns; the framework never names a specific grads type.
 *  - `Eval` must expose interior bounds via `imin()/imax()`,
 *    `jmin()/jmax()`, `kmin()/kmax()`; a linear index
 *    `idx(ix,iy,iz) -> std::size_t`; an optional `prepare()` for any
 *    once-per-step backend pre-processing (e.g. spectral FFTs); and
 *    `operator()(ix,iy,iz) -> G`.
 *
 * Multi-field models opt in to the tuple protocol (`as_tuple()` member
 * or a plain `std::tuple`, see `tuple_protocol.hpp`) so the driver can
 * scatter the increments into per-field `du` buffers passed as a
 * tuple-like `du` argument.
 *
 * Cells outside `[imin,imax) x [jmin,jmax) x [kmin,kmax)` are left
 * untouched (`du` is not cleared here). Parallelized with
 * `#pragma omp parallel for collapse(2)` over `(iz, iy)`.
 *
 * @see openpfc/kernel/field/grad_concepts.hpp for the per-member detection
 *      concepts that backends use to fill only requested members
 * @see openpfc/kernel/field/tuple_protocol.hpp for the multi-field
 *      bundling convention
 * @see openpfc/kernel/field/fd_gradient.hpp for the FD evaluator
 * @see openpfc/kernel/field/spectral_gradient.hpp for the spectral evaluator
 * @see openpfc/kernel/simulation/steppers/euler.hpp for a stepper that
 *      uses this driver
 */

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/field/tuple_protocol.hpp>

namespace pfc::sim {

namespace detail {

/**
 * @brief Scatter one model RHS return value into one or more `du` buffers.
 *
 * Single-field path: `du` is a `double*` and `inc` is a scalar — write one
 * cell. Multi-field path: `du` is a tuple-like of `double*` and `inc`
 * exposes a tuple-like view via the tuple protocol — fan out element by
 * element with `std::apply`.
 */
template <class DuOut, class Inc>
inline void scatter(DuOut &du, std::size_t k, const Inc &inc) {
  if constexpr (std::is_pointer_v<std::remove_reference_t<DuOut>>) {
    // Single-field leaf case.
    du[k] = static_cast<double>(inc);
  } else {
    auto &&du_tuple = pfc::field::detail::to_tuple(du);
    auto &&inc_tuple = pfc::field::detail::to_tuple(inc);
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(du_tuple)>> ==
                      std::tuple_size_v<std::remove_cvref_t<decltype(inc_tuple)>>,
                  "for_each_interior: number of du buffers does not match "
                  "the arity of the increments returned by model.rhs(...).");
    std::apply(
        [&](auto &...du_ptrs) {
          std::apply(
              [&](const auto &...inc_vals) {
                ((du_ptrs[k] = static_cast<double>(inc_vals)), ...);
              },
              inc_tuple);
        },
        du_tuple);
  }
}

} // namespace detail

/**
 * @brief Apply `model.rhs(t, eval(i,j,k))` over every interior cell.
 *
 * @tparam Model  Type providing `rhs(double, const G&) -> Inc`, where `G`
 *                is the grads aggregate `Eval` returns and `Inc` is either
 *                a scalar (single-field) or a tuple-protocol bundle
 *                (multi-field).
 * @tparam Eval   Per-point evaluator (e.g. `pfc::field::FdGradient<G>`).
 * @tparam DuOut  Either `double*` (single-field) or a tuple-protocol
 *                bundle of `double*` (multi-field).
 *
 * @param model  Physics model.
 * @param eval   Backend evaluator. `eval.prepare()` runs once per call.
 * @param du     Output buffer(s); see `DuOut`.
 * @param t      Simulation time forwarded to `rhs`.
 */
template <class Model, class Eval, class DuOut>
inline void for_each_interior(const Model &model, Eval &eval, DuOut du, double t) {
  eval.prepare();
  const int kmin = eval.kmin();
  const int kmax = eval.kmax();
  const int jmin = eval.jmin();
  const int jmax = eval.jmax();
  const int imin = eval.imin();
  const int imax = eval.imax();
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int iz = kmin; iz < kmax; ++iz) {
    for (int iy = jmin; iy < jmax; ++iy) {
      for (int ix = imin; ix < imax; ++ix) {
        const auto g = eval(ix, iy, iz);
        const auto inc = model.rhs(t, g);
        detail::scatter(du, eval.idx(ix, iy, iz), inc);
      }
    }
  }
}

} // namespace pfc::sim
