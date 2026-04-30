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
 *  - `Model` must expose `rhs(double t, const pfc::field::GradPoint&)
 *    -> double`.
 *  - `Eval` must expose interior bounds via `imin()/imax()`, `jmin()/jmax()`,
 *    `kmin()/kmax()`; a linear index `idx(ix,iy,iz) -> std::size_t`; an
 *    optional `prepare()` for any once-per-step backend pre-processing
 *    (e.g. spectral FFTs); and `operator()(ix,iy,iz) -> pfc::field::GradPoint`.
 *
 * Cells outside `[imin,imax) x [jmin,jmax) x [kmin,kmax)` are left untouched
 * (`du` is not cleared here). Parallelized with `#pragma omp parallel for
 * collapse(2)` over `(iz, iy)`.
 *
 * @see openpfc/kernel/field/grad_point.hpp for the interface struct
 * @see openpfc/kernel/field/fd_gradient.hpp for the FD evaluator
 * @see openpfc/kernel/field/spectral_gradient.hpp for the spectral evaluator
 * @see openpfc/kernel/simulation/steppers/euler.hpp for a stepper that uses
 *      this driver
 */

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <openpfc/kernel/field/grad_point.hpp>

namespace pfc::sim {

template <class Model, class Eval>
inline void for_each_interior(const Model &model, Eval &eval, double *du, double t) {
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
        const pfc::field::GradPoint g = eval(ix, iy, iz);
        du[eval.idx(ix, iy, iz)] = model.rhs(t, g);
      }
    }
  }
}

} // namespace pfc::sim
