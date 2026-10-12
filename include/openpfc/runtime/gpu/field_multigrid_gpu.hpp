// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_multigrid_gpu.hpp
 * @brief Device launchers of `pfc::solvers::FieldMultigrid` (internal).
 *
 * Each launcher runs one kernel over the owned cells of a level on the default
 * stream, applying the shared point functions of field_multigrid_kernels.hpp.
 */

#include <cstddef>

#include <openpfc/solvers/field_multigrid_kernels.hpp>

namespace pfc::solvers::gpu::fmg {

using Level = ::pfc::solvers::fmg::Level;
using Coefficients = ::pfc::solvers::fmg::Coefficients;

#define OPENPFC_FMG_DECLARE(sfx)                                                    \
  void fill_##sfx(double *p, std::size_t n, double v);                              \
  void fill_owned_##sfx(const Level &L, double *p, double v);                       \
  void copy_owned_##sfx(const Level &L, const double *src, double *dst);            \
  void apply_##sfx(const Level &L, const Coefficients &C, const double *u,          \
                   double *out);                                                    \
  void smooth_##sfx(const Level &L, const Coefficients &C, const double *f,         \
                    double *u, int colour);                                         \
  void residual_##sfx(const Level &L, const Coefficients &C, const double *f,       \
                      const double *u, double *r);                                  \
  void restrict_coefficients_##sfx(const Level &F, const Coefficients &Cf,          \
                                   const Level &G, double *c, double *dx,           \
                                   double *dy, double *dz, double *m);              \
  void restrict_residual_##sfx(const Level &F, const double *rf, const double *mf,  \
                               const Level &G, const double *mg, double *fg);       \
  void prolong_add_##sfx(const Level &F, const double *mf, double *uf,              \
                         const Level &G, const double *ug);

#if defined(OpenPFC_ENABLE_CUDA)
OPENPFC_FMG_DECLARE(cuda)
#endif
#if defined(OpenPFC_ENABLE_HIP)
OPENPFC_FMG_DECLARE(hip)
#endif
#undef OPENPFC_FMG_DECLARE

} // namespace pfc::solvers::gpu::fmg
