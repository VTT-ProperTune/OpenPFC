// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file nodal_multigrid_gpu.hpp
 * @brief Device launchers of `pfc::solvers::NodalMultigrid` (internal).
 *
 * Each launcher runs one kernel over the owned nodes (or a coarse region) of a
 * level on the default stream, applying the shared point functions of
 * nodal_multigrid_kernels.hpp.
 */

#include <cstddef>

#include <openpfc/solvers/nodal_multigrid_kernels.hpp>

namespace pfc::solvers::gpu::nmg {

using Level = ::pfc::solvers::nmg::Level;
using Operator = ::pfc::solvers::nmg::Operator;

#define OPENPFC_NMG_DECLARE(sfx)                                                    \
  void fill_##sfx(double *p, std::size_t n, double v);                              \
  void axpy_owned_##sfx(const Level &L, double a, const double *x, double *y);      \
  void add_owned_##sfx(const Level &L, double *y, double v);                        \
  double dot_owned_##sfx(const Level &L, const double *x, const double *y);         \
  void hash_owned_##sfx(const Level &L, double *x);                                 \
  void copy_coefficient_##sfx(const Level &L, const double *src, int src_halo,      \
                              double *dst);                                         \
  void apply_##sfx(const Level &L, const Operator &A, const double *u,              \
                   double *out);                                                    \
  void scaled_apply_##sfx(const Level &L, const Operator &A, const double *invd,    \
                          const double *u, double *out);                            \
  void inverse_diagonal_##sfx(const Level &L, const Operator &A, double *invd);     \
  void residual_##sfx(const Level &L, const Operator &A, const double *f,           \
                      const double *u, double *r, int zero_u);                      \
  void chebyshev_##sfx(const Level &L, const Operator &A, const double *f,          \
                       const double *u, const double *invd, double *d, double c1,   \
                       double c2, int zero_u);                                      \
  void gauss_seidel_##sfx(const Level &L, const Operator &A, const double *f,       \
                          const double *invd, double *u, int colour);               \
  void restrict_residual_##sfx(const Level &F, const double *rf, const Level &G,    \
                               double *fg, const int off[3]);                       \
  void prolong_add_##sfx(const Level &F, double *uf, const Level &G,                \
                         const double *ug, const int off[3]);                       \
  void restrict_coefficient_##sfx(const Level &F, const double *af, const Level &G, \
                                  double *ag, const int off[3]);                    \
  void galerkin_##sfx(const Level &F, const Operator &Af, const Level &G,           \
                      double *sg, std::size_t gplane, const int off[3]);

#if defined(OpenPFC_ENABLE_CUDA)
OPENPFC_NMG_DECLARE(cuda)
#endif
#if defined(OpenPFC_ENABLE_HIP)
OPENPFC_NMG_DECLARE(hip)
#endif
#undef OPENPFC_NMG_DECLARE

} // namespace pfc::solvers::gpu::nmg
