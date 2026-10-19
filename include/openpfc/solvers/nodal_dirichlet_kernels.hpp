// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_kernels.hpp
 * @brief Shared CPU/CUDA/HIP point mathematics for free-node Q1 Dirichlet FEM.
 *
 * This is a mathematical kernel, not a second Decomposition, Field or
 * HaloExchange. Use the same nmg::Level geometry as the periodic nodal
 * multigrid but a two-cell padded vector, with fixed physical boundary
 * nodes eliminated. The host solver validates each rank's transfer reach
 * before a stencil or P/P^T is evaluated. Device wrappers reuse the
 * existing nodal_multigrid GPU translation units and memory runtime.
 */

#include <openpfc/solvers/nodal_multigrid_kernels.hpp>

namespace pfc::solvers::nmg::dirichlet {

inline constexpr int kHalo = 2;

/// Fixed-coefficient Q1 Galerkin matrix row, in the existing offset ordering.
/// Trivially copyable by value into a CUDA/HIP kernel launch.
struct Weights {
  double v[27]{};
};

/// An interior row touches 8 hexahedra. The element stiffness and consistent
/// mass are shared with nmg::set_element_data and its periodic counterpart.
inline Weights make_weights(const double spacing[3], double diffusion,
                            double reaction) {
  ::pfc::solvers::nmg::Level element;
  ::pfc::solvers::nmg::set_element_data(element, spacing);
  Weights result;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int slot = ::pfc::solvers::nmg::offset_index(dx, dy, dz);
        const double count = (dx == 0 ? 2.0 : 1.0) *
                             (dy == 0 ? 2.0 : 1.0) *
                             (dz == 0 ? 2.0 : 1.0);
        result.v[slot] =
            diffusion * count * element.ke[slot] +
            reaction * element.mass[slot];
      }
  return result;
}

/// Global free-node coordinates: outside the physical domain the coefficient
/// is exactly zero, irrespective of what happens to live in the halo buffer.
/// Coordinates INSIDE the physical free-node box must lie within the caller's
/// two-cell padded local Field, as guaranteed by the hierarchy constructor.
OPENPFC_HD inline double sample(const Level &L, const double *padded,
                                 int gi, int gj, int gk) {
  if (gi < 0 || gj < 0 || gk < 0 || gi >= L.gn[0] || gj >= L.gn[1] ||
      gk >= L.gn[2])
    return 0.0;
  return padded[index(L, kHalo, gi - L.low[0], gj - L.low[1],
                      gk - L.low[2])];
}

/// Matrix-free row at one OWNED free node; zero Dirichlet boundaries have
/// already been eliminated. No special-case periodic wrap is performed.
OPENPFC_HD inline double action(const Level &L, const Weights &weights,
                                 const double *padded, int i, int j, int k) {
  const int gi = L.low[0] + i, gj = L.low[1] + j, gk = L.low[2] + k;
  double result = 0.0;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
        result += weights.v[offset_index(dx, dy, dz)] *
                  sample(L, padded, gi + dx, gj + dy, gk + dz);
  return result;
}

/// A fused row-owned damped Jacobi update. The caller keeps the old iterate
/// and its exchanged halo constant until every owned node was updated.
OPENPFC_HD inline double jacobi(const Level &L, const Weights &weights,
                                 const double *padded_old, double old_value,
                                 double rhs_value, double omega_over_diagonal,
                                 int i, int j, int k) {
  return old_value + omega_over_diagonal *
                         (rhs_value - action(L, weights, padded_old, i, j, k));
}

/// R = P^T, NOT volume-normalized full weighting. Coarse free-node index c
/// maps to fine free-node index 2c+1 on a nested Dirichlet hierarchy.
OPENPFC_HD inline double restrict_pt(const Level &F,
                                      const double *fine_padded,
                                      int ci, int cj, int ck) {
  const int fi = 2 * ci + 1, fj = 2 * cj + 1, fk = 2 * ck + 1;
  double value = 0.0;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const double weight = (dx ? 0.5 : 1.0) *
                              (dy ? 0.5 : 1.0) * (dz ? 0.5 : 1.0);
        value += weight * sample(F, fine_padded, fi + dx, fj + dy, fk + dz);
      }
  return value;
}

/// Trilinear P from nested coarse free nodes to one fine free node.
OPENPFC_HD inline double prolong_p(const Level &C,
                                    const double *coarse_padded,
                                    int fi, int fj, int fk) {
  const int f[3] = {fi + 1, fj + 1, fk + 1};
  const int base[3] = {f[0] / 2 - 1, f[1] / 2 - 1, f[2] / 2 - 1};
  const int choices[3] = {(f[0] & 1) ? 2 : 1, (f[1] & 1) ? 2 : 1,
                          (f[2] & 1) ? 2 : 1};
  const double weight = ((f[0] & 1) ? 0.5 : 1.0) *
                        ((f[1] & 1) ? 0.5 : 1.0) *
                        ((f[2] & 1) ? 0.5 : 1.0);
  double value = 0.0;
  for (int dz = 0; dz < choices[2]; ++dz)
    for (int dy = 0; dy < choices[1]; ++dy)
      for (int dx = 0; dx < choices[0]; ++dx)
        value += weight *
                 sample(C, coarse_padded, base[0] + dx, base[1] + dy,
                        base[2] + dz);
  return value;
}

} // namespace pfc::solvers::nmg::dirichlet
