// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity_hip_kernels.hpp
 * @brief HIP launch surface for the Green operator, Eyre–Milton local
 *        reflection, residual reduction, and equation-(7) finalise.
 *
 * Transcription of the host loops in `microelasticity.hpp`. Not a
 * `SpectralETDOps` path: six-component cubic contractions with a spatially
 * varying stiffness. The inner loop never copies the tensor fields to the
 * host; the residual reduction is a per-block max of two doubles.
 */

#include <cstddef>

namespace pfc::apps::hip_detail {

inline constexpr int kSym = 6;

struct MEStiffness {
  double c11{0.0};
  double c12{0.0};
  double c44{0.0};
};

struct MESym6 {
  double *c[kSym]{};
};

struct MESym6Const {
  const double *c[kSym]{};
};

struct MEParams {
  MEStiffness c_solid{};
  MEStiffness c_liquid{};
  MEStiffness c0{};
  MEStiffness dc{};
  double pattern[kSym]{};
  long long n{0};
};

void me_build_polarisation(const double *h, const double *amp,
                           const MESym6Const &eps, const MESym6 &tau,
                           const MEParams &p);

void me_eyre_milton_local(const double *h, const double *amp,
                          const MESym6Const &zin, const MESym6 &eps,
                          const MESym6 &zout, const MEParams &p,
                          double *block_max, int n_blocks);

[[nodiscard]] int me_local_block_count(long long n) noexcept;

void me_green_multiply(const MESym6 &hat, const double *kx, const double *ky,
                       const double *kz, const MESym6Const &g,
                       long long n_outbox, long long zero_mode,
                       const double applied[kSym], double n_global);

void me_finalise(const double *h, const double *amp, const double *dh_dphi,
                 const double *damp_dphi, const MESym6Const &eps,
                 const MESym6 &sig, double *f_el, double *dfel_dphi,
                 const MEParams &p, int want_dfel);

void me_fill(double *dst, double value, long long n);

void me_block_sum(const double *x, double *partial, long long n, int n_blocks);

void me_assemble_from_padded(const double *phi, const double *U,
                             const double *theta, double *h, double *amp,
                             double *damp, int nx, int ny, int nz, int hw,
                             long long sy, long long sz, double eps_c,
                             double eps_T, double u_ref, double theta_ref,
                             double *block_sum, int n_blocks);

void me_copy_owned_to_padded(const double *src, double *dst, int nx, int ny,
                             int nz, int hw, long long sy, long long sz);

/// Per-block `(sum p, max |dfel|, max σ_vm)`. `block` length `3 * n_blocks`.
void me_report_stats(const MESym6Const &sig, const double *dfel, double *block,
                      long long n, int n_blocks);

} // namespace pfc::apps::hip_detail
