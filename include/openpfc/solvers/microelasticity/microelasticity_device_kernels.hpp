// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity_device_kernels.hpp
 * @brief Device launch surface for the Green operator, Eyre–Milton local
 *        reflection, residual reduction, and equation-(7) finalise.
 *
 * One declaration set for both vendors. The definitions are compiled from
 * `microelasticity_kernels_gpu.inc` into `openpfc_microelasticity_cuda` or
 * `openpfc_microelasticity_hip`; a build links exactly one of them.
 *
 * Transcription of the host loops in `microelasticity.hpp`. Not a
 * `SpectralETDOps` path: six-component cubic contractions with a spatially
 * varying stiffness. The inner loop never copies the tensor fields to the
 * host; the residual reduction is a per-block max of two doubles.
 */

#include <cstddef>

namespace pfc::solvers::device_detail {

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
  MEStiffness stiffness_at_one{};
  MEStiffness stiffness_at_zero{};
  MEStiffness reference{};
  MEStiffness stiffness_difference{};
  double pattern[kSym]{};
  long long n{0};
};

void me_build_polarisation(const double *h, const double *amp,
                           const MESym6Const &eps, const MESym6 &tau,
                           const MEParams &p);

void me_eyre_milton_local(const double *h, const double *amp, const MESym6Const &zin,
                          const MESym6 &eps, const MESym6 &zout, const MEParams &p,
                          double *block_max, int n_blocks);

[[nodiscard]] int me_local_block_count(long long n) noexcept;

void me_green_multiply(const MESym6 &hat, const double *kx, const double *ky,
                       const double *kz, const MESym6Const &g, long long n_outbox,
                       long long zero_mode, const double applied[kSym],
                       double n_global);

void me_finalise(const double *h, const double *amp, const double *dh_dphi,
                 const double *damp_dphi, const MESym6Const &eps, const MESym6 &sig,
                 double *f_el, double *elastic_energy_derivative, const MEParams &p,
                 int want_dfel);

void me_fill(double *dst, double value, long long n);

/// `dst[i] = src[i]` on the device.
void me_copy(const double *src, double *dst, long long n);

/// SIMP weight and slope of a density, \f$\phi\f$ clamped to \f$[0,1]\f$:
/// `w = phi^p`, `dw = p phi^(p-1)` (`w = phi`, `dw = 1` for \f$p=1\f$), as
/// `SimpInterpolation` on the host.
void me_simp_weights(const double *phi, double *w, double *dw, double exponent,
                     long long n);

void me_block_sum(const double *x, double *partial, long long n, int n_blocks);

void me_assemble_from_padded(const double *phi, const double *U, const double *theta,
                             double *h, double *amp, double *damp, int nx, int ny,
                             int nz, int hw, long long sy, long long sz,
                             double eps_c, double eps_T, double u_ref,
                             double theta_ref, double *block_sum, int n_blocks);

void me_copy_owned_to_padded(const double *src, double *dst, int nx, int ny, int nz,
                             int hw, long long sy, long long sz);

/// Per-block `(sum p, max |dfel|, max σ_vm)`. `block` length `3 * n_blocks`.
void me_report_stats(const MESym6Const &sig, const double *dfel, double *block,
                     long long n, int n_blocks);

/// Per-block sum of \f$\boldsymbol\sigma:\boldsymbol\varepsilon^{*}\f$.
void me_sigma_estar_sum(const MESym6Const &sig, const double *amp, const MEParams &p,
                        double *partial, int n_blocks);

} // namespace pfc::solvers::device_detail
