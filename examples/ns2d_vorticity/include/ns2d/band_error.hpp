// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file band_error.hpp
 * @brief Common-band field comparison for the 2-D spectral NS ladder (#224).
 *
 * A fine-grid vorticity coefficient vector is restricted with
 * `restrict_hat_by_k` onto a coarser plane. The inverse transform of that
 * band is the fine solution's content on the coarse modes. The coarse
 * solution is then subtracted on those same modes. Grids are not compared
 * by interpolating unmatched collocation points.
 *
 * Norms are discrete RMS values, `sqrt(mean(error^2))` over the global
 * cell count, and the matching relative norm. `omega_linf` is the max
 * absolute vorticity difference. One MPI rank owns every mode; a split
 * pencil would drop coefficients that the other rank holds.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/fft/kspace_iterator.hpp>

#include <ns2d/spectral.hpp>
#include <ns2d/vorticity_stream.hpp>

namespace ns2d {

struct CommonBandError {
  double omega_l2{0.0};
  double omega_l2_rel{0.0};
  double omega_linf{0.0};
  double velocity_l2{0.0};
  double velocity_l2_rel{0.0};
};

/// Modes with `max(|k_i|, |k_j|) > N/6` over the full vorticity RMS.
/// The 2/3 rule already zeros modes past about `N/3`, so this fraction
/// is the outer half of the retained band, not energy in the masked tail.
inline double outer_band_rms_fraction(VorticityStreamCPU &solver, MPI_Comm comm) {
  std::vector<SpectralPlane::Complex> hat;
  solver.copy_omega_hat(hat);
  auto &plane = solver.plane();
  const auto n = plane.gsize();
  const int k_cut = n[0] / 6;
  pfc::fft::kspace::for_each_kpoint(
      plane.outbox(), n, plane.spacing(),
      [&](std::size_t idx, double, double, double, int i, int j, int) {
        const int ki = std::abs(signed_wave_index(i, n[0]));
        const int kj = std::abs(signed_wave_index(j, n[1]));
        if (std::max(ki, kj) <= k_cut) hat[idx] = SpectralPlane::Complex{};
      });
  std::vector<double> high(plane.in_n(), 0.0);
  plane.fft().backward(hat, high);

  double high_sq = 0.0;
  double full_sq = 0.0;
  solver.omega().for_each_owned([&](int i, int j, int k) {
    const auto c = plane.real_idx(i, j, k);
    const double w = solver.omega()(i, j, k);
    high_sq += high[c] * high[c];
    full_sq += w * w;
  });
  double g_high = 0.0;
  double g_full = 0.0;
  MPI_Allreduce(&high_sq, &g_high, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&full_sq, &g_full, 1, MPI_DOUBLE, MPI_SUM, comm);
  if (g_full <= 0.0) return 0.0;
  return std::sqrt(g_high / g_full);
}

inline CommonBandError
common_band_error_from_hat(SpectralPlane &reference_plane,
                           const std::vector<SpectralPlane::Complex> &reference_hat,
                           VorticityStreamCPU &coarse, MPI_Comm comm) {
  coarse.recover_velocity_from_omega();
  std::vector<SpectralPlane::Complex> restricted;
  restrict_hat_by_k(reference_plane, reference_hat, coarse.plane(), restricted);

  auto &plane = coarse.plane();
  std::vector<double> omega_ref(plane.in_n(), 0.0);
  plane.fft().backward(restricted, omega_ref);

  std::vector<SpectralPlane::Complex> psi(restricted.size());
  plane.poisson(restricted, psi);
  std::vector<double> u_ref(plane.in_n(), 0.0);
  std::vector<double> v_ref(plane.in_n(), 0.0);
  plane.curl_from_hat(psi, u_ref, v_ref);

  double e2 = 0.0;
  double r2 = 0.0;
  double linf = 0.0;
  double ve2 = 0.0;
  double vr2 = 0.0;
  coarse.omega().for_each_owned([&](int i, int j, int k) {
    const auto c = plane.real_idx(i, j, k);
    const double dw = coarse.omega()(i, j, k) - omega_ref[c];
    e2 += dw * dw;
    r2 += omega_ref[c] * omega_ref[c];
    linf = std::max(linf, std::abs(dw));
    const double du = coarse.u()[c] - u_ref[c];
    const double dv = coarse.v()[c] - v_ref[c];
    ve2 += du * du + dv * dv;
    vr2 += u_ref[c] * u_ref[c] + v_ref[c] * v_ref[c];
  });

  double ge2 = 0.0;
  double gr2 = 0.0;
  double glinf = 0.0;
  double gve2 = 0.0;
  double gvr2 = 0.0;
  MPI_Allreduce(&e2, &ge2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&r2, &gr2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&linf, &glinf, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&ve2, &gve2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&vr2, &gvr2, 1, MPI_DOUBLE, MPI_SUM, comm);

  const auto gs = plane.gsize();
  const double ncells =
      static_cast<double>(gs[0]) * static_cast<double>(gs[1]) * gs[2];
  CommonBandError err;
  err.omega_l2 = std::sqrt(ge2 / ncells);
  err.omega_linf = glinf;
  const double ref_l2 = std::sqrt(gr2 / ncells);
  err.omega_l2_rel = (ref_l2 > 0.0) ? err.omega_l2 / ref_l2 : 0.0;
  err.velocity_l2 = std::sqrt(gve2 / ncells);
  const double vref = std::sqrt(gvr2 / ncells);
  err.velocity_l2_rel = (vref > 0.0) ? err.velocity_l2 / vref : 0.0;
  return err;
}

} // namespace ns2d
