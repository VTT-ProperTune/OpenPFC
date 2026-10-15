// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file overlap.hpp
 * @brief Gaussian overlap energy between labelled indicators.
 *
 * For one amplitude field phi and persistent labels L, the indicators are
 * m_i = phi 1[L = i]. Their pairwise overlap energy
 *
 *     F = omega sum_{i<j} int (G_s * m_i)(G_s * m_j) dx
 *       = omega/2 int [ M (K * M) - sum_i m_i (K * m_i) ] dx,
 *
 * with G_s a normalized Gaussian of standard deviation s, K = G_s * G_s
 * (variance 2 s^2, spectrum exp(-s^2 |k|^2)) and M = sum_i m_i, is the
 * heat-kernel approximation of interface area used in threshold dynamics
 * (Esedoglu and Otto, Comm. Pure Appl. Math. 68 (2015) 808-864, eq. 13).
 * Two labelled half-spaces with phi = 1 have F = omega s / sqrt(pi) per unit
 * area, so `weight(gamma, s)` = sqrt(pi) gamma / s gives a boundary energy
 * gamma per unit area. The kernel only measures area; gamma is an input.
 *
 * `evaluate` returns the volume average F / V and the force
 * f(x) = N dF/V/dphi(x) = omega [K * (M - m_{L(x)})](x) on labelled cells
 * (zero elsewhere), so that F / V = (1 / 2N) sum_x phi(x) f(x). The labels
 * are constants: the derivative is taken at fixed L. Cost: one forward and
 * one backward transform per identity.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/grain/topology.hpp>

namespace pfc::grain::overlap {

/// omega = sqrt(pi) gamma / s: boundary energy gamma per unit area.
[[nodiscard]] inline double weight(double boundary_energy, double sigma) {
  if (!(sigma > 0.0) || !std::isfinite(sigma) || !(boundary_energy >= 0.0) ||
      !std::isfinite(boundary_energy))
    throw std::invalid_argument(
        "overlap weight needs sigma > 0 and a finite gamma >= 0");
  return std::sqrt(std::numbers::pi) * boundary_energy / sigma;
}

/// Spectrum of K = G_s * G_s, exp(-s^2 |k|^2), on `outbox` with the physical
/// wavenumbers of `domain`.
[[nodiscard]] inline std::vector<double> kernel(const Box3i &outbox,
                                                const Domain &domain, double sigma) {
  if (!(sigma > 0.0) || !std::isfinite(sigma))
    throw std::invalid_argument("overlap kernel needs sigma > 0");
  std::vector<double> out(static_cast<std::size_t>(outbox.size[0]) * outbox.size[1] *
                          outbox.size[2]);
  fft::kspace::for_each_kpoint(
      outbox, domain,
      [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
        out[idx] = std::exp(-sigma * sigma * (kx * kx + ky * ky + kz * kz));
      });
  return out;
}

/**
 * @brief F / V and its force on the local inbox.
 * @param fft distributed host transform of the field layout
 * @param spectrum `kernel(...)` on the transform's outbox
 * @param omega `weight(...)`
 * @param phi local amplitude
 * @param labels local labels (background 0 carries no indicator)
 * @param identities every identity present on any rank, each once
 * @param force local output, same size as phi
 * @param comm communicator of the decomposition
 */
inline double evaluate(fft::IHostFFT &fft, std::span<const double> spectrum,
                       double omega, std::span<const double> phi,
                       std::span<const Id> labels, std::span<const Id> identities,
                       std::span<double> force, MPI_Comm comm) {
  const std::size_t n = fft.size_inbox();
  if (phi.size() != n || labels.size() != n || force.size() != n ||
      spectrum.size() != fft.size_outbox())
    throw std::invalid_argument(
        "overlap: field, label, force or kernel size mismatch");
  std::vector<double> mask(n), smooth(n), total(n, 0.0), own(n, 0.0);
  std::vector<std::complex<double>> hat(fft.size_outbox());
  for (Id id : identities) {
    if (id == background) continue;
    for (std::size_t i = 0; i < n; ++i) mask[i] = labels[i] == id ? phi[i] : 0.0;
    fft.forward(mask, hat);
    for (std::size_t c = 0; c < hat.size(); ++c) hat[c] *= spectrum[c];
    fft.backward(hat, smooth);
    for (std::size_t i = 0; i < n; ++i) {
      total[i] += smooth[i];
      if (labels[i] == id) own[i] = smooth[i];
    }
  }
  double local = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    force[i] = labels[i] != background ? omega * (total[i] - own[i]) : 0.0;
    local += phi[i] * force[i];
  }
  double sums[2] = {local, static_cast<double>(n)};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
  return sums[1] > 0.0 ? 0.5 * sums[0] / sums[1] : 0.0;
}

} // namespace pfc::grain::overlap
