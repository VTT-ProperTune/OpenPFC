// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file structure_factor.hpp
 * @brief Azimuthally averaged structure factor and the domain length it gives.
 *
 * Shared by the applications whose science case is a selected length scale:
 * Fe-Cr coarsening, thin-film dewetting, surface-diffusion patterning and
 * higher-order PFC crystal selection all read the same two numbers off it.
 *
 * @details
 * The observable that turns a picture of a decomposing alloy into a number.
 * For the composition fluctuation \f$\delta c = c - \bar c\f$,
 *
 * \f[
 *   S(\mathbf k,t) = \bigl|\widehat{\delta c}(\mathbf k,t)\bigr|^2 ,
 * \f]
 *
 * averaged over shells of constant \f$|\mathbf k|\f$. Two numbers are read off
 * it:
 *
 * \f[
 *   k_1(t) = \frac{\sum_k k\,S(k,t)}{\sum_k S(k,t)},
 *   \qquad
 *   L(t) = \frac{2\pi}{k_1(t)} .
 * \f]
 *
 * \f$k_1\f$ is the first moment rather than the peak position: the peak is a
 * single noisy bin, while the moment uses the whole spectrum and is far
 * steadier late in a run when \f$S\f$ is broad. \f$L\f$ is the standard
 * characteristic domain size, and it is the quantity whose growth exponent is
 * compared against the \f$t^{1/3}\f$ law for conserved dynamics.
 *
 * The \f$\mathbf k = 0\f$ bin is excluded throughout; it carries the mean
 * composition, which conserved dynamics holds fixed and which would otherwise
 * dominate the moment.
 *
 * Shell accumulation is a distributed reduction: each rank bins the part of
 * the transform it owns and the histograms are summed across the communicator,
 * so the result does not depend on the decomposition.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace pfc::apps {

/// One azimuthally averaged spectrum plus the scalars derived from it.
struct StructureFactor {
  std::vector<double> k;      ///< shell centre wave numbers
  std::vector<double> S;      ///< shell-averaged \f$S(k)\f$
  double k1{0.0};             ///< first moment \f$\sum kS/\sum S\f$
  double k_peak{0.0};         ///< wave number of the largest shell
  double S_peak{0.0};         ///< that shell's value
  double total_power{0.0};    ///< \f$\sum S\f$ over all non-zero shells

  /// Characteristic domain size \f$L = 2\pi/k_1\f$, in code length units.
  [[nodiscard]] double domain_length() const {
    return (k1 > 0.0) ? 2.0 * std::numbers::pi / k1 : 0.0;
  }
  /// Wavelength of the dominant shell, \f$2\pi/k_{\mathrm{peak}}\f$.
  [[nodiscard]] double dominant_wavelength() const {
    return (k_peak > 0.0) ? 2.0 * std::numbers::pi / k_peak : 0.0;
  }
};

/**
 * @brief Bin \f$|\hat{\delta c}|^2\f$ into shells of \f$|\mathbf k|\f$.
 *
 * @param outbox   spectral bounds owned by this rank
 * @param domain   grid geometry, for the wave numbers
 * @param spectrum transform values over @p outbox, as the FFT leaves them
 * @param comm     communicator to reduce the histograms over
 * @param n_bins   number of shells between 0 and the Nyquist radius
 *
 * The caller supplies the transform of the *fluctuation*; whether the mean was
 * removed before transforming or the zero bin is simply skipped does not
 * matter, because the \f$k=0\f$ shell is dropped either way.
 */
[[nodiscard]] inline StructureFactor
shell_average(const pfc::Box3i &outbox, const pfc::Domain &domain,
              const std::complex<double> *spectrum, MPI_Comm comm,
              int n_bins = 64) {
  const auto size = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);

  // Nyquist radius of the coarsest active axis bounds the useful range.
  double k_max = 0.0;
  for (int d = 0; d < 3; ++d)
    if (size[d] > 1) k_max = std::max(k_max, std::numbers::pi / dx[d]);
  if (k_max <= 0.0 || n_bins < 1) return {};

  const double bin_width = k_max / n_bins;
  std::vector<double> power(n_bins, 0.0), counts(n_bins, 0.0);

  pfc::fft::kspace::for_each_kpoint(
      outbox, domain,
      [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
        const double kk = std::sqrt(kx * kx + ky * ky + kz * kz);
        if (kk <= 0.0) return; // mean composition, not a fluctuation
        const int bin = static_cast<int>(kk / bin_width);
        if (bin < 0 || bin >= n_bins) return;
        power[bin] += std::norm(spectrum[i]);
        counts[bin] += 1.0;
      });

  std::vector<double> global_power(n_bins, 0.0), global_counts(n_bins, 0.0);
  MPI_Allreduce(power.data(), global_power.data(), n_bins, MPI_DOUBLE, MPI_SUM,
                comm);
  MPI_Allreduce(counts.data(), global_counts.data(), n_bins, MPI_DOUBLE,
                MPI_SUM, comm);

  StructureFactor out;
  out.k.reserve(n_bins);
  out.S.reserve(n_bins);
  double moment = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    if (global_counts[b] <= 0.0) continue;
    const double kc = (b + 0.5) * bin_width;
    const double s = global_power[b] / global_counts[b];
    out.k.push_back(kc);
    out.S.push_back(s);
    out.total_power += s;
    moment += kc * s;
    if (s > out.S_peak) {
      out.S_peak = s;
      out.k_peak = kc;
    }
  }
  if (out.total_power > 0.0) out.k1 = moment / out.total_power;
  return out;
}

/**
 * @brief Least-squares exponent @p n in \f$L(t)\propto t^{n}\f$.
 *
 * Fits a straight line to \f$(\ln t,\ln L)\f$ over the samples given. Only
 * points with positive time and length take part, so an initial sample at
 * \f$t=0\f$ can be passed harmlessly. Returns 0 when fewer than two usable
 * points remain.
 *
 * The caller decides which interval is late enough for a power law to mean
 * anything; this routine does not guess.
 */
[[nodiscard]] inline double coarsening_exponent(const std::vector<double> &t,
                                                const std::vector<double> &L) {
  const std::size_t n = std::min(t.size(), L.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0, count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (t[i] <= 0.0 || L[i] <= 0.0) continue;
    const double x = std::log(t[i]), y = std::log(L[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    count += 1.0;
  }
  if (count < 2.0) return 0.0;
  const double denom = count * sxx - sx * sx;
  if (std::abs(denom) < 1.0e-300) return 0.0;
  return (count * sxy - sx * sy) / denom;
}

} // namespace pfc::apps
