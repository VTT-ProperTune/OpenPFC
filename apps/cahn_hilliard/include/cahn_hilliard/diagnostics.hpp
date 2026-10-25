// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <vector>

#include <cahn_hilliard/cahn_hilliard_physics.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/fft/power_spectrum.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

namespace cahn_hilliard {

struct DiagnosticSample {
  double mean{}, mass{}, minimum{}, maximum{}, bulk_energy{}, gradient_energy{};
  double invalid_cells{};
  /// Spectral observables; zero when the spectrum carries no power.
  double k1{}, domain_length{}, k_peak{}, dominant_wavelength{};
  [[nodiscard]] double total_energy() const { return bulk_energy + gradient_energy; }
};

/**
 * @brief Least-squares exponent n in L(t) proportional to t^n.
 *
 * Fits (ln t, ln L). Non-positive samples are ignored. Returns 0 when fewer
 * than two usable points remain. The caller chooses the time window.
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

/** @brief Collective diagnostics of the current field, not the lagged ETD RHS.
 * Computes the periodic spectral energy as integral(f(c)-kappa*c*lap(c)/2).
 * Uses the session FFT/backend, with host reductions at output cadence only.
 * Scratch fields are separate from integrator state and are not checkpointed.
 */
template <class MemorySpace> class Diagnostics {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;
  using RealField = typename Ops::RealField;
  using ComplexField = typename Ops::ComplexField;

public:
  Diagnostics(const pfc::Domain &domain, typename Ops::FFT &fft, MPI_Comm comm)
      : m_domain(domain), m_fft(fft), m_comm(comm),
        m_hat(domain, fft.get_outbox_bounds(), 0),
        m_lap(domain, fft.get_inbox_bounds(), 0),
        m_weights(Ops::make_real(fft.size_outbox())),
        m_work(Ops::make_complex(fft.size_outbox())) {
    std::vector<double> weights(fft.size_outbox());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double x, double y, double z, int, int, int) {
          weights[i] = -(x * x + y * y + z * z);
        });
    Ops::upload(m_weights, weights);
  }

  DiagnosticSample sample(RealField &field, const CahnHilliardParams &params) {
    DiagnosticSample result;
    const auto stats = pfc::sim::reduce_owned(field, m_comm);
    const auto invalid = pfc::sim::reduce_owned(field, m_comm, [](double c) {
      return (!std::isfinite(c) || c <= 0.0 || c >= 1.0) ? 1.0 : 0.0;
    });
    const auto bulk = pfc::sim::reduce_owned(field, m_comm, [&](double c) {
      if (!std::isfinite(c) || c <= 0.0 || c >= 1.0) return 0.0;
      // No evaluator clamp: report the actual logarithmic energy for
      // every admissible concentration, including values near 0 and 1.
      return params.omega_nd * c * (1.0 - c) + c * std::log(c) +
             (1.0 - c) * std::log1p(-c);
    });
    const auto n = pfc::domain::get_size(m_domain);
    const auto dx = pfc::domain::get_spacing(m_domain);
    const double cell_volume = dx[0] * dx[1] * dx[2];
    const double ncell = double(n[0]) * n[1] * n[2];
    result.mean = stats.sum / ncell;
    result.mass = stats.sum * cell_volume;
    result.minimum = stats.min;
    result.maximum = stats.max;
    result.invalid_cells = invalid.sum;
    if (result.invalid_cells != 0.0) {
      result.bulk_energy = result.gradient_energy =
          std::numeric_limits<double>::quiet_NaN();
      return result;
    }
    result.bulk_energy = bulk.sum * cell_volume;
    Ops::forward(m_fft, field, m_hat);
    // The same transform serves the gradient energy and the spectrum.
    // radial_average drops k=0, so the mean does not have to be removed first.
    m_hat.with_host_read([&](const typename Ops::Complex *hat, std::size_t) {
      const auto spectrum = pfc::fft::radial_average(
          m_fft.get_outbox_bounds(), m_domain, hat, m_comm, m_sf_bins);
      result.k1 = spectrum.first_moment;
      result.domain_length = spectrum.mean_wavelength();
      result.k_peak = spectrum.peak_wavenumber;
      result.dominant_wavelength = spectrum.dominant_wavelength();
    });
    Ops::multiply(m_hat, m_weights, m_work);
    Ops::backward(m_fft, m_work, m_lap);
    double gradient = 0;
    field.with_host_read([&](const double *c, std::size_t count) {
      m_lap.with_host_read([&](const double *lap, std::size_t) {
        for (std::size_t i = 0; i < count; ++i) gradient -= c[i] * lap[i];
      });
    });
    MPI_Allreduce(&gradient, &result.gradient_energy, 1, MPI_DOUBLE, MPI_SUM,
                  m_comm);
    result.gradient_energy *= 0.5 * params.kappa * cell_volume;
    return result;
  }

private:
  pfc::Domain m_domain;
  typename Ops::FFT &m_fft;
  MPI_Comm m_comm;
  ComplexField m_hat;
  RealField m_lap;
  typename Ops::real_coeffs m_weights;
  typename Ops::complex_scratch m_work;
  int m_sf_bins{64};
};

} // namespace cahn_hilliard
