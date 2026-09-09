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
#include <openpfc_apps/structure_factor.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

namespace cahn_hilliard {

struct DiagnosticSample {
  double mean{}, mass{}, minimum{}, maximum{}, bulk_energy{}, gradient_energy{};
  double invalid_cells{};
  /// Structure-factor observables; zero when the spectrum carries no power.
  double k1{}, domain_length{}, k_peak{}, dominant_wavelength{};
  [[nodiscard]] double total_energy() const { return bulk_energy + gradient_energy; }
};

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
    double local[3]{}; // sum, bulk density sum, invalid count
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    field.with_host_view([&](double *data, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) {
        const double c = data[i];
        local[0] += c;
        if (!std::isfinite(c) || c <= 0 || c >= 1)
          ++local[2];
        else
          // No evaluator clamp: report the actual logarithmic energy for
          // every admissible concentration, including values near 0 and 1.
          local[1] += params.omega_nd * c * (1 - c) + c * std::log(c) +
                      (1 - c) * std::log1p(-c);
        lo = std::min(lo, c);
        hi = std::max(hi, c);
      }
    });
    double global[3]{};
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&lo, &result.minimum, 1, MPI_DOUBLE, MPI_MIN, m_comm);
    MPI_Allreduce(&hi, &result.maximum, 1, MPI_DOUBLE, MPI_MAX, m_comm);
    const auto n = pfc::domain::get_size(m_domain);
    const auto dx = pfc::domain::get_spacing(m_domain);
    const double cell_volume = dx[0] * dx[1] * dx[2];
    result.mean = global[0] / (double(n[0]) * n[1] * n[2]);
    result.mass = global[0] * cell_volume;
    result.invalid_cells = global[2];
    if (global[2] != 0) {
      result.bulk_energy = result.gradient_energy =
          std::numeric_limits<double>::quiet_NaN();
      return result;
    }
    result.bulk_energy = global[1] * cell_volume;
    Ops::forward(m_fft, field, m_hat);
    // The same transform serves the gradient energy and the structure factor;
    // shell_average drops k=0, so the mean composition does not have to be
    // subtracted from the field first.
    m_hat.with_host_view([&](typename Ops::Complex *hat, std::size_t) {
      const auto sf = pfc::apps::shell_average(m_fft.get_outbox_bounds(), m_domain, hat,
                                    m_comm, m_sf_bins);
      result.k1 = sf.k1;
      result.domain_length = sf.domain_length();
      result.k_peak = sf.k_peak;
      result.dominant_wavelength = sf.dominant_wavelength();
    });
    Ops::multiply(m_hat, m_weights, m_work);
    Ops::backward(m_fft, m_work, m_lap);
    double gradient = 0;
    field.with_host_view([&](double *c, std::size_t count) {
      m_lap.with_host_view([&](double *lap, std::size_t) {
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

/// Rank-zero CSV with collective failure propagation. Never replaces old data.
class DiagnosticCSV {
public:
  DiagnosticCSV(const std::filesystem::path &path, MPI_Comm comm) : m_comm(comm) {
    MPI_Comm_rank(comm, &m_rank);
    int ok = 1;
    if (m_rank == 0) {
      try {
        if (path.has_parent_path())
          std::filesystem::create_directories(path.parent_path());
        // Exclusive creation also protects against concurrent jobs and links.
        m_out.reset(std::fopen(path.string().c_str(), "wx"));
        if (!m_out) throw std::runtime_error("open failed");
        ok = publish("step,time,mean,mass,min,max,bulk_energy,gradient_energy,"
                     "total_energy,invalid_cells,k1,domain_length,k_peak,"
                     "dominant_wavelength\n");
      } catch (const std::exception &) {
        ok = 0;
      }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if (!ok)
      throw std::runtime_error("diagnostics: cannot create fresh CSV: " +
                               path.string());
  }

  void write(int step, double time, const DiagnosticSample &s) {
    int ok = 1;
    if (m_rank == 0) {
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << std::setprecision(17) << step << ',' << time << ',' << s.mean << ','
           << s.mass << ',' << s.minimum << ',' << s.maximum << ',' << s.bulk_energy
           << ',' << s.gradient_energy << ',' << s.total_energy() << ','
           << s.invalid_cells << ',' << s.k1 << ',' << s.domain_length << ','
           << s.k_peak << ',' << s.dominant_wavelength << '\n';
      ok = publish(line.str());
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, m_comm);
    if (!ok) throw std::runtime_error("diagnostics: CSV write failed");
    if (s.invalid_cells != 0)
      throw std::runtime_error(
          "Cahn-Hilliard diagnostics: nonfinite or out-of-range composition; "
          "require 0<c<1 (reduce dt/check input)");
  }

private:
  bool publish(const std::string &line) {
    return std::fputs(line.c_str(), m_out.get()) >= 0 &&
           std::fflush(m_out.get()) == 0;
  }
  MPI_Comm m_comm;
  int m_rank{};
  std::unique_ptr<std::FILE, decltype(&std::fclose)> m_out{nullptr, &std::fclose};
};
} // namespace cahn_hilliard
