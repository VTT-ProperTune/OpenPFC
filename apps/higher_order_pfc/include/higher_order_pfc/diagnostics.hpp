// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file diagnostics.hpp
 * @brief Crystal-selection benchmark sample: energy, reciprocal peaks, real-
 *        space symmetry -- all three per `#118`, not ring power alone.
 *
 * @details
 * Wraps `FreeEnergySampler` (free-energy density, mean density, the
 * azimuthally averaged structure factor) and `bond_orientational_order`
 * (real-space \f$\psi_4\f$/\f$\psi_6\f$) into one collective sample plus a
 * never-overwriting rank-0 CSV, the same shape as
 * `cahn_hilliard::Diagnostics`/`DiagnosticCSV`.
 *
 * The real-space order metric needs the whole grid on one rank -- finding a
 * peak's nearest neighbours across a rank boundary is not a reduction, unlike
 * the structure factor. So it is only computed when `nproc == 1`; on more
 * ranks the sample still reports free energy and reciprocal peaks (both
 * genuinely collective), and `psi4`/`psi6` come back `NaN` with `n_peaks =
 * 0`, which is the honest signal that the real-space classification was not
 * run, not that the crystal has no order. Every case shipped with this
 * benchmark runs single-rank for exactly this reason; see the app README.
 */

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <higher_order_pfc/free_energy.hpp>
#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <higher_order_pfc/order_parameter.hpp>

namespace higher_order_pfc {

struct DiagnosticSample {
  double mean_psi{0.0};
  double free_energy_density{0.0};
  double k1{0.0}, domain_length{0.0}, k_peak{0.0}, dominant_wavelength{0.0};
  double S_at_1{0.0}, S_at_q1{0.0};
  double psi4_global{0.0}, psi4_local{0.0};
  double psi6_global{0.0}, psi6_local{0.0};
  double n_peaks{0.0};
  double mean_neighbours{0.0};
};

template <class MemorySpace> class Diagnostics {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;

public:
  Diagnostics(const pfc::Domain &domain, typename Ops::FFT &fft, MPI_Comm comm)
      : m_domain(domain), m_comm(comm), m_energy(domain, fft, comm) {}

  DiagnosticSample sample(typename Ops::RealField &psi, const HigherOrderPFCParams &p,
                          int sf_bins = 64, double neighbour_cutoff_factor = 1.3) {
    DiagnosticSample out;
    const auto fe = m_energy.sample(psi, p, sf_bins);
    out.mean_psi = fe.mean_psi;
    out.free_energy_density = fe.free_energy_density;
    out.k1 = fe.sf.k1;
    out.domain_length = fe.sf.domain_length();
    out.k_peak = fe.sf.k_peak;
    out.dominant_wavelength = fe.sf.dominant_wavelength();
    out.S_at_1 = fe.S_at_1;
    out.S_at_q1 = fe.S_at_q1;

    int rank = 0, nproc = 1;
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &nproc);
    if (nproc == 1) {
      const auto n = pfc::domain::get_size(m_domain);
      const auto dx = pfc::domain::get_spacing(m_domain);
      const auto peaks = detect_peaks(psi, n[0], n[1], dx[0], dx[1]);
      const double Lx = dx[0] * double(n[0]);
      const double Ly = dx[1] * double(n[1]);
      const auto bo = bond_orientational_order(peaks, Lx, Ly, neighbour_cutoff_factor);
      out.psi4_global = bo.psi4.global;
      out.psi4_local = bo.psi4.local;
      out.psi6_global = bo.psi6.global;
      out.psi6_local = bo.psi6.local;
      out.n_peaks = double(bo.n_points);
      out.mean_neighbours = bo.mean_neighbours;
    } else {
      out.psi4_global = out.psi4_local = std::numeric_limits<double>::quiet_NaN();
      out.psi6_global = out.psi6_local = std::numeric_limits<double>::quiet_NaN();
      out.n_peaks = 0.0;
    }
    return out;
  }

private:
  pfc::Domain m_domain;
  MPI_Comm m_comm;
  FreeEnergySampler<MemorySpace> m_energy;
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
        m_out.reset(std::fopen(path.string().c_str(), "wx"));
        if (!m_out) throw std::runtime_error("open failed");
        ok = publish(
            "step,time,mean_psi,free_energy_density,k1,domain_length,k_peak,"
            "dominant_wavelength,S_at_1,S_at_q1,psi4_global,psi4_local,"
            "psi6_global,psi6_local,n_peaks,mean_neighbours\n");
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
      line << std::setprecision(17) << step << ',' << time << ',' << s.mean_psi
           << ',' << s.free_energy_density << ',' << s.k1 << ',' << s.domain_length
           << ',' << s.k_peak << ',' << s.dominant_wavelength << ',' << s.S_at_1
           << ',' << s.S_at_q1 << ',' << s.psi4_global << ',' << s.psi4_local << ','
           << s.psi6_global << ',' << s.psi6_local << ',' << s.n_peaks << ','
           << s.mean_neighbours << '\n';
      ok = publish(line.str());
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, m_comm);
    if (!ok) throw std::runtime_error("diagnostics: CSV write failed");
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

} // namespace higher_order_pfc
