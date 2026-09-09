// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file wave_packet_diagnostics.hpp
 * @brief Group velocity / phase velocity / spreading diagnostics for a
 *        narrow-band Kawahara wave packet (`#119`).
 *
 * @details
 * Two independent quantities are extracted from the current field `u`,
 * assumed 1D (`Ny=Nz=1`) and single-rank (spectral diagnostics need the whole
 * line):
 *
 * - **Carrier phase**: project `u` onto `cos(k0 x)`/`sin(k0 x)` directly (a
 *   discrete Fourier coefficient when `k0` is an exact grid mode, as chosen
 *   in every shipped wave-packet case). For `u=A*cos(k0 x + phi)`,
 *   `mode_phase = atan2(s,c) = -phi`, and the linear (`alpha=0`) evolution
 *   `u_k(t) = u_k(0)*exp(-i*omega(k0)*t)` gives `phi(t) = phi(0) - omega*t`,
 *   so `mode_phase(t) = mode_phase(0) + omega*t`: the wrapped phase
 *   difference over an interval gives `omega(k0)` (and hence
 *   `c_p=omega/k0`) essentially exactly, independent of the packet's
 *   envelope/bandwidth, because each Fourier mode of a linear PDE evolves
 *   independently.
 * - **Envelope**: for `u = A(x)cos(k0 x + phi(x))` with slowly varying
 *   `A`,`phi` (narrow-band assumption), `u^2 = A(x)^2/2 + O(k0)` after
 *   removing the fast oscillations at `k0` and `2k0` by spectral low-pass
 *   filtering (Gaussian mask, cutoff a fixed fraction of `k0`). This gives
 *   the envelope-intensity `A(x)^2` up to the constant factor below, from
 *   which the energy-weighted centroid (group-velocity tracer) follows by
 *   direct quadrature; `width2` is twice that intensity-weighted spatial
 *   variance, which corrects for `A(x)^2` (a Gaussian of std `sigma/sqrt(2)`
 *   when `A` has std `sigma`) being narrower than `A` itself, so
 *   `sqrt(width2)` estimates the amplitude-envelope std directly. This is a
 *   standard narrowband quadrature-demodulation technique (not specific to
 *   Kawahara), valid as long as the envelope's own spectral content sits
 *   well below the low-pass cutoff -- i.e. the packet stays narrow-band,
 *   which the caller should check via the Fourier spectrum.
 *
 * Also reports `edge_fraction`: the largest envelope value in the outer 5%
 * of the domain on either side, relative to the peak envelope. Small values
 * (this app requires callers to check it is < 1e-2) are the automatic,
 * reproducible check that the packet has not reached the periodic boundary
 * (no wraparound) over the reported interval.
 */

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

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

namespace kawahara {

struct WavePacketSample {
  double mean{};
  /// Discrete projection onto exactly k0. For a narrowband envelope this is
  /// proportional to the envelope's own DC Fourier content (its area) over
  /// Lx, *not* the envelope's peak amplitude -- useful as a relative/decay
  /// tracer at fixed k0, not as an absolute amplitude reading. `mode_phase`
  /// (below) is the quantity with an exact physical meaning (phase velocity).
  double mode_amplitude{};
  double mode_phase{}; ///< wrapped, atan2(s,c) in (-pi,pi]
  double centroid{};
  double width2{};
  double peak_envelope{};
  double edge_fraction{};
};

/// Envelope/phase diagnostics for the Kawahara primary field `u`.
/// HostSpace only (single-line MPI-collective reductions).
class WavePacketDiagnostics {
  using Ops = pfc::sim::SpectralETDOps<pfc::HostSpace>;
  using RealField = typename Ops::RealField;
  using ComplexField = typename Ops::ComplexField;

public:
  WavePacketDiagnostics(const pfc::Domain &domain, typename Ops::FFT &fft,
                        MPI_Comm comm, double k0, double cutoff_factor = 0.5)
      : m_domain(domain), m_fft(fft), m_comm(comm), m_k0(k0),
        m_u2(domain, fft.get_inbox_bounds(), 0),
        m_hat(domain, fft.get_outbox_bounds(), 0),
        m_mask(Ops::make_real(fft.size_outbox())) {
    const double kc = cutoff_factor * k0;
    std::vector<double> mask(fft.size_outbox());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          const double k2 = kx * kx + ky * ky + kz * kz;
          const double r2 = k2 / (kc * kc);
          mask[i] = std::exp(-r2 * r2); // quartic: flatter passband, sharper cutoff
        });
    Ops::upload(m_mask, mask);
  }

  WavePacketSample sample(RealField &u) {
    WavePacketSample result;

    // Carrier phase/amplitude: direct projection onto cos(k0 x), sin(k0 x).
    double local_mean = 0.0, local_c = 0.0, local_s = 0.0;
    std::size_t local_n = 0;
    u.for_each_owned([&](const pfc::Real3 &x, double v) {
      local_mean += v;
      local_c += v * std::cos(m_k0 * x[0]);
      local_s += v * std::sin(m_k0 * x[0]);
      ++local_n;
    });
    double global_mean = 0.0, global_c = 0.0, global_s = 0.0;
    unsigned long long global_n = 0, local_n_ll = local_n;
    MPI_Allreduce(&local_mean, &global_mean, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_c, &global_c, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_s, &global_s, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_n_ll, &global_n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, m_comm);
    result.mean = (global_n > 0) ? global_mean / static_cast<double>(global_n) : 0.0;
    // Orthogonality over a full period gives den = N/2 for a matching lattice mode.
    const double den = 0.5 * static_cast<double>(global_n);
    result.mode_amplitude = (den > 0.0) ? std::sqrt(global_c * global_c +
                                                     global_s * global_s) /
                                              den
                                        : 0.0;
    result.mode_phase = std::atan2(global_s, global_c);

    // Envelope intensity: low-pass filter u^2 (removes k0, 2k0 oscillations).
    // Fill u^2 from u's owned values (m_u2 shares u's domain/box/halo=0).
    {
      auto dst = m_u2.data();
      const auto *src = u.data();
      for (std::size_t i = 0; i < u.size(); ++i) dst[i] = src[i] * src[i];
      m_u2.note_host_write();
    }
    Ops::forward(m_fft, m_u2, m_hat);
    Ops::multiply(m_hat, m_mask, m_hat);
    Ops::backward(m_fft, m_hat, m_u2);

    double local_w = 0.0, local_wx = 0.0;
    double local_peak = 0.0;
    m_u2.for_each_owned([&](const pfc::Real3 &x, double lp) {
      const double env2 = std::max(0.0, 2.0 * lp - 2.0 * result.mean * result.mean);
      local_w += env2;
      local_wx += env2 * x[0];
      local_peak = std::max(local_peak, env2);
    });
    double global_w = 0.0, global_wx = 0.0, global_peak = 0.0;
    MPI_Allreduce(&local_w, &global_w, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_wx, &global_wx, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_peak, &global_peak, 1, MPI_DOUBLE, MPI_MAX, m_comm);
    result.centroid = (global_w > 0.0) ? global_wx / global_w : 0.0;
    result.peak_envelope = std::sqrt(std::max(0.0, global_peak));

    double local_var = 0.0;
    const double xc = result.centroid;
    m_u2.for_each_owned([&](const pfc::Real3 &x, double lp) {
      const double env2 = std::max(0.0, 2.0 * lp - 2.0 * result.mean * result.mean);
      const double dx = x[0] - xc;
      local_var += env2 * dx * dx;
    });
    double global_var = 0.0;
    MPI_Allreduce(&local_var, &global_var, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    // The intensity env2(x)=A(x)^2 is itself a Gaussian of std sigma/sqrt(2)
    // when A(x) has std sigma (squaring a Gaussian narrows it by sqrt(2)), so
    // its energy-weighted spatial variance underestimates the *amplitude*
    // envelope's variance by a factor of 2; correct for that here so
    // sqrt(width2) reports the amplitude-envelope std sigma directly.
    result.width2 = (global_w > 0.0) ? 2.0 * global_var / global_w : 0.0;

    // Edge sentinel: peak envelope amplitude in the outer 5% of the domain.
    const auto size = pfc::domain::get_size(m_domain);
    const auto spacing = pfc::domain::get_spacing(m_domain);
    const double Lx = spacing[0] * static_cast<double>(size[0]);
    const double edge = 0.05 * Lx;
    double local_edge_peak = 0.0;
    m_u2.for_each_owned([&](const pfc::Real3 &x, double lp) {
      if (x[0] < edge || x[0] > Lx - edge) {
        const double env2 = std::max(0.0, 2.0 * lp - 2.0 * result.mean * result.mean);
        local_edge_peak = std::max(local_edge_peak, env2);
      }
    });
    double global_edge_peak = 0.0;
    MPI_Allreduce(&local_edge_peak, &global_edge_peak, 1, MPI_DOUBLE, MPI_MAX, m_comm);
    const double edge_env = std::sqrt(std::max(0.0, global_edge_peak));
    result.edge_fraction =
        (result.peak_envelope > 0.0) ? edge_env / result.peak_envelope : 0.0;

    return result;
  }

private:
  pfc::Domain m_domain;
  typename Ops::FFT &m_fft;
  MPI_Comm m_comm;
  double m_k0;
  RealField m_u2;
  ComplexField m_hat;
  typename Ops::real_coeffs m_mask;
};

/// Rank-zero CSV, never overwrites. Columns: step,time,mean,mode_amplitude,
/// mode_phase,centroid,width,peak_envelope,edge_fraction.
class WavePacketCSV {
public:
  WavePacketCSV(const std::filesystem::path &path, MPI_Comm comm) : m_comm(comm) {
    MPI_Comm_rank(comm, &m_rank);
    int ok = 1;
    if (m_rank == 0) {
      try {
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        m_out.reset(std::fopen(path.string().c_str(), "wx"));
        if (!m_out) throw std::runtime_error("open failed");
        ok = publish("step,time,mean,mode_amplitude,mode_phase,centroid,width,"
                     "peak_envelope,edge_fraction\n");
      } catch (const std::exception &) {
        ok = 0;
      }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if (!ok)
      throw std::runtime_error("wave_packet diagnostics: cannot create fresh CSV: " +
                               path.string());
  }

  void write(int step, double time, const WavePacketSample &s) {
    int ok = 1;
    if (m_rank == 0) {
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << std::setprecision(17) << step << ',' << time << ',' << s.mean << ','
           << s.mode_amplitude << ',' << s.mode_phase << ',' << s.centroid << ','
           << std::sqrt(std::max(0.0, s.width2)) << ',' << s.peak_envelope << ','
           << s.edge_fraction << '\n';
      ok = publish(line.str());
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, m_comm);
    if (!ok) throw std::runtime_error("wave_packet diagnostics: CSV write failed");
  }

private:
  bool publish(const std::string &line) {
    return std::fputs(line.c_str(), m_out.get()) >= 0 && std::fflush(m_out.get()) == 0;
  }
  MPI_Comm m_comm;
  int m_rank{};
  std::unique_ptr<std::FILE, decltype(&std::fclose)> m_out{nullptr, &std::fclose};
};

/**
 * @brief Case-B (nonlinear localized pulse) diagnostics: peak amplitude and
 * location plus a trailing-radiation RMS measured outside a fixed window
 * around the peak. Comparing this RMS between a third-order-only run
 * (`gamma=0`) and the full third+fifth-order run is the automated,
 * reproducible fifth-order-term effect the app README/tests report.
 */
struct PulseSample {
  double mean{};
  double peak_amplitude{}; ///< max(|u|)
  double peak_x{};         ///< location of the peak
  double tail_rms{};       ///< RMS of u outside +-window/2 of the peak
};

class PulseDiagnostics {
public:
  explicit PulseDiagnostics(MPI_Comm comm, double window) : m_comm(comm), m_window(window) {}

  template <class RealField> PulseSample sample(RealField &u) {
    PulseSample result;
    double local_mean = 0.0;
    unsigned long long local_n = 0;
    double local_peak = 0.0, local_peak_x = 0.0;
    u.for_each_owned([&](const pfc::Real3 &x, double v) {
      local_mean += v;
      ++local_n;
      if (std::abs(v) > std::abs(local_peak)) {
        local_peak = v;
        local_peak_x = x[0];
      }
    });
    struct {
      double v;
      int rank;
    } local_in{}, global_in{};
    MPI_Comm_rank(m_comm, &local_in.rank);
    local_in.v = std::abs(local_peak);
    MPI_Allreduce(&local_in, &global_in, 1, MPI_DOUBLE_INT, MPI_MAXLOC, m_comm);
    double peak_and_x[2] = {local_peak, local_peak_x};
    MPI_Bcast(peak_and_x, 2, MPI_DOUBLE, global_in.rank, m_comm);
    result.peak_amplitude = std::abs(peak_and_x[0]);
    result.peak_x = peak_and_x[1];

    double global_mean = 0.0;
    unsigned long long global_n = 0;
    MPI_Allreduce(&local_mean, &global_mean, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_n, &global_n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, m_comm);
    result.mean = (global_n > 0) ? global_mean / static_cast<double>(global_n) : 0.0;

    double local_tail_sumsq = 0.0;
    unsigned long long local_tail_n = 0;
    const double half = 0.5 * m_window;
    u.for_each_owned([&](const pfc::Real3 &x, double v) {
      if (std::abs(x[0] - result.peak_x) > half) {
        local_tail_sumsq += v * v;
        ++local_tail_n;
      }
    });
    double global_tail_sumsq = 0.0;
    unsigned long long global_tail_n = 0;
    MPI_Allreduce(&local_tail_sumsq, &global_tail_sumsq, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    MPI_Allreduce(&local_tail_n, &global_tail_n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  m_comm);
    result.tail_rms = (global_tail_n > 0)
                          ? std::sqrt(global_tail_sumsq / static_cast<double>(global_tail_n))
                          : 0.0;
    return result;
  }

private:
  MPI_Comm m_comm;
  double m_window;
};

/// Rank-zero CSV, never overwrites. Columns: step,time,mean,peak_amplitude,peak_x,tail_rms.
class PulseCSV {
public:
  PulseCSV(const std::filesystem::path &path, MPI_Comm comm) : m_comm(comm) {
    MPI_Comm_rank(comm, &m_rank);
    int ok = 1;
    if (m_rank == 0) {
      try {
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        m_out.reset(std::fopen(path.string().c_str(), "wx"));
        if (!m_out) throw std::runtime_error("open failed");
        ok = publish("step,time,mean,peak_amplitude,peak_x,tail_rms\n");
      } catch (const std::exception &) {
        ok = 0;
      }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    if (!ok)
      throw std::runtime_error("pulse diagnostics: cannot create fresh CSV: " +
                               path.string());
  }

  void write(int step, double time, const PulseSample &s) {
    int ok = 1;
    if (m_rank == 0) {
      std::ostringstream line;
      line.imbue(std::locale::classic());
      line << std::setprecision(17) << step << ',' << time << ',' << s.mean << ','
           << s.peak_amplitude << ',' << s.peak_x << ',' << s.tail_rms << '\n';
      ok = publish(line.str());
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, m_comm);
    if (!ok) throw std::runtime_error("pulse diagnostics: CSV write failed");
  }

private:
  bool publish(const std::string &line) {
    return std::fputs(line.c_str(), m_out.get()) >= 0 && std::fflush(m_out.get()) == 0;
  }
  MPI_Comm m_comm;
  int m_rank{};
  std::unique_ptr<std::FILE, decltype(&std::fclose)> m_out{nullptr, &std::fclose};
};

} // namespace kawahara
