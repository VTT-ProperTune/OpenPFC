// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file periodic_spectra.hpp
 * @brief Periodic Fourier families for spatial-operator comparison.
 *
 * @details
 * Occupancy \f$f\in(0,1]\f$ is the fraction of the Nyquist wavenumber
 * \f$k_{\mathrm{Nyq}}=\pi/h\f$ at which a family's **amplitude** prescription
 * reaches the declared content threshold (default \f$10^{-3}\f$ of peak).
 * Band-limited (top-hat) families use the same \f$f\f$ as the support cutoff.
 *
 * Geometry is either an isotropic function of \f$|\mathbf n|\f$ or a
 * separable product of one-dimensional amplitudes. A Gaussian is both:
 * \f$\prod_i\exp(-c n_i^2)=\exp(-c|\mathbf n|^2)\f$. Exponential families
 * differ: isotropic uses \f$|\mathbf n|_2\f$, separable uses
 * \f$|n_x|+|n_y|+|n_z|\f$.
 *
 * Even-grid Nyquist modes (\f$n=N/2\f$) are omitted by default, matching the
 * truncated Fourier series used by the Heat3D Gaussian study and the
 * even-grid convention that a real r2c Nyquist bin is not a unique paired
 * mode. Odd \f$N\f$ has no Nyquist bin; every resolved mode is kept.
 *
 * This is not a general DSP toolkit: it only evaluates Parseval errors of
 * OpenPFC's Laplacian symbols (spectral \f$-k^2\f$ and shipped FD D2
 * stencils) on deterministic, real, even spectra.
 *
 * @see fd_symbols.hpp
 * @see kernel/fft/kspace.hpp
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/field/fd_symbols.hpp>

namespace pfc::field::spectra {

inline constexpr double kContentThreshold = 1.0e-3;
inline constexpr double kDiffusionTime = 1.0;
inline constexpr int kMapGrid = 128;

/// How a family's amplitude is assembled from integer modes.
enum class WeightGeometry {
  /// \f$a(\mathbf n)=\prod_i a_{1\mathrm{D}}(|n_i|/k_c)\f$.
  SeparableProduct,
  /// \f$a(\mathbf n)=a_{\mathrm{rad}}(|\mathbf n|/k_c)\f$.
  IsotropicRadial,
};

struct SpectrumFamily {
  const char *id;
  const char *name;
  WeightGeometry geometry;
  /// Amplitude at \f$\nu=k/k_c\f$ (1-D or radial), relative to the peak.
  double (*amplitude)(double nu);
};

[[nodiscard]] inline double occupancy_log() noexcept {
  return std::log(1.0 / kContentThreshold);
}

[[nodiscard]] inline double gaussian_amplitude(double nu) noexcept {
  return std::exp(-occupancy_log() * nu * nu);
}

[[nodiscard]] inline double exponential_amplitude(double nu) noexcept {
  return std::exp(-occupancy_log() * std::abs(nu));
}

[[nodiscard]] inline double tophat_amplitude(double nu) noexcept {
  return (std::abs(nu) <= 1.0) ? 1.0 : 0.0;
}

[[nodiscard]] inline SpectrumFamily gaussian() noexcept {
  return {"gaussian", "periodic Gaussian (isotropic / separable identity)",
          WeightGeometry::IsotropicRadial, &gaussian_amplitude};
}

[[nodiscard]] inline SpectrumFamily isotropic_tophat() noexcept {
  return {"isotropic_tophat", "isotropic band-limited (top-hat) ball",
          WeightGeometry::IsotropicRadial, &tophat_amplitude};
}

[[nodiscard]] inline SpectrumFamily isotropic_exponential() noexcept {
  return {"isotropic_exponential", "isotropic exponential tail (L2 radius)",
          WeightGeometry::IsotropicRadial, &exponential_amplitude};
}

[[nodiscard]] inline SpectrumFamily separable_exponential() noexcept {
  return {"separable_exponential", "separable exponential tail (L1 product)",
          WeightGeometry::SeparableProduct, &exponential_amplitude};
}

[[nodiscard]] inline std::array<SpectrumFamily, 4> builtin_families() noexcept {
  return {gaussian(), isotropic_tophat(), isotropic_exponential(),
          separable_exponential()};
}

[[nodiscard]] inline const SpectrumFamily *find_family(std::string_view id) {
  for (const SpectrumFamily &fam : builtin_families()) {
    if (id == fam.id) return &fam;
  }
  return nullptr;
}

/// Nyquist wavenumber on an axis of an \f$N\f$-point grid of length \f$2\pi\f$.
[[nodiscard]] inline double nyquist_wavenumber(int N) {
  if (N < 2) throw std::invalid_argument("nyquist_wavenumber: N >= 2");
  return static_cast<double>(N) / 2.0;
}

/// Inclusive 1-D mode bounds. Even \f$N\f$ omits \f$\pm N/2\f$ unless requested.
[[nodiscard]] inline std::pair<int, int> mode_bounds(int N,
                                                     bool include_nyquist = false) {
  if (N < 2) throw std::invalid_argument("mode_bounds: N >= 2");
  if (N % 2 != 0) {
    const int m = (N - 1) / 2;
    return {-m, m};
  }
  const int n_hi = include_nyquist ? (N / 2) : (N / 2 - 1);
  return {1 - N / 2, n_hi};
}

[[nodiscard]] inline int max_resolved_mode(int N, bool include_nyquist = false) {
  return mode_bounds(N, include_nyquist).second;
}

[[nodiscard]] inline double content_wavenumber(double f, int N) {
  if (!(f > 0.0) || f > 1.0)
    throw std::invalid_argument("content_wavenumber: need 0 < f <= 1");
  return f * nyquist_wavenumber(N);
}

[[nodiscard]] inline double mode_amplitude(const SpectrumFamily &fam, int nx, int ny,
                                           int nz, double k_c, int dim) {
  if (!(k_c > 0.0)) throw std::invalid_argument("mode_amplitude: k_c > 0");
  if (dim < 1 || dim > 3)
    throw std::invalid_argument("mode_amplitude: dim must be 1, 2, or 3");
  if (fam.geometry == WeightGeometry::IsotropicRadial) {
    double r2 = static_cast<double>(nx) * static_cast<double>(nx);
    if (dim > 1) r2 += static_cast<double>(ny) * static_cast<double>(ny);
    if (dim > 2) r2 += static_cast<double>(nz) * static_cast<double>(nz);
    return fam.amplitude(std::sqrt(r2) / k_c);
  }
  double a = fam.amplitude(std::abs(static_cast<double>(nx)) / k_c);
  if (dim > 1) a *= fam.amplitude(std::abs(static_cast<double>(ny)) / k_c);
  if (dim > 2) a *= fam.amplitude(std::abs(static_cast<double>(nz)) / k_c);
  return a;
}

/// Grid with enough integer modes near \f$k_c\f$ for a stable occupancy sum.
[[nodiscard]] inline int auto_map_grid(double f) {
  if (!(f > 0.0) || f > 1.0)
    throw std::invalid_argument("auto_map_grid: need 0 < f <= 1");
  const int N = 2 * static_cast<int>(std::ceil(12.0 / f));
  return std::max(N, 16);
}

/// Largest \f$|\nu|\f$ retained: amplitude below ~1e-18, else Nyquist.
[[nodiscard]] inline double amplitude_support_nu(const SpectrumFamily &fam) {
  if (fam.amplitude(1.0 + 1.0e-15) == 0.0 && fam.amplitude(1.0) != 0.0) {
    return 1.0;
  }
  for (double nu = 0.25; nu <= 16.0; nu += 0.25) {
    if (fam.amplitude(nu) < 1.0e-18) return nu;
  }
  return 16.0;
}

struct ParsevalError {
  double l2_relative{0.0};
  double energy{0.0};
};

struct SpectrumDiagnostics {
  double energy{0.0};
  /// Parseval mass with \f$|\mathbf k|/k_{\mathrm{Nyq}} > 0.5\f$.
  double energy_high_k{0.0};
  /// Parseval mass with \f$|\mathbf n| > 0.8\,k_c\f$.
  double energy_near_cutoff{0.0};
  /// Energy-weighted \f$\langle k^2\rangle / k_{\mathrm{Nyq}}^2\f$.
  double moment2_over_nyquist2{0.0};
  /// Energy-weighted \f$\langle k^4\rangle / k_{\mathrm{Nyq}}^4\f$.
  double moment4_over_nyquist4{0.0};
  /// Energy-weighted FD D2 defect (dimensionless, sum of axes) at `fd_order`.
  double weighted_d2_defect{0.0};
};

namespace detail {

inline void check_grid(int N, int dim) {
  if (N < 4) throw std::invalid_argument("periodic spectra: N >= 4");
  if (dim < 1 || dim > 3)
    throw std::invalid_argument("periodic spectra: dim must be 1, 2, or 3");
}

template <class Body>
void for_each_mode(int N, int dim, bool include_nyquist, int n_box, Body &&body) {
  auto [n_lo, n_hi] = mode_bounds(N, include_nyquist);
  n_lo = std::max(n_lo, -n_box);
  n_hi = std::min(n_hi, n_box);
  if (dim == 1) {
    for (int i = n_lo; i <= n_hi; ++i) body(i, 0, 0);
    return;
  }
  if (dim == 2) {
    for (int i = n_lo; i <= n_hi; ++i) {
      for (int j = n_lo; j <= n_hi; ++j) body(i, j, 0);
    }
    return;
  }
  for (int i = n_lo; i <= n_hi; ++i) {
    for (int j = n_lo; j <= n_hi; ++j) {
      for (int k = n_lo; k <= n_hi; ++k) body(i, j, k);
    }
  }
}

inline double radial_mode(int nx, int ny, int nz, int dim) {
  double r2 = static_cast<double>(nx) * static_cast<double>(nx);
  if (dim > 1) r2 += static_cast<double>(ny) * static_cast<double>(ny);
  if (dim > 2) r2 += static_cast<double>(nz) * static_cast<double>(nz);
  return std::sqrt(r2);
}

} // namespace detail

/**
 * @brief Parseval \f$L^2\f$ heat-equation error of an FD Laplacian vs spectral.
 *
 * Each mode decays as \f$e^{-D|k|^2 t}\f$ spectrally and
 * \f$e^{D\lambda_p(\mathbf k)t}\f$ under FD. Time is
 * \f$\tau=D t k_c^2\f$. The reported value is RMS error over RMS of the
 * initial spectrum. The spectral operator itself has error 0.
 */
[[nodiscard]] inline ParsevalError predict_heat_error(const SpectrumFamily &fam,
                                                      int fd_order, double f, int N,
                                                      double tau = kDiffusionTime,
                                                      int dim = 3,
                                                      bool include_nyquist = false) {
  detail::check_grid(N, dim);
  if (!(f > 0.0) || f > 1.0)
    throw std::invalid_argument("predict_heat_error: need 0 < f <= 1");
  if (fd_order != 0 && (fd_order < 2 || fd_order % 2 != 0)) {
    throw std::invalid_argument("predict_heat_error: fd_order even and >= 2, or 0");
  }

  const double k_c = content_wavenumber(f, N);
  const int n_hi = max_resolved_mode(N, include_nyquist);
  const double nu_cut = amplitude_support_nu(fam);
  const int n_box = std::min(n_hi, static_cast<int>(std::ceil(nu_cut * k_c)) + 2);

  double energy = 0.0;
  double err_sq = 0.0;
  detail::for_each_mode(N, dim, include_nyquist, n_box, [&](int nx, int ny, int nz) {
    const double amp = mode_amplitude(fam, nx, ny, nz, k_c, dim);
    const double w = amp * amp;
    if (w <= 0.0) return;
    energy += w;
    if (fd_order == 0) return;
    const double theta_x =
        pfc::two_pi * static_cast<double>(nx) / static_cast<double>(N);
    double defect = pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_x));
    if (dim > 1) {
      const double theta_y =
          pfc::two_pi * static_cast<double>(ny) / static_cast<double>(N);
      defect += pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_y));
    }
    if (dim > 2) {
      const double theta_z =
          pfc::two_pi * static_cast<double>(nz) / static_cast<double>(N);
      defect += pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_z));
    }
    const double nu_r = detail::radial_mode(nx, ny, nz, dim) / k_c;
    const double decay = std::exp(-tau * nu_r * nu_r);
    const double delta = defect * tau / (pfc::pi * pfc::pi * f * f);
    const double diff = decay * std::expm1(delta);
    err_sq += w * diff * diff;
  });
  if (!(energy > 0.0))
    throw std::runtime_error("predict_heat_error: empty spectrum");
  ParsevalError out;
  out.energy = energy;
  out.l2_relative = (fd_order == 0) ? 0.0 : std::sqrt(err_sq / energy);
  return out;
}

[[nodiscard]] inline double predict_heat_l2_error(const SpectrumFamily &fam,
                                                  int fd_order, double f, int N,
                                                  double tau = kDiffusionTime,
                                                  int dim = 3,
                                                  bool include_nyquist = false) {
  return predict_heat_error(fam, fd_order, f, N, tau, dim, include_nyquist)
      .l2_relative;
}

[[nodiscard]] inline SpectrumDiagnostics
diagnose_spectrum(const SpectrumFamily &fam, double f, int N, int fd_order = 2,
                  int dim = 3, bool include_nyquist = false) {
  detail::check_grid(N, dim);
  const double k_c = content_wavenumber(f, N);
  const double k_nyq = nyquist_wavenumber(N);
  const int n_hi = max_resolved_mode(N, include_nyquist);
  const double nu_cut = amplitude_support_nu(fam);
  const int n_box = std::min(n_hi, static_cast<int>(std::ceil(nu_cut * k_c)) + 2);

  SpectrumDiagnostics d{};
  double w_k2 = 0.0;
  double w_k4 = 0.0;
  double w_def = 0.0;
  detail::for_each_mode(N, dim, include_nyquist, n_box, [&](int nx, int ny, int nz) {
    const double amp = mode_amplitude(fam, nx, ny, nz, k_c, dim);
    const double w = amp * amp;
    if (w <= 0.0) return;
    d.energy += w;
    const double r = detail::radial_mode(nx, ny, nz, dim);
    const double k_over = r / k_nyq;
    if (k_over > 0.5) d.energy_high_k += w;
    if (r > 0.8 * k_c) d.energy_near_cutoff += w;
    const double k2 = (r / k_nyq) * (r / k_nyq);
    w_k2 += w * k2;
    w_k4 += w * k2 * k2;
    if (fd_order >= 2) {
      const double theta_x =
          pfc::two_pi * static_cast<double>(nx) / static_cast<double>(N);
      double defect = pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_x));
      if (dim > 1) {
        const double theta_y =
            pfc::two_pi * static_cast<double>(ny) / static_cast<double>(N);
        defect += pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_y));
      }
      if (dim > 2) {
        const double theta_z =
            pfc::two_pi * static_cast<double>(nz) / static_cast<double>(N);
        defect += pfc::field::fd::d2_symbol_defect(fd_order, std::abs(theta_z));
      }
      w_def += w * defect;
    }
  });
  if (!(d.energy > 0.0))
    throw std::runtime_error("diagnose_spectrum: empty spectrum");
  d.energy_high_k /= d.energy;
  d.energy_near_cutoff /= d.energy;
  d.moment2_over_nyquist2 = w_k2 / d.energy;
  d.moment4_over_nyquist4 = w_k4 / d.energy;
  d.weighted_d2_defect = w_def / d.energy;
  return d;
}

/// Largest admitted content fraction that meets an error target, or an
/// explicit miss when no admitted fraction attains `eps`.
struct ContentFractionMatch {
  bool attainable{false};
  double f{0.0};
};

[[nodiscard]] inline ContentFractionMatch
content_fraction_at(const SpectrumFamily &fam, int fd_order, double eps,
                    double tau = kDiffusionTime, int dim = 3,
                    double f_min = 1.0e-3) {
  const auto err = [&](double f) {
    return predict_heat_l2_error(fam, fd_order, f, auto_map_grid(f), tau, dim);
  };
  double lo = f_min, hi = 1.0;
  if (err(hi) <= eps) return ContentFractionMatch{true, hi};
  if (err(lo) > eps) return ContentFractionMatch{false, lo};
  for (int it = 0; it < 40; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (err(mid) <= eps) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  if (err(lo) <= eps) return ContentFractionMatch{true, lo};
  return ContentFractionMatch{false, f_min};
}

[[nodiscard]] inline double crossover_fraction(double cost_fd,
                                               double cost_spectral) {
  if (!(cost_fd > 0.0) || !(cost_spectral > 0.0))
    throw std::invalid_argument("crossover_fraction: costs must be positive");
  return std::cbrt(cost_fd / cost_spectral);
}

[[nodiscard]] inline double equal_accuracy_cost_ratio(double f_star, double cost_fd,
                                                      double cost_spectral) {
  if (!(f_star > 0.0))
    throw std::invalid_argument("equal_accuracy_cost_ratio: f_star > 0");
  return (cost_fd / (f_star * f_star * f_star)) / cost_spectral;
}

[[nodiscard]] inline bool fd_cheaper_than_spectral(double f_star, double cost_fd,
                                                   double cost_spectral) {
  return f_star > crossover_fraction(cost_fd, cost_spectral);
}

/// Identity of the cheapest operator at equal accuracy; 0 = spectral.
struct EqualAccuracyPick {
  int fd_order{0};
  double relative_cost{1.0};
  double f_star{1.0};
  bool attainable{true};
};

/**
 * @brief Cheapest method among spectral and the supplied FD orders.
 *
 * `cost_by_order[i]` pairs FD order with per-step cost; spectral cost is
 * separate. Missing FD timings and unattainable accuracy targets are skipped.
 */
[[nodiscard]] inline EqualAccuracyPick
pick_equal_accuracy(const SpectrumFamily &fam, double eps,
                    const std::vector<std::pair<int, double>> &cost_by_order,
                    double cost_spectral, double tau = kDiffusionTime, int dim = 3) {
  if (!(cost_spectral > 0.0))
    throw std::invalid_argument("pick_equal_accuracy: spectral cost > 0");
  EqualAccuracyPick best;
  best.fd_order = 0;
  best.relative_cost = 1.0;
  best.f_star = 1.0;
  best.attainable = true;
  for (const auto &[order, cost] : cost_by_order) {
    if (!(cost > 0.0)) continue;
    const auto match = content_fraction_at(fam, order, eps, tau, dim);
    if (!match.attainable) continue;
    const double rel = equal_accuracy_cost_ratio(match.f, cost, cost_spectral);
    if (rel < best.relative_cost) {
      best.fd_order = order;
      best.relative_cost = rel;
      best.f_star = match.f;
      best.attainable = true;
    }
  }
  return best;
}

/**
 * @brief Real-space value of a real even spectrum, optionally heat-evolved.
 *
 * \f$u(\mathbf x,t)=\sum_{\mathbf n} a_{\mathbf n}e^{-D|\mathbf k|^2 t}
 * \cos(\mathbf n\cdot\mathbf x)\f$ on the \f$L=2\pi\f$ box. Intended for
 * small grids (validation), not production ICs.
 */
[[nodiscard]] inline double evaluate_periodic_field(const SpectrumFamily &fam,
                                                    double x, double y, double z,
                                                    double t, double f, int N,
                                                    int dim = 3, double D = 1.0,
                                                    bool include_nyquist = false) {
  detail::check_grid(N, dim);
  const double k_c = content_wavenumber(f, N);
  const int n_hi = max_resolved_mode(N, include_nyquist);
  const double nu_cut = amplitude_support_nu(fam);
  const int n_box = std::min(n_hi, static_cast<int>(std::ceil(nu_cut * k_c)) + 2);
  double u = 0.0;
  detail::for_each_mode(N, dim, include_nyquist, n_box, [&](int nx, int ny, int nz) {
    const double amp = mode_amplitude(fam, nx, ny, nz, k_c, dim);
    if (amp == 0.0) return;
    const double r2 =
        static_cast<double>(nx) * static_cast<double>(nx) +
        (dim > 1 ? static_cast<double>(ny) * static_cast<double>(ny) : 0.0) +
        (dim > 2 ? static_cast<double>(nz) * static_cast<double>(nz) : 0.0);
    const double phase = static_cast<double>(nx) * x +
                         (dim > 1 ? static_cast<double>(ny) * y : 0.0) +
                         (dim > 2 ? static_cast<double>(nz) * z : 0.0);
    u += amp * std::exp(-D * r2 * t) * std::cos(phase);
  });
  return u;
}

} // namespace pfc::field::spectra
