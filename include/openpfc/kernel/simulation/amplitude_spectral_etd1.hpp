// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file amplitude_spectral_etd1.hpp
 * @brief ETD1 for several complex amplitudes on one real-to-complex FFT.
 *
 * Each amplitude is a pair of real fields. On the half spectrum,
 * `c(q) = re^ + i im^` and `conj(eta^(-q)) = re^ - i im^` advance with
 * independent real symbols `s(q)` and `s(-q)`:
 *
 *   d = exp(s dt) c + phi1(s, dt) N
 *   phi1 = dt (1 + s dt / 2)    when |s dt| < 1e-8
 *   phi1 = (exp(s dt) - 1) / s  otherwise
 *
 * The caller supplies both symbols, including any Nyquist reduction.
 * `SpectralETDSystem` stays one real field. This header is host-only.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/fft/fft_interface.hpp>

namespace pfc::sim {

/// `exp(s dt)` and the APFC ETD1 weight `phi1` for one real symbol.
struct AmplitudeETD1Weight {
  double exp_s{1.0};
  double phi1{0.0};
};

/**
 * @brief Reference ETD1 weight used by the amplitude stepper.
 *
 * The small-argument branch matches `|s dt| < 1e-8`, not a threshold on
 * `|s|` alone. `phi1` already multiplies `N` with no extra factor of `dt`.
 */
[[nodiscard]] inline AmplitudeETD1Weight amplitude_etd1_weight(double symbol,
                                                               double dt) {
  const double z = symbol * dt;
  AmplitudeETD1Weight w;
  w.exp_s = std::exp(z);
  if (std::abs(z) < 1e-8) {
    w.phi1 = dt * (1.0 + 0.5 * z);
  } else {
    // expm1 avoids subtractive cancellation near the Taylor threshold.
    w.phi1 = std::expm1(z) / symbol;
  }
  if (!std::isfinite(w.exp_s) || !std::isfinite(w.phi1)) {
    throw std::domain_error("amplitude ETD1 weight is not finite");
  }
  return w;
}

/**
 * @brief Several complex amplitudes, one ETD1 step, host FFT.
 *
 * The nonlinear callback reads every accepted amplitude and writes `N`.
 * A failed check leaves the stored fields unchanged.
 */
class AmplitudeSpectralETD1 {
public:
  using Complex = std::complex<double>;

  AmplitudeSpectralETD1(fft::IHostFFT &fft, std::size_t amplitude_count)
      : m_fft(fft), m_count(amplitude_count), m_n(fft.size_inbox()),
        m_m(fft.size_outbox()) {
    if (m_count == 0) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1: need at least one amplitude");
    }
    m_re.assign(m_count, std::vector<double>(m_n, 0.0));
    m_im.assign(m_count, std::vector<double>(m_n, 0.0));
    m_hold_re = m_re;
    m_hold_im = m_im;
    m_n_re = m_re;
    m_n_im = m_im;
    m_next_re = m_re;
    m_next_im = m_im;
    m_symbol_q.assign(m_count, {});
    m_symbol_m.assign(m_count, {});
    m_exp_q.assign(m_count, {});
    m_phi_q.assign(m_count, {});
    m_exp_m.assign(m_count, {});
    m_phi_m.assign(m_count, {});
    m_ready.assign(m_count, false);
    m_hat_re.assign(m_m, Complex{});
    m_hat_im.assign(m_m, Complex{});
    m_hat_nr.assign(m_m, Complex{});
    m_hat_ni.assign(m_m, Complex{});
  }

  AmplitudeSpectralETD1(const AmplitudeSpectralETD1 &) = delete;
  AmplitudeSpectralETD1 &operator=(const AmplitudeSpectralETD1 &) = delete;

  [[nodiscard]] std::size_t amplitude_count() const noexcept { return m_count; }
  [[nodiscard]] std::size_t inbox_size() const noexcept { return m_n; }
  [[nodiscard]] std::size_t outbox_size() const noexcept { return m_m; }
  [[nodiscard]] double dt() const noexcept { return m_dt; }

  [[nodiscard]] std::span<double> real(std::size_t amplitude) {
    return mutable_field(m_re, amplitude);
  }
  [[nodiscard]] std::span<double> imag(std::size_t amplitude) {
    return mutable_field(m_im, amplitude);
  }
  [[nodiscard]] std::span<const double> real(std::size_t amplitude) const {
    return const_field(m_re, amplitude);
  }
  [[nodiscard]] std::span<const double> imag(std::size_t amplitude) const {
    return const_field(m_im, amplitude);
  }

  /// Store `s(q)` and `s(-q)` on this rank's outbox. Both spans are copied.
  void set_symbol(std::size_t amplitude, std::span<const double> symbol_q,
                  std::span<const double> symbol_minus_q) {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1: amplitude index");
    }
    if (symbol_q.size() != m_m || symbol_minus_q.size() != m_m) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1: symbol size != FFT outbox size");
    }
    require_finite(symbol_q, "symbol");
    require_finite(symbol_minus_q, "symbol");
    m_ready[amplitude] = false;
    m_symbol_q[amplitude].assign(symbol_q.begin(), symbol_q.end());
    m_symbol_m[amplitude].assign(symbol_minus_q.begin(), symbol_minus_q.end());
    if (m_dt_set) {
      rebuild_weights(amplitude, m_dt, m_exp_q, m_phi_q, m_exp_m, m_phi_m);
    }
    m_ready[amplitude] = true;
  }

  void set_dt(double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1: dt must be positive and finite");
    }
    // Build the complete coefficient family before publishing the new time step.
    // An overflow on a later amplitude must not leave earlier ones updated.
    auto exp_q = m_exp_q;
    auto phi_q = m_phi_q;
    auto exp_m = m_exp_m;
    auto phi_m = m_phi_m;
    for (std::size_t j = 0; j < m_count; ++j) {
      if (m_ready[j]) {
        rebuild_weights(j, dt, exp_q, phi_q, exp_m, phi_m);
      }
    }
    m_exp_q.swap(exp_q);
    m_phi_q.swap(phi_q);
    m_exp_m.swap(exp_m);
    m_phi_m.swap(phi_m);
    m_dt = dt;
    m_dt_set = true;
  }

  /**
   * @brief One ETD1 step from the stored amplitudes.
   *
   * `nonlinear` is called as
   * `nonlinear(real, imag, n_real, n_imag)` with one span per amplitude.
   * The real and imag spans are the accepted state. Write `N` into the
   * last two ranges. Returns `time + dt`.
   */
  template <class Nonlinear>
  [[nodiscard]] double step(double time, Nonlinear &&nonlinear) {
    if (!std::isfinite(time)) {
      throw std::invalid_argument("AmplitudeSpectralETD1: time is not finite");
    }
    if (!m_dt_set) {
      throw std::invalid_argument("AmplitudeSpectralETD1: set_dt before step");
    }
    for (std::size_t j = 0; j < m_count; ++j) {
      if (!m_ready[j]) {
        throw std::invalid_argument("AmplitudeSpectralETD1: symbol is not set");
      }
    }
    for (std::size_t j = 0; j < m_count; ++j) {
      m_hold_re[j] = m_re[j];
      m_hold_im[j] = m_im[j];
      std::fill(m_n_re[j].begin(), m_n_re[j].end(), 0.0);
      std::fill(m_n_im[j].begin(), m_n_im[j].end(), 0.0);
    }
    std::vector<std::span<const double>> re_view;
    std::vector<std::span<const double>> im_view;
    std::vector<std::span<double>> nre_view;
    std::vector<std::span<double>> nim_view;
    re_view.reserve(m_count);
    im_view.reserve(m_count);
    nre_view.reserve(m_count);
    nim_view.reserve(m_count);
    for (std::size_t j = 0; j < m_count; ++j) {
      re_view.emplace_back(m_hold_re[j]);
      im_view.emplace_back(m_hold_im[j]);
      nre_view.emplace_back(m_n_re[j]);
      nim_view.emplace_back(m_n_im[j]);
    }
    nonlinear(re_view, im_view, nre_view, nim_view);
    for (std::size_t j = 0; j < m_count; ++j) {
      require_finite(m_n_re[j], "nonlinear real part");
      require_finite(m_n_im[j], "nonlinear imag part");
      integrate_one(j);
      require_finite(m_next_re[j], "updated real part");
      require_finite(m_next_im[j], "updated imag part");
    }
    for (std::size_t j = 0; j < m_count; ++j) {
      std::copy(m_next_re[j].begin(), m_next_re[j].end(), m_re[j].begin());
      std::copy(m_next_im[j].begin(), m_next_im[j].end(), m_im[j].begin());
    }
    return time + m_dt;
  }

private:
  fft::IHostFFT &m_fft;
  std::size_t m_count{0};
  std::size_t m_n{0};
  std::size_t m_m{0};
  double m_dt{0.0};
  bool m_dt_set{false};
  std::vector<std::vector<double>> m_re;
  std::vector<std::vector<double>> m_im;
  std::vector<std::vector<double>> m_hold_re;
  std::vector<std::vector<double>> m_hold_im;
  std::vector<std::vector<double>> m_n_re;
  std::vector<std::vector<double>> m_n_im;
  std::vector<std::vector<double>> m_next_re;
  std::vector<std::vector<double>> m_next_im;
  std::vector<std::vector<double>> m_symbol_q;
  std::vector<std::vector<double>> m_symbol_m;
  std::vector<std::vector<double>> m_exp_q;
  std::vector<std::vector<double>> m_phi_q;
  std::vector<std::vector<double>> m_exp_m;
  std::vector<std::vector<double>> m_phi_m;
  std::vector<char> m_ready;
  std::vector<Complex> m_hat_re;
  std::vector<Complex> m_hat_im;
  std::vector<Complex> m_hat_nr;
  std::vector<Complex> m_hat_ni;

  [[nodiscard]] std::span<double>
  mutable_field(std::vector<std::vector<double>> &field, std::size_t amplitude) {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1: amplitude index");
    }
    return field[amplitude];
  }
  [[nodiscard]] std::span<const double>
  const_field(const std::vector<std::vector<double>> &field,
              std::size_t amplitude) const {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1: amplitude index");
    }
    return field[amplitude];
  }

  static void require_finite(std::span<const double> values, const char *what) {
    for (double v : values) {
      if (!std::isfinite(v)) {
        throw std::domain_error(std::string("AmplitudeSpectralETD1: non-finite ") +
                                what);
      }
    }
  }

  void rebuild_weights(std::size_t amplitude, double dt,
                       std::vector<std::vector<double>> &exp_q_table,
                       std::vector<std::vector<double>> &phi_q_table,
                       std::vector<std::vector<double>> &exp_m_table,
                       std::vector<std::vector<double>> &phi_m_table) {
    std::vector<double> exp_q(m_m);
    std::vector<double> phi_q(m_m);
    std::vector<double> exp_m(m_m);
    std::vector<double> phi_m(m_m);
    for (std::size_t q = 0; q < m_m; ++q) {
      const AmplitudeETD1Weight wq =
          amplitude_etd1_weight(m_symbol_q[amplitude][q], dt);
      const AmplitudeETD1Weight wm =
          amplitude_etd1_weight(m_symbol_m[amplitude][q], dt);
      exp_q[q] = wq.exp_s;
      phi_q[q] = wq.phi1;
      exp_m[q] = wm.exp_s;
      phi_m[q] = wm.phi1;
    }
    exp_q_table[amplitude].swap(exp_q);
    phi_q_table[amplitude].swap(phi_q);
    exp_m_table[amplitude].swap(exp_m);
    phi_m_table[amplitude].swap(phi_m);
  }

  void integrate_one(std::size_t amplitude) {
    m_fft.forward(m_hold_re[amplitude], m_hat_re);
    m_fft.forward(m_hold_im[amplitude], m_hat_im);
    m_fft.forward(m_n_re[amplitude], m_hat_nr);
    m_fft.forward(m_n_im[amplitude], m_hat_ni);
    const Complex i(0.0, 1.0);
    for (std::size_t q = 0; q < m_m; ++q) {
      const Complex c1 = m_hat_re[q] + i * m_hat_im[q];
      const Complex c2 = m_hat_re[q] - i * m_hat_im[q];
      const Complex n1 = m_hat_nr[q] + i * m_hat_ni[q];
      const Complex n2 = m_hat_nr[q] - i * m_hat_ni[q];
      const Complex d1 = m_exp_q[amplitude][q] * c1 + m_phi_q[amplitude][q] * n1;
      const Complex d2 = m_exp_m[amplitude][q] * c2 + m_phi_m[amplitude][q] * n2;
      m_hat_re[q] = 0.5 * (d1 + d2);
      m_hat_im[q] = (d1 - d2) / (2.0 * i);
    }
    m_fft.backward(m_hat_re, m_next_re[amplitude]);
    m_fft.backward(m_hat_im, m_next_im[amplitude]);
  }
};

} // namespace pfc::sim
