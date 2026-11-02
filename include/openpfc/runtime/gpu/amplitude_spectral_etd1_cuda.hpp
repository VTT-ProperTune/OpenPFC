// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file amplitude_spectral_etd1_cuda.hpp
 * @brief CUDA ETD1 for several complex amplitudes on `IDeviceFFT`.
 *
 * The nonlinear callback still runs on the host. The forward transform,
 * the half-spectrum update, and the backward transform stay on the device
 * FFT. HIP is a separate kernel entry and is not executed here.
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/simulation/amplitude_spectral_etd1.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim {

/// Device-resident spectral update for @ref AmplitudeSpectralETD1.
class AmplitudeSpectralETD1Cuda {
public:
  using Complex = std::complex<double>;
  using FFT = fft::IDeviceFFT<pfc::CUDASpace>;
  using RealBuffer = FFT::RealBuffer;
  using ComplexBuffer = FFT::ComplexBuffer;

  AmplitudeSpectralETD1Cuda(FFT &fft, std::size_t amplitude_count)
      : m_fft(fft), m_count(amplitude_count), m_n(fft.size_inbox()),
        m_m(fft.size_outbox()), m_re_d(m_n), m_im_d(m_n), m_nre_d(m_n), m_nim_d(m_n),
        m_e1(m_m), m_p1(m_m), m_e2(m_m), m_p2(m_m), m_ha(m_m), m_hb(m_m), m_na(m_m),
        m_nb(m_m), m_oa(m_m), m_ob(m_m) {
    if (m_count == 0) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1Cuda: need at least one amplitude");
    }
    const std::vector<double> zeros(m_n, 0.0);
    m_re.assign(m_count, zeros);
    m_im.assign(m_count, zeros);
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
  }

  AmplitudeSpectralETD1Cuda(const AmplitudeSpectralETD1Cuda &) = delete;
  AmplitudeSpectralETD1Cuda &operator=(const AmplitudeSpectralETD1Cuda &) = delete;

  [[nodiscard]] std::size_t amplitude_count() const noexcept { return m_count; }
  [[nodiscard]] std::size_t inbox_size() const noexcept { return m_n; }
  [[nodiscard]] std::size_t outbox_size() const noexcept { return m_m; }

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

  void set_symbol(std::size_t amplitude, std::span<const double> symbol_q,
                  std::span<const double> symbol_minus_q) {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1Cuda: amplitude index");
    }
    if (symbol_q.size() != m_m || symbol_minus_q.size() != m_m) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1Cuda: symbol size != FFT outbox size");
    }
    require_finite(symbol_q, "symbol");
    require_finite(symbol_minus_q, "symbol");
    m_ready[amplitude] = false;
    m_symbol_q[amplitude].assign(symbol_q.begin(), symbol_q.end());
    m_symbol_m[amplitude].assign(symbol_minus_q.begin(), symbol_minus_q.end());
    if (m_dt_set) {
      rebuild_weights(amplitude);
    }
    m_ready[amplitude] = true;
  }

  void set_dt(double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt)) {
      throw std::invalid_argument(
          "AmplitudeSpectralETD1Cuda: dt must be positive and finite");
    }
    m_dt = dt;
    m_dt_set = true;
    for (std::size_t j = 0; j < m_count; ++j) {
      if (m_ready[j]) {
        rebuild_weights(j);
      }
    }
  }

  template <class Nonlinear>
  [[nodiscard]] double step(double time, Nonlinear &&nonlinear) {
    if (!std::isfinite(time)) {
      throw std::invalid_argument("AmplitudeSpectralETD1Cuda: time is not finite");
    }
    if (!m_dt_set) {
      throw std::invalid_argument("AmplitudeSpectralETD1Cuda: set_dt before step");
    }
    for (std::size_t j = 0; j < m_count; ++j) {
      if (!m_ready[j]) {
        throw std::invalid_argument("AmplitudeSpectralETD1Cuda: symbol is not set");
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
  FFT &m_fft;
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
  RealBuffer m_re_d;
  RealBuffer m_im_d;
  RealBuffer m_nre_d;
  RealBuffer m_nim_d;
  RealBuffer m_e1;
  RealBuffer m_p1;
  RealBuffer m_e2;
  RealBuffer m_p2;
  ComplexBuffer m_ha;
  ComplexBuffer m_hb;
  ComplexBuffer m_na;
  ComplexBuffer m_nb;
  ComplexBuffer m_oa;
  ComplexBuffer m_ob;

  [[nodiscard]] std::span<double>
  mutable_field(std::vector<std::vector<double>> &field, std::size_t amplitude) {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1Cuda: amplitude index");
    }
    return field[amplitude];
  }
  [[nodiscard]] std::span<const double>
  const_field(const std::vector<std::vector<double>> &field,
              std::size_t amplitude) const {
    if (amplitude >= m_count) {
      throw std::out_of_range("AmplitudeSpectralETD1Cuda: amplitude index");
    }
    return field[amplitude];
  }

  static void require_finite(std::span<const double> values, const char *what) {
    for (double v : values) {
      if (!std::isfinite(v)) {
        throw std::domain_error(
            std::string("AmplitudeSpectralETD1Cuda: non-finite ") + what);
      }
    }
  }

  void rebuild_weights(std::size_t amplitude) {
    std::vector<double> exp_q(m_m);
    std::vector<double> phi_q(m_m);
    std::vector<double> exp_m(m_m);
    std::vector<double> phi_m(m_m);
    for (std::size_t q = 0; q < m_m; ++q) {
      const AmplitudeETD1Weight wq =
          amplitude_etd1_weight(m_symbol_q[amplitude][q], m_dt);
      const AmplitudeETD1Weight wm =
          amplitude_etd1_weight(m_symbol_m[amplitude][q], m_dt);
      exp_q[q] = wq.exp_s;
      phi_q[q] = wq.phi1;
      exp_m[q] = wm.exp_s;
      phi_m[q] = wm.phi1;
    }
    m_exp_q[amplitude].swap(exp_q);
    m_phi_q[amplitude].swap(phi_q);
    m_exp_m[amplitude].swap(exp_m);
    m_phi_m[amplitude].swap(phi_m);
  }

  void integrate_one(std::size_t amplitude) {
    m_re_d.copy_from_host(m_hold_re[amplitude]);
    m_im_d.copy_from_host(m_hold_im[amplitude]);
    m_nre_d.copy_from_host(m_n_re[amplitude]);
    m_nim_d.copy_from_host(m_n_im[amplitude]);
    m_e1.copy_from_host(m_exp_q[amplitude]);
    m_p1.copy_from_host(m_phi_q[amplitude]);
    m_e2.copy_from_host(m_exp_m[amplitude]);
    m_p2.copy_from_host(m_phi_m[amplitude]);
    m_fft.forward(m_re_d, m_ha);
    m_fft.forward(m_im_d, m_hb);
    m_fft.forward(m_nre_d, m_na);
    m_fft.forward(m_nim_d, m_nb);
    amplitude_pair_etd1_cuda_impl(m_ha.data(), m_hb.data(), m_na.data(), m_nb.data(),
                                  m_e1.data(), m_p1.data(), m_e2.data(), m_p2.data(),
                                  m_oa.data(), m_ob.data(), m_m);
    m_fft.backward(m_oa, m_re_d);
    m_fft.backward(m_ob, m_im_d);
    m_re_d.copy_to_host(m_next_re[amplitude].data(), m_n);
    m_im_d.copy_to_host(m_next_im[amplitude].data(), m_n);
  }
};

} // namespace pfc::sim
