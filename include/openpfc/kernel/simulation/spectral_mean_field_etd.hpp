// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_mean_field_etd.hpp
 * @brief Host spectral ETD with a mean-field filter (tungsten / aluminum).
 *
 * @details
 * Like `SpectralETDSystem`, but:
 * 1. @f$\hat\psi_{\mathrm{MF}} = \chi(k)\,\hat\psi@f$ then iFFT,
 * 2. @f$N = N(\psi,\psi_{\mathrm{MF}})@f$,
 * 3. ETD combine uses @f$n_{\mathrm{weight}} = k_{\mathrm{lap}}\,\phi_1@f$
 *    (PFC form @f$\partial_t\hat\psi = k_{\mathrm{lap}}(C\hat\psi + \hat N)@f$).
 */

#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>

namespace pfc::sim {

struct SpectralMeanFieldETDOptions : SpectralETDOptions {
  std::string psi_mf_name{"psi_mf"};
  std::string psi_mf_hat_name{"psi_mf_hat"};
};

template <class Physics>
  requires MeanFieldETDPhysics<Physics>
class SpectralMeanFieldETDSystem {
public:
  using Complex = std::complex<double>;

  SpectralMeanFieldETDSystem(Physics physics, fft::IHostFFT &fft,
                             SimulationState &state, double dt,
                             SpectralMeanFieldETDOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument("SpectralMeanFieldETDSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument("SpectralMeanFieldETDSystem: primary field '" +
                                  m_opt.psi_name + "' is missing");
    }
    auto &psi = m_state.get_field<double>(m_opt.psi_name);
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "SpectralMeanFieldETDSystem: psi.size() != FFT inbox size");
    }
    allocate_work_fields(psi.domain());
    prepare_operators();
  }

  void prepare_operators() {
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom = m_state.get_field<double>(m_opt.psi_name).domain();
    const std::size_t n = m_fft.size_outbox();
    m_L.assign(n, 0.0);
    m_filter.assign(n, 0.0);
    m_k_lap.assign(n, 0.0);
    fft::kspace::for_each_kpoint(
        outbox, dom,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
          m_k_lap[idx] = k_lap;
          m_L[idx] = m_physics.linear_symbol(k_lap);
          m_filter[idx] = m_physics.filter_mf(k_lap);
        });
    m_cache.ensure(std::span<const double>(m_L), m_dt,
                   integrator::SpectralExpOperatorId{.value = 1},
                   integrator::SpectralExpDtId::from_bits(m_dt),
                   integrator::SpectralExpConfigId{.value = 3});
    const auto phi1 = m_cache.phi1_L();
    m_n_weight.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      m_n_weight[i] = m_k_lap[i] * phi1[i];
    }
  }

  double step(double t) {
    auto &psi = m_state.get_field<double>(m_opt.psi_name).vec();
    auto &psi_mf = m_state.get_field<double>(m_opt.psi_mf_name).vec();
    auto &n_real = m_state.get_field<double>(m_opt.n_name).vec();
    auto &psi_hat = m_state.get_field<Complex>(m_opt.psi_hat_name).vec();
    auto &psi_mf_hat = m_state.get_field<Complex>(m_opt.psi_mf_hat_name).vec();
    auto &n_hat = m_state.get_field<Complex>(m_opt.n_hat_name).vec();

    m_fft.forward(psi, psi_hat);
    for (std::size_t i = 0; i < psi_hat.size(); ++i) {
      psi_mf_hat[i] = m_filter[i] * psi_hat[i];
    }
    m_fft.backward(psi_mf_hat, psi_mf);
    for (std::size_t i = 0; i < psi.size(); ++i) {
      n_real[i] = m_physics.nonlinearity(psi[i], psi_mf[i]);
    }
    m_fft.forward(n_real, n_hat);
    integrator::apply_etd1_update(
        std::span<const double>(m_cache.exp_Ldt()),
        std::span<const double>(m_n_weight), std::span<const Complex>(psi_hat),
        std::span<const Complex>(n_hat), std::span<Complex>(psi_hat));
    m_fft.backward(psi_hat, psi);
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }
  [[nodiscard]] const std::vector<double> &filter_mf() const noexcept {
    return m_filter;
  }

private:
  void allocate_work_fields(const Domain &domain) {
    const auto inbox = m_fft.get_inbox_bounds();
    const auto outbox = m_fft.get_outbox_bounds();
    add_if_missing<double>(m_opt.n_name, domain, inbox);
    add_if_missing<double>(m_opt.psi_mf_name, domain, inbox);
    add_if_missing<Complex>(m_opt.psi_hat_name, domain, outbox);
    add_if_missing<Complex>(m_opt.psi_mf_hat_name, domain, outbox);
    add_if_missing<Complex>(m_opt.n_hat_name, domain, outbox);
  }

  template <class T>
  void add_if_missing(const std::string &name, const Domain &domain,
                      const Box3i &box) {
    if (m_state.has_field(name)) {
      return;
    }
    add_declared_field<T>(m_state, name, domain, box, 0);
  }

  Physics m_physics;
  fft::IHostFFT &m_fft;
  SimulationState &m_state;
  double m_dt{};
  SpectralMeanFieldETDOptions m_opt{};
  std::vector<double> m_L;
  std::vector<double> m_filter;
  std::vector<double> m_k_lap;
  std::vector<double> m_n_weight;
  integrator::SpectralExpCoefficientCache<> m_cache{};
};

} // namespace pfc::sim
