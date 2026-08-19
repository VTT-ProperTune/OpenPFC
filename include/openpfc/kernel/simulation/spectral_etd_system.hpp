// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_system.hpp
 * @brief Framework-owned pseudo-spectral ETD driver on `SimulationState`.
 *
 * @details
 * Host path: physics supplies `linear_symbol(k)` and `nonlinearity(psi)`
 * (`SpectralEtdPhysics`). This type owns spectral work fields on the
 * caller's `SimulationState`, prepares ETD coefficients with
 * `for_each_kpoint` + `SpectralExpCoefficientCache`, and advances via
 * `Etd1Stepper` on the complex hat:
 *
 *   N = N(psi)  →  FFT(psi), FFT(N)  →  ETD1 combine  →  iFFT(psi)
 *
 * Optional Orszag 2/3-rule mask (M5) multiplies `N̂` when enabled.
 * Device `IDeviceFFT` instantiations follow in a later M7 slice.
 */

#include <complex>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/steppers/etd1.hpp>
#include <openpfc/kernel/simulation/steppers/step_attempt.hpp>

namespace pfc::sim {

struct SpectralEtdOptions {
  std::string psi_name{"psi"};
  std::string psi_hat_name{"psi_hat"};
  std::string n_name{"N"};
  std::string n_hat_name{"N_hat"};
  bool dealias{false};
};

/**
 * @brief Host spectral-ETD driver for one real field + its Fourier hat.
 *
 * @tparam Physics Models @ref SpectralEtdPhysics.
 */
template <class Physics>
  requires SpectralEtdPhysics<Physics>
class SpectralEtdSystem {
public:
  using Complex = std::complex<double>;

  SpectralEtdSystem(Physics physics, fft::IHostFFT &fft, SimulationState &state,
                    double dt, SpectralEtdOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument("SpectralEtdSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument(
          "SpectralEtdSystem: primary field '" + m_opt.psi_name +
          "' is missing; call physics.declare_fields first");
    }
    auto &psi = m_state.get_field<double>(m_opt.psi_name);
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "SpectralEtdSystem: psi.size() != FFT inbox size");
    }
    allocate_work_fields(psi.domain());
    m_n_hat = &m_state.get_field<Complex>(m_opt.n_hat_name).vec();
    m_etd = std::make_unique<steppers::Etd1Stepper<CopyNhat, Complex>>(
        m_dt, m_fft.size_outbox(), CopyNhat{m_n_hat});
    prepare_operators();
  }

  void prepare_operators() {
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom = m_state.get_field<double>(m_opt.psi_name).domain();
    m_L.assign(m_fft.size_outbox(), 0.0);
    fft::kspace::for_each_kpoint(
        outbox, dom,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
          m_L[idx] = m_physics.linear_symbol(k_lap);
        });
    m_cache.ensure(std::span<const double>(m_L), m_dt,
                   integrator::SpectralExpOperatorId{.value = 1},
                   integrator::SpectralExpDtId::from_bits(m_dt),
                   integrator::SpectralExpConfigId{.value = m_opt.dealias ? 2
                                                                         : 1});
    m_etd->set_coefficients(m_cache);
    if (m_opt.dealias) {
      m_dealias_mask.assign(m_fft.size_outbox(), 0.0);
      fft::kspace::fill_two_thirds_mask(
          outbox, pfc::domain::get_size(dom), pfc::domain::get_spacing(dom),
          m_dealias_mask.data(), m_dealias_mask.size());
    } else {
      m_dealias_mask.clear();
    }
  }

  /**
   * @brief One ETD1 step. Returns the candidate time `t + dt`.
   */
  double step(double t) {
    auto &psi = m_state.get_field<double>(m_opt.psi_name).vec();
    auto &n_real = m_state.get_field<double>(m_opt.n_name).vec();
    auto &psi_hat = m_state.get_field<Complex>(m_opt.psi_hat_name).vec();
    auto &n_hat = m_state.get_field<Complex>(m_opt.n_hat_name).vec();

    for (std::size_t i = 0; i < psi.size(); ++i) {
      n_real[i] = m_physics.nonlinearity(psi[i]);
    }
    m_fft.forward(psi, psi_hat);
    m_fft.forward(n_real, n_hat);
    if (m_opt.dealias) {
      for (std::size_t i = 0; i < n_hat.size(); ++i) {
        n_hat[i] *= m_dealias_mask[i];
      }
    }
    const auto attempt = m_etd->attempt(t, psi_hat);
    if (!attempt.success) {
      throw std::runtime_error("SpectralEtdSystem: ETD1 attempt failed: " +
                               m_etd->last_reason());
    }
    steppers::commit_step_attempt(psi_hat, attempt);
    m_fft.backward(psi_hat, psi);
    return attempt.t1;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }
  [[nodiscard]] const Physics &physics() const noexcept { return m_physics; }

private:
  struct CopyNhat {
    const std::vector<Complex> *n_hat{};
    void operator()(double, std::vector<Complex> &,
                    std::vector<Complex> &du) const {
      du = *n_hat;
    }
  };

  void allocate_work_fields(const Domain &domain) {
    const auto inbox = m_fft.get_inbox_bounds();
    const auto outbox = m_fft.get_outbox_bounds();
    add_if_missing<double>(m_opt.n_name, domain, inbox);
    add_if_missing<Complex>(m_opt.psi_hat_name, domain, outbox);
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
  SpectralEtdOptions m_opt{};
  std::vector<double> m_L;
  std::vector<double> m_dealias_mask;
  integrator::SpectralExpCoefficientCache<> m_cache{};
  std::vector<Complex> *m_n_hat{nullptr};
  std::unique_ptr<steppers::Etd1Stepper<CopyNhat, Complex>> m_etd;
};

} // namespace pfc::sim
