// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file moving_frame_mean_field_etd.hpp
 * @brief Host spectral ETD with mean-field filter, @f$P(k)@f$, and
 * @f$T_{\mathrm{var}}@f$.
 *
 * @details
 * Like `SpectralMeanFieldETDSystem`, but:
 * 1. @f$\hat\psi_{\mathrm{MF}} = \chi(k)\,\hat\psi@f$ then iFFT,
 * 2. @f$P*\psi@f$ from @f$P(k)\,\hat\psi@f$ then iFFT,
 * 3. @f$N = N(\psi,\psi_{\mathrm{MF}},P*\psi,T_{\mathrm{var}}(x,t))@f$,
 * 4. ETD combine uses @f$n_{\mathrm{weight}} = k_{\mathrm{lap}}\,\phi_1@f$,
 * 5. rank-local free-energy density is reduced with `integrate_owned`.
 */

#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/etd1_apply.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_mean_field_etd.hpp>

namespace pfc::sim {

struct MovingFrameMeanFieldETDOptions : SpectralMeanFieldETDOptions {
  std::string p_star_name{"P_star_psi"};
  std::string p_hat_name{"P_hat"};
  std::string fe_name{"fe_density"};
  MPI_Comm comm{MPI_COMM_WORLD};
};

template <class Physics>
  requires MovingFrameMeanFieldETDPhysics<Physics>
class MovingFrameMeanFieldETDSystem {
public:
  using Complex = std::complex<double>;

  MovingFrameMeanFieldETDSystem(Physics physics, fft::IHostFFT &fft,
                                SimulationState &state, double dt,
                                MovingFrameMeanFieldETDOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument("MovingFrameMeanFieldETDSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument("MovingFrameMeanFieldETDSystem: primary field '" +
                                  m_opt.psi_name + "' is missing");
    }
    auto &psi = m_state.get_field<double>(m_opt.psi_name);
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "MovingFrameMeanFieldETDSystem: psi.size() != FFT inbox size");
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
    m_p_k.assign(n, 0.0);
    m_k_lap.assign(n, 0.0);
    fft::kspace::for_each_kpoint(
        outbox, dom,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          const double k_lap = fft::kspace::k_laplacian_value(kx, ky, kz);
          m_k_lap[idx] = k_lap;
          m_L[idx] = m_physics.linear_symbol(k_lap);
          m_filter[idx] = m_physics.filter_mf(k_lap);
          m_p_k[idx] = m_physics.correlation_kernel(k_lap);
        });
    m_cache.ensure(std::span<const double>(m_L), m_dt,
                   integrator::SpectralExpOperatorId{.value = 1},
                   integrator::SpectralExpDtId::from_bits(m_dt),
                   integrator::SpectralExpConfigId{.value = 4});
    const auto phi1 = m_cache.phi1_L();
    m_n_weight.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      m_n_weight[i] = m_k_lap[i] * phi1[i];
    }
  }

  double step(double t) {
    auto &psi_field = m_state.get_field<double>(m_opt.psi_name);
    auto &psi = psi_field.vec();
    auto &psi_mf = m_state.get_field<double>(m_opt.psi_mf_name).vec();
    auto &p_star = m_state.get_field<double>(m_opt.p_star_name).vec();
    auto &n_real = m_state.get_field<double>(m_opt.n_name).vec();
    auto &fe = m_state.get_field<double>(m_opt.fe_name).vec();
    auto &psi_hat = m_state.get_field<Complex>(m_opt.psi_hat_name).vec();
    auto &psi_mf_hat = m_state.get_field<Complex>(m_opt.psi_mf_hat_name).vec();
    auto &p_hat = m_state.get_field<Complex>(m_opt.p_hat_name).vec();
    auto &n_hat = m_state.get_field<Complex>(m_opt.n_hat_name).vec();

    m_fft.forward(psi, psi_hat);
    for (std::size_t i = 0; i < psi_hat.size(); ++i) {
      psi_mf_hat[i] = m_filter[i] * psi_hat[i];
      p_hat[i] = m_p_k[i] * psi_hat[i];
    }
    m_fft.backward(psi_mf_hat, psi_mf);
    m_fft.backward(p_hat, p_star);

    const int nx = psi_field.box().size[0];
    const int ny = psi_field.box().size[1];
    const int nz = psi_field.box().size[2];
    std::size_t idx = 0;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const double T_var =
              m_physics.temperature_variation(psi_field.coords(i, j, k)[0], t);
          n_real[idx] =
              m_physics.nonlinearity(psi[idx], psi_mf[idx], p_star[idx], T_var);
          fe[idx] = m_physics.free_energy_density(psi[idx], psi_mf[idx], p_star[idx],
                                                  T_var);
          ++idx;
        }
      }
    }
    psi_field.note_host_write();
    m_state.get_field<double>(m_opt.n_name).note_host_write();
    m_state.get_field<double>(m_opt.fe_name).note_host_write();

    m_fft.forward(n_real, n_hat);
    integrator::apply_etd1_update(
        std::span<const double>(m_cache.exp_Ldt()),
        std::span<const double>(m_n_weight), std::span<const Complex>(psi_hat),
        std::span<const Complex>(n_hat), std::span<Complex>(psi_hat));
    m_fft.backward(psi_hat, psi);
    auto &fe_field = m_state.get_field<double>(m_opt.fe_name);
    m_fe_sum = sum_owned(fe_field);
    int mpi_ready = 0;
    int mpi_done = 0;
    MPI_Initialized(&mpi_ready);
    MPI_Finalized(&mpi_done);
    if (mpi_ready != 0 && mpi_done == 0) {
      m_fe_integral = integrate_owned(fe_field, m_opt.comm);
    } else {
      m_fe_integral = m_fe_sum * cell_volume(fe_field.domain());
    }
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }
  [[nodiscard]] const std::vector<double> &filter_mf() const noexcept {
    return m_filter;
  }
  [[nodiscard]] const std::vector<double> &correlation_kernel() const noexcept {
    return m_p_k;
  }
  [[nodiscard]] double last_free_energy_sum() const noexcept { return m_fe_sum; }
  [[nodiscard]] double last_free_energy() const noexcept { return m_fe_integral; }

private:
  void allocate_work_fields(const Domain &domain) {
    const auto inbox = m_fft.get_inbox_bounds();
    const auto outbox = m_fft.get_outbox_bounds();
    add_if_missing<double>(m_opt.n_name, domain, inbox);
    add_if_missing<double>(m_opt.psi_mf_name, domain, inbox);
    add_if_missing<double>(m_opt.p_star_name, domain, inbox);
    add_if_missing<double>(m_opt.fe_name, domain, inbox);
    add_if_missing<Complex>(m_opt.psi_hat_name, domain, outbox);
    add_if_missing<Complex>(m_opt.psi_mf_hat_name, domain, outbox);
    add_if_missing<Complex>(m_opt.p_hat_name, domain, outbox);
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
  MovingFrameMeanFieldETDOptions m_opt{};
  std::vector<double> m_L;
  std::vector<double> m_filter;
  std::vector<double> m_p_k;
  std::vector<double> m_k_lap;
  std::vector<double> m_n_weight;
  integrator::SpectralExpCoefficientCache<> m_cache{};
  double m_fe_sum{0.0};
  double m_fe_integral{0.0};
};

} // namespace pfc::sim
