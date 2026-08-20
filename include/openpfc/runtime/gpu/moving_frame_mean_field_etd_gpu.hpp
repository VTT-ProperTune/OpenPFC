// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file moving_frame_mean_field_etd_gpu.hpp
 * @brief Device `IDeviceFFT` moving-frame mean-field spectral-ETD driver.
 *
 * Sibling of host `pfc::sim::MovingFrameMeanFieldETDSystem`. FFT, \(P(k)\) /
 * \(\chi(k)\) multiply, and ETD1 run on device; \(N\) and free-energy density
 * are formed on a host view (temperature is \(T_{\mathrm{var}}(x,t)\)).
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/moving_frame_mean_field_etd.hpp>
#include <openpfc/kernel/simulation/observable_reduce.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/etd1_apply_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim {

template <class Physics, class MemorySpace>
  requires MovingFrameMeanFieldETDPhysics<Physics>
class DeviceMovingFrameMeanFieldETDSystem {
public:
  using Complex = std::complex<double>;
  using FFT = fft::IDeviceFFT<MemorySpace>;
  using RealBuffer = typename FFT::RealBuffer;
  using ComplexBuffer = typename FFT::ComplexBuffer;

  DeviceMovingFrameMeanFieldETDSystem(Physics physics, FFT &fft,
                                      SimulationState &state, double dt,
                                      MovingFrameMeanFieldETDOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)), m_exp(fft.size_outbox()),
        m_n_weight_dev(fft.size_outbox()), m_filter_dev(fft.size_outbox()),
        m_p_k_dev(fft.size_outbox()), m_candidate(fft.size_outbox()) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument(
          "DeviceMovingFrameMeanFieldETDSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument(
          "DeviceMovingFrameMeanFieldETDSystem: primary field '" + m_opt.psi_name +
          "' is missing");
    }
    auto &psi = m_state.get_field<double, MemorySpace>(m_opt.psi_name);
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "DeviceMovingFrameMeanFieldETDSystem: psi.size() != FFT inbox size");
    }
    allocate_work_fields(psi.domain());
    prepare_operators();
  }

  void prepare_operators() {
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom =
        m_state.get_field<double, MemorySpace>(m_opt.psi_name).domain();
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
    m_exp.copy_from_host(m_cache.exp_Ldt());
    m_n_weight_dev.copy_from_host(m_n_weight);
    m_filter_dev.copy_from_host(m_filter);
    m_p_k_dev.copy_from_host(m_p_k);
  }

  double step(double t) {
    auto &psi = m_state.get_field<double, MemorySpace>(m_opt.psi_name);
    auto &psi_mf = m_state.get_field<double, MemorySpace>(m_opt.psi_mf_name);
    auto &p_star = m_state.get_field<double, MemorySpace>(m_opt.p_star_name);
    auto &n_real = m_state.get_field<double, MemorySpace>(m_opt.n_name);
    auto &fe = m_state.get_field<double, MemorySpace>(m_opt.fe_name);
    auto &psi_hat = m_state.get_field<Complex, MemorySpace>(m_opt.psi_hat_name);
    auto &psi_mf_hat =
        m_state.get_field<Complex, MemorySpace>(m_opt.psi_mf_hat_name);
    auto &p_hat = m_state.get_field<Complex, MemorySpace>(m_opt.p_hat_name);
    auto &n_hat = m_state.get_field<Complex, MemorySpace>(m_opt.n_hat_name);

    psi.sync_to_device();
    m_fft.forward(psi.buffer(), psi_hat.buffer());
    psi_hat.note_device_write();

    apply_filter(psi_hat.data(), m_filter_dev.data(), psi_mf_hat.data(),
                 m_fft.size_outbox());
    psi_mf_hat.note_device_write();
    m_fft.backward(psi_mf_hat.buffer(), psi_mf.buffer());
    psi_mf.note_device_write();

    apply_filter(psi_hat.data(), m_p_k_dev.data(), p_hat.data(),
                 m_fft.size_outbox());
    p_hat.note_device_write();
    m_fft.backward(p_hat.buffer(), p_star.buffer());
    p_star.note_device_write();

    const int nx = psi.box().size[0];
    const int ny = psi.box().size[1];
    const int nz = psi.box().size[2];
    psi.with_host_view([&](double *pd, std::size_t) {
      psi_mf.with_host_view([&](double *md, std::size_t) {
        p_star.with_host_view([&](double *pp, std::size_t) {
          n_real.with_host_view([&](double *nd, std::size_t) {
            fe.with_host_view([&](double *fd, std::size_t) {
              std::size_t idx = 0;
              for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                  for (int i = 0; i < nx; ++i) {
                    const double T_var =
                        m_physics.temperature_variation(psi.coords(i, j, k)[0], t);
                    nd[idx] =
                        m_physics.nonlinearity(pd[idx], md[idx], pp[idx], T_var);
                    fd[idx] = m_physics.free_energy_density(pd[idx], md[idx],
                                                            pp[idx], T_var);
                    ++idx;
                  }
                }
              }
            });
          });
        });
      });
    });
    n_real.sync_to_device();

    m_fft.forward(n_real.buffer(), n_hat.buffer());
    n_hat.note_device_write();
    apply_etd1(psi_hat.data(), n_hat.data(), m_exp.data(), m_n_weight_dev.data(),
               m_candidate.data(), m_fft.size_outbox());
    m_fft.backward(m_candidate, psi.buffer());
    psi.note_device_write();

    m_fe_sum = sum_owned(fe);
    int mpi_ready = 0;
    int mpi_done = 0;
    MPI_Initialized(&mpi_ready);
    MPI_Finalized(&mpi_done);
    if (mpi_ready != 0 && mpi_done == 0) {
      m_fe_integral = integrate_owned(fe, m_opt.comm);
    } else {
      m_fe_integral = m_fe_sum * cell_volume(fe.domain());
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
    add_declared_field<T, MemorySpace>(m_state, name, domain, box, 0);
  }

  static void apply_filter(const Complex *in, const double *chi, Complex *out,
                           std::size_t n) {
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HIPSpace>) {
      pfc::multiply_complex_real_hip_impl(in, chi, out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CUDASpace>) {
      pfc::multiply_complex_real_cuda_impl(in, chi, out, n);
      return;
    }
#endif
    (void)in;
    (void)chi;
    (void)out;
    (void)n;
    throw std::logic_error(
        "DeviceMovingFrameMeanFieldETDSystem: no k-space multiply for this "
        "MemorySpace");
  }

  static void apply_etd1(const Complex *u, const Complex *nlin,
                         const double *exp_Ldt, const double *n_weight, Complex *out,
                         std::size_t n) {
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HIPSpace>) {
      integrator::apply_etd1_update_hip(u, nlin, exp_Ldt, n_weight, out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CUDASpace>) {
      integrator::apply_etd1_update_cuda(u, nlin, exp_Ldt, n_weight, out, n);
      return;
    }
#endif
    throw std::logic_error(
        "DeviceMovingFrameMeanFieldETDSystem: no device ETD1 combine for this "
        "MemorySpace");
  }

  Physics m_physics;
  FFT &m_fft;
  SimulationState &m_state;
  double m_dt{};
  MovingFrameMeanFieldETDOptions m_opt{};
  std::vector<double> m_L;
  std::vector<double> m_filter;
  std::vector<double> m_p_k;
  std::vector<double> m_k_lap;
  std::vector<double> m_n_weight;
  integrator::SpectralExpCoefficientCache<> m_cache{};
  RealBuffer m_exp;
  RealBuffer m_n_weight_dev;
  RealBuffer m_filter_dev;
  RealBuffer m_p_k_dev;
  ComplexBuffer m_candidate;
  double m_fe_sum{0.0};
  double m_fe_integral{0.0};
};

} // namespace pfc::sim

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
