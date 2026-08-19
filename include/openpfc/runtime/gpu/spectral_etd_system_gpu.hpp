// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_etd_system_gpu.hpp
 * @brief Device `IDeviceFFT` spectral-ETD driver on `SimulationState`.
 *
 * @details
 * Sibling of host `pfc::sim::SpectralEtdSystem`. Lives in runtime because
 * the combine uses `apply_etd1_update_{cuda,hip}` (kernel must not include
 * runtime). Coefficients and `N(psi)` are formed on the host; FFT and the
 * ETD1 combine run on device buffers.
 *
 *   N = N(psi) [host view] → sync → FFT → ETD1 combine → iFFT
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/dealias.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/integrator/spectral_exp_coefficients.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/spectral_etd_system.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/etd1_apply_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim {

/**
 * @brief Device spectral-ETD driver for one real field + its Fourier hat.
 *
 * @tparam Physics     Models @ref SpectralEtdPhysics.
 * @tparam MemorySpace `CudaSpace` or `HipSpace`.
 */
template <class Physics, class MemorySpace>
  requires SpectralEtdPhysics<Physics>
class DeviceSpectralEtdSystem {
public:
  using Complex = std::complex<double>;
  using FFT = fft::IDeviceFFT<MemorySpace>;
  using RealBuffer = typename FFT::RealBuffer;
  using ComplexBuffer = typename FFT::ComplexBuffer;

  DeviceSpectralEtdSystem(Physics physics, FFT &fft, SimulationState &state,
                          double dt, SpectralEtdOptions opt = {})
      : m_physics(std::move(physics)), m_fft(fft), m_state(state), m_dt(dt),
        m_opt(std::move(opt)), m_exp(fft.size_outbox()),
        m_phi(fft.size_outbox()), m_candidate(fft.size_outbox()),
        m_n_work(fft.size_outbox()), m_mask_real(fft.size_outbox()) {
    if (m_dt <= 0.0) {
      throw std::invalid_argument("DeviceSpectralEtdSystem: dt must be > 0");
    }
    if (!m_state.has_field(m_opt.psi_name)) {
      throw std::invalid_argument(
          "DeviceSpectralEtdSystem: primary field '" + m_opt.psi_name +
          "' is missing");
    }
    auto &psi = m_state.get_field<double, MemorySpace>(m_opt.psi_name);
    if (psi.size() != m_fft.size_inbox()) {
      throw std::invalid_argument(
          "DeviceSpectralEtdSystem: psi.size() != FFT inbox size");
    }
    allocate_work_fields(psi.domain());
    prepare_operators();
  }

  void prepare_operators() {
    const auto outbox = m_fft.get_outbox_bounds();
    const auto &dom =
        m_state.get_field<double, MemorySpace>(m_opt.psi_name).domain();
    m_L.assign(m_fft.size_outbox(), 0.0);
    fft::kspace::for_each_kpoint(
        outbox, dom,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          m_L[idx] = m_physics.linear_symbol(
              fft::kspace::k_laplacian_value(kx, ky, kz));
        });
    m_cache.ensure(std::span<const double>(m_L), m_dt,
                   integrator::SpectralExpOperatorId{.value = 1},
                   integrator::SpectralExpDtId::from_bits(m_dt),
                   integrator::SpectralExpConfigId{.value = m_opt.dealias ? 2
                                                                         : 1});
    m_exp.copy_from_host(m_cache.exp_Ldt());
    m_phi.copy_from_host(m_cache.phi1_L());
    if (m_opt.dealias) {
      m_dealias_mask.assign(m_fft.size_outbox(), 0.0);
      fft::kspace::fill_two_thirds_mask(
          outbox, pfc::domain::get_size(dom), pfc::domain::get_spacing(dom),
          m_dealias_mask.data(), m_dealias_mask.size());
      m_mask_real.copy_from_host(m_dealias_mask);
    }
  }

  double step(double t) {
    auto &psi = m_state.get_field<double, MemorySpace>(m_opt.psi_name);
    auto &n_real = m_state.get_field<double, MemorySpace>(m_opt.n_name);
    auto &psi_hat = m_state.get_field<Complex, MemorySpace>(m_opt.psi_hat_name);
    auto &n_hat = m_state.get_field<Complex, MemorySpace>(m_opt.n_hat_name);

    psi.with_host_view([&](double *pd, std::size_t n) {
      n_real.with_host_view([&](double *nd, std::size_t) {
        for (std::size_t i = 0; i < n; ++i) {
          nd[i] = m_physics.nonlinearity(pd[i]);
        }
      });
    });
    psi.sync_to_device();
    n_real.sync_to_device();

    m_fft.forward(psi.buffer(), psi_hat.buffer());
    m_fft.forward(n_real.buffer(), n_hat.buffer());
    psi_hat.note_device_write();
    n_hat.note_device_write();

    const Complex *n_ptr = n_hat.data();
    if (m_opt.dealias) {
      apply_dealias(n_hat.data(), m_n_work.data(), m_fft.size_outbox());
      n_ptr = m_n_work.data();
    }
    apply_etd1(psi_hat.data(), n_ptr, m_exp.data(), m_phi.data(),
               m_candidate.data(), m_fft.size_outbox());
    m_fft.backward(m_candidate, psi.buffer());
    psi.note_device_write();
    return t + m_dt;
  }

  [[nodiscard]] double dt() const noexcept { return m_dt; }
  [[nodiscard]] const std::vector<double> &linear_symbol() const noexcept {
    return m_L;
  }

private:
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
    add_declared_field<T, MemorySpace>(m_state, name, domain, box, 0);
  }

  static void apply_etd1(const Complex *u, const Complex *nlin,
                         const double *exp_Ldt, const double *phi1_L,
                         Complex *out, std::size_t n) {
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HipSpace>) {
      integrator::apply_etd1_update_hip(u, nlin, exp_Ldt, phi1_L, out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CudaSpace>) {
      integrator::apply_etd1_update_cuda(u, nlin, exp_Ldt, phi1_L, out, n);
      return;
    }
#endif
    throw std::logic_error(
        "DeviceSpectralEtdSystem: no device ETD1 combine for this MemorySpace");
  }

  void apply_dealias(const Complex *in, Complex *out, std::size_t n) {
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<MemorySpace, HipSpace>) {
      pfc::multiply_complex_real_hip_impl(in, m_mask_real.data(), out, n);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<MemorySpace, CudaSpace>) {
      pfc::multiply_complex_real_cuda_impl(in, m_mask_real.data(), out, n);
      return;
    }
#endif
    (void)in;
    (void)out;
    (void)n;
    throw std::logic_error(
        "DeviceSpectralEtdSystem: no dealias multiply for this MemorySpace");
  }

  Physics m_physics;
  FFT &m_fft;
  SimulationState &m_state;
  double m_dt{};
  SpectralEtdOptions m_opt{};
  std::vector<double> m_L;
  std::vector<double> m_dealias_mask;
  integrator::SpectralExpCoefficientCache<> m_cache{};
  RealBuffer m_exp;
  RealBuffer m_phi;
  ComplexBuffer m_candidate;
  ComplexBuffer m_n_work;
  RealBuffer m_mask_real;
};

} // namespace pfc::sim

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
