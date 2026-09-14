// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity_hip.hpp
 * @brief Device FFT + Green-operator path for `EigenstrainMicroelasticity`.
 *
 * Same Eyre–Milton scheme as the host solver. Transforms use
 * `pfc::fft::IDeviceFFT` (rocFFT HeFFTe). Polarisation, Green contraction,
 * local reflection, residual reduction, and equation (7) stay on device.
 * The inner loop does not copy the six tensor fields to the host.
 *
 * Host `Field<double>` `solve()` is the homogenization API: upload, iterate,
 * download strain/stress/`f_el`/`dfel`. Dendrite coupling writes the device
 * `h`/`amp` buffers directly and calls `solve_resident()`.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "microelasticity_hip.hpp requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc_apps/microelasticity.hpp>
#include <openpfc_apps/microelasticity_hip_kernels.hpp>

namespace pfc::apps {

class DeviceEigenstrainMicroelasticity {
public:
  using RealField = pfc::data::Field<double>;
  using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;
  using RealBuf = FFT::RealBuffer;
  using CplxBuf = FFT::ComplexBuffer;
  using SymRealFields = EigenstrainMicroelasticity::SymRealFields;

  DeviceEigenstrainMicroelasticity(const pfc::Domain &domain, FFT &fft,
                                   MicroelasticityParams params)
      : m_fft(fft), m_params(params),
        m_c0(EigenstrainMicroelasticity::optimal_reference(
            params.scheme, params.c_solid, params.c_liquid)),
        m_n_local(fft.size_inbox()), m_n_outbox(fft.size_outbox()) {
    if (m_params.scheme != MicroelasticityScheme::EyreMilton) {
      throw std::invalid_argument(
          "DeviceEigenstrainMicroelasticity: only EyreMilton is on the "
          "device; Basic is the host-side reference");
    }
    require_invertible(Stiffness::blend(m_params.c_solid, 1.0, m_c0, 1.0),
                       "c_solid + reference");
    require_invertible(Stiffness::blend(m_params.c_liquid, 1.0, m_c0, 1.0),
                       "c_liquid + reference");
    const auto box = fft.get_inbox_bounds();
    for (int c = 0; c < kSymComponents; ++c) {
      m_strain_host[static_cast<std::size_t>(c)] =
          pfc::data::field_from_inbox<double>(domain, box);
      m_stress_host[static_cast<std::size_t>(c)] =
          pfc::data::field_from_inbox<double>(domain, box);
      m_d_strain[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_tau[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_tau_prev[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_stress[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_hat[static_cast<std::size_t>(c)] = CplxBuf(m_n_outbox);
      m_d_g[static_cast<std::size_t>(c)] = RealBuf(m_n_outbox);
    }
    m_f_el_host = pfc::data::field_from_inbox<double>(domain, box);
    m_dfel_host = pfc::data::field_from_inbox<double>(domain, box);
    m_d_h = RealBuf(m_n_local);
    m_d_amp = RealBuf(m_n_local);
    m_d_dh = RealBuf(m_n_local);
    m_d_damp = RealBuf(m_n_local);
    m_d_f_el = RealBuf(m_n_local);
    m_d_dfel = RealBuf(m_n_local);
    m_d_kx = RealBuf(m_n_outbox);
    m_d_ky = RealBuf(m_n_outbox);
    m_d_kz = RealBuf(m_n_outbox);
    m_n_blocks = hip_detail::me_local_block_count(
        static_cast<long long>(m_n_local));
    m_d_block = RealBuf(static_cast<std::size_t>(std::max(3 * m_n_blocks, 3)));
    m_block_host.assign(static_cast<std::size_t>(std::max(3 * m_n_blocks, 3)),
                        0.0);
    hip_detail::me_fill(m_d_dh.data(), 0.5, static_cast<long long>(m_n_local));
    build_and_upload_green(domain, fft);
    bind_packs();
    const auto gs = m_strain_host[0].global_size();
    m_n_global = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                 static_cast<double>(gs[2]);
    const auto sp = m_strain_host[0].spacing();
    m_cell = sp[0] * sp[1] * sp[2];
  }

  [[nodiscard]] MicroelasticityParams &params() noexcept { return m_params; }
  [[nodiscard]] const MicroelasticityParams &params() const noexcept {
    return m_params;
  }
  [[nodiscard]] const SymRealFields &strain() const noexcept {
    return m_strain_host;
  }
  [[nodiscard]] const SymRealFields &stress() const noexcept {
    return m_stress_host;
  }
  [[nodiscard]] const RealField &elastic_energy_density() const noexcept {
    return m_f_el_host;
  }
  [[nodiscard]] const RealField &dfel_dphi() const noexcept { return m_dfel_host; }

  [[nodiscard]] double *h_device() noexcept { return m_d_h.data(); }
  [[nodiscard]] double *amp_device() noexcept { return m_d_amp.data(); }
  [[nodiscard]] double *damp_device() noexcept { return m_d_damp.data(); }
  [[nodiscard]] const double *dfel_device() const noexcept {
    return m_d_dfel.data();
  }
  [[nodiscard]] const double *f_el_device() const noexcept {
    return m_d_f_el.data();
  }
  [[nodiscard]] const double *stress_device(int c) const noexcept {
    return m_d_stress[static_cast<std::size_t>(c)].data();
  }
  [[nodiscard]] std::size_t n_local() const noexcept { return m_n_local; }
  [[nodiscard]] int n_blocks() const noexcept { return m_n_blocks; }
  [[nodiscard]] double *block_device() noexcept { return m_d_block.data(); }

  void reset() {
    m_has = false;
  }

  MicroelasticityReport solve(const RealField &h, const RealField &amp,
                              const RealField *dh_dphi = nullptr,
                              const RealField *damp_dphi = nullptr) {
    m_d_h.copy_from_host(h.data(), m_n_local);
    m_d_amp.copy_from_host(amp.data(), m_n_local);
    const bool want = (dh_dphi != nullptr) && (damp_dphi != nullptr);
    if (want) {
      m_d_dh.copy_from_host(dh_dphi->data(), m_n_local);
      m_d_damp.copy_from_host(damp_dphi->data(), m_n_local);
    }
    const auto report = solve_resident(want);
    download_outputs();
    return report;
  }

  MicroelasticityReport solve_resident(bool want_dfel) {
    if (!m_params.warm_start || !m_has) {
      for (int c = 0; c < kSymComponents; ++c) {
        hip_detail::me_fill(m_d_strain[static_cast<std::size_t>(c)].data(),
                            m_params.applied_strain[c],
                            static_cast<long long>(m_n_local));
      }
    }
    rebuild_device_params();
    bind_packs();
    polarise();
    MicroelasticityReport report;
    for (int it = 1; it <= m_params.n_el_iter; ++it) {
      for (int c = 0; c < kSymComponents; ++c) {
        std::swap(m_d_tau[static_cast<std::size_t>(c)],
                  m_d_tau_prev[static_cast<std::size_t>(c)]);
      }
      bind_packs();
      apply_green();
      const double res = eyre_milton_local();
      report.iterations = it;
      report.residual = res;
      report.residual_history.push_back(res);
      if (res < m_params.tol_el) {
        report.converged = true;
        break;
      }
    }
    finalise(want_dfel);
    m_has = true;
    return report;
  }

  [[nodiscard]] double total_elastic_energy() {
    hip_detail::me_block_sum(m_d_f_el.data(), m_d_block.data(),
                            static_cast<long long>(m_n_local), m_n_blocks);
    return finish_block_sum() * m_cell;
  }

  [[nodiscard]] double finish_block_sum() {
    m_d_block.copy_to_host(m_block_host.data(), m_block_host.size());
    double local = 0.0;
    for (int b = 0; b < m_n_blocks; ++b)
      local += m_block_host[static_cast<std::size_t>(b)];
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, m_params.comm);
    return global;
  }

  void download_outputs() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_d_strain[static_cast<std::size_t>(c)].copy_to_host(
          m_strain_host[static_cast<std::size_t>(c)].data(), m_n_local);
      m_strain_host[static_cast<std::size_t>(c)].note_host_write();
    }
    download_stress_energy();
  }

  /// Hydrostatic mean, max |dfel|, max von Mises (same σ_vm as the host
  /// dendrite snapshots). Uses `m_d_block`.
  void stress_invariants(double &mean_p, double &max_abs_dfel,
                         double &max_vm) {
    hip_detail::MESym6Const sig_ro{};
    for (int c = 0; c < kSymComponents; ++c)
      sig_ro.c[c] = m_d_stress[static_cast<std::size_t>(c)].data();
    hip_detail::me_report_stats(sig_ro, m_d_dfel.data(), m_d_block.data(),
                                static_cast<long long>(m_n_local), m_n_blocks);
    m_d_block.copy_to_host(m_block_host.data(), m_block_host.size());
    double p_local = 0.0, adfel = 0.0, vm = 0.0;
    for (int b = 0; b < m_n_blocks; ++b) {
      p_local += m_block_host[static_cast<std::size_t>(3 * b)];
      adfel = std::max(adfel, m_block_host[static_cast<std::size_t>(3 * b + 1)]);
      vm = std::max(vm, m_block_host[static_cast<std::size_t>(3 * b + 2)]);
    }
    double p_sum = 0.0;
    MPI_Allreduce(&p_local, &p_sum, 1, MPI_DOUBLE, MPI_SUM, m_params.comm);
    double extra[2] = {adfel, vm};
    double gextra[2] = {0.0, 0.0};
    MPI_Allreduce(extra, gextra, 2, MPI_DOUBLE, MPI_MAX, m_params.comm);
    mean_p = p_sum / m_n_global;
    max_abs_dfel = gextra[0];
    max_vm = gextra[1];
  }

private:
  static void require_invertible(const Stiffness &s, const char *what) {
    const double d = s.c11 - s.c12;
    const double t = s.c11 + 2.0 * s.c12;
    if (d == 0.0 || t == 0.0 || s.c44 == 0.0) {
      throw std::invalid_argument(
          std::string("DeviceEigenstrainMicroelasticity: singular ") + what);
    }
  }

  static hip_detail::MEStiffness flat(const Stiffness &s) noexcept {
    return hip_detail::MEStiffness{s.c11, s.c12, s.c44};
  }

  void rebuild_device_params() {
    m_dp.c_solid = flat(m_params.c_solid);
    m_dp.c_liquid = flat(m_params.c_liquid);
    m_dp.c0 = flat(m_c0);
    m_dp.dc = flat(Stiffness::blend(m_params.c_solid, 1.0, m_params.c_liquid,
                                    -1.0));
    for (int c = 0; c < kSymComponents; ++c)
      m_dp.pattern[c] = m_params.eigenstrain_pattern[c];
    m_dp.n = static_cast<long long>(m_n_local);
  }

  void bind_packs() {
    for (int c = 0; c < kSymComponents; ++c) {
      const auto ci = static_cast<std::size_t>(c);
      m_eps_rw.c[c] = m_d_strain[ci].data();
      m_eps_ro.c[c] = m_d_strain[ci].data();
      m_tau_rw.c[c] = m_d_tau[ci].data();
      m_tau_prev_ro.c[c] = m_d_tau_prev[ci].data();
      m_hat_rw.c[c] = reinterpret_cast<double *>(m_d_hat[ci].data());
      m_g_ro.c[c] = m_d_g[ci].data();
      m_sig_rw.c[c] = m_d_stress[ci].data();
    }
  }

  void polarise() {
    hip_detail::me_build_polarisation(m_d_h.data(), m_d_amp.data(), m_eps_ro,
                                      m_tau_rw, m_dp);
  }

  void apply_green() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.forward(m_d_tau_prev[static_cast<std::size_t>(c)],
                    m_d_hat[static_cast<std::size_t>(c)]);
    }
    double eapp[6];
    for (int c = 0; c < kSymComponents; ++c) eapp[c] = m_params.applied_strain[c];
    hip_detail::me_green_multiply(
        m_hat_rw, m_d_kx.data(), m_d_ky.data(), m_d_kz.data(), m_g_ro,
        static_cast<long long>(m_n_outbox),
        (m_zero_mode == static_cast<std::size_t>(-1))
            ? -1
            : static_cast<long long>(m_zero_mode),
        eapp, m_n_global);
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.backward(m_d_hat[static_cast<std::size_t>(c)],
                     m_d_strain[static_cast<std::size_t>(c)]);
    }
  }

  double eyre_milton_local() {
    hip_detail::me_eyre_milton_local(m_d_h.data(), m_d_amp.data(), m_tau_prev_ro,
                                     m_eps_rw, m_tau_rw, m_dp, m_d_block.data(),
                                     m_n_blocks);
    m_d_block.copy_to_host(m_block_host.data(), m_block_host.size());
    double diff = 0.0, scale = 0.0;
    for (int b = 0; b < m_n_blocks; ++b) {
      diff = std::max(diff, m_block_host[static_cast<std::size_t>(2 * b)]);
      scale =
          std::max(scale, m_block_host[static_cast<std::size_t>(2 * b + 1)]);
    }
    double local[2] = {diff, scale};
    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, m_params.comm);
    return (global[1] > 0.0) ? global[0] / global[1] : 0.0;
  }

  void finalise(bool want_dfel) {
    hip_detail::me_finalise(m_d_h.data(), m_d_amp.data(), m_d_dh.data(),
                           m_d_damp.data(), m_eps_ro, m_sig_rw, m_d_f_el.data(),
                           m_d_dfel.data(), m_dp, want_dfel ? 1 : 0);
  }

  void download_stress_energy() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_d_stress[static_cast<std::size_t>(c)].copy_to_host(
          m_stress_host[static_cast<std::size_t>(c)].data(), m_n_local);
      m_stress_host[static_cast<std::size_t>(c)].note_host_write();
    }
    m_d_f_el.copy_to_host(m_f_el_host.data(), m_n_local);
    m_f_el_host.note_host_write();
    m_d_dfel.copy_to_host(m_dfel_host.data(), m_n_local);
    m_dfel_host.note_host_write();
  }

  void build_and_upload_green(const pfc::Domain &domain, FFT &fft) {
    std::vector<double> kx(m_n_outbox, 0.0), ky(m_n_outbox, 0.0),
        kz(m_n_outbox, 0.0);
    std::array<std::vector<double>, 6> g;
    for (auto &v : g) v.assign(m_n_outbox, 0.0);
    m_zero_mode = static_cast<std::size_t>(-1);
    const double c12 = m_c0.c12, c44 = m_c0.c44;
    const double aniso = m_c0.c11 - m_c0.c12 - 2.0 * m_c0.c44;
    const auto gsz = pfc::domain::get_size(domain);
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t idx, double kx_raw, double ky_raw, double kz_raw, int i,
            int j, int k) {
          if (i == 0 && j == 0 && k == 0) {
            m_zero_mode = idx;
            return;
          }
          double kxv = pfc::fft::kspace::is_nyquist_index(i, gsz[0]) ? 0.0 : kx_raw;
          double kyv = pfc::fft::kspace::is_nyquist_index(j, gsz[1]) ? 0.0 : ky_raw;
          double kzv = pfc::fft::kspace::is_nyquist_index(k, gsz[2]) ? 0.0 : kz_raw;
          if (kxv == 0.0 && kyv == 0.0 && kzv == 0.0) {
            kxv = kx_raw;
            kyv = ky_raw;
            kzv = kz_raw;
          }
          kx[idx] = kxv;
          ky[idx] = kyv;
          kz[idx] = kzv;
          const double k2 = kxv * kxv + kyv * kyv + kzv * kzv;
          const double axx = (c12 + c44) * kxv * kxv + c44 * k2 + aniso * kxv * kxv;
          const double ayy = (c12 + c44) * kyv * kyv + c44 * k2 + aniso * kyv * kyv;
          const double azz = (c12 + c44) * kzv * kzv + c44 * k2 + aniso * kzv * kzv;
          const double ayz = (c12 + c44) * kyv * kzv;
          const double axz = (c12 + c44) * kxv * kzv;
          const double axy = (c12 + c44) * kxv * kyv;
          const double cof_xx = ayy * azz - ayz * ayz;
          const double cof_xy = ayz * axz - axy * azz;
          const double cof_xz = axy * ayz - ayy * axz;
          const double cof_yy = axx * azz - axz * axz;
          const double cof_yz = axy * axz - axx * ayz;
          const double cof_zz = axx * ayy - axy * axy;
          const double det = axx * cof_xx + axy * cof_xy + axz * cof_xz;
          const double inv = 1.0 / det;
          g[0][idx] = cof_xx * inv;
          g[1][idx] = cof_yy * inv;
          g[2][idx] = cof_zz * inv;
          g[3][idx] = cof_yz * inv;
          g[4][idx] = cof_xz * inv;
          g[5][idx] = cof_xy * inv;
        });
    if (m_n_outbox > 0) {
      m_d_kx.copy_from_host(kx);
      m_d_ky.copy_from_host(ky);
      m_d_kz.copy_from_host(kz);
      for (int c = 0; c < kSymComponents; ++c)
        m_d_g[static_cast<std::size_t>(c)].copy_from_host(
            g[static_cast<std::size_t>(c)]);
    }
  }

  FFT &m_fft;
  MicroelasticityParams m_params;
  Stiffness m_c0;
  SymRealFields m_strain_host{};
  SymRealFields m_stress_host{};
  RealField m_f_el_host{};
  RealField m_dfel_host{};
  std::array<RealBuf, 6> m_d_strain{}, m_d_tau{}, m_d_tau_prev{}, m_d_stress{},
      m_d_g{};
  std::array<CplxBuf, 6> m_d_hat{};
  RealBuf m_d_h, m_d_amp, m_d_dh, m_d_damp, m_d_f_el, m_d_dfel, m_d_kx, m_d_ky,
      m_d_kz, m_d_block;
  hip_detail::MEParams m_dp{};
  hip_detail::MESym6 m_eps_rw{}, m_tau_rw{}, m_hat_rw{}, m_sig_rw{};
  hip_detail::MESym6Const m_eps_ro{}, m_tau_prev_ro{}, m_g_ro{};
  std::vector<double> m_block_host;
  std::size_t m_n_local{0}, m_n_outbox{0},
      m_zero_mode{static_cast<std::size_t>(-1)};
  double m_n_global{1.0};
  double m_cell{1.0};
  int m_n_blocks{1};
  bool m_has{false};
};

} // namespace pfc::apps
