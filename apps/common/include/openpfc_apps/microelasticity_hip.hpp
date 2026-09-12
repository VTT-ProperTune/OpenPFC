// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file microelasticity_hip.hpp
 * @brief Device FFT + Green-operator path for the existing eigenstrain solver.
 *
 * Same physics as `EigenstrainMicroelasticity`. Not a second elasticity
 * implementation: the Green operator, Eyre–Milton local reflection, and
 * polarisation are the host formulae executed on HIP with rocFFT HeFFTe.
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "microelasticity_hip.hpp requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

namespace pfc::apps {

void hip_green_kspace(void *txx, void *tyy, void *tzz, void *tyz, void *txz,
                      void *txy, const double *kx, const double *ky, const double *kz,
                      const double *gxx, const double *gyy, const double *gzz,
                      const double *gyz, const double *gxz, const double *gxy,
                      std::size_t n, std::size_t zero_mode, double n_global,
                      const double eapp[6]);
void hip_polarisation(const double *h, const double *amp, const double *eps[6],
                      double *tau[6], std::size_t n, const double cs[3],
                      const double cl[3], const double c0[3], const double pat[6]);
void hip_eyre_milton_local(const double *h, const double *amp, const double *zin[6],
                           double *eps[6], double *zout[6], std::size_t n,
                           const double cs[3], const double cl[3], const double c0[3],
                           const double pat[6]);
void hip_memcpy_d2d(void *dst, const void *src, std::size_t bytes);

/**
 * @brief HIP Eyre–Milton solver. Host `Field<double>` API; FFT/Green/local
 *        reflection stay on device for the inner loop.
 */
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
    const auto box = fft.get_inbox_bounds();
    for (int c = 0; c < kSymComponents; ++c) {
      m_strain_host[static_cast<std::size_t>(c)] =
          pfc::data::field_from_inbox<double>(domain, box);
      m_d_strain[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_tau[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_tau_prev[static_cast<std::size_t>(c)] = RealBuf(m_n_local);
      m_d_hat[static_cast<std::size_t>(c)] = CplxBuf(m_n_outbox);
      m_d_g[static_cast<std::size_t>(c)] = RealBuf(m_n_outbox);
    }
    m_d_h = RealBuf(m_n_local);
    m_d_amp = RealBuf(m_n_local);
    m_d_kx = RealBuf(m_n_outbox);
    m_d_ky = RealBuf(m_n_outbox);
    m_d_kz = RealBuf(m_n_outbox);
    build_and_upload_green(domain, fft);
    const auto gs = m_strain_host[0].global_size();
    m_n_global = static_cast<double>(gs[0]) * static_cast<double>(gs[1]) *
                 static_cast<double>(gs[2]);
  }

  [[nodiscard]] MicroelasticityParams &params() noexcept { return m_params; }
  [[nodiscard]] const MicroelasticityParams &params() const noexcept {
    return m_params;
  }
  [[nodiscard]] const SymRealFields &strain() const noexcept { return m_strain_host; }

  void reset() {
    for (int c = 0; c < kSymComponents; ++c) {
      std::vector<double> z(m_n_local, 0.0);
      m_d_strain[static_cast<std::size_t>(c)].copy_from_host(z);
    }
    m_has = false;
  }

  MicroelasticityReport solve(const RealField &h, const RealField &amp) {
    m_d_h.copy_from_host(h.data(), m_n_local);
    m_d_amp.copy_from_host(amp.data(), m_n_local);
    if (!m_params.warm_start || !m_has) {
      for (int c = 0; c < kSymComponents; ++c) {
        std::vector<double> fill(m_n_local, m_params.applied_strain[c]);
        m_d_strain[static_cast<std::size_t>(c)].copy_from_host(fill);
      }
    }
    const double cs[3] = {m_params.c_solid.c11, m_params.c_solid.c12,
                          m_params.c_solid.c44};
    const double cl[3] = {m_params.c_liquid.c11, m_params.c_liquid.c12,
                          m_params.c_liquid.c44};
    const double c0[3] = {m_c0.c11, m_c0.c12, m_c0.c44};
    const double pat[6] = {m_params.eigenstrain_pattern[0],
                           m_params.eigenstrain_pattern[1],
                           m_params.eigenstrain_pattern[2],
                           m_params.eigenstrain_pattern[3],
                           m_params.eigenstrain_pattern[4],
                           m_params.eigenstrain_pattern[5]};

    polarise(cs, cl, c0, pat);
    MicroelasticityReport report;
    for (int it = 1; it <= m_params.n_el_iter; ++it) {
      for (int c = 0; c < kSymComponents; ++c) {
        hip_memcpy_d2d(m_d_tau_prev[static_cast<std::size_t>(c)].data(),
                       m_d_tau[static_cast<std::size_t>(c)].data(),
                       m_n_local * sizeof(double));
      }
      apply_green();
      const double *zin[6], *eps_c[6];
      double *eps[6], *zout[6];
      for (int c = 0; c < kSymComponents; ++c) {
        zin[c] = m_d_tau_prev[static_cast<std::size_t>(c)].data();
        eps[c] = m_d_strain[static_cast<std::size_t>(c)].data();
        zout[c] = m_d_tau[static_cast<std::size_t>(c)].data();
        (void)eps_c;
      }
      hip_eyre_milton_local(m_d_h.data(), m_d_amp.data(), zin, eps, zout, m_n_local,
                            cs, cl, c0, pat);
      const double res = residual_host();
      report.iterations = it;
      report.residual = res;
      report.residual_history.push_back(res);
      if (res < m_params.tol_el) {
        report.converged = true;
        break;
      }
    }
    download_strain();
    m_has = true;
    return report;
  }

private:
  void polarise(const double cs[3], const double cl[3], const double c0[3],
                const double pat[6]) {
    const double *eps[6];
    double *tau[6];
    for (int c = 0; c < kSymComponents; ++c) {
      eps[c] = m_d_strain[static_cast<std::size_t>(c)].data();
      tau[c] = m_d_tau[static_cast<std::size_t>(c)].data();
    }
    hip_polarisation(m_d_h.data(), m_d_amp.data(), eps, tau, m_n_local, cs, cl, c0,
                     pat);
  }

  void apply_green() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.forward(m_d_tau[static_cast<std::size_t>(c)],
                    m_d_hat[static_cast<std::size_t>(c)]);
    }
    double eapp[6];
    for (int c = 0; c < kSymComponents; ++c) eapp[c] = m_params.applied_strain[c];
    hip_green_kspace(m_d_hat[0].data(), m_d_hat[1].data(), m_d_hat[2].data(),
                     m_d_hat[3].data(), m_d_hat[4].data(), m_d_hat[5].data(),
                     m_d_kx.data(), m_d_ky.data(), m_d_kz.data(), m_d_g[0].data(),
                     m_d_g[1].data(), m_d_g[2].data(), m_d_g[3].data(),
                     m_d_g[4].data(), m_d_g[5].data(), m_n_outbox, m_zero_mode,
                     m_n_global, eapp);
    for (int c = 0; c < kSymComponents; ++c) {
      m_fft.backward(m_d_hat[static_cast<std::size_t>(c)],
                     m_d_strain[static_cast<std::size_t>(c)]);
    }
  }

  double residual_host() {
    double diff = 0.0, scale = 0.0;
    for (int c = 0; c < kSymComponents; ++c) {
      auto a = m_d_tau[static_cast<std::size_t>(c)].to_host();
      auto b = m_d_tau_prev[static_cast<std::size_t>(c)].to_host();
      for (std::size_t i = 0; i < m_n_local; ++i) {
        diff = std::max(diff, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(a[i]));
      }
    }
    double local[2] = {diff, scale};
    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, m_params.comm);
    return (global[1] > 0.0) ? global[0] / global[1] : 0.0;
  }

  void download_strain() {
    for (int c = 0; c < kSymComponents; ++c) {
      m_d_strain[static_cast<std::size_t>(c)].copy_to_host(
          m_strain_host[static_cast<std::size_t>(c)].data(), m_n_local);
      m_strain_host[static_cast<std::size_t>(c)].note_host_write();
    }
  }

  void build_and_upload_green(const pfc::Domain &domain, FFT &fft) {
    std::vector<double> kx(m_n_outbox, 0.0), ky(m_n_outbox, 0.0), kz(m_n_outbox, 0.0);
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
    m_d_kx.copy_from_host(kx);
    m_d_ky.copy_from_host(ky);
    m_d_kz.copy_from_host(kz);
    for (int c = 0; c < kSymComponents; ++c)
      m_d_g[static_cast<std::size_t>(c)].copy_from_host(g[static_cast<std::size_t>(c)]);
  }

  FFT &m_fft;
  MicroelasticityParams m_params;
  Stiffness m_c0;
  SymRealFields m_strain_host{};
  std::array<RealBuf, 6> m_d_strain{}, m_d_tau{}, m_d_tau_prev{};
  std::array<CplxBuf, 6> m_d_hat{};
  std::array<RealBuf, 6> m_d_g{};
  RealBuf m_d_h, m_d_amp, m_d_kx, m_d_ky, m_d_kz;
  std::size_t m_n_local{0}, m_n_outbox{0}, m_zero_mode{static_cast<std::size_t>(-1)};
  double m_n_global{1.0};
  bool m_has{false};
};

} // namespace pfc::apps
