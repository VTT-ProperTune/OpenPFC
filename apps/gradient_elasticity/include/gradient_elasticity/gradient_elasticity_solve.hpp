// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gradient_elasticity_solve.hpp
 * @brief One-shot spectral 2×2 Helmholtz–Navier invert.
 *
 * FFT of the inclusion \(g\), invert hats on the host (even for HIP),
 * inverse FFT of \(u_x,u_y\). `SpectralDiagonalSolver` is scalar-only,
 * so the \(2\times 2\) split stays in-app.
 */

#include <complex>
#include <cstddef>
#include <type_traits>

#include <gradient_elasticity/gradient_elasticity_physics.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

namespace gradient_elasticity {

template <class Physics, class ComplexField>
void invert_hats(const Physics &phys, const pfc::fft::IFFTQueries &fft,
                 const pfc::Domain &domain, ComplexField &g_hat,
                 ComplexField &ux_hat, ComplexField &uy_hat) {
  const auto outbox = fft.get_outbox_bounds();
  g_hat.with_host_view([&](auto *gh, std::size_t) {
    ux_hat.with_host_view([&](auto *uxh, std::size_t) {
      uy_hat.with_host_view([&](auto *uyh, std::size_t) {
        pfc::fft::kspace::for_each_kpoint(
            outbox, domain,
            [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
              const auto f = phys.eigenstrain_force(kx, ky, kz, gh[idx]);
              const auto u = phys.invert(kx, ky, kz, f);
              uxh[idx] = u.ux;
              uyh[idx] = u.uy;
            });
      });
    });
  });
}

template <class MemorySpace, class FFT, class Physics>
void solve_displacement(FFT &fft, const Physics &phys,
                        pfc::data::Field<double, MemorySpace> &g,
                        pfc::data::Field<double, MemorySpace> &ux,
                        pfc::data::Field<double, MemorySpace> &uy) {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;
  using Complex = std::complex<double>;
  using ComplexField = pfc::data::Field<Complex, MemorySpace>;
  ComplexField g_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField ux_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField uy_hat(g.domain(), fft.get_outbox_bounds(), 0);
  Ops::forward(fft, g, g_hat);
  invert_hats(phys, fft, g.domain(), g_hat, ux_hat, uy_hat);
  if constexpr (!std::is_same_v<MemorySpace, pfc::HostSpace>) {
    ux_hat.sync_to_device();
    uy_hat.sync_to_device();
  }
  Ops::backward(fft, ux_hat, ux);
  Ops::backward(fft, uy_hat, uy);
}

/**
 * @brief Same one-shot invert as `invert_hats`, but also emits the spectral
 * strain of the resulting displacement (`#117` derived diagnostics).
 *
 * Computed in the same \(\mathbf{k}\)-loop as the displacement invert (no
 * extra forward FFT of \(u_x,u_y\) is needed): \(\hat\varepsilon\) follows
 * directly from \(\hat{\mathbf{u}}(\mathbf{k})\) and
 * `Physics::strain_from_displacement`.
 */
template <class Physics, class ComplexField>
void invert_hats_with_strain(const Physics &phys, const pfc::fft::IFFTQueries &fft,
                             const pfc::Domain &domain, ComplexField &g_hat,
                             ComplexField &ux_hat, ComplexField &uy_hat,
                             ComplexField &exx_hat, ComplexField &eyy_hat,
                             ComplexField &exy_hat) {
  const auto outbox = fft.get_outbox_bounds();
  g_hat.with_host_view([&](auto *gh, std::size_t) {
    ux_hat.with_host_view([&](auto *uxh, std::size_t) {
      uy_hat.with_host_view([&](auto *uyh, std::size_t) {
        exx_hat.with_host_view([&](auto *exxh, std::size_t) {
          eyy_hat.with_host_view([&](auto *eyyh, std::size_t) {
            exy_hat.with_host_view([&](auto *exyh, std::size_t) {
              pfc::fft::kspace::for_each_kpoint(
                  outbox, domain,
                  [&](std::size_t idx, double kx, double ky, double kz, int, int,
                      int) {
                    const auto f = phys.eigenstrain_force(kx, ky, kz, gh[idx]);
                    const auto u = phys.invert(kx, ky, kz, f);
                    const auto e = phys.strain_from_displacement(kx, ky, u);
                    uxh[idx] = u.ux;
                    uyh[idx] = u.uy;
                    exxh[idx] = e.exx;
                    eyyh[idx] = e.eyy;
                    exyh[idx] = e.exy;
                  });
            });
          });
        });
      });
    });
  });
}

/**
 * @brief Solve for displacement and its compatible strain in one pass:
 * forward FFT of \(g\), a single \(\mathbf{k}\)-loop invert (displacement +
 * strain hats), five inverse FFTs.
 */
template <class MemorySpace, class FFT, class Physics>
void solve_displacement_and_strain(FFT &fft, const Physics &phys,
                                   pfc::data::Field<double, MemorySpace> &g,
                                   pfc::data::Field<double, MemorySpace> &ux,
                                   pfc::data::Field<double, MemorySpace> &uy,
                                   pfc::data::Field<double, MemorySpace> &exx,
                                   pfc::data::Field<double, MemorySpace> &eyy,
                                   pfc::data::Field<double, MemorySpace> &exy) {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;
  using Complex = std::complex<double>;
  using ComplexField = pfc::data::Field<Complex, MemorySpace>;
  ComplexField g_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField ux_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField uy_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField exx_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField eyy_hat(g.domain(), fft.get_outbox_bounds(), 0);
  ComplexField exy_hat(g.domain(), fft.get_outbox_bounds(), 0);
  Ops::forward(fft, g, g_hat);
  invert_hats_with_strain(phys, fft, g.domain(), g_hat, ux_hat, uy_hat, exx_hat,
                          eyy_hat, exy_hat);
  if constexpr (!std::is_same_v<MemorySpace, pfc::HostSpace>) {
    ux_hat.sync_to_device();
    uy_hat.sync_to_device();
    exx_hat.sync_to_device();
    eyy_hat.sync_to_device();
    exy_hat.sync_to_device();
  }
  Ops::backward(fft, ux_hat, ux);
  Ops::backward(fft, uy_hat, uy);
  Ops::backward(fft, exx_hat, exx);
  Ops::backward(fft, eyy_hat, eyy);
  Ops::backward(fft, exy_hat, exy);
}

} // namespace gradient_elasticity
