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

} // namespace gradient_elasticity
