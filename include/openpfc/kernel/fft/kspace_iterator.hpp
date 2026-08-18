// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kspace_iterator.hpp
 * @brief Visit every local Fourier-space point with its wave vector (M5).
 *
 * @details
 * `for_each_kpoint` walks an FFT outbox in the same x-fastest order as
 * HeFFTe r2c buffers and calls `fn(idx, kx, ky, kz)`. Wavenumbers use
 * `kspace::k_component` (Nyquist folding). A device kernel iterator is a
 * later increment; this host helper is what `SpectralGradient` uses to
 * build operator tables.
 */

#include <array>
#include <cstddef>
#include <utility>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/kspace.hpp>

namespace pfc::fft::kspace {

/**
 * @brief Visit every local outbox mode.
 *
 * @param outbox      Inclusive local Fourier-space index box.
 * @param global_size Global real-space grid `{Nx, Ny, Nz}`.
 * @param spacing     Grid spacing `{dx, dy, dz}`.
 * @param fn          Callable `void(std::size_t idx, double kx, double ky, double kz)`.
 */
template <typename Fn>
void for_each_kpoint(const Box3i &outbox, std::array<int, 3> global_size,
                     std::array<double, 3> spacing, Fn &&fn) {
  const double fx = two_pi / (spacing[0] * static_cast<double>(global_size[0]));
  const double fy = two_pi / (spacing[1] * static_cast<double>(global_size[1]));
  const double fz = two_pi / (spacing[2] * static_cast<double>(global_size[2]));
  std::size_t idx = 0;
  for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
    const double kz = k_component(k, global_size[2], fz);
    for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
      const double ky = k_component(j, global_size[1], fy);
      for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
        const double kx = k_component(i, global_size[0], fx);
        fn(idx, kx, ky, kz);
        ++idx;
      }
    }
  }
}

/// Domain overload: take size and spacing from `domain`.
template <typename Fn>
void for_each_kpoint(const Box3i &outbox, const Domain &domain, Fn &&fn) {
  for_each_kpoint(outbox, domain::get_size(domain), domain::get_spacing(domain),
                  std::forward<Fn>(fn));
}

} // namespace pfc::fft::kspace
