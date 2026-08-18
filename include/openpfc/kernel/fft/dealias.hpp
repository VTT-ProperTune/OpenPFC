// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file dealias.hpp
 * @brief Optional 2/3-rule dealiasing mask for cubic nonlinearities (M5).
 *
 * @details
 * Orszag's 2/3 rule zeros Fourier modes with `|k_i| >= (2/3) k_{Nyquist,i}`
 * on any axis (`k_Nyquist,i = π / Δx_i`). Off by default — callers fill a
 * diagonal mask and multiply the nonlinear spectrum when they want it.
 *
 * A cubic term `u³` aliases the highest third of the spectrum onto
 * retained modes if those modes are left live. See
 * `docs/science/numerics_limits.md`.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/fft/box3i.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace pfc::fft::kspace {

/// Per-axis cutoff: keep `|k_i| < (2/3) π / spacing[i]`.
[[nodiscard]] inline bool two_thirds_keep(double kx, double ky, double kz,
                                          std::array<double, 3> spacing) noexcept {
  const double cx = (2.0 / 3.0) * (pfc::pi / spacing[0]);
  const double cy = (2.0 / 3.0) * (pfc::pi / spacing[1]);
  const double cz = (2.0 / 3.0) * (pfc::pi / spacing[2]);
  return std::abs(kx) < cx && std::abs(ky) < cy && std::abs(kz) < cz;
}

/**
 * @brief Fill a 0/1 mask over a local outbox (`1` = keep).
 *
 * @throws std::invalid_argument if `n` does not match the outbox volume.
 */
inline void fill_two_thirds_mask(const Box3i &outbox,
                                 std::array<int, 3> global_size,
                                 std::array<double, 3> spacing, double *mask,
                                 std::size_t n) {
  const std::size_t expected =
      static_cast<std::size_t>(outbox.size[0]) *
      static_cast<std::size_t>(outbox.size[1]) *
      static_cast<std::size_t>(outbox.size[2]);
  if (mask == nullptr || n != expected) {
    throw std::invalid_argument(
        "fill_two_thirds_mask: mask length must equal outbox volume");
  }
  for_each_kpoint(outbox, global_size, spacing,
                  [&](std::size_t idx, double kx, double ky, double kz, int, int,
                      int) {
                    mask[idx] = two_thirds_keep(kx, ky, kz, spacing) ? 1.0 : 0.0;
                  });
}

} // namespace pfc::fft::kspace
