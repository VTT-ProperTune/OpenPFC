// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <alloy_pf_directional/defaults.hpp>
#include <alloy_pf_directional/pad_field.hpp>

/**
 * @file window.hpp
 * @brief Moving window (slab around the DS front) and optional 16³/32³ block skip.
 *
 * Still a regular grid. Classic octree AMR is intentionally not implemented;
 * revisit only if the window + skip cannot fit the target brick (see README).
 */
namespace alloy_pf_directional {

struct WindowState {
  int shift_cells = 0;
  int lab_Nx = 0;
  double lab_Lx = 0.0;
};

struct BlockSkip {
  int bs = 0;
  int nbx = 0;
  int nby = 0;
  int nbz = 0;
  double tol_phi = 1.0e-4;
  double tol_c = 1.0e-4;
  int refresh = 10;
  std::vector<char> active;

  bool enabled() const noexcept { return bs > 0; }

  bool is_active(int i, int j, int k) const noexcept {
    if (!enabled() || active.empty()) {
      return true;
    }
    const int bi = (i - 1) / bs;
    const int bj = (j - 1) / bs;
    const int bk = (k - 1) / bs;
    const int idx = bi + bj * nbx + bk * nbx * nby;
    return active[static_cast<std::size_t>(idx)] != 0;
  }
};

inline BlockSkip make_block_skip(int Nx, int Ny, int Nz, int bs, double tol_phi,
                                 double tol_c, int refresh) {
  BlockSkip s;
  if (bs != 16 && bs != 32) {
    return s;
  }
  s.bs = bs;
  s.nbx = (Nx + bs - 1) / bs;
  s.nby = (Ny + bs - 1) / bs;
  s.nbz = (Nz + bs - 1) / bs;
  s.tol_phi = tol_phi;
  s.tol_c = tol_c;
  s.refresh = std::max(1, refresh);
  s.active.assign(static_cast<std::size_t>(s.nbx * s.nby * s.nbz), 1);
  return s;
}

inline void refresh_block_skip(BlockSkip &s, const PadField &phi1, const PadField &phi2,
                               const PadField &c, const Physics &phys, int n_grains) {
  if (!s.enabled()) {
    return;
  }
  std::fill(s.active.begin(), s.active.end(), 0);
  const double k = phys.ke;
  const double clo = phys.clo;
  for (int kcell = 1; kcell <= phi1.Nz; ++kcell) {
    for (int j = 1; j <= phi1.Ny; ++j) {
      for (int i = 1; i <= phi1.Nx; ++i) {
        const double ph =
            (n_grains == 1) ? phi1(i, j, kcell) : phi_eff_two(phi1(i, j, kcell), phi2(i, j, kcell));
        const double idle_phi = std::abs(ph * ph - 1.0);
        const double idle_c = std::abs(c(i, j, kcell) - c_eq(ph, k, clo));
        if (idle_phi > s.tol_phi || idle_c > s.tol_c) {
          const int bi = (i - 1) / s.bs;
          const int bj = (j - 1) / s.bs;
          const int bk = (kcell - 1) / s.bs;
          s.active[static_cast<std::size_t>(bi + bj * s.nbx + bk * s.nbx * s.nby)] = 1;
        }
      }
    }
  }
}

/** How many cells to drop on the left so the tip stays ~margin_left from the slab edge. */
inline int window_shift_count(double x_tip, int shift_cells, double dx, double margin_left,
                              int Nx, double margin_right) noexcept {
  if (!(dx > 0.0) || Nx < 8) {
    return 0;
  }
  const double left = static_cast<double>(shift_cells) * dx;
  const double excess = x_tip - left - margin_left;
  if (excess < dx) {
    return 0;
  }
  int n = static_cast<int>(std::floor(excess / dx));
  const int keep = std::max(8, static_cast<int>(std::ceil(margin_right / dx)));
  n = std::min(n, std::max(0, Nx - keep));
  return n;
}

inline void apply_window_shift(PadField &phi1, PadField &phi2, PadField &c, PadField *psi1,
                               PadField *psi2, const Physics &phys, int n_grains,
                               bool use_glasner, int nshift) {
  if (nshift <= 0) {
    return;
  }
  const double c_liq = c_eq(-1.0, phys.ke, phys.clo);
  shift_left(phi1, nshift, -1.0);
  shift_left(c, nshift, c_liq);
  if (n_grains == 2) {
    shift_left(phi2, nshift, -1.0);
  }
  if (use_glasner && psi1 != nullptr) {
    shift_left(*psi1, nshift, -8.0);
    if (n_grains == 2 && psi2 != nullptr) {
      shift_left(*psi2, nshift, -8.0);
    }
  }
}

} // namespace alloy_pf_directional
