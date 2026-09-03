// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <vector>

/**
 * @file pad_field.hpp
 * @brief Regular-grid padded brick for the OpenMP Al-Cu FTA engine.
 *
 * Interior is 1-based: `(i,j,k) ∈ [1,Nx]×[1,Ny]×[1,Nz]`. Ghosts live at 0 and
 * `N+1`. `Nz = 1` is the 2D path (single plane at `k = 1`).
 */
namespace alloy_pf_directional {

struct PadField {
  int Nx = 0;
  int Ny = 0;
  int Nz = 1;
  int sx = 0;
  int sxy = 0;
  std::vector<double> a;

  PadField() = default;
  PadField(int nx, int ny, int nz = 1)
      : Nx(nx), Ny(ny), Nz(nz < 1 ? 1 : nz), sx(nx + 2),
        sxy(sx * (ny + 2)),
        a(static_cast<std::size_t>(sxy) * static_cast<std::size_t>(Nz + 2), 0.0) {}

  double &operator()(int i, int j, int k = 1) noexcept {
    return a[static_cast<std::size_t>(i + j * sx + k * sxy)];
  }
  double operator()(int i, int j, int k = 1) const noexcept {
    return a[static_cast<std::size_t>(i + j * sx + k * sxy)];
  }
};

/** 3-index view so Ji operators can take any PadField. */
struct FieldAs3 {
  const PadField &f;
  double operator()(int i, int j, int k) const noexcept { return f(i, j, k); }
};

inline double x_of(int i, double dx, int shift_cells = 0) noexcept {
  return (static_cast<double>(i) - 0.5 + static_cast<double>(shift_cells)) * dx;
}
inline double y_of(int j, double dx) noexcept {
  return (static_cast<double>(j) - 0.5) * dx;
}
inline double z_of(int k, double dx) noexcept {
  return (static_cast<double>(k) - 0.5) * dx;
}

/**
 * Ghosts: no-flux in x. y/z are periodic or no-flux.
 * Sequential x → y → z so edges and corners pick up the widened values
 * (same idea as Full-connectivity `pfc::comm::HaloExchange`).
 *
 * `Nz == 1` is the 2D path: Ji \(\bar S_{2,1}\) and 2D `div(α∇β)` never read
 * `k±1`, so the k = 0 and k = 2 planes are not filled. Those copies are a
 * full \(N_x\times N_y\) traffic per field and dominated OpenMP wall time
 * on large LUMI-C DS bricks. 3D (`Nz > 1`) still fills z ghosts.
 */
inline void fill_ghosts(PadField &f, bool periodic_y, bool periodic_z) {
  const int Nx = f.Nx;
  const int Ny = f.Ny;
  const int Nz = f.Nz;
  // x/y rings are O(N) per plane — serial is cheaper than an OpenMP fork.
  for (int k = 1; k <= Nz; ++k) {
    for (int j = 1; j <= Ny; ++j) {
      f(0, j, k) = f(1, j, k);
      f(Nx + 1, j, k) = f(Nx, j, k);
    }
  }
  if (periodic_y) {
    for (int k = 1; k <= Nz; ++k) {
      for (int i = 0; i <= Nx + 1; ++i) {
        f(i, 0, k) = f(i, Ny, k);
        f(i, Ny + 1, k) = f(i, 1, k);
      }
    }
  } else {
    for (int k = 1; k <= Nz; ++k) {
      for (int i = 0; i <= Nx + 1; ++i) {
        f(i, 0, k) = f(i, 1, k);
        f(i, Ny + 1, k) = f(i, Ny, k);
      }
    }
  }
  if (Nz <= 1) {
    return;
  }
  if (periodic_z) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j <= Ny + 1; ++j) {
      for (int i = 0; i <= Nx + 1; ++i) {
        f(i, j, 0) = f(i, j, Nz);
        f(i, j, Nz + 1) = f(i, j, 1);
      }
    }
  } else {
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j <= Ny + 1; ++j) {
      for (int i = 0; i <= Nx + 1; ++i) {
        f(i, j, 0) = f(i, j, 1);
        f(i, j, Nz + 1) = f(i, j, Nz);
      }
    }
  }
}

inline void shift_left(PadField &f, int nshift, double fill) {
  if (nshift <= 0) {
    return;
  }
  const int Nx = f.Nx;
  const int Ny = f.Ny;
  const int Nz = f.Nz;
  const int n = nshift < Nx ? nshift : Nx;
  for (int k = 1; k <= Nz; ++k) {
    for (int j = 1; j <= Ny; ++j) {
      for (int i = 1; i <= Nx - n; ++i) {
        f(i, j, k) = f(i + n, j, k);
      }
      for (int i = Nx - n + 1; i <= Nx; ++i) {
        f(i, j, k) = fill;
      }
    }
  }
}

} // namespace alloy_pf_directional
