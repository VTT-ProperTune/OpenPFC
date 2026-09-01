// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <alloy_pf_directional/cli.hpp>
#include <alloy_pf_directional/defaults.hpp>
#include <alloy_pf_directional/recompute.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>

#include <cmath>
#include <mpi.h>
#include <vector>

namespace alloy_pf_directional::mpi_util {

using Field = pfc::data::Field<double, pfc::HostSpace>;

inline bool owns_lo(const Field &f, int axis) noexcept { return f.box().low[axis] == 0; }
inline bool owns_hi(const Field &f, int axis, int nglob) noexcept {
  return f.box().high[axis] == nglob - 1;
}

/** After a periodic halo, overwrite no-flux faces this rank owns.
 *  `dim3 == false` (Nz=1 2D): skip z-face copies (full \(N_x\times N_y\)).
 *  x/y faces still include in-plane corner ghosts for Ji \(\bar S_{2,1}\). */
inline void apply_noflux(Field &f, bool noflux_x, bool noflux_y, bool noflux_z, int Nx,
                         int Ny, int Nz, bool dim3 = true) {
  const int nx = f.local_size()[0];
  const int ny = f.local_size()[1];
  const int nz = f.local_size()[2];
  const int k0 = dim3 ? -1 : 0;
  const int k1 = dim3 ? nz : nz - 1;
  if (noflux_x && owns_lo(f, 0)) {
    for (int k = k0; k <= k1; ++k) {
      for (int j = -1; j <= ny; ++j) {
        f(-1, j, k) = f(0, j, k);
      }
    }
  }
  if (noflux_x && owns_hi(f, 0, Nx)) {
    for (int k = k0; k <= k1; ++k) {
      for (int j = -1; j <= ny; ++j) {
        f(nx, j, k) = f(nx - 1, j, k);
      }
    }
  }
  if (noflux_y && owns_lo(f, 1)) {
    for (int k = k0; k <= k1; ++k) {
      for (int i = -1; i <= nx; ++i) {
        f(i, -1, k) = f(i, 0, k);
      }
    }
  }
  if (noflux_y && owns_hi(f, 1, Ny)) {
    for (int k = k0; k <= k1; ++k) {
      for (int i = -1; i <= nx; ++i) {
        f(i, ny, k) = f(i, ny - 1, k);
      }
    }
  }
  if (dim3 && noflux_z && owns_lo(f, 2)) {
    for (int j = -1; j <= ny; ++j) {
      for (int i = -1; i <= nx; ++i) {
        f(i, j, -1) = f(i, j, 0);
      }
    }
  }
  if (dim3 && noflux_z && owns_hi(f, 2, Nz)) {
    for (int j = -1; j <= ny; ++j) {
      for (int i = -1; i <= nx; ++i) {
        f(i, j, nz) = f(i, j, nz - 1);
      }
    }
  }
}

inline void init_fields(Field &phi1, Field &phi2, Field &psi1, Field &psi2, Field &c,
                        const RunConfig &cfg, bool use_glasner, bool dim3) {
  const Physics &phys = cfg.phys;
  const double dx = phys.dx;
  const int n_grains = cfg.n_grains;
  const int Ny = cfg.Ny;
  const int Nz = cfg.Nz < 1 ? 1 : cfg.Nz;
  const double Ly = (static_cast<double>(Ny) - 0.5) * dx;
  const double Lz = dim3 ? (static_cast<double>(Nz) - 0.5) * dx : 0.0;
  double y1 = 0.0;
  double y2 = 0.0;
  two_grain_seed_ys(Ny, dx, y1, y2);
  const double ymid = 0.5 * (Ly + 0.5 * dx);
  const double zmid = dim3 ? 0.5 * (Lz + 0.5 * dx) : 0.0;
  const double Rseed = phys.r_seed;
  const double k = phys.ke;
  const double clo = phys.clo;

  phi1.for_each_owned([&](int i, int j, int kcell) {
    const auto g = phi1.global(i, j, kcell);
    const double x = (static_cast<double>(g[0]) + 0.5) * dx;
    const double y = (static_cast<double>(g[1]) + 0.5) * dx;
    const double z = dim3 ? (static_cast<double>(g[2]) + 0.5) * dx : 0.0;
    if (n_grains == 1) {
      const double dy = y - ymid;
      const double dz = dim3 ? (z - zmid) : 0.0;
      const double xint =
          cfg.seed_depth +
          cfg.seed_bump * std::exp(-0.5 * (dy * dy + dz * dz) /
                                   (cfg.seed_bump_sigma * cfg.seed_bump_sigma));
      const double s = -(x - xint) / phys.W0;
      if (use_glasner) {
        psi1(i, j, kcell) = std::max(-8.0, std::min(8.0, s));
        psi2(i, j, kcell) = -8.0;
        phi1(i, j, kcell) = phi_from_psi(psi1(i, j, kcell));
        phi2(i, j, kcell) = -1.0;
      } else {
        phi1(i, j, kcell) = -std::tanh((x - xint) / (std::sqrt(2.0) * phys.W0));
        phi2(i, j, kcell) = -1.0;
      }
    } else {
      double s1 = 0.0;
      double s2 = 0.0;
      two_grain_seed_s(x, y, z, y1, y2, zmid, Rseed, phys.W0, dim3, s1, s2);
      if (use_glasner) {
        apply_two_grain_seed(s1, s2, true, phi1(i, j, kcell), phi2(i, j, kcell),
                             &psi1(i, j, kcell), &psi2(i, j, kcell));
      } else {
        apply_two_grain_seed(s1, s2, false, phi1(i, j, kcell), phi2(i, j, kcell), nullptr,
                             nullptr);
      }
    }
    const double ph =
        (n_grains == 1) ? phi1(i, j, kcell) : phi_eff_two(phi1(i, j, kcell), phi2(i, j, kcell));
    c(i, j, kcell) = c_eq(ph, k, clo);
  });
}

/** Shift owned data one cell toward −x; ingest `fill` on the global +x face. */
inline void shift_left_one(Field &f, double fill, const pfc::decomposition::Decomposition &decomp,
                           int rank, int Nx) {
  const int nx = f.local_size()[0];
  const int ny = f.local_size()[1];
  const int nz = f.local_size()[2];
  const int right =
      pfc::decomposition::get_neighbor_rank(decomp, rank, pfc::types::Int3{1, 0, 0});
  const int left =
      pfc::decomposition::get_neighbor_rank(decomp, rank, pfc::types::Int3{-1, 0, 0});
  const bool last = owns_hi(f, 0, Nx);
  const std::size_t plane = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  std::vector<double> send(plane, 0.0), recv(plane, fill);
  std::size_t p = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      send[p++] = f(0, j, k);
    }
  }
  if (left >= 0 && left != rank) {
    MPI_Sendrecv(send.data(), static_cast<int>(plane), MPI_DOUBLE, left, 710,
                 recv.data(), static_cast<int>(plane), MPI_DOUBLE,
                 (right >= 0 && right != rank && !last) ? right : MPI_PROC_NULL, 710,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (!last && right >= 0 && right != rank) {
    MPI_Recv(recv.data(), static_cast<int>(plane), MPI_DOUBLE, right, 710, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        f(i, j, k) = f(i + 1, j, k);
      }
      f(nx - 1, j, k) = last ? fill : recv[static_cast<std::size_t>(j + k * ny)];
    }
  }
  (void)right;
}

} // namespace alloy_pf_directional::mpi_util
