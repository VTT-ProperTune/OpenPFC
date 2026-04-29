// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/finite_difference.hpp>

namespace allen_cahn {

struct RunConfig {
  int nx_glob = 64;
  int ny_glob = 64;
  /** Default length chosen so a center seed visibly coarsens in 2D (raise M or
   * lower ε for faster interface motion; reduce dt if explicit Euler diverges). */
  int n_steps = 3000;
  double dt = 0.0015;
  double M = 2.0;
  double epsilon = 0.35;
  /** If non-empty, gather the final scalar field on rank 0 and write a grayscale
   * PNG. */
  std::string png_output;
  /** If non-empty (optional argv[8]), write the field right after IC, before time
   * stepping. */
  std::string png_output_initial;
  static constexpr int kHaloWidth = 1;
};

inline RunConfig parse_args(int argc, char **argv) {
  RunConfig c;
  if (argc > 1) {
    c.nx_glob = std::atoi(argv[1]);
  }
  if (argc > 2) {
    c.ny_glob = std::atoi(argv[2]);
  }
  if (argc > 3) {
    c.n_steps = std::atoi(argv[3]);
  }
  if (argc > 4) {
    c.dt = std::atof(argv[4]);
  }
  if (argc > 5) {
    c.M = std::atof(argv[5]);
  }
  if (argc > 6) {
    c.epsilon = std::atof(argv[6]);
  }
  if (argc > 7) {
    if (argc > 8) {
      c.png_output_initial = argv[7];
      c.png_output = argv[8];
    } else {
      c.png_output = argv[7];
    }
  }
  return c;
}

/**
 * @brief Phase field at t=0: one "grain" as a Gaussian bump on a φ≈-1 matrix.
 *
 * φ(g) = -1 + 2 exp( -r² / (2σ²) ), with r measured from the domain center in
 * index space. At the center φ→+1; far from the center φ→-1 (Allen–Cahn
 * equilibria). Diffusion and the double-well term then let the +1 region grow.
 */
inline void fill_initial_condition(std::vector<double> *u,
                                   const pfc::decomposition::Decomposition &decomp,
                                   int rank) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  auto gsz = pfc::world::get_size(gw);
  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  auto lo = pfc::world::get_lower(local);
  auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int sxy = nx * ny;
  const double cx = 0.5 * static_cast<double>(gsz[0] - 1);
  const double cy = 0.5 * static_cast<double>(gsz[1] - 1);
  const int gmin = std::min(gsz[0], gsz[1]);
  const double sigma =
      std::max(1.5, 0.06 * static_cast<double>(gmin)); // small seed, ~6% of span
  const double denom = 2.0 * sigma * sigma;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = lo[0] + ix;
        const int gy = lo[1] + iy;
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
        const double dxg = static_cast<double>(gx) - cx;
        const double dyg = static_cast<double>(gy) - cy;
        const double r2 = dxg * dxg + dyg * dyg;
        (*u)[idx] = -1.0 + 2.0 * std::exp(-r2 / denom);
      }
    }
  }
}

inline void
step_explicit_euler_cpu(std::vector<double> *u, std::vector<double> *lap,
                        std::array<std::vector<double>, 6> *face_halos,
                        pfc::SeparatedFaceHaloExchanger<double> *exchanger, int nx,
                        int ny, int nz, double inv_dx2, double inv_dy2, double dt,
                        double M, double inv_eps2) {
  constexpr int hw = RunConfig::kHaloWidth;
  exchanger->exchange_halos(u->data(), u->size(), *face_halos);
  std::fill(lap->begin(), lap->end(), 0.0);
  std::array<const double *, 6> face_ptrs{};
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<std::size_t>(i)] =
        (*face_halos)[static_cast<std::size_t>(i)].data();
  }
  pfc::field::fd::laplacian_5point_xy_interior_separated(
      u->data(), face_ptrs, lap->data(), nx, ny, nz, inv_dx2, inv_dy2, hw);
  for (std::size_t i = 0; i < u->size(); ++i) {
    const double p = (*u)[i];
    (*u)[i] += dt * (M * (*lap)[i] - inv_eps2 * (p * p * p - p));
  }
}

} // namespace allen_cahn
