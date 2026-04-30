// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
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
  /** Defaults are tuned so the superlevel area grows enough for the built-in check.
   */
  int n_steps = 5000;
  double dt = 0.00009;
  double M = 8.0;
  double epsilon = 0.19;
  /** Positive bulk driving term that favors the φ≈+1 seed over the φ≈-1 matrix. */
  double driving_force = 10.0;
  /** If non-empty, gather the final scalar field on rank 0 and write a grayscale
   * PNG. */
  std::string png_output;
  /** If non-empty, write the field right after IC, before time stepping. */
  std::string png_output_initial;
  static constexpr int kHaloWidth = 1;
  /** Superlevel for the seed-area metric: φ > 0 matches the visible seed in PNGs. */
  static constexpr double kLevelSetThreshold = 0.0;
  /** End-of-run: global cell count above threshold must grow by this factor. */
  static constexpr double kMinLevelSetAreaGrowthFactor = 5.0;
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
    char *end = nullptr;
    const double parsed = std::strtod(argv[7], &end);
    int png_arg = 7;
    if (end != argv[7] && *end == '\0') {
      c.driving_force = parsed;
      png_arg = 8;
    }
    if (argc > png_arg) {
      if (argc > png_arg + 1) {
        c.png_output_initial = argv[png_arg];
        c.png_output = argv[png_arg + 1];
      } else {
        c.png_output = argv[png_arg];
      }
    }
  }
  return c;
}

/**
 * @brief Phase field at t=0: one "grain" as a Gaussian bump on a φ≈-1 matrix.
 *
 * φ(g) = -1 + 2 exp( -r² / (2σ²) ), with r measured from the domain center in
 * index space. At the center φ→+1; far from the center φ→-1.
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
  const double sigma = std::max(2.0, 0.055 * static_cast<double>(gmin));
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

/** Local count of cells with φ strictly above @p threshold. */
inline std::int64_t count_cells_above(const std::vector<double> &u,
                                      double threshold) {
  std::int64_t n = 0;
  for (double v : u) {
    if (v > threshold) {
      ++n;
    }
  }
  return n;
}

/**
 * @brief Global superlevel area (cell count) at t=0 vs end; require N1 ≥ factor×N0.
 *        Interprets φ>threshold as the visible +1-phase seed.
 */
inline bool verify_level_set_area_growth(MPI_Comm comm, int rank,
                                         std::int64_t n_local_initial,
                                         std::int64_t n_local_final,
                                         double min_factor, double threshold_used) {
  std::int64_t n0 = 0;
  std::int64_t n1 = 0;
  MPI_Reduce(&n_local_initial, &n0, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&n_local_final, &n1, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  int ok = 1;
  if (rank == 0) {
    std::cout << "Superlevel area (cells with phi > " << threshold_used
              << "): N0=" << n0 << ", N1=" << n1;
    if (n0 > 0) {
      const double ratio = static_cast<double>(n1) / static_cast<double>(n0);
      std::cout << ", N1/N0=" << ratio << "\n";
      if (static_cast<double>(n1) < min_factor * static_cast<double>(n0)) {
        std::cout << "Growth check (require N1 >= " << min_factor
                  << " * N0): FAIL\n";
        ok = 0;
      } else {
        std::cout << "Growth check (require N1 >= " << min_factor
                  << " * N0): PASS\n";
      }
    } else {
      std::cout << "\nGrowth check: FAIL (N0 == 0)\n";
      ok = 0;
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
  return ok != 0;
}

inline void report_step_timing(MPI_Comm comm, int rank, int n_steps,
                               double elapsed_local_s) {
  double elapsed_min_s = 0.0;
  double elapsed_max_s = 0.0;
  double elapsed_sum_s = 0.0;
  MPI_Reduce(&elapsed_local_s, &elapsed_min_s, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(&elapsed_local_s, &elapsed_max_s, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&elapsed_local_s, &elapsed_sum_s, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  int nproc = 1;
  MPI_Comm_size(comm, &nproc);
  if (rank == 0) {
    const double avg_elapsed_s = elapsed_sum_s / static_cast<double>(nproc);
    const double avg_step_time_s = elapsed_max_s / static_cast<double>(n_steps);
    std::cout << "Step timing: elapsed_min_s=" << elapsed_min_s
              << ", elapsed_max_s=" << elapsed_max_s
              << ", elapsed_avg_s=" << avg_elapsed_s
              << ", avg_step_time_s=" << avg_step_time_s << "\n";
  }
}

inline void
step_explicit_euler_cpu(std::vector<double> *u, std::vector<double> *lap,
                        std::array<std::vector<double>, 6> *face_halos,
                        pfc::SeparatedFaceHaloExchanger<double> *exchanger, int nx,
                        int ny, int nz, double inv_dx2, double inv_dy2, double dt,
                        double M, double inv_eps2, double driving_force) {
  constexpr int hw = RunConfig::kHaloWidth;
  exchanger->exchange_halos(u->data(), u->size(), *face_halos);
  std::fill(lap->begin(), lap->end(), 0.0);
  std::array<const double *, 6> face_ptrs{};
  for (int i = 0; i < 6; ++i) {
    face_ptrs[static_cast<std::size_t>(i)] =
        (*face_halos)[static_cast<std::size_t>(i)].data();
  }
  pfc::field::fd::laplacian2d_xy_periodic_separated<2>(
      u->data(), face_ptrs, lap->data(), nx, ny, nz, inv_dx2, inv_dy2, hw);
  for (std::size_t i = 0; i < u->size(); ++i) {
    const double p = (*u)[i];
    (*u)[i] += dt * (M * (*lap)[i] - inv_eps2 * (p * p * p - p) + driving_force);
  }
}

} // namespace allen_cahn
