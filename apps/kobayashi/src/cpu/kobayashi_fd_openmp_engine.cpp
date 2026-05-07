// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_openmp_engine.cpp
 * @brief Periodic FD Kobayashi kernel with OpenMP (no MPI halos — torus indexing).
 */

#include <kobayashi/defaults.hpp>
#include <kobayashi/openmp_engine.hpp>

#include <openpfc/frontend/io/png_writer.hpp>

#include <cstdio>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <vector>

#include <omp.h>

namespace {

constexpr inline int wrap_x(int i, int nx) noexcept {
  const int m = i % nx;
  return m < 0 ? m + nx : m;
}

constexpr inline int wrap_y(int j, int ny) noexcept {
  const int m = j % ny;
  return m < 0 ? m + ny : m;
}

inline std::size_t ijx(int i, int j, int nx) noexcept {
  return static_cast<std::size_t>(i + j * nx);
}

void write_phi_png(const char *path, int nx, int ny, const std::vector<double> &phi) {
  pfc::io::write_png_grayscale_from_doubles(path, nx, ny, phi.data(), 0.0, 1.0);
}

} // namespace

namespace kobayashi::openmp_engine {

RunResult run(const RunConfigOpenMP &cfg, bool skip_png, bool quiet) {
  const double dx = cfg.dx;
  const double dy = dx;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_lap_den = 1.0 / (dx * dy);

  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const std::size_t n =
      static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny);

  if (cfg.num_threads > 0) {
    omp_set_num_threads(cfg.num_threads);
  }
  const int nthr = omp_get_max_threads();
  const bool use_team = (nthr > 1);

  std::vector<double> phi(n);
  std::vector<double> tempr(n);
  std::vector<double> lap_phi(n);
  std::vector<double> lap_t(n);
  std::vector<double> phidx(n);
  std::vector<double> phidy(n);
  std::vector<double> epsilon(n);
  std::vector<double> epsilon_deriv(n);

  const int ci = Nx / 2;
  const int cj = Ny / 2;

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
  for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
      const double ddx = static_cast<double>(i - ci);
      const double ddy = static_cast<double>(j - cj);
      phi[ijx(i, j, Nx)] =
          (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
      tempr[ijx(i, j, Nx)] = 0.0;
    }
  }

  // Parallel first-touch for workspace buffers so pages fault under the same thread team as the
  // integration loop (NUMA placement), and not during the timed section.
#pragma omp parallel for schedule(static) if (use_team)
  for (std::size_t k = 0; k < n; ++k) {
    lap_phi[k] = 0.0;
    lap_t[k] = 0.0;
    phidx[k] = 0.0;
    phidy[k] = 0.0;
    epsilon[k] = 0.0;
    epsilon_deriv[k] = 0.0;
  }

  int filenum = 0;
  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
    std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    write_phi_png(path, Nx, Ny, phi);
    ++filenum;
  }

  const int nprint_eff = quiet ? 0 : cfg.nprint;

  const double t_loop0 = omp_get_wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        const std::size_t c = ijx(i, j, Nx);
        const int ip = wrap_x(i + 1, Nx);
        const int im = wrap_x(i - 1, Nx);
        const int jp = wrap_y(j + 1, Ny);
        const int jm = wrap_y(j - 1, Ny);

        const double hne = phi[ijx(ip, j, Nx)];
        const double hnw = phi[ijx(im, j, Nx)];
        const double hns = phi[ijx(i, jm, Nx)];
        const double hnn = phi[ijx(i, jp, Nx)];
        const double hnc = phi[c];
        lap_phi[c] = (hne + hnw + hns + hnn - 4.0 * hnc) * inv_lap_den;

        const double Tne = tempr[ijx(ip, j, Nx)];
        const double Tnw = tempr[ijx(im, j, Nx)];
        const double Tns = tempr[ijx(i, jm, Nx)];
        const double Tnn = tempr[ijx(i, jp, Nx)];
        const double Tnc = tempr[c];
        lap_t[c] = (Tne + Tnw + Tns + Tnn - 4.0 * Tnc) * inv_lap_den;

        const double dpx = (phi[ijx(ip, j, Nx)] - phi[ijx(im, j, Nx)]) * inv_dx;
        const double dpy = (phi[ijx(i, jp, Nx)] - phi[ijx(i, jm, Nx)]) * inv_dy;
        phidx[c] = dpx;
        phidy[c] = dpy;

        const double theta = std::atan2(dpy, dpx);
        epsilon[c] =
            kobayashi::kEpsilonb *
            (1.0 + kobayashi::kDelta *
                       std::cos(kobayashi::kAniso * (theta - kobayashi::kTheta0)));
        epsilon_deriv[c] = -kobayashi::kEpsilonb * kobayashi::kAniso * kobayashi::kDelta *
                           std::sin(kobayashi::kAniso * (theta - kobayashi::kTheta0));
      }
    }

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        const std::size_t c = ijx(i, j, Nx);
        const int ip = wrap_x(i + 1, Nx);
        const int im = wrap_x(i - 1, Nx);
        const int jp = wrap_y(j + 1, Ny);
        const int jm = wrap_y(j - 1, Ny);

        const double phiold = phi[c];

        const double term1 = (epsilon[ijx(i, jp, Nx)] * epsilon_deriv[ijx(i, jp, Nx)] *
                                      phidx[ijx(i, jp, Nx)] -
                                  epsilon[ijx(i, jm, Nx)] * epsilon_deriv[ijx(i, jm, Nx)] *
                                      phidx[ijx(i, jm, Nx)]) *
                             inv_dy;

        const double term2 = -(epsilon[ijx(ip, j, Nx)] * epsilon_deriv[ijx(ip, j, Nx)] *
                                    phidy[ijx(ip, j, Nx)] -
                                epsilon[ijx(im, j, Nx)] * epsilon_deriv[ijx(im, j, Nx)] *
                                    phidy[ijx(im, j, Nx)]) *
                               inv_dx;

        const double ep = epsilon[c];
        const double term3 = ep * ep * lap_phi[c];

        const double m =
            kobayashi::kAlpha / std::numbers::pi *
            std::atan(kobayashi::kGamma * (kobayashi::kTeq - tempr[c]));
        const double term4 = phiold * (1.0 - phiold) * (phiold - 0.5 + m);

        phi[c] = phiold + (cfg.dt / kobayashi::kTau) * (term1 + term2 + term3 + term4);

        tempr[c] = tempr[c] + cfg.dt * lap_t[c] +
                   kobayashi::kKappa * (phi[c] - phiold);
      }
    }

    if (nprint_eff > 0 && istep % nprint_eff == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(), filenum);
      std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file " << path
                << "\n";
      write_phi_png(path, Nx, Ny, phi);
      ++filenum;
    }
  }

  const double t_loop1 = omp_get_wtime();

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    std::cout << "saving final field to " << path << "\n";
    write_phi_png(path, Nx, Ny, phi);
  }

  RunResult out;
  out.phi_xy = std::move(phi);
  out.tempr_xy = std::move(tempr);
  out.wall_loop_s = t_loop1 - t_loop0;
  out.nthreads = nthr;
  return out;
}

} // namespace kobayashi::openmp_engine
