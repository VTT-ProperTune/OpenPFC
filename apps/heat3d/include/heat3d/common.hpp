// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace heat3d {

enum class Method { Fd, Spectral };

struct RunConfig {
  Method method = Method::Fd;
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  double D = 1.0;
  /** Spatial order for FD: even 2, 4, …, 20 (ignored for spectral). */
  int fd_order = 2;
};

inline void print_usage(const char *exe) {
  std::cerr
      << "Usage:\n  " << exe << " fd <N> <n_steps> <dt> <D> <fd_order>\n  " << exe
      << " spectral <N> <n_steps> <dt> <D>\n"
      << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n";
}

/** @param argc argv count; caller must ensure enough arguments for the chosen mode
 */
inline RunConfig parse_args(int argc, char **argv) {
  RunConfig c;
  if (argc < 2) {
    return c;
  }
  if (std::strcmp(argv[1], "fd") == 0) {
    c.method = Method::Fd;
    if (argc < 7) {
      return c;
    }
    c.fd_order = std::atoi(argv[6]);
  } else if (std::strcmp(argv[1], "spectral") == 0) {
    c.method = Method::Spectral;
    if (argc < 6) {
      return c;
    }
  } else {
    return c;
  }
  c.N = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.D = std::atof(argv[5]);
  return c;
}

inline bool validate(const RunConfig &c) {
  if (c.N < 8 || c.n_steps < 1 || c.dt <= 0.0 || c.D <= 0.0) {
    return false;
  }
  if (c.method == Method::Fd) {
    if (c.fd_order < 2 || c.fd_order > 20 || (c.fd_order % 2) != 0) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Gaussian IC matching examples/diffusion_model.hpp: exp(-r²/(4D)).
 */
inline void fill_gaussian_subdomain(std::vector<double> *u,
                                    const pfc::decomposition::Decomposition &decomp,
                                    int rank, double D) {
  const auto &gw = pfc::decomposition::get_world(decomp);
  const auto &local = pfc::decomposition::get_subworld(decomp, rank);
  auto lo = pfc::world::get_lower(local);
  auto sz = pfc::world::get_size(local);
  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int sxy = nx * ny;
  const auto origin = pfc::world::get_origin(gw);
  const auto spacing = pfc::world::get_spacing(gw);
  u->resize(static_cast<size_t>(nx * ny * nz));
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gi = lo[0] + ix;
        const int gj = lo[1] + iy;
        const int gk = lo[2] + iz;
        const double x = origin[0] + static_cast<double>(gi) * spacing[0];
        const double y = origin[1] + static_cast<double>(gj) * spacing[1];
        const double z = origin[2] + static_cast<double>(gk) * spacing[2];
        const double r2 = x * x + y * y + z * z;
        const size_t idx = static_cast<size_t>(ix) +
                           static_cast<size_t>(iy) * static_cast<size_t>(nx) +
                           static_cast<size_t>(iz) * static_cast<size_t>(sxy);
        (*u)[idx] = std::exp(-r2 / (4.0 * D));
      }
    }
  }
}

/**
 * @brief Fundamental Gaussian solution on ℝ³ for IC u(x,0)=exp(-|x|²/(4D)), ∂_t u =
 * D∇²u: u(x,t)=(1+t)^{-3/2} exp(-|x|²/(4D(1+t))).
 */
inline double analytic_gaussian(double r2, double t, double D) {
  const double s = 1.0 + t;
  return std::pow(s, -1.5) * std::exp(-r2 / (4.0 * D * s));
}

} // namespace heat3d
