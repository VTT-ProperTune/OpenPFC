// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace heat3d {

enum class Method { Fd, Spectral, SpectralPointwise };

struct RunConfig {
  Method method = Method::Fd;
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  double D = 1.0;
  /** Spatial order for FD: even 2, 4, …, 20 (ignored for spectral methods). */
  int fd_order = 2;
};

inline void print_usage(const char *exe) {
  std::cerr
      << "Usage:\n  " << exe << " fd <N> <n_steps> <dt> <D> <fd_order>\n  " << exe
      << " spectral <N> <n_steps> <dt> <D>\n  " << exe
      << " spectral_pw <N> <n_steps> <dt> <D>\n"
      << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n"
      << "  spectral:    implicit Euler in Fourier space (2 FFTs/step)\n"
      << "  spectral_pw: explicit Euler with point-wise RHS over materialized\n"
      << "               second-derivative fields (1 fwd + 3 inv FFTs/step)\n";
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
  } else if (std::strcmp(argv[1], "spectral_pw") == 0) {
    c.method = Method::SpectralPointwise;
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
 * @brief Fundamental Gaussian solution on ℝ³ for IC u(x,0)=exp(-|x|²/(4D)), ∂_t u =
 * D∇²u: u(x,t)=(1+t)^{-3/2} exp(-|x|²/(4D(1+t))).
 */
inline double analytic_gaussian(double r2, double t, double D) {
  const double s = 1.0 + t;
  return std::pow(s, -1.5) * std::exp(-r2 / (4.0 * D * s));
}

} // namespace heat3d
