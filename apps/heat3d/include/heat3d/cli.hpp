// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief Command-line parsing for the heat3d binary lineup
 *        (`heat3d_fd`, `heat3d_fd_hip`, `heat3d_fd_manual`, `heat3d_fd_scratch`,
 *        `heat3d_spectral`, `heat3d_spectral_hip`, `heat3d_spectral_pointwise`).
 *
 * @details
 * Header-only, MPI-free, OpenPFC-free. Lives next to `heat_model.hpp`
 * so the same ergonomics apply: physicists can edit the CLI surface in
 * one tiny self-contained file, and the parsers are trivially
 * unit-testable (see `apps/heat3d/tests/test_heat3d.cpp`).
 *
 * The diffusion coefficient `D` is **not** a CLI knob — it is hard-coded
 * once in `heat_model.hpp` (`heat3d::kD`) so all binaries share the same
 * fixed value and their L2 numbers stay comparable. Each binary owns
 * its method, so the parsers do **not** consume an `argv[1]`
 * discriminator. Two parser families:
 *
 *  - `parse_fd` / `parse_fd_or_print_usage` — cubic
 *    `<N> <n_steps> <dt> <fd_order>` or rectangular
 *    `<Nx> <Ny> <Nz> <n_steps> <dt> <fd_order>`.
 *  - `parse_spectral` / `parse_spectral_or_print_usage` —
 *    `<N> <n_steps> <dt>`, used by `heat3d_fd_manual`,
 *    `heat3d_fd_scratch`, `heat3d_spectral`, `heat3d_spectral_hip`, and
 *    `heat3d_spectral_pointwise`.
 *
 * Both return `std::optional<RunConfig>`: `nullopt` on insufficient
 * args or out-of-range values; the `_or_print_usage` wrappers print
 * the per-binary usage line on rank 0 before returning `nullopt`.
 */

#include <cstdlib>
#include <iostream>
#include <optional>
#include <ostream>

#include <heat3d/parse_support.hpp>

namespace heat3d {

/**
 * @brief Parsed CLI configuration for one heat3d run.
 *
 * `fd_order` is meaningful only for the compact FD binary (`heat3d_fd`);
 * the manual/scratch FD binaries hard-code 2nd-order central, and the
 * spectral binaries leave it at the default `2`.
 *
 * Cubic CLI sets `N = Nx = Ny = Nz`. Rectangular FD CLI sets all three
 * extents and keeps `N = Nx` for callers that still print a single N.
 */
struct RunConfig {
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  /** Spatial order for compact FD: even 2, 4, …, 20 (ignored elsewhere). */
  int fd_order = 2;
  int Nx = 32;
  int Ny = 32;
  int Nz = 32;
};

/// Per-binary usage line for the compact FD executable.
inline void print_usage_fd(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <fd_order>\n"
     << "  " << exe << " <Nx> <Ny> <Nz> <n_steps> <dt> <fd_order>\n"
     << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n";
}

/// Per-binary usage line for the spectral / manual / scratch executables.
inline void print_usage_spectral(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt>\n"
     << "  " << exe << " <Nx> <Ny> <Nz> <n_steps> <dt>\n";
}

namespace detail {

inline void sync_cube(RunConfig &c) noexcept { c.Nx = c.Ny = c.Nz = c.N; }

/// Common value-range check shared by both parser families.
inline bool valid_values(const RunConfig &c, bool needs_fd_order) noexcept {
  if (c.Nx < 8 || c.Ny < 8 || c.Nz < 8 || c.n_steps < 1 || c.dt <= 0.0) {
    return false;
  }
  if (needs_fd_order && !heat3d::even_fd_order(c.fd_order)) return false;
  return true;
}

} // namespace detail

/// Fill a cubic `RunConfig` (CPU drivers that still parse locally).
inline RunConfig make_cubic_config(int N, int n_steps, double dt, int fd_order) {
  RunConfig c;
  c.N = N;
  c.n_steps = n_steps;
  c.dt = dt;
  c.fd_order = fd_order;
  detail::sync_cube(c);
  return c;
}

/**
 * @brief Parse the compact-FD positional CLI.
 *
 * Cubic: `<N> <n_steps> <dt> <fd_order>` (`argc == 5`).
 * Rectangular: `<Nx> <Ny> <Nz> <n_steps> <dt> <fd_order>` (`argc == 7`).
 *
 * Returns `std::nullopt` on insufficient args or out-of-range values.
 */
inline std::optional<RunConfig> parse_fd(int argc, char **argv) noexcept {
  RunConfig c;
  if (argc == 7) {
    c.Nx = std::atoi(argv[1]);
    c.Ny = std::atoi(argv[2]);
    c.Nz = std::atoi(argv[3]);
    c.N = c.Nx;
    c.n_steps = std::atoi(argv[4]);
    c.dt = std::atof(argv[5]);
    c.fd_order = std::atoi(argv[6]);
  } else if (argc == 5) {
    c.N = std::atoi(argv[1]);
    c.n_steps = std::atoi(argv[2]);
    c.dt = std::atof(argv[3]);
    c.fd_order = std::atoi(argv[4]);
    detail::sync_cube(c);
  } else {
    return std::nullopt;
  }
  if (!detail::valid_values(c, /*needs_fd_order=*/true)) return std::nullopt;
  return c;
}

/**
 * @brief Parse the spectral / manual / scratch CLI.
 *
 * Cubic: `<N> <n_steps> <dt>` (`argc == 4`).
 * Rectangular: `<Nx> <Ny> <Nz> <n_steps> <dt>` (`argc == 6`).
 *
 * `fd_order` is left at the `RunConfig` default (unused).
 */
inline std::optional<RunConfig> parse_spectral(int argc, char **argv) noexcept {
  RunConfig c;
  if (argc == 4) {
    c.N = std::atoi(argv[1]);
    c.n_steps = std::atoi(argv[2]);
    c.dt = std::atof(argv[3]);
    detail::sync_cube(c);
  } else if (argc == 6) {
    c.Nx = std::atoi(argv[1]);
    c.Ny = std::atoi(argv[2]);
    c.Nz = std::atoi(argv[3]);
    c.n_steps = std::atoi(argv[4]);
    c.dt = std::atof(argv[5]);
    c.N = c.Nx;
  } else {
    return std::nullopt;
  }
  if (!detail::valid_values(c, /*needs_fd_order=*/false)) return std::nullopt;
  return c;
}

/// `parse_fd` + rank-0 usage print — drop-in for compact FD `main`.
inline std::optional<RunConfig> parse_fd_or_print_usage(int argc, char **argv,
                                                        int rank) {
  return heat3d::parse_or_print_usage(argc, argv, rank, parse_fd, print_usage_fd);
}

/// `parse_spectral` + rank-0 usage print — drop-in for the simpler binaries.
inline std::optional<RunConfig> parse_spectral_or_print_usage(int argc, char **argv,
                                                              int rank) {
  return heat3d::parse_or_print_usage(argc, argv, rank, parse_spectral,
                                      print_usage_spectral);
}

} // namespace heat3d
