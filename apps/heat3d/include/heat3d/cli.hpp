// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief Command-line parsing for the heat3d binary quartet
 *        (`heat3d_fd`, `heat3d_fd_manual`, `heat3d_spectral`,
 *        `heat3d_spectral_pointwise`).
 *
 * @details
 * Header-only, MPI-free, OpenPFC-free. Lives next to `heat_model.hpp`
 * so the same ergonomics apply: physicists can edit the CLI surface in
 * one tiny self-contained file, and the parsers are trivially
 * unit-testable (see `apps/heat3d/tests/test_heat3d.cpp`).
 *
 * Each binary owns its method, so the parsers do **not** consume an
 * `argv[1]` discriminator. Two parser families:
 *
 *  - `parse_fd` / `parse_fd_or_print_usage` — `<N> <n_steps> <dt> <D> <fd_order>`,
 *    used by `heat3d_fd` and `heat3d_fd_manual`.
 *  - `parse_spectral` / `parse_spectral_or_print_usage` —
 *    `<N> <n_steps> <dt> <D>`, used by `heat3d_spectral` and
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

namespace heat3d {

/**
 * @brief Parsed CLI configuration for one heat3d run.
 *
 * `fd_order` is meaningful only for the FD-style binaries; the
 * spectral binaries leave it at the default `2`.
 */
struct RunConfig {
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  double D = 1.0;
  /** Spatial order for FD: even 2, 4, …, 20 (ignored for spectral binaries). */
  int fd_order = 2;
};

/// Per-binary usage line for FD-style executables.
inline void print_usage_fd(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <D> <fd_order>\n"
     << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n";
}

/// Per-binary usage line for spectral-style executables.
inline void print_usage_spectral(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <D>\n";
}

namespace detail {

/// Common value-range check shared by both parser families.
inline bool valid_values(const RunConfig &c, bool needs_fd_order) noexcept {
  if (c.N < 8 || c.n_steps < 1 || c.dt <= 0.0 || c.D <= 0.0) return false;
  if (needs_fd_order) {
    if (c.fd_order < 2 || c.fd_order > 20 || (c.fd_order % 2) != 0) return false;
  }
  return true;
}

} // namespace detail

/**
 * @brief Parse the FD-style positional CLI: `<N> <n_steps> <dt> <D> <fd_order>`.
 *
 * Returns `std::nullopt` on insufficient args or out-of-range values.
 */
inline std::optional<RunConfig> parse_fd(int argc, char **argv) noexcept {
  if (argc < 6) return std::nullopt;
  RunConfig c;
  c.N = std::atoi(argv[1]);
  c.n_steps = std::atoi(argv[2]);
  c.dt = std::atof(argv[3]);
  c.D = std::atof(argv[4]);
  c.fd_order = std::atoi(argv[5]);
  if (!detail::valid_values(c, /*needs_fd_order=*/true)) return std::nullopt;
  return c;
}

/**
 * @brief Parse the spectral-style positional CLI: `<N> <n_steps> <dt> <D>`.
 *
 * `fd_order` is left at the `RunConfig` default (unused).
 */
inline std::optional<RunConfig> parse_spectral(int argc, char **argv) noexcept {
  if (argc < 5) return std::nullopt;
  RunConfig c;
  c.N = std::atoi(argv[1]);
  c.n_steps = std::atoi(argv[2]);
  c.dt = std::atof(argv[3]);
  c.D = std::atof(argv[4]);
  if (!detail::valid_values(c, /*needs_fd_order=*/false)) return std::nullopt;
  return c;
}

/// `parse_fd` + rank-0 usage print — drop-in for FD binary `main`s.
inline std::optional<RunConfig> parse_fd_or_print_usage(int argc, char **argv,
                                                        int rank) {
  auto cfg = parse_fd(argc, argv);
  if (!cfg && rank == 0) {
    print_usage_fd(std::cerr, argc >= 1 ? argv[0] : "heat3d_fd");
  }
  return cfg;
}

/// `parse_spectral` + rank-0 usage print — drop-in for spectral binary `main`s.
inline std::optional<RunConfig> parse_spectral_or_print_usage(int argc, char **argv,
                                                              int rank) {
  auto cfg = parse_spectral(argc, argv);
  if (!cfg && rank == 0) {
    print_usage_spectral(std::cerr, argc >= 1 ? argv[0] : "heat3d_spectral");
  }
  return cfg;
}

} // namespace heat3d
