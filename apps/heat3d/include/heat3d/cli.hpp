// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief Command-line parsing for the heat3d application.
 *
 * @details
 * Header-only, MPI-free, OpenPFC-free. Lives next to `heat_model.hpp` so
 * the same ergonomics apply: physicists can edit the CLI surface in one
 * tiny self-contained file, and the parser is trivially unit-testable
 * (see `apps/heat3d/tests/test_heat3d.cpp`).
 *
 * The legacy "if argc < 7 then default" sentinel API of earlier
 * `parse_args` implementations is replaced by `std::optional<RunConfig>`:
 * `parse` returns nullopt on any failure (insufficient args, unknown
 * method, out-of-range values), and `parse_or_print_usage` is the
 * convenience wrapper that prints the usage line on rank 0 when parsing
 * fails.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <ostream>

namespace heat3d {

/**
 * @brief Time-stepping / discretisation backend.
 *
 * @deprecated Retained only for the unified `heat3d.cpp` driver during
 * the per-binary split. Once each method has its own executable each
 * binary parses its arguments via `parse_fd` / `parse_spectral` and the
 * `Method` discriminator becomes unnecessary.
 */
enum class Method { Fd, Spectral, SpectralPointwise };

/**
 * @brief Parsed CLI configuration for one heat3d run.
 *
 * `method` is set by the unified `parse` (legacy `argv[1]` dispatch).
 * The slim per-binary `parse_fd` / `parse_spectral` overloads do **not**
 * touch this field — each binary already knows its own method.
 *
 * `fd_order` defaults to 2 and is left at the default by `parse_spectral`.
 */
struct RunConfig {
  Method method = Method::Fd;
  int N = 32;
  int n_steps = 100;
  double dt = 0.01;
  double D = 1.0;
  /** Spatial order for FD: even 2, 4, …, 20 (ignored for spectral methods). */
  int fd_order = 2;
};

/**
 * @brief Print the unified usage line covering all three sub-commands.
 *        Used by the legacy `heat3d` binary's `parse_or_print_usage`.
 */
inline void print_usage(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " fd <N> <n_steps> <dt> <D> <fd_order>\n  " << exe
     << " spectral <N> <n_steps> <dt> <D>\n  " << exe
     << " spectral_pw <N> <n_steps> <dt> <D>\n"
     << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n"
     << "  spectral:    implicit Euler in Fourier space (2 FFTs/step)\n"
     << "  spectral_pw: explicit Euler with point-wise RHS over materialized\n"
     << "               second-derivative fields (1 fwd + 3 inv FFTs/step)\n";
}

/**
 * @brief Per-binary usage line for the FD-style executables
 *        (`heat3d_fd`, `heat3d_fd_manual`).
 *
 * Each binary already knows its own discretisation, so the usage drops
 * the `fd` / `spectral` sub-command and just lists the positionals.
 */
inline void print_usage_fd(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <D> <fd_order>\n"
     << "  fd_order: even 2,4,...,20 (central Laplacian; halo width order/2)\n";
}

/**
 * @brief Per-binary usage line for the spectral-style executables
 *        (`heat3d_spectral`, `heat3d_spectral_pointwise`).
 */
inline void print_usage_spectral(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe << " <N> <n_steps> <dt> <D>\n";
}

namespace detail {

/**
 * @brief Sanity-check parsed values (post-positional-extraction).
 *
 * Cheap to call separately so test cases can target value ranges
 * without reproducing the argv plumbing.
 */
inline bool valid_values(const RunConfig &c) noexcept {
  if (c.N < 8 || c.n_steps < 1 || c.dt <= 0.0 || c.D <= 0.0) return false;
  if (c.method == Method::Fd) {
    if (c.fd_order < 2 || c.fd_order > 20 || (c.fd_order % 2) != 0) return false;
  }
  return true;
}

} // namespace detail

/**
 * @brief Parse `argv` into a `RunConfig`.
 *
 * @return The parsed config on success, `std::nullopt` on any failure
 *         (missing or unknown method, too few positional args, or
 *         out-of-range values).
 *
 * Pure: no I/O, no exit, no MPI. Use `parse_or_print_usage` for the
 * usual "print usage on rank 0 and bail" wrapper.
 */
inline std::optional<RunConfig> parse(int argc, char **argv) noexcept {
  if (argc < 2) return std::nullopt;
  RunConfig c;
  int min_argc = 0;
  if (std::strcmp(argv[1], "fd") == 0) {
    c.method = Method::Fd;
    min_argc = 7;
  } else if (std::strcmp(argv[1], "spectral") == 0) {
    c.method = Method::Spectral;
    min_argc = 6;
  } else if (std::strcmp(argv[1], "spectral_pw") == 0) {
    c.method = Method::SpectralPointwise;
    min_argc = 6;
  } else {
    return std::nullopt;
  }
  if (argc < min_argc) return std::nullopt;

  c.N = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.D = std::atof(argv[5]);
  if (c.method == Method::Fd) c.fd_order = std::atoi(argv[6]);

  if (!detail::valid_values(c)) return std::nullopt;
  return c;
}

/**
 * @brief Parse CLI; on failure, print usage on rank 0 and return nullopt.
 *
 * Handy main()-side wrapper that absorbs the "print on rank 0 only"
 * dance every MPI app needs.
 */
inline std::optional<RunConfig> parse_or_print_usage(int argc, char **argv,
                                                     int rank) {
  auto cfg = parse(argc, argv);
  if (!cfg && rank == 0) {
    print_usage(std::cerr, argc >= 1 ? argv[0] : "heat3d");
  }
  return cfg;
}

// -----------------------------------------------------------------------------
// Slim per-binary parsers — used by `heat3d_fd`, `heat3d_fd_manual`,
// `heat3d_spectral`, and `heat3d_spectral_pointwise`. Each binary knows
// its own method, so the parser does NOT consume an `argv[1]` discriminator.
// -----------------------------------------------------------------------------

/**
 * @brief Parse the FD-style positional CLI: `<N> <n_steps> <dt> <D> <fd_order>`.
 *
 * Returns `std::nullopt` on insufficient args or out-of-range values.
 * `RunConfig::method` is set to `Method::Fd` so existing helpers
 * (`heat3d::report`, `heat3d::fd_extra_metadata`) keep working.
 */
inline std::optional<RunConfig> parse_fd(int argc, char **argv) noexcept {
  if (argc < 6) return std::nullopt;
  RunConfig c;
  c.method = Method::Fd;
  c.N = std::atoi(argv[1]);
  c.n_steps = std::atoi(argv[2]);
  c.dt = std::atof(argv[3]);
  c.D = std::atof(argv[4]);
  c.fd_order = std::atoi(argv[5]);
  if (!detail::valid_values(c)) return std::nullopt;
  return c;
}

/**
 * @brief Parse the spectral-style positional CLI: `<N> <n_steps> <dt> <D>`.
 *
 * `RunConfig::method` is set to `Method::Spectral` so existing helpers
 * keep working; the SpectralPointwise binary may overwrite the field
 * after parsing.
 */
inline std::optional<RunConfig> parse_spectral(int argc, char **argv) noexcept {
  if (argc < 5) return std::nullopt;
  RunConfig c;
  c.method = Method::Spectral;
  c.N = std::atoi(argv[1]);
  c.n_steps = std::atoi(argv[2]);
  c.dt = std::atof(argv[3]);
  c.D = std::atof(argv[4]);
  if (!detail::valid_values(c)) return std::nullopt;
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
