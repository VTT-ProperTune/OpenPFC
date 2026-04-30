// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief MPI-free CLI for `wave2d_fd`, `wave2d_fd_manual`, and GPU `wave2d_cuda` /
 *        `wave2d_hip` (same positional layout as manual).
 */

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <wave2d/wave_boundary.hpp>

namespace wave2d {

struct RunConfig {
  int Nx = 64;
  int Ny = 64;
  int n_steps = 200;
  double dt = 0.01;
  /** Even spatial order for `wave2d_fd` (ignored by manual). */
  int fd_order = 2;
  YBoundaryKind y_bc = YBoundaryKind::Dirichlet;
  /** Dirichlet value on y walls (Neumann ignores). */
  double u_wall = 0.0;
  /** Empty disables VTK; pattern may include `%04d` for time index (see
   * `VTKWriter`). */
  std::string vtk_pattern;
  /** Write VTK every this many completed steps (after initial frame 0). */
  int vtk_every = 1;
};

inline void print_usage(std::ostream &os, const char *exe, bool with_fd_order) {
  os << "Usage:\n  " << exe << " <Nx> <Ny> <n_steps> <dt>";
  if (with_fd_order) {
    os << " <fd_order> <y_bc> [u_wall]";
  } else {
    os << " <y_bc> [u_wall]";
  }
  os << " [--vtk <path_%04d.vti>] [--vtk-every <k>]\n";
  if (with_fd_order) {
    os << "  fd_order: even 2,4,...,20\n";
  }
  os << "  y_bc: dirichlet | neumann\n";
  os << "  u_wall: optional, default 0 (Dirichlet value on y=0 and y=Ny-1)\n";
  os << "  --vtk: ParaView VTK ImageData series (omit to skip file output)\n";
  os << "  --vtk-every: save every k completed steps after t=0 (default 1)\n";
}

namespace detail {

[[nodiscard]] inline bool is_cli_flag(const char *s) noexcept {
  return s != nullptr && s[0] == '-' && s[1] == '-';
}

inline bool parse_vtk_tail(int argc, char **argv, int start, RunConfig &c) {
  for (int i = start; i < argc;) {
    const std::string_view a = argv[i];
    if (a == "--vtk") {
      if (i + 1 >= argc) {
        return false;
      }
      c.vtk_pattern = argv[i + 1];
      i += 2;
    } else if (a == "--vtk-every") {
      if (i + 1 >= argc) {
        return false;
      }
      c.vtk_every = std::atoi(argv[i + 1]);
      i += 2;
    } else {
      return false;
    }
  }
  if (!c.vtk_pattern.empty() && c.vtk_every < 1) {
    return false;
  }
  return true;
}

} // namespace detail

inline std::optional<YBoundaryKind> parse_y_bc(std::string_view s) {
  if (s == "dirichlet" || s == "d" || s == "D") return YBoundaryKind::Dirichlet;
  if (s == "neumann" || s == "n" || s == "N") return YBoundaryKind::Neumann;
  return std::nullopt;
}

namespace detail {

inline bool valid_grid(const RunConfig &c) noexcept {
  return c.Nx >= 4 && c.Ny >= 4 && c.n_steps >= 1 && c.dt > 0.0;
}

inline bool valid_fd_order(const RunConfig &c) noexcept {
  return c.fd_order >= 2 && c.fd_order <= 20 && (c.fd_order % 2) == 0;
}

} // namespace detail

inline std::optional<RunConfig> parse_fd(int argc, char **argv) {
  if (argc < 7) return std::nullopt;
  RunConfig c;
  c.Nx = std::atoi(argv[1]);
  c.Ny = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.fd_order = std::atoi(argv[5]);
  const auto yb = parse_y_bc(argv[6]);
  if (!yb) return std::nullopt;
  c.y_bc = *yb;
  int opt = 7;
  if (argc > 7 && !detail::is_cli_flag(argv[7])) {
    c.u_wall = std::atof(argv[7]);
    opt = 8;
  }
  if (!detail::parse_vtk_tail(argc, argv, opt, c)) return std::nullopt;
  if (!detail::valid_grid(c) || !detail::valid_fd_order(c)) return std::nullopt;
  return c;
}

inline std::optional<RunConfig> parse_manual(int argc, char **argv) {
  if (argc < 6) return std::nullopt;
  RunConfig c;
  c.Nx = std::atoi(argv[1]);
  c.Ny = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  const auto yb = parse_y_bc(argv[5]);
  if (!yb) return std::nullopt;
  c.y_bc = *yb;
  int opt = 6;
  if (argc > 6 && !detail::is_cli_flag(argv[6])) {
    c.u_wall = std::atof(argv[6]);
    opt = 7;
  }
  if (!detail::parse_vtk_tail(argc, argv, opt, c)) return std::nullopt;
  if (!detail::valid_grid(c)) return std::nullopt;
  return c;
}

inline std::optional<RunConfig> parse_fd_or_print_usage(int argc, char **argv,
                                                        int rank) {
  auto c = parse_fd(argc, argv);
  if (!c && rank == 0) print_usage(std::cerr, argv[0], true);
  return c;
}

inline std::optional<RunConfig> parse_manual_or_print_usage(int argc, char **argv,
                                                            int rank) {
  auto c = parse_manual(argc, argv);
  if (!c && rank == 0) print_usage(std::cerr, argv[0], false);
  return c;
}

} // namespace wave2d
