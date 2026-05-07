// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief CLI for `kobayashi_fd_manual` (MPI-free parsing helpers).
 */

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <kobayashi/defaults.hpp>

namespace kobayashi {

struct RunConfig {
  int Nx = 256;
  int Ny = 256;
  int n_steps = 2000;
  double dt = 1.0e-4;
  double dx = 0.03;
  std::string output_dir = "results/kobayashi_v1";
  int nprint = kNprint;
  int nsave = kNsave;
};

/** Same workload keys as @ref RunConfig plus optional OpenMP thread override (0 = runtime default). */
struct RunConfigOpenMP : RunConfig {
  int num_threads = 0;
};

inline void print_usage(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe
     << " [<Nx> <Ny> <n_steps> <dt> <dx> [output_dir]]\n"
     << "All arguments optional; defaults match Julia kobayashi_v1:\n"
     << "  Nx Ny n_steps dt dx  →  256 256 2000 1e-4 0.03\n"
     << "  output_dir           →  results/kobayashi_v1\n"
     << "PNG snapshots: initial + every nsave steps (" << kNsave << ").\n"
     << "Progress print every nprint steps (" << kNprint << ").\n";
}

inline void print_usage_openmp(std::ostream &os, const char *exe) {
  print_usage(os, exe);
  os << "OpenMP: respects OMP_NUM_THREADS unless an 8th argument sets an explicit thread count:\n"
     << "  … <output_dir> <num_threads>\n";
}

inline std::optional<RunConfig> parse_args(int argc, char **argv) {
  RunConfig c;
  if (argc == 1) {
    return c;
  }
  if (argc != 6 && argc != 7) {
    return std::nullopt;
  }
  c.Nx = std::atoi(argv[1]);
  c.Ny = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.dx = std::atof(argv[5]);
  if (argc == 7) {
    c.output_dir = argv[6];
  }
  if (c.Nx < 4 || c.Ny < 4 || c.n_steps < 1 || !(c.dt > 0.0) || !(c.dx > 0.0)) {
    return std::nullopt;
  }
  return c;
}

inline std::optional<RunConfig> parse_or_print_usage(int argc, char **argv, int rank) {
  auto cfg = parse_args(argc, argv);
  if (!cfg && rank == 0) {
    print_usage(std::cerr, argv[0]);
  }
  return cfg;
}

/**
 * Parses argc in {1, 6, 7, 8} like @ref parse_args, plus optional 8th integer thread count
 * (requires explicit `output_dir` as argv[6]).
 */
inline std::optional<RunConfigOpenMP> parse_args_openmp(int argc, char **argv) {
  RunConfigOpenMP c;
  if (argc == 1) {
    return c;
  }
  if (argc != 6 && argc != 7 && argc != 8) {
    return std::nullopt;
  }
  c.Nx = std::atoi(argv[1]);
  c.Ny = std::atoi(argv[2]);
  c.n_steps = std::atoi(argv[3]);
  c.dt = std::atof(argv[4]);
  c.dx = std::atof(argv[5]);
  if (argc >= 7) {
    c.output_dir = argv[6];
  }
  if (argc == 8) {
    c.num_threads = std::atoi(argv[7]);
    if (c.num_threads < 1) {
      return std::nullopt;
    }
  }
  if (c.Nx < 4 || c.Ny < 4 || c.n_steps < 1 || !(c.dt > 0.0) || !(c.dx > 0.0)) {
    return std::nullopt;
  }
  return c;
}

inline std::optional<RunConfigOpenMP> parse_or_print_usage_openmp(int argc, char **argv) {
  auto cfg = parse_args_openmp(argc, argv);
  if (!cfg) {
    print_usage_openmp(std::cerr, argv[0]);
  }
  return cfg;
}

} // namespace kobayashi
