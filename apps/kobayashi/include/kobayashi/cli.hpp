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
#include <string_view>
#include <vector>

#include <kobayashi/defaults.hpp>
#include <openpfc_apps/cli.hpp>

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
  /// Schema-v2 profiling JSON path (empty = do not export).
  std::string profile_json;
  /// Untimed steps discarded before JSON frames (HIP/CUDA perf capture).
  int warmup = 0;
};

/** Same workload keys as @ref RunConfig plus optional OpenMP thread override (0 =
 * runtime default). */
struct RunConfigOpenMP : RunConfig {
  int num_threads = 0;
};

inline void print_usage(std::ostream &os, const char *exe) {
  os << "Usage:\n  " << exe
     << " [<Nx> <Ny> <n_steps> <dt> <dx> [output_dir]] "
        "[--output PATH.json] [--warmup N]\n"
     << "All positional arguments optional; defaults match Julia kobayashi_v1:\n"
     << "  Nx Ny n_steps dt dx  →  256 256 2000 1e-4 0.03\n"
     << "  output_dir           →  results/kobayashi_v1\n"
     << "  --output PATH.json   →  schema-v2 profiling JSON (optional)\n"
     << "  --warmup N           →  untimed steps before JSON frames (default 0)\n"
     << "PNG snapshots: initial + every nsave steps (" << kNsave << ").\n"
     << "Progress print every nprint steps (" << kNprint << ").\n";
}

inline void print_usage_openmp(std::ostream &os, const char *exe) {
  print_usage(os, exe);
  os << "OpenMP: respects OMP_NUM_THREADS unless an 8th argument sets an explicit "
        "thread count:\n"
     << "  … <output_dir> <num_threads>\n";
}

namespace detail {

struct SplitArgv {
  std::vector<std::string> positional;
  std::string profile_json;
  int warmup = 0;
  bool help = false;
  bool error = false;
};

inline SplitArgv split_argv(int argc, char **argv) {
  SplitArgv out;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a{argv[i]};
    if (a == "--help" || a == "-h") {
      out.help = true;
      return out;
    }
    if (a == "--output") {
      if (i + 1 >= argc) {
        out.error = true;
        return out;
      }
      out.profile_json = argv[++i];
      continue;
    }
    if (a == "--warmup") {
      if (i + 1 >= argc) {
        out.error = true;
        return out;
      }
      out.warmup = std::atoi(argv[++i]);
      if (out.warmup < 0) {
        out.error = true;
        return out;
      }
      continue;
    }
    if (!a.empty() && a.front() == '-') {
      out.error = true;
      return out;
    }
    out.positional.emplace_back(argv[i]);
  }
  return out;
}

inline bool apply_positionals(RunConfig &c, const std::vector<std::string> &pos) {
  if (pos.empty()) {
    return true;
  }
  if (pos.size() != 5 && pos.size() != 6) {
    return false;
  }
  c.Nx = std::atoi(pos[0].c_str());
  c.Ny = std::atoi(pos[1].c_str());
  c.n_steps = std::atoi(pos[2].c_str());
  c.dt = std::atof(pos[3].c_str());
  c.dx = std::atof(pos[4].c_str());
  if (pos.size() == 6) {
    c.output_dir = pos[5];
  }
  return c.Nx >= 4 && c.Ny >= 4 && c.n_steps >= 1 && c.dt > 0.0 && c.dx > 0.0;
}

} // namespace detail

inline std::optional<RunConfig> parse_args(int argc, char **argv) {
  const auto split = detail::split_argv(argc, argv);
  if (split.help || split.error) {
    return std::nullopt;
  }
  RunConfig c;
  c.profile_json = split.profile_json;
  c.warmup = split.warmup;
  if (!detail::apply_positionals(c, split.positional)) {
    return std::nullopt;
  }
  return c;
}

inline std::optional<RunConfig> parse_or_print_usage(int argc, char **argv,
                                                     int rank) {
  return pfc::apps::parse_or_print_usage(argc, argv, rank, parse_args, print_usage);
}

/**
 * Parses argc in {1, 6, 7, 8} like @ref parse_args, plus optional 8th integer thread
 * count (requires explicit `output_dir` as argv[6]).
 */
inline std::optional<RunConfigOpenMP> parse_args_openmp(int argc, char **argv) {
  const auto split = detail::split_argv(argc, argv);
  if (split.help || split.error) {
    return std::nullopt;
  }
  RunConfigOpenMP c;
  c.profile_json = split.profile_json;
  c.warmup = split.warmup;
  auto pos = split.positional;
  if (!pos.empty() && pos.size() == 7) {
    c.num_threads = std::atoi(pos.back().c_str());
    if (c.num_threads < 1) {
      return std::nullopt;
    }
    pos.pop_back();
  }
  if (!detail::apply_positionals(c, pos)) {
    return std::nullopt;
  }
  return c;
}

inline std::optional<RunConfigOpenMP> parse_or_print_usage_openmp(int argc,
                                                                  char **argv) {
  auto cfg = parse_args_openmp(argc, argv);
  if (!cfg) {
    print_usage_openmp(std::cerr, argv[0]);
  }
  return cfg;
}

} // namespace kobayashi
