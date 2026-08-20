// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file cli.hpp
 * @brief Shared MPI-free CLI helpers for FD demo apps.
 *
 * Per-app `RunConfig` and usage strings stay in the app. This header is
 * the common parse-or-print wrapper, `--flag` detection, and even FD-order
 * check used by heat3d / wave2d / kobayashi.
 */

#include <iostream>

namespace pfc::apps {

/// True for tokens that start with `--` (optional-flag tails).
[[nodiscard]] inline bool is_long_flag(const char *s) noexcept {
  return s != nullptr && s[0] == '-' && s[1] == '-';
}

/// Compact-FD spatial order: even in `[2, 20]`.
[[nodiscard]] inline bool even_fd_order(int order) noexcept {
  return order >= 2 && order <= 20 && (order % 2) == 0;
}

/**
 * Run @p parse; on failure print @p print_usage on rank 0.
 *
 * @p parse is `optional<Config>(int, char**)` and @p print_usage is
 * `void(ostream&, const char *exe)`.
 */
template <class Parse, class PrintUsage>
[[nodiscard]] auto parse_or_print_usage(int argc, char **argv, int rank,
                                        Parse &&parse, PrintUsage &&print_usage)
    -> decltype(parse(argc, argv)) {
  auto cfg = parse(argc, argv);
  if (!cfg && rank == 0) {
    print_usage(std::cerr, argc >= 1 ? argv[0] : "app");
  }
  return cfg;
}

} // namespace pfc::apps
