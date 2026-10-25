// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once
/** @file parse_support.hpp
 * @brief Shared command-line flag and finite-difference order parsing.
 */

#include <iostream>

namespace pfc::ui::app_support {

[[nodiscard]] inline bool is_long_flag(const char *s) noexcept {
  return s != nullptr && s[0] == '-' && s[1] == '-';
}

[[nodiscard]] inline bool even_fd_order(int order) noexcept {
  return order >= 2 && order <= 20 && (order % 2) == 0;
}

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

} // namespace pfc::ui::app_support
