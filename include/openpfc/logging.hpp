// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file logging.hpp
 * @brief Minimal structured logging utilities for OpenPFC
 *
 * Transparent API following OpenPFC philosophy: structs + free functions.
 */

#include <iosfwd>
#include <string>
#include <string_view>

namespace pfc {

/**
 * @brief Log severity levels
 */
enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3 };

/**
 * @brief Lightweight logger configuration
 *
 * Holds minimal immutable state and is safe to pass by value.
 */
struct Logger {
  const LogLevel m_min_level;
  const int m_rank; // MPI rank (or -1 if unknown)
};

/**
 * @brief Write a log message if level >= logger.m_min_level
 *
 * Messages at Warning/Error are written to std::cerr, otherwise to std::clog.
 * When m_rank >= 0, messages are prefixed with "rank <N>: ".
 */
void log(const Logger &logger, LogLevel level, std::string_view message);

/**
 * @brief Convenience helpers
 */
inline void log_error(const Logger &lg, std::string_view msg) {
  log(lg, LogLevel::Error, msg);
}
inline void log_warning(const Logger &lg, std::string_view msg) {
  log(lg, LogLevel::Warning, msg);
}
inline void log_info(const Logger &lg, std::string_view msg) {
  log(lg, LogLevel::Info, msg);
}
inline void log_debug(const Logger &lg, std::string_view msg) {
  log(lg, LogLevel::Debug, msg);
}

} // namespace pfc
