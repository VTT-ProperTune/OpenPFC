// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/utils/logging.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace pfc {

static const char *to_string(LogLevel lvl) {
  switch (lvl) {
  case LogLevel::Debug: return "DEBUG";
  case LogLevel::Info: return "INFO";
  case LogLevel::Warning: return "WARN";
  case LogLevel::Error: return "ERROR";
  }
  return "?";
}

void log(const Logger &logger, LogLevel level, std::string_view message) {
  if (static_cast<int>(level) < static_cast<int>(logger.m_min_level)) return;

  // Timestamp (UTC)
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  const bool utc_ok = (gmtime_s(&tm, &t) == 0);
#else
  const bool utc_ok = (gmtime_r(&t, &tm) != nullptr);
#endif

  std::ostringstream oss;
  if (utc_ok) {
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  } else {
    // gmtime_s / gmtime_r can fail for unrepresentable instants
    oss << "<time-unavailable>";
  }
  oss << " [" << to_string(level) << "] ";
  if (logger.m_rank >= 0) {
    oss << "rank " << logger.m_rank << ": ";
  }
  oss << message;

  if (level == LogLevel::Warning || level == LogLevel::Error) {
    std::cerr << oss.str() << std::endl;
  } else {
    std::clog << oss.str() << std::endl;
  }
}

} // namespace pfc
