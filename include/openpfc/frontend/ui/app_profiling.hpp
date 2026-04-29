// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file app_profiling.hpp
 * @brief Optional profiling session lifecycle for JSON-driven `App` runs
 *
 * @details
 * Parses the `"profiling"` section from root settings, owns the
 * `ProfilingSession` when enabled, and performs export / optional MPI report at
 * shutdown. Implementation of `AppProfilingController` lives in `app_profiling.cpp`
 * to keep this header light. Inline helpers remain for unit tests.
 */

#ifndef PFC_UI_APP_PROFILING_HPP
#define PFC_UI_APP_PROFILING_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <mpi.h>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/utils/logging.hpp>

namespace pfc::profiling {
class ProfilingSession;
struct ProfilingExportOptions;
} // namespace pfc::profiling

namespace pfc::ui {

using json = nlohmann::json;

/**
 * @brief Return profiling JSON keys that are not in the supported set (for tests
 * and diagnostics).
 */
[[nodiscard]] inline std::vector<std::string>
list_unknown_profiling_keys(const json &profiling) {
  std::vector<std::string> out;
  if (!profiling.is_object()) {
    return out;
  }
  static constexpr std::array<std::string_view, 8> known_keys = {
      "enabled",      "format",  "output", "memory_samples",
      "print_report", "regions", "run_id", "export_metadata"};

  for (const auto &[key, _] : profiling.items()) {
    if (std::find(known_keys.begin(), known_keys.end(), key) == known_keys.end()) {
      out.push_back(std::string("unknown profiling config key '") + key + "'");
    }
  }
  return out;
}

inline void warn_unknown_profiling_keys(const json &profiling,
                                        const pfc::Logger &lg) {
  for (const auto &msg : list_unknown_profiling_keys(profiling)) {
    pfc::log_warning(lg, msg);
  }
}

/**
 * @brief Owns parsed profiling settings and optional `ProfilingSession`
 */
class AppProfilingController {
public:
  void configure_from_root_settings(const json &root_settings, int mpi_rank,
                                    bool rank0);

  [[nodiscard]] bool enabled() const noexcept { return m_enabled; }
  [[nodiscard]] bool memory_samples() const noexcept { return m_memory_samples; }
  [[nodiscard]] bool print_report() const noexcept { return m_print_report; }

  [[nodiscard]] pfc::profiling::ProfilingSession *session() noexcept;
  [[nodiscard]] const pfc::profiling::ProfilingSession *session() const noexcept;

  void finalize_and_export_if_active(const json &root_settings, MPI_Comm comm,
                                     bool rank0, const pfc::Logger &app_lg);

private:
  bool m_enabled = false;
  std::string m_format = "json";
  std::string m_output = "openpfc_profile";
  bool m_memory_samples = false;
  bool m_print_report = false;
  std::vector<std::string> m_extra_regions;
  std::string m_run_id;
  json m_export_metadata;
  std::unique_ptr<pfc::profiling::ProfilingSession> m_session;

  void populate_export_metadata(const json &root_settings,
                                pfc::profiling::ProfilingExportOptions &exp) const;
};

inline pfc::profiling::ProfilingSession *AppProfilingController::session() noexcept {
  return m_session.get();
}

inline const pfc::profiling::ProfilingSession *
AppProfilingController::session() const noexcept {
  return m_session.get();
}

} // namespace pfc::ui

#endif // PFC_UI_APP_PROFILING_HPP
