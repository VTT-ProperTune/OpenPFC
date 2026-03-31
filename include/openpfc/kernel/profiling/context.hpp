// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file context.hpp
 * @brief Thread-local active profiling session for low-coupling instrumentation
 */

#ifndef PFC_KERNEL_PROFILING_CONTEXT_HPP
#define PFC_KERNEL_PROFILING_CONTEXT_HPP

#include <string_view>

namespace pfc {
namespace profiling {

class ProfilingSession;

/// Active session for this OS thread (typically one thread per MPI rank).
inline ProfilingSession *&current_session_ptr() noexcept {
  thread_local ProfilingSession *p = nullptr;
  return p;
}

inline void set_current_session(ProfilingSession *s) noexcept {
  current_session_ptr() = s;
}

inline ProfilingSession *current_session() noexcept { return current_session_ptr(); }

/**
 * @brief Add elapsed time to a named catalog region (inclusive == exclusive).
 *        No-op if no active session or path not in catalog.
 */
void record_time(std::string_view path, double seconds) noexcept;

/**
 * @brief RAII: bind @p session for the duration of a scope (e.g. one step()).
 */
class ProfilingContextScope {
  ProfilingSession *prev_;

public:
  explicit ProfilingContextScope(ProfilingSession *session) noexcept
      : prev_(current_session_ptr()) {
    current_session_ptr() = session;
  }

  ProfilingContextScope(const ProfilingContextScope &) = delete;
  ProfilingContextScope &operator=(const ProfilingContextScope &) = delete;

  ~ProfilingContextScope() { current_session_ptr() = prev_; }
};

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_CONTEXT_HPP
