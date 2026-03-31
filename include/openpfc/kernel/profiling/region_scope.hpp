// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file region_scope.hpp
 * @brief RAII nested wall-time scopes (steady_clock) for catalog regions
 */

#ifndef PFC_KERNEL_PROFILING_REGION_SCOPE_HPP
#define PFC_KERNEL_PROFILING_REGION_SCOPE_HPP

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#include <string_view>

namespace pfc {
namespace profiling {

/**
 * @brief Push a timed scope on construction, pop on destruction (LIFO with other
 * scopes). No-op if there is no active session. Unknown paths are registered via
 * ensure_path.
 */
class ProfilingTimedScope {
  bool active_{false};

public:
  explicit ProfilingTimedScope(std::string_view path) noexcept {
    ProfilingSession *s = current_session();
    if (s) {
      s->push_timed_scope(path);
      active_ = true;
    }
  }

  ProfilingTimedScope(const ProfilingTimedScope &) = delete;
  ProfilingTimedScope &operator=(const ProfilingTimedScope &) = delete;

  ~ProfilingTimedScope() noexcept {
    if (!active_) return;
    ProfilingSession *s = current_session();
    if (s) s->pop_timed_scope();
  }
};

/**
 * @brief Same stack semantics as ProfilingTimedScope, but you may call stop() early
 * or restart(path) to reuse one object without nested { } blocks. Still LIFO with
 *        other scopes on the active session.
 */
class ProfilingManualScope {
  bool active_{false};

  void pop_if_active() noexcept {
    if (!active_) return;
    active_ = false;
    ProfilingSession *s = current_session();
    if (s) s->pop_timed_scope();
  }

public:
  ProfilingManualScope() noexcept = default;

  explicit ProfilingManualScope(std::string_view path) noexcept {
    ProfilingSession *s = current_session();
    if (s) {
      s->push_timed_scope(path);
      active_ = true;
    }
  }

  ProfilingManualScope(const ProfilingManualScope &) = delete;
  ProfilingManualScope &operator=(const ProfilingManualScope &) = delete;

  ProfilingManualScope(ProfilingManualScope &&o) noexcept : active_(o.active_) {
    o.active_ = false;
  }

  ProfilingManualScope &operator=(ProfilingManualScope &&o) noexcept {
    if (this == &o) return *this;
    pop_if_active();
    active_ = o.active_;
    o.active_ = false;
    return *this;
  }

  ~ProfilingManualScope() noexcept { pop_if_active(); }

  [[nodiscard]] bool is_active() const noexcept { return active_; }

  /// End the current interval now (idempotent). Safe if session was cleared.
  void stop() noexcept { pop_if_active(); }

  /// stop() then push @p path; reuses one variable for sequential regions.
  void restart(std::string_view path) noexcept {
    pop_if_active();
    ProfilingSession *s = current_session();
    if (s) {
      s->push_timed_scope(path);
      active_ = true;
    }
  }

  /// Begin timing @p path if currently inactive (no-op if already active).
  void start(std::string_view path) noexcept {
    if (active_) return;
    ProfilingSession *s = current_session();
    if (s) {
      s->push_timed_scope(path);
      active_ = true;
    }
  }
};

} // namespace profiling
} // namespace pfc

#endif // PFC_KERNEL_PROFILING_REGION_SCOPE_HPP
