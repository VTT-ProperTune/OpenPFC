// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file integration_count.hpp
 * @brief Diagnostic count of local J2 and FCC constitutive integrations.
 *
 * Each entry to the J2 return map or the FCC slip update adds one.
 * Applying a tangent that those updates already returned does not.
 * Saint-Venant–Kirchhoff has no integration and does not add one.
 * The counter is a relaxed atomic so a test can read it. It is not a
 * material result.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace pfc::finite_strain {

namespace detail {

inline std::atomic<std::uint64_t> constitutive_integration_count{0};

inline void note_constitutive_integration() noexcept {
  constitutive_integration_count.fetch_add(1, std::memory_order_relaxed);
}

} // namespace detail

[[nodiscard]] inline std::uint64_t constitutive_integrations() noexcept {
  return detail::constitutive_integration_count.load(std::memory_order_relaxed);
}

inline void reset_constitutive_integrations() noexcept {
  detail::constitutive_integration_count.store(0, std::memory_order_relaxed);
}

} // namespace pfc::finite_strain
