// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file payloads.hpp
 * @brief Serialization-agnostic versioned field and component checkpoint payloads.
 *
 * @details
 * These types are value carriers for a future checkpoint manager. They do not
 * prescribe a file format. Field payloads describe accepted solution fields;
 * component payloads hold irreducible integrator/controller cross-step state.
 *
 * Exclusions (not represented here): stage buffers, FFT plans, operator caches,
 * halo rings, and driver-owned @c Time / step counters.
 *
 * @see openpfc/kernel/checkpoint/state_capture.hpp
 * @see docs/development/checkpoint_state_capture.md
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <openpfc/kernel/data/types.hpp>

namespace pfc::checkpoint {

/// Format version for @ref FieldPayload layout/semantics.
inline constexpr std::uint32_t kFieldPayloadFormatVersion = 1;

/// Format version for @ref ComponentPayload layout/semantics.
inline constexpr std::uint32_t kComponentPayloadFormatVersion = 1;

/**
 * @brief Numeric element type of a field payload's contiguous bytes.
 */
enum class FieldDtype : std::uint8_t {
  Float64 = 1,
  Complex128 = 2,
};

/**
 * @brief Memory layout / iteration order of field bytes.
 *
 * OpenPFC uses row-major storage with the x-axis fastest (same as
 * @c PaddedBrick / @c LocalField / FD stack).
 */
enum class CoordinateOrder : std::uint8_t {
  XFastest = 1,
};

/**
 * @brief Map a C++ element type to @ref FieldDtype.
 * @tparam T Element type (@c double or @c std::complex<double>).
 */
template <typename T> constexpr FieldDtype dtype_of() noexcept {
  static_assert(sizeof(T) == 0,
                "pfc::checkpoint::dtype_of: unsupported element type");
  return FieldDtype::Float64;
}

template <> constexpr FieldDtype dtype_of<double>() noexcept {
  return FieldDtype::Float64;
}

template <> constexpr FieldDtype dtype_of<std::complex<double>>() noexcept {
  return FieldDtype::Complex128;
}

/**
 * @brief Optional MPI decomposition descriptors attached to a field payload.
 *
 * When present on restore, the destination must match exactly.
 */
struct DecompositionMeta {
  int rank_count{};
  int rank{};
  pfc::types::Int3 global_extents{};
  pfc::types::Int3 local_extents{};
  pfc::types::Int3 local_offset{};

  friend bool operator==(const DecompositionMeta &,
                         const DecompositionMeta &) = default;
};

/**
 * @brief Versioned named field payload (accepted solution values + metadata).
 *
 * Bytes are a contiguous x-fastest copy of owned cells only (no halo ring).
 */
struct FieldPayload {
  std::string field_id;
  FieldDtype dtype{};
  pfc::types::Int3 extents{};
  CoordinateOrder coordinate_order{CoordinateOrder::XFastest};
  std::uint32_t version{kFieldPayloadFormatVersion};
  std::optional<DecompositionMeta> decomposition;
  std::vector<std::byte> bytes;
};

/**
 * @brief Versioned named component payload for irreducible cross-step state.
 *
 * Empty @c bytes means the component has no irreducible state (e.g. Explicit
 * Euler). Do not store driver-owned @c Time here.
 */
struct ComponentPayload {
  std::string component_id;
  std::uint32_t version{kComponentPayloadFormatVersion};
  std::vector<std::byte> bytes;
};

/**
 * @brief Bundle of field and component payloads for one persistent snapshot.
 */
struct PersistentState {
  std::vector<FieldPayload> fields;
  std::vector<ComponentPayload> components;
};

} // namespace pfc::checkpoint
