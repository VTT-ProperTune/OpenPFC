// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file state_capture.hpp
 * @brief Capture and validate-before-mutate restore for checkpoint payloads.
 *
 * @details
 * Free functions copy accepted field/component state into
 * @ref FieldPayload / @ref ComponentPayload and restore into caller-owned
 * buffers only after all metadata and byte-length checks succeed.
 *
 * **Exclusions:** stage buffers (@c Workspace stages), FFT plans, operator
 * caches, and halo rings are not captured. Driver-owned @c Time / increment /
 * config identity are not component payloads.
 *
 * @see openpfc/kernel/checkpoint/payloads.hpp
 * @see docs/development/checkpoint_state_capture.md
 */

#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <openpfc/kernel/checkpoint/payloads.hpp>

namespace pfc::checkpoint {

/**
 * @brief Why a restore was rejected (destination left unchanged).
 */
enum class RestoreError {
  VersionMismatch,
  FieldIdMismatch,
  DtypeMismatch,
  ShapeMismatch,
  CoordinateOrderMismatch,
  DecompositionMismatch,
  BytesSizeMismatch,
  BufferTooSmall,
  ComponentIdMismatch,
};

/**
 * @brief Result of a restore attempt.
 *
 * Factories are named @c make_restore_ok / @c make_restore_rejected to avoid
 * GCC 11 name collisions with the @c ok data member.
 */
struct RestoreOutcome {
  bool ok{false};
  RestoreError error{};
};

[[nodiscard]] inline RestoreOutcome make_restore_ok() noexcept {
  return RestoreOutcome{.ok = true, .error = {}};
}

[[nodiscard]] inline RestoreOutcome
make_restore_rejected(RestoreError error) noexcept {
  return RestoreOutcome{.ok = false, .error = error};
}

/**
 * @brief Element size in bytes for a @ref FieldDtype.
 */
[[nodiscard]] constexpr std::size_t dtype_nbytes(FieldDtype dtype) noexcept {
  switch (dtype) {
  case FieldDtype::Float64: return 8;
  case FieldDtype::Complex128: return 16;
  }
  return 0;
}

/**
 * @brief Expected byte length for a field with the given dtype and extents.
 */
[[nodiscard]] inline std::size_t
field_expected_nbytes(FieldDtype dtype, pfc::types::Int3 extents) noexcept {
  const auto nx = static_cast<std::size_t>(extents[0]);
  const auto ny = static_cast<std::size_t>(extents[1]);
  const auto nz = static_cast<std::size_t>(extents[2]);
  return nx * ny * nz * dtype_nbytes(dtype);
}

/**
 * @brief Capture a contiguous owned field into a @ref FieldPayload.
 *
 * @tparam T Element type (@c double or @c std::complex<double>).
 * @param field_id Stable field identifier.
 * @param extents Owned grid extents (nx, ny, nz).
 * @param values Contiguous x-fastest values; size must equal @c nx*ny*nz.
 * @param version Payload format version (default @ref kFieldPayloadFormatVersion).
 * @param decomposition Optional MPI decomposition metadata.
 * @throws std::invalid_argument if @p values size does not match extents.
 */
template <typename T>
[[nodiscard]] FieldPayload
capture_field(std::string_view field_id, pfc::types::Int3 extents,
              std::span<const T> values,
              std::uint32_t version = kFieldPayloadFormatVersion,
              std::optional<DecompositionMeta> decomposition = std::nullopt) {
  const auto expected = static_cast<std::size_t>(extents[0]) *
                        static_cast<std::size_t>(extents[1]) *
                        static_cast<std::size_t>(extents[2]);
  if (values.size() != expected) {
    throw std::invalid_argument(
        "pfc::checkpoint::capture_field: values.size() must equal nx*ny*nz");
  }

  FieldPayload payload{
      .field_id = std::string(field_id),
      .dtype = dtype_of<T>(),
      .extents = extents,
      .coordinate_order = CoordinateOrder::XFastest,
      .version = version,
      .decomposition = std::move(decomposition),
      .bytes = {},
  };
  const auto nbytes = values.size() * sizeof(T);
  payload.bytes.resize(nbytes);
  if (nbytes > 0) {
    std::memcpy(payload.bytes.data(), values.data(), nbytes);
  }
  return payload;
}

/**
 * @brief Restore a field payload into @p destination after full validation.
 *
 * Validates version, id, dtype, extents, coordinate order, optional
 * decomposition, exact @c payload.bytes.size() == expected nbytes, and
 * destination capacity — **before** any write. On failure the destination
 * is left unchanged.
 */
[[nodiscard]] inline RestoreOutcome restore_field(
    const FieldPayload &payload, std::string_view expected_field_id,
    FieldDtype expected_dtype, pfc::types::Int3 expected_extents,
    std::span<std::byte> destination,
    std::optional<DecompositionMeta> expected_decomposition = std::nullopt) {
  if (payload.version != kFieldPayloadFormatVersion) {
    return make_restore_rejected(RestoreError::VersionMismatch);
  }
  if (payload.field_id != expected_field_id) {
    return make_restore_rejected(RestoreError::FieldIdMismatch);
  }
  if (payload.dtype != expected_dtype) {
    return make_restore_rejected(RestoreError::DtypeMismatch);
  }
  if (payload.extents != expected_extents) {
    return make_restore_rejected(RestoreError::ShapeMismatch);
  }
  if (payload.coordinate_order != CoordinateOrder::XFastest) {
    return make_restore_rejected(RestoreError::CoordinateOrderMismatch);
  }
  if (expected_decomposition.has_value()) {
    if (!payload.decomposition.has_value() ||
        *payload.decomposition != *expected_decomposition) {
      return make_restore_rejected(RestoreError::DecompositionMismatch);
    }
  }

  const std::size_t expected_nbytes =
      field_expected_nbytes(expected_dtype, expected_extents);
  if (payload.bytes.size() != expected_nbytes) {
    return make_restore_rejected(RestoreError::BytesSizeMismatch);
  }
  if (destination.size() < expected_nbytes) {
    return make_restore_rejected(RestoreError::BufferTooSmall);
  }

  if (expected_nbytes > 0) {
    std::memcpy(destination.data(), payload.bytes.data(), expected_nbytes);
  }
  return make_restore_ok();
}

/**
 * @brief Capture irreducible component bytes (may be empty).
 */
[[nodiscard]] inline ComponentPayload
capture_component(std::string_view id, std::span<const std::byte> bytes = {},
                  std::uint32_t version = kComponentPayloadFormatVersion) {
  ComponentPayload payload{
      .component_id = std::string(id),
      .version = version,
      .bytes = {},
  };
  payload.bytes.assign(bytes.begin(), bytes.end());
  return payload;
}

/**
 * @brief Empty component payload (e.g. Explicit Euler / Heun with no state).
 */
[[nodiscard]] inline ComponentPayload empty_component_payload(std::string_view id) {
  return capture_component(id, {});
}

/**
 * @brief Restore a component payload with validate-before-mutate semantics.
 *
 * Requires @c payload.bytes.size() == @p expected_nbytes (use 0 for empty
 * components). Destination is unchanged on any rejection.
 */
[[nodiscard]] inline RestoreOutcome
restore_component(const ComponentPayload &payload, std::string_view expected_id,
                  std::size_t expected_nbytes, std::span<std::byte> destination,
                  std::uint32_t expected_version = kComponentPayloadFormatVersion) {
  if (payload.version != expected_version) {
    return make_restore_rejected(RestoreError::VersionMismatch);
  }
  if (payload.component_id != expected_id) {
    return make_restore_rejected(RestoreError::ComponentIdMismatch);
  }
  if (payload.bytes.size() != expected_nbytes) {
    return make_restore_rejected(RestoreError::BytesSizeMismatch);
  }
  if (destination.size() < expected_nbytes) {
    return make_restore_rejected(RestoreError::BufferTooSmall);
  }
  if (expected_nbytes > 0) {
    std::memcpy(destination.data(), payload.bytes.data(), expected_nbytes);
  }
  return make_restore_ok();
}

} // namespace pfc::checkpoint
