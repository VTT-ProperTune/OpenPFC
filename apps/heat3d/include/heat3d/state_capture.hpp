// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file state_capture.hpp
 * @brief Heat3D scalar temperature field capture/restore adapters.
 *
 * @details
 * Packs accepted owned temperature cells into @c pfc::checkpoint payloads.
 * Halo rings are excluded (recomputable via exchange). Driver-owned Time is
 * never stuffed into component payloads.
 *
 * Supports both @c pfc::data::Field and legacy @c pfc::field::PaddedBrick
 * storage containers, providing overloads for @c capture_u and @c restore_u
 * that work seamlessly with either layout.
 *
 * @see openpfc/kernel/checkpoint/state_capture.hpp
 * @see docs/development/checkpoint_state_capture.md
 */

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <openpfc/kernel/checkpoint/state_capture.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace heat3d {

/// Stable field id for the Heat3D temperature field.
inline constexpr std::string_view kTemperatureFieldId = "heat3d.u";

/**
 * @brief Capture owned contiguous temperature values.
 */
[[nodiscard]] inline pfc::checkpoint::FieldPayload
capture_u(std::span<const double> owned_u, pfc::types::Int3 extents,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  return pfc::checkpoint::capture_field<double>(
      kTemperatureFieldId, extents, owned_u,
      pfc::checkpoint::kFieldPayloadFormatVersion, std::move(decomp));
}

/**
 * @brief Restore temperature into a caller-owned contiguous buffer.
 */
[[nodiscard]] inline pfc::checkpoint::RestoreOutcome
restore_u(const pfc::checkpoint::FieldPayload &payload, std::span<double> dest,
          pfc::types::Int3 extents,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  return pfc::checkpoint::restore_field(
      payload, kTemperatureFieldId, pfc::checkpoint::FieldDtype::Float64, extents,
      std::as_writable_bytes(dest), std::move(decomp));
}

namespace detail {

[[nodiscard]] inline std::vector<double>
pack_owned_core(const pfc::field::PaddedBrick<double> &brick) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  std::vector<double> owned(static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz));
  std::size_t n = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        owned[n++] = brick(i, j, k);
      }
    }
  }
  return owned;
}

[[nodiscard]] inline std::vector<double>
pack_owned_core(const pfc::data::Field<double, pfc::HostSpace> &field) {
  const auto size = field.local_size();
  const int nx = size[0];
  const int ny = size[1];
  const int nz = size[2];
  std::vector<double> owned(static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz));
  std::size_t n = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        owned[n++] = field(i, j, k);
      }
    }
  }
  return owned;
}

inline void unpack_owned_core(pfc::field::PaddedBrick<double> &brick,
                              std::span<const double> owned) {
  const int nx = brick.nx();
  const int ny = brick.ny();
  const int nz = brick.nz();
  std::size_t n = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        brick(i, j, k) = owned[n++];
      }
    }
  }
}

inline void unpack_owned_core(pfc::data::Field<double, pfc::HostSpace> &field,
                              std::span<const double> owned) {
  const auto size = field.local_size();
  const int nx = size[0];
  const int ny = size[1];
  const int nz = size[2];
  std::size_t n = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        field(i, j, k) = owned[n++];
      }
    }
  }
}

} // namespace detail

/**
 * @brief Capture owned cells from a @c PaddedBrick (halo excluded).
 */
[[nodiscard]] inline pfc::checkpoint::FieldPayload
capture_u(const pfc::field::PaddedBrick<double> &brick,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const pfc::types::Int3 extents{brick.nx(), brick.ny(), brick.nz()};
  const auto owned = detail::pack_owned_core(brick);
  return capture_u(std::span<const double>(owned), extents, std::move(decomp));
}

/**
 * @brief Restore into owned cells of a @c PaddedBrick (halo untouched on
 *        success; unchanged on reject).
 */
[[nodiscard]] inline pfc::checkpoint::RestoreOutcome
restore_u(const pfc::checkpoint::FieldPayload &payload,
          pfc::field::PaddedBrick<double> &brick,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const pfc::types::Int3 extents{brick.nx(), brick.ny(), brick.nz()};
  const std::size_t n = static_cast<std::size_t>(extents[0]) *
                        static_cast<std::size_t>(extents[1]) *
                        static_cast<std::size_t>(extents[2]);
  std::vector<double> owned(n);
  // Validate into a temporary first so a reject leaves the brick untouched.
  const auto outcome =
      restore_u(payload, std::span<double>(owned), extents, std::move(decomp));
  if (outcome.ok) {
    detail::unpack_owned_core(brick, owned);
  }
  return outcome;
}

/**
 * @brief Capture owned cells from a @c pfc::data::Field (halo excluded).
 */
[[nodiscard]] inline pfc::checkpoint::FieldPayload
capture_u(const pfc::data::Field<double, pfc::HostSpace> &field,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const auto size = field.local_size();
  const pfc::types::Int3 extents{size[0], size[1], size[2]};
  const auto owned = detail::pack_owned_core(field);
  return capture_u(std::span<const double>(owned), extents, std::move(decomp));
}

/**
 * @brief Restore into owned cells of a @c pfc::data::Field (halo untouched on
 *        success; unchanged on reject).
 */
[[nodiscard]] inline pfc::checkpoint::RestoreOutcome
restore_u(const pfc::checkpoint::FieldPayload &payload,
          pfc::data::Field<double, pfc::HostSpace> &field,
          std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const auto size = field.local_size();
  const pfc::types::Int3 extents{size[0], size[1], size[2]};
  const std::size_t n = static_cast<std::size_t>(extents[0]) *
                        static_cast<std::size_t>(extents[1]) *
                        static_cast<std::size_t>(extents[2]);
  std::vector<double> owned(n);
  // Validate into a temporary first so a reject leaves the field untouched.
  const auto outcome =
      restore_u(payload, std::span<double>(owned), extents, std::move(decomp));
  if (outcome.ok) {
    detail::unpack_owned_core(field, owned);
  }
  return outcome;
}

/**
 * @brief Capture temperature plus an empty Explicit-Euler component payload.
 *
 * Does not include driver-owned @c Time.
 */
[[nodiscard]] inline pfc::checkpoint::PersistentState capture_persistent_state(
    std::span<const double> owned_u, pfc::types::Int3 extents,
    std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  pfc::checkpoint::PersistentState state;
  state.fields.push_back(capture_u(owned_u, extents, std::move(decomp)));
  state.components.push_back(pfc::checkpoint::empty_component_payload("euler"));
  return state;
}

} // namespace heat3d
