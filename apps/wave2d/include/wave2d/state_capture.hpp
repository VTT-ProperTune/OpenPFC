// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file state_capture.hpp
 * @brief Wave2D coupled displacement/velocity capture/restore adapters.
 *
 * @details
 * Captures accepted owned @c u and @c v into a @c PersistentState. Restore
 * validates **both** field payloads fully before mutating either destination.
 * Halo rings are excluded. Driver-owned Time is never a component payload.
 *
 * @see openpfc/kernel/checkpoint/state_capture.hpp
 * @see docs/development/checkpoint_state_capture.md
 */

#include <cstddef>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <openpfc/kernel/checkpoint/state_capture.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

namespace wave2d {

/// Stable field id for Wave2D displacement.
inline constexpr std::string_view kDisplacementFieldId = "wave2d.u";
/// Stable field id for Wave2D velocity.
inline constexpr std::string_view kVelocityFieldId = "wave2d.v";

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

[[nodiscard]] inline const pfc::checkpoint::FieldPayload *
find_field(const pfc::checkpoint::PersistentState &state, std::string_view id) {
  for (const auto &f : state.fields) {
    if (f.field_id == id) {
      return &f;
    }
  }
  return nullptr;
}

} // namespace detail

/**
 * @brief Capture coupled @c u / @c v into a @ref PersistentState.
 */
[[nodiscard]] inline pfc::checkpoint::PersistentState
capture_uv(std::span<const double> u, std::span<const double> v,
           pfc::types::Int3 extents,
           std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  pfc::checkpoint::PersistentState state;
  state.fields.push_back(pfc::checkpoint::capture_field<double>(
      kDisplacementFieldId, extents, u, pfc::checkpoint::kFieldPayloadFormatVersion,
      decomp));
  state.fields.push_back(pfc::checkpoint::capture_field<double>(
      kVelocityFieldId, extents, v, pfc::checkpoint::kFieldPayloadFormatVersion,
      std::move(decomp)));
  state.components.push_back(pfc::checkpoint::empty_component_payload("euler"));
  return state;
}

/**
 * @brief Restore @c u and @c v after validating both payloads fully.
 *
 * If either payload would reject, neither destination is mutated.
 */
[[nodiscard]] inline pfc::checkpoint::RestoreOutcome
restore_uv(const pfc::checkpoint::PersistentState &state, std::span<double> u_dest,
           std::span<double> v_dest, pfc::types::Int3 extents,
           std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const auto *u_payload = detail::find_field(state, kDisplacementFieldId);
  const auto *v_payload = detail::find_field(state, kVelocityFieldId);
  if (u_payload == nullptr || v_payload == nullptr) {
    return pfc::checkpoint::make_restore_rejected(
        pfc::checkpoint::RestoreError::FieldIdMismatch);
  }

  const std::size_t nbytes = pfc::checkpoint::field_expected_nbytes(
      pfc::checkpoint::FieldDtype::Float64, extents);

  // Pre-validate both without writing either destination.
  std::vector<std::byte> u_tmp(nbytes);
  std::vector<std::byte> v_tmp(nbytes);
  const auto u_check = pfc::checkpoint::restore_field(
      *u_payload, kDisplacementFieldId, pfc::checkpoint::FieldDtype::Float64,
      extents, std::span<std::byte>(u_tmp), decomp);
  if (!u_check.ok) {
    return u_check;
  }
  const auto v_check = pfc::checkpoint::restore_field(
      *v_payload, kVelocityFieldId, pfc::checkpoint::FieldDtype::Float64, extents,
      std::span<std::byte>(v_tmp), std::move(decomp));
  if (!v_check.ok) {
    return v_check;
  }

  // Capacity check against caller buffers (restore_field already checked tmps).
  if (u_dest.size_bytes() < nbytes || v_dest.size_bytes() < nbytes) {
    return pfc::checkpoint::make_restore_rejected(
        pfc::checkpoint::RestoreError::BufferTooSmall);
  }

  // Both validated: commit to caller buffers.
  if (nbytes > 0) {
    std::memcpy(u_dest.data(), u_tmp.data(), nbytes);
    std::memcpy(v_dest.data(), v_tmp.data(), nbytes);
  }
  return pfc::checkpoint::make_restore_ok();
}

/**
 * @brief Capture owned cores from @c PaddedBrick pair (halos excluded).
 */
[[nodiscard]] inline pfc::checkpoint::PersistentState
capture_uv(const pfc::field::PaddedBrick<double> &u,
           const pfc::field::PaddedBrick<double> &v,
           std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const pfc::types::Int3 extents{u.nx(), u.ny(), u.nz()};
  const auto u_owned = detail::pack_owned_core(u);
  const auto v_owned = detail::pack_owned_core(v);
  return capture_uv(std::span<const double>(u_owned),
                    std::span<const double>(v_owned), extents, std::move(decomp));
}

/**
 * @brief Restore into owned cells of two bricks; reject leaves both unchanged.
 */
[[nodiscard]] inline pfc::checkpoint::RestoreOutcome
restore_uv(const pfc::checkpoint::PersistentState &state,
           pfc::field::PaddedBrick<double> &u, pfc::field::PaddedBrick<double> &v,
           std::optional<pfc::checkpoint::DecompositionMeta> decomp = std::nullopt) {
  const pfc::types::Int3 extents{u.nx(), u.ny(), u.nz()};
  const std::size_t n = static_cast<std::size_t>(extents[0]) *
                        static_cast<std::size_t>(extents[1]) *
                        static_cast<std::size_t>(extents[2]);
  std::vector<double> u_buf(n);
  std::vector<double> v_buf(n);
  const auto outcome =
      restore_uv(state, std::span<double>(u_buf), std::span<double>(v_buf), extents,
                 std::move(decomp));
  if (outcome.ok) {
    detail::unpack_owned_core(u, u_buf);
    detail::unpack_owned_core(v, v_buf);
  }
  return outcome;
}

} // namespace wave2d
