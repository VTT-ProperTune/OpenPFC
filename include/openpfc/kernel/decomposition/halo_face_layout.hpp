// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_face_layout.hpp
 * @brief Per-face halo element counts, allocation, and structured-grid
 *        helpers for `pfc::SparseHaloExchanger`.
 *
 * @details
 * Face order matches `HaloExchanger` / `create_face_types_6`:
 * 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
 *
 * Element counts match `halo::create_recv_halo(...).size()` for each direction.
 *
 * This header also hosts:
 *   - `make_structured_halos<T>(...)` — build a `RemoteHalo` list for the
 *     standard structured (axis / edge / corner) exchange driven by a
 *     `HaloDirectionSet`. Tag layout follows the canonical opposite-slot
 *     scheme (`direction_to_canonical_tag`) so structured face exchanges
 *     are deterministic and rank-symmetric.
 *   - `copy_to_face_layout<T>(...)` — adapter that copies the recv buffers
 *     of a `SparseHaloExchanger` driven by `make_structured_halos` into the
 *     `std::array<std::vector<T>, 6>` layout that the templated
 *     periodic-separated FD Laplacians (`laplacian_periodic_separated<Order>`,
 *     `laplacian2d_xy_periodic_separated<Order>`) consume.
 *
 * @see sparse_halo_exchange.hpp for `RemoteHalo<T>` and `SparseHaloExchanger<T>`.
 * @see docs/concepts/halo_exchange.md for "which exchanger when".
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/decomposition/sparse_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc::halo {

using Int3 = pfc::types::Int3;

/**
 * @brief Six face halo sizes (element counts), order +X,-X,+Y,-Y,+Z,-Z
 */
struct FaceHaloCounts {
  std::array<size_t, 6> counts{};
};

/**
 * @brief Compute per-face recv halo sizes for a rank (matches halo patterns).
 */
inline FaceHaloCounts face_halo_counts(const decomposition::Decomposition &decomp,
                                       int rank, int halo_width) {
  FaceHaloCounts out{};
  const std::array<Int3, 6> dirs = {
      {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
  for (int i = 0; i < 6; ++i) {
    auto recv = create_recv_halo<backend::CpuTag>(decomp, rank, dirs[i], halo_width);
    out.counts[static_cast<size_t>(i)] = recv.size();
  }
  return out;
}

/**
 * @brief Analytic face sizes (must match create_recv_halo for interior boxes).
 */
inline FaceHaloCounts face_halo_counts_analytic(int nx, int ny, int nz, int hw) {
  FaceHaloCounts out{};
  out.counts[0] = static_cast<size_t>(hw) * static_cast<size_t>(ny) *
                  static_cast<size_t>(nz); // +X
  out.counts[1] = out.counts[0];           // -X
  out.counts[2] = static_cast<size_t>(nx) * static_cast<size_t>(hw) *
                  static_cast<size_t>(nz); // +Y
  out.counts[3] = out.counts[2];           // -Y
  out.counts[4] = static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                  static_cast<size_t>(hw); // +Z
  out.counts[5] = out.counts[4];           // -Z
  return out;
}

/**
 * @brief Allocate six contiguous face buffers with the given counts.
 */
template <typename T>
std::array<std::vector<T>, 6> allocate_face_halos(const FaceHaloCounts &c) {
  std::array<std::vector<T>, 6> out{};
  for (int i = 0; i < 6; ++i) {
    out[static_cast<size_t>(i)].assign(c.counts[static_cast<size_t>(i)], T{});
  }
  return out;
}

/**
 * @brief Convenience: counts from subworld size + allocate.
 */
template <typename T>
std::array<std::vector<T>, 6>
allocate_face_halos(const decomposition::Decomposition &decomp, int rank,
                    int halo_width) {
  return allocate_face_halos<T>(face_halo_counts(decomp, rank, halo_width));
}

/**
 * @brief Build a `RemoteHalo<T>` list for the standard structured exchange.
 *
 * @details
 * For every direction `d` in `dirs`:
 *
 *   1. Look up `peer = get_neighbor_rank(decomp, rank, d)`. With OpenPFC's
 *      periodic decomposition the lookup never returns `< 0`; an entry is
 *      still emitted when `peer == rank` (single-rank wrap) so MPI handles
 *      the local self-send cleanly.
 *   2. Build send indices via `create_send_halo<CpuTag>(decomp, rank, d, hw)`
 *      (cells on **this** rank's `+d` boundary, in row-major order).
 *   3. Build recv indices via `create_recv_halo<CpuTag>(decomp, rank, d, hw)`
 *      (cells on **this** rank's `-d` boundary). The recv buffer's data
 *      layout matches the SparseVector sort order — the row-major slab
 *      ordering the templated periodic-separated FD Laplacians consume.
 *   4. Tags: `send_tag = base_tag + direction_to_canonical_tag(d)`,
 *      `recv_tag = base_tag + direction_to_canonical_tag(-d)`. For axis
 *      faces this collapses to the legacy `(slot, opposite_slot)` scheme.
 *
 * `scatter_after_recv` is left at its default `false`. The caller normally
 * reads `recv_values.data()` directly (e.g. via `copy_to_face_layout` for
 * the array-of-six Laplacian input). Flip the flag in the returned vector
 * if a particular caller wants the exchanger to scatter into the local
 * field after the wait.
 *
 * @tparam T        Field element type (`double`, `float`, …).
 * @param decomp    Domain decomposition.
 * @param rank      This MPI rank.
 * @param halo_width Halo thickness (cells) along each active axis.
 * @param dirs      Direction set (defaults to `Axes3D()` — the historical
 *                  6-face exchange).
 * @param base_tag  Tag offset; the canonical tag scheme is added on top.
 *
 * @return Vector of `RemoteHalo<T>` ready to feed into `SparseHaloExchanger`.
 */
template <typename T>
[[nodiscard]] std::vector<RemoteHalo<T>> make_structured_halos(
    const decomposition::Decomposition &decomp, int rank, int halo_width,
    const HaloDirectionSet &dirs = presets::Axes3D(), int base_tag = 0) {
  std::vector<RemoteHalo<T>> out;
  out.reserve(dirs.size());

  for (const auto &d : dirs.dirs) {
    const int peer = decomposition::get_neighbor_rank(decomp, rank, d);
    if (peer < 0) {
      continue; // Non-periodic dead end; skip.
    }

    const auto send_idx_sv =
        create_send_halo<backend::CpuTag>(decomp, rank, d, halo_width);
    const auto recv_idx_sv =
        create_recv_halo<backend::CpuTag>(decomp, rank, d, halo_width);

    std::vector<std::size_t> send_idx = sparsevector::get_index(send_idx_sv);
    std::vector<std::size_t> recv_idx = sparsevector::get_index(recv_idx_sv);

    RemoteHalo<T> h;
    h.peer_rank = peer;
    h.send_tag = base_tag + direction_to_canonical_tag(d);
    h.recv_tag = base_tag + direction_to_canonical_tag(
                                HaloDirectionSet::Int3{-d[0], -d[1], -d[2]});
    h.send_values = core::SparseVector<backend::CpuTag, T>(std::move(send_idx));
    h.recv_values = core::SparseVector<backend::CpuTag, T>(std::move(recv_idx));
    h.scatter_after_recv = false;
    h.direction = d;
    out.push_back(std::move(h));
  }

  return out;
}

/**
 * @brief Copy recv buffers of a structured `SparseHaloExchanger` into a
 *        six-slot face layout (`std::array<std::vector<T>, 6>`).
 *
 * @details
 * Designed for callers who built `ex` with `make_structured_halos<T>(...)`
 * over an axis subset of the 26-direction set. For every entry whose
 * canonical tag corresponds to one of the 6 face slots (0:+X, 1:-X, 2:+Y,
 * 3:-Y, 4:+Z, 5:-Z), the recv buffer (`recv_values.data()`) is `memcpy`-ed
 * into `face_halos[slot]`. Slots without a corresponding entry are left
 * untouched (consistent with the inactive-direction behaviour of the
 * legacy face exchanger under a `HaloDirectionSet`).
 *
 * Non-axis entries (edges/corners) are silently ignored — the array-of-six
 * layout has no slots for them. If you exchange a `Full3D()` set and need
 * the diagonal data, read the relevant `RemoteHalo::recv_values.data()`
 * directly via `ex.halos()`.
 *
 * @throws std::runtime_error if a slot's `face_halos[slot].size()` is too
 *         small to hold the matching recv buffer.
 */
template <typename T>
void copy_to_face_layout(const SparseHaloExchanger<T> &ex,
                         std::array<std::vector<T>, 6> &face_halos) {
  for (const auto &h : ex.halos()) {
    // Use the direction hint that make_structured_halos stamps onto each
    // RemoteHalo. Non-axis directions (and entries with no associated
    // direction, e.g. user-built lists) return slot < 0 and are skipped.
    const int slot = direction_to_face_slot(h.direction);
    if (slot < 0) {
      continue;
    }
    auto &dst = face_halos[static_cast<std::size_t>(slot)];
    const std::size_t n = h.recv_values.size();
    if (n == 0) {
      continue;
    }
    if (dst.size() < n) {
      throw std::runtime_error("copy_to_face_layout: face buffer " +
                               std::to_string(slot) + " too small (need " +
                               std::to_string(n) + ", have " +
                               std::to_string(dst.size()) + ")");
    }
    std::memcpy(dst.data(), h.recv_values.data().data(), n * sizeof(T));
  }
}

} // namespace pfc::halo
