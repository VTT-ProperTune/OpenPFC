// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_face_layout.hpp
 * @brief Per-face halo element counts and allocation (separated halo layout)
 *
 * @details
 * Face order matches `HaloExchanger` / `create_face_types_6`:
 * 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
 *
 * Element counts match `halo::create_recv_halo(...).size()` for each direction.
 *
 * @see separated_halo_exchange.hpp
 * @see docs/halo_exchange.md
 */

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc {
namespace halo {

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
  const std::array<Int3, 6> dirs = {{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
                                     {0, 0, 1}, {0, 0, -1}}};
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
  out.counts[1] = out.counts[0];          // -X
  out.counts[2] = static_cast<size_t>(nx) * static_cast<size_t>(hw) *
                  static_cast<size_t>(nz); // +Y
  out.counts[3] = out.counts[2];          // -Y
  out.counts[4] = static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                  static_cast<size_t>(hw); // +Z
  out.counts[5] = out.counts[4];            // -Z
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
std::array<std::vector<T>, 6> allocate_face_halos(const decomposition::Decomposition &decomp,
                                                  int rank, int halo_width) {
  return allocate_face_halos<T>(face_halo_counts(decomp, rank, halo_width));
}

} // namespace halo
} // namespace pfc
