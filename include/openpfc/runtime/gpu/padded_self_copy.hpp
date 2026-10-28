// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_self_copy.hpp
 * @brief Kernel parameter block for fused self-neighbor halo copies (internal).
 *
 * One launch copies both faces of one axis for up to `kMaxSelfCopyFields`
 * padded device buffers: `pads[f][dst] = pads[f][src]` over each slab. Shared
 * by the host halo code and the vendor kernels in `padded_halo_faces_gpu.inc`.
 */

#include <cstdint>
#include <type_traits>

namespace pfc::gpu {

/// Element types the device halo exchangers and their bitwise pack kernels
/// support. Each needs an explicit kernel instantiation in
/// `padded_halo_faces_gpu.inc` and an `exchange::detail::get_mpi_type`.
template <typename T>
inline constexpr bool is_halo_element_v =
    std::is_same_v<T, double> || std::is_same_v<T, float> ||
    std::is_same_v<T, std::uint32_t>;

} // namespace pfc::gpu

namespace pfc::gpu::detail {

inline constexpr int kMaxSelfCopyFields = 8;

template <typename T> struct PaddedSelfCopy {
  T *pads[kMaxSelfCopyFields] = {};
  int nfields = 0;
  int nslab = 0;      ///< 1 or 2 slab pairs (faces) in this launch
  int src[2][3] = {}; ///< slab origin to read (padded coordinates)
  int dst[2][3] = {}; ///< slab origin to write (padded coordinates)
  int ext[2][3] = {}; ///< slab extents
  int nxp = 0;        ///< padded x extent
  int nyp = 0;        ///< padded y extent
};

} // namespace pfc::gpu::detail
