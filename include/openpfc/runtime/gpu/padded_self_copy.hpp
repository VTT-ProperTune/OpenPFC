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

namespace pfc::gpu::detail {

inline constexpr int kMaxSelfCopyFields = 8;

struct PaddedSelfCopy {
  double *pads[kMaxSelfCopyFields] = {};
  int nfields = 0;
  int nslab = 0;      ///< 1 or 2 slab pairs (faces) in this launch
  int src[2][3] = {}; ///< slab origin to read (padded coordinates)
  int dst[2][3] = {}; ///< slab origin to write (padded coordinates)
  int ext[2][3] = {}; ///< slab extents
  int nxp = 0;        ///< padded x extent
  int nyp = 0;        ///< padded y extent
};

} // namespace pfc::gpu::detail
