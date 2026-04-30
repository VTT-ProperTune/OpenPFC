// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_halo_mpi_types.hpp
 * @brief MPI derived types for in-place face halo exchange on a
 *        **halo-padded** brick.
 *
 * @details
 * Sibling of `halo_mpi_types.hpp`. The original `create_face_types_6`
 * documents itself as "no extra padding" — its send subarrays read the
 * outermost owned cells and its recv subarrays overwrite the same
 * outermost owned cells with neighbor data, so it cannot fill a true
 * ghost ring.
 *
 * `create_padded_face_types_6` builds 6 send/recv face subarrays
 * over a **padded** buffer of outer extents
 * `(nx + 2*hw, ny + 2*hw, nz + 2*hw)` (row-major, x fastest):
 *
 *   - The **owned** core lives at `[hw, hw+nx) x [hw, hw+ny) x [hw, hw+nz)`.
 *   - The **+X halo ring** lives at `[hw+nx, hw+nx+hw) x [hw, hw+ny) x [hw, hw+nz)`,
 *     and so on for the other five faces.
 *
 * For each direction the **send** type reads the `hw`-thick slab on
 * that side of the owned region; the **recv** type writes the
 * `hw`-thick halo ring on that side. In the orthogonal axes both
 * subarrays span the **owned** extent (so corner halo cells are not
 * filled — they are unnecessary for face-only stencils such as the 7-
 * or 27-point Laplacian, and a face-only exchange is one MPI message
 * per neighbor).
 *
 * The element layout matches `pfc::field::PaddedBrick<T>` exactly:
 * the same `idx(i, j, k)` formula computes the offsets into the same
 * buffer, so the resulting MPI types are zero-copy when used with
 * `brick.data()` / `brick.size()`.
 *
 * @see pfc::field::PaddedBrick — the matching data layout.
 * @see pfc::halo::create_face_types_6 — the original "no padding"
 *      face-type builder used by `pfc::HaloExchanger`.
 */

#include <array>
#include <mpi.h>

#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>

namespace pfc::halo {

/**
 * @brief Create send/recv MPI face subarrays for the **padded** brick layout.
 *
 * @param nx,ny,nz   Owned extents (the brick's `nx()/ny()/nz()`).
 * @param halo_width Halo ring thickness `hw` on every side.
 * @param element_type MPI datatype of one element (e.g. `MPI_DOUBLE`).
 * @return Six send/recv `FaceTypes` in canonical order +X, -X, +Y, -Y, +Z, -Z.
 *
 * Each direction's send subarray covers the `hw`-thick owned slab
 * adjacent to that face; its recv subarray covers the matching
 * `hw`-thick halo slab. Both subarrays use the **owned** extent in the
 * orthogonal axes; corners are not filled by face-only exchange.
 *
 * Built on the existing `create_face_type(...)` helper, just with the
 * **padded** outer extents `(nx + 2hw, ny + 2hw, nz + 2hw)` and the
 * appropriate `(hw, ...)` offsets baked into the start vectors.
 */
inline std::array<FaceTypes, 6>
create_padded_face_types_6(int nx, int ny, int nz, int halo_width,
                           MPI_Datatype element_type) {
  const int hw = halo_width;
  const int nxp = nx + 2 * hw;
  const int nyp = ny + 2 * hw;
  const int nzp = nz + 2 * hw;

  std::array<FaceTypes, 6> out = {};

  struct FaceDirSpec {
    int send_sx, send_sy, send_sz, send_ox, send_oy, send_oz;
    int recv_sx, recv_sy, recv_sz, recv_ox, recv_oy, recv_oz;
  };

  const std::array<FaceDirSpec, 6> dirs = {{
      // +X : send last hw owned x-slab, recv into +X halo ring
      {hw, ny, nz, /*send start*/ nx, hw, hw, hw, ny, nz, /*recv start*/ nx + hw, hw,
       hw},
      // -X : send first hw owned x-slab, recv into -X halo ring
      {hw, ny, nz, /*send start*/ hw, hw, hw, hw, ny, nz, /*recv start*/ 0, hw, hw},
      // +Y : send last hw owned y-slab, recv into +Y halo ring
      {nx, hw, nz, /*send start*/ hw, ny, hw, nx, hw, nz, /*recv start*/ hw, ny + hw,
       hw},
      // -Y : send first hw owned y-slab, recv into -Y halo ring
      {nx, hw, nz, /*send start*/ hw, hw, hw, nx, hw, nz, /*recv start*/ hw, 0, hw},
      // +Z : send last hw owned z-slab, recv into +Z halo ring
      {nx, ny, hw, /*send start*/ hw, hw, nz, nx, ny, hw, /*recv start*/ hw, hw,
       nz + hw},
      // -Z : send first hw owned z-slab, recv into -Z halo ring
      {nx, ny, hw, /*send start*/ hw, hw, hw, nx, ny, hw, /*recv start*/ hw, hw, 0},
  }};

  for (int d = 0; d < 6; ++d) {
    const auto di = static_cast<std::size_t>(d);
    const auto &q = dirs[di];
    out[di].send_type =
        create_face_type(nxp, nyp, nzp, q.send_ox, q.send_oy, q.send_oz, q.send_sx,
                         q.send_sy, q.send_sz, element_type);
    out[di].recv_type =
        create_face_type(nxp, nyp, nzp, q.recv_ox, q.recv_oy, q.recv_oz, q.recv_sx,
                         q.recv_sy, q.recv_sz, element_type);
  }
  return out;
}

} // namespace pfc::halo
