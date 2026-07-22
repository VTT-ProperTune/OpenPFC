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
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>

namespace pfc::halo {

namespace {

/**
 * @brief Compute padded extent `n + 2*hw` with overflow check.
 *
 * @param n Base extent.
 * @param hw Halo width (non-negative).
 * @return Padded extent.
 * @throws std::overflow_error if `n + 2*hw` would overflow `int`.
 * @throws std::invalid_argument if `hw < 0`.
 */
inline int checked_padded_extent(int n, int hw) {
  if (hw < 0) {
    throw std::invalid_argument("padded extent: halo width must be non-negative (got " +
                                std::to_string(hw) + ")");
  }
  // Use long long to detect overflow in addition
  const long long result = static_cast<long long>(n) + 2LL * static_cast<long long>(hw);
  if (result > static_cast<long long>(std::numeric_limits<int>::max()) ||
      result < static_cast<long long>(std::numeric_limits<int>::min())) {
    throw std::overflow_error("padded extent overflow in create_padded_face_types_6: " +
                              std::to_string(n) + " + 2*" + std::to_string(hw) +
                              " exceeds int range");
  }
  return static_cast<int>(result);
}

} // anonymous namespace

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
 *
 * When `halo_width > 0`, each owned extent must be `>= halo_width` so the
 * owned send slab fits inside the core. Same per-axis fit rule as
 * non-padded `create_face_types_6`. The LocalField `> 2*hw` interior rule
 * does **not** apply here (`PaddedBrick` semantics).
 *
 * @throws std::invalid_argument if `halo_width < 0`, owned extents are
 *         non-positive, or (when `halo_width > 0`) any owned axis is
 *         `< halo_width`
 */
inline std::array<FaceTypes, 6>
create_padded_face_types_6(int nx, int ny, int nz, int halo_width,
                           MPI_Datatype element_type) {
  if (halo_width < 0) {
    throw std::invalid_argument(
        "pfc::halo::create_padded_face_types_6: halo_width must be "
        "non-negative (got " +
        std::to_string(halo_width) + ")");
  }
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument(
        "pfc::halo::create_padded_face_types_6: owned extents must be "
        "positive (got " +
        std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) +
        ")");
  }
  if (halo_width > 0 &&
      (nx < halo_width || ny < halo_width || nz < halo_width)) {
    throw std::invalid_argument(
        std::string("pfc::halo::create_padded_face_types_6: owned extents ") +
        std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) +
        " cannot host halo_width=" + std::to_string(halo_width) +
        " owned send slabs (need >= halo_width points per owned dimension)");
  }

  const int hw = halo_width;
  const int nxp = checked_padded_extent(nx, hw);
  const int nyp = checked_padded_extent(ny, hw);
  const int nzp = checked_padded_extent(nz, hw);

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
