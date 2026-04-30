// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_mpi_types.hpp
 * @brief MPI derived types for zero-copy face halo exchange
 *
 * @details
 * Builds MPI_Type_create_subarray for each of the 6 face regions (send and
 * recv) so MPI can read/write directly from the field buffer. Row-major
 * [nx, ny, nz]; x varies fastest (linear index z*nx*ny + y*nx + x). For
 * MPI_ORDER_C, dimension 0 is slowest (z), then y, then x. RAII wrapper frees
 * types in destructor.
 *
 * @see docs/halo_exchange.md
 */

#pragma once

#include <array>
#include <mpi.h>
#include <stdexcept>

#include <openpfc/kernel/data/world_types.hpp>

namespace pfc::halo {

using Int3 = pfc::types::Int3;

/**
 * @brief RAII holder for MPI_Datatype (calls MPI_Type_free in destructor)
 */
struct MPI_Type_guard {
  MPI_Datatype type = MPI_DATATYPE_NULL;
  MPI_Type_guard() = default;
  explicit MPI_Type_guard(MPI_Datatype t) : type(t) {}
  ~MPI_Type_guard() {
    if (type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&type);
    }
  }
  MPI_Type_guard(MPI_Type_guard &&other) noexcept : type(other.type) {
    other.type = MPI_DATATYPE_NULL;
  }
  MPI_Type_guard &operator=(MPI_Type_guard &&other) noexcept {
    if (this != &other) {
      if (type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&type);
      }
      type = other.type;
      other.type = MPI_DATATYPE_NULL;
    }
    return *this;
  }
  MPI_Type_guard(const MPI_Type_guard &) = delete;
  MPI_Type_guard &operator=(const MPI_Type_guard &) = delete;

  [[nodiscard]] MPI_Datatype get() const { return type; }
};

/**
 * @brief Create MPI_Datatype for a 3D face (subarray) in row-major [nx,ny,nz]
 * @param nx, ny, nz Full array dimensions
 * @param start_x, start_y, start_z Start of subarray (0-based)
 * @param size_x, size_y, size_z Extent of subarray
 * @param element_type MPI_Datatype for one element (e.g. MPI_DOUBLE)
 * @return RAII guard holding committed type
 */
[[nodiscard]] inline MPI_Type_guard
create_face_type(int nx, int ny, int nz, int start_x, int start_y, int start_z,
                 int size_x, int size_y, int size_z, MPI_Datatype element_type) {
  const int ndims = 3;
  // Field layout: x fastest, then y, then z (see finite_difference.hpp). MPI_ORDER_C
  // lists dimensions slowest → fastest, i.e. z, y, x.
  std::array<int, 3> sizes = {nz, ny, nx};
  std::array<int, 3> subsizes = {size_z, size_y, size_x};
  std::array<int, 3> starts = {start_z, start_y, start_x};

  MPI_Datatype dt;
  int err = MPI_Type_create_subarray(ndims, sizes.data(), subsizes.data(),
                                     starts.data(), MPI_ORDER_C, element_type, &dt);
  if (err != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Type_create_subarray failed");
  }
  err = MPI_Type_commit(&dt);
  if (err != MPI_SUCCESS) {
    MPI_Type_free(&dt);
    throw std::runtime_error("MPI_Type_commit failed");
  }
  return MPI_Type_guard(dt);
}

/**
 * @brief Send and recv face types for one direction (e.g. +X)
 */
struct FaceTypes {
  MPI_Type_guard send_type;
  MPI_Type_guard recv_type;
};

/**
 * @brief Create send and recv face types for all 6 face directions
 *
 * Field layout: row-major [nx, ny, nz], boundary layers are the halo (no
 * extra padding). For direction +X: send right face, recv into left face.
 */
inline std::array<FaceTypes, 6> create_face_types_6(int nx, int ny, int nz,
                                                    int halo_width,
                                                    MPI_Datatype element_type) {
  std::array<FaceTypes, 6> out = {};
  // Order: +X, -X, +Y, -Y, +Z, -Z
  struct FaceDirSpec {
    int send_sx, send_sy, send_sz, send_ox, send_oy, send_oz;
    int recv_sx, recv_sy, recv_sz, recv_ox, recv_oy, recv_oz;
  };
  const std::array<FaceDirSpec, 6> dirs = {{
      {halo_width, ny, nz, nx - halo_width, 0, 0, halo_width, ny, nz, 0, 0, 0}, // +X
      {halo_width, ny, nz, 0, 0, 0, halo_width, ny, nz, nx - halo_width, 0, 0}, // -X
      {nx, halo_width, nz, 0, ny - halo_width, 0, nx, halo_width, nz, 0, 0, 0}, // +Y
      {nx, halo_width, nz, 0, 0, 0, nx, halo_width, nz, 0, ny - halo_width, 0}, // -Y
      {nx, ny, halo_width, 0, 0, nz - halo_width, nx, ny, halo_width, 0, 0, 0}, // +Z
      {nx, ny, halo_width, 0, 0, 0, nx, ny, halo_width, 0, 0, nz - halo_width}, // -Z
  }};

  for (int d = 0; d < 6; ++d) {
    const auto di = static_cast<size_t>(d);
    const auto &q = dirs[di];
    out[di].send_type =
        create_face_type(nx, ny, nz, q.send_ox, q.send_oy, q.send_oz, q.send_sx,
                         q.send_sy, q.send_sz, element_type);
    out[di].recv_type =
        create_face_type(nx, ny, nz, q.recv_ox, q.recv_oy, q.recv_oz, q.recv_sx,
                         q.recv_sy, q.recv_sz, element_type);
  }
  return out;
}

} // namespace pfc::halo
