// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file owned_block_gather.hpp
 * @brief Gather and scatter the owned cells of a decomposed field.
 *
 * The transfer moves tightly packed owned cells. A field halo is not part of
 * the block: pack the owned interior first, then pass that span. Global
 * buffers are x-fastest over the domain index box that starts at the origin,
 * the same order as `Box3i::to_linear`.
 */

#pragma once

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::decomposition {

/// Owned cells packed x-fastest, in the same order as `box.to_linear`.
struct OwnedBlockView {
  Box3i box{};
  std::span<const double> values{};
};

/// Writable owned cells packed x-fastest, in the same order as `box.to_linear`.
struct OwnedBlockBuffer {
  Box3i box{};
  std::span<double> values{};
};

/**
 * @brief Gather owned blocks onto `root`.
 *
 * Every rank passes the same decomposition, communicator, root and field
 * count. `blocks[f].box` is that rank's owned box and `blocks[f].values`
 * holds `box.count()` doubles. On `root` the result has one buffer per field,
 * each of length `Nx * Ny * Nz`. Every other rank receives an empty vector.
 *
 * @throws std::invalid_argument on a null communicator, a root outside the
 *         communicator, a rank-count or field-count mismatch, a box that is
 *         not the rank's owned box, or a span whose length is not the box
 *         count. Peers of a failing rank throw as well.
 * @throws std::overflow_error when a block count or the gathered displacement
 *         sum does not fit in an MPI `int`.
 */
[[nodiscard]] std::vector<std::vector<double>>
gather_owned_blocks(const Decomposition &decomposition, MPI_Comm comm, int root,
                    std::span<const OwnedBlockView> blocks);

/**
 * @brief Scatter global buffers from `root` into owned blocks.
 *
 * `global` on `root` is the layout `gather_owned_blocks` returns. Every other
 * rank passes an empty vector. Each `blocks[f].values` span receives that
 * rank's owned cells.
 *
 * @throws std::invalid_argument on the same mismatches as
 *         `gather_owned_blocks`, and when `root` does not hold one full-domain
 *         buffer per field or another rank passes a non-empty `global`.
 * @throws std::overflow_error when a block count does not fit in an MPI `int`.
 */
void scatter_owned_blocks(const Decomposition &decomposition, MPI_Comm comm,
                          int root, const std::vector<std::vector<double>> &global,
                          std::span<OwnedBlockBuffer> blocks);

namespace detail {

/// Cell count of a consistent box, or an exception if the box is not usable.
[[nodiscard]] inline std::size_t owned_cell_count(const Box3i &box) {
  if (!box.is_consistent()) {
    throw std::invalid_argument("owned-block transfer: owned box is not consistent");
  }
  std::size_t count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const auto extent = static_cast<std::size_t>(box.size[axis]);
    if (count > (std::numeric_limits<std::size_t>::max)() / extent) {
      throw std::overflow_error("owned-block transfer: block count exceeds size_t");
    }
    count *= extent;
  }
  return count;
}

} // namespace detail

/**
 * @brief Copy the owned interior of a field into an x-fastest block.
 *
 * `field.box()` is the owned index box. `field(i, j, k)` is addressed in
 * local owned coordinates, with `(0, 0, 0)` at `box.low`, which is how
 * `Field::operator()` is defined. Halo cells are not read.
 */
template <typename Field>
[[nodiscard]] std::vector<double> pack_owned_block(const Field &field) {
  const Box3i &box = field.box();
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  std::vector<double> packed(detail::owned_cell_count(box));
  std::size_t at = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        packed[at++] = static_cast<double>(field(i, j, k));
      }
    }
  }
  return packed;
}

/**
 * @brief Write an x-fastest owned block back into a field.
 *
 * Indices match `pack_owned_block`. Cells outside the owned box, including a
 * storage halo, are left unchanged.
 */
template <typename Field>
void unpack_owned_block(Field &field, std::span<const double> values) {
  const Box3i &box = field.box();
  if (values.size() != detail::owned_cell_count(box)) {
    throw std::invalid_argument(
        "owned-block transfer: block length does not match its box");
  }
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  std::size_t at = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        field(i, j, k) = values[at++];
      }
    }
  }
}

} // namespace pfc::decomposition
