// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file padded_sparse_halos.hpp
 * @brief Six-face padded index lists for the existing typed sparse transport.
 */

#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>

namespace pfc::halo {

/**
 * @brief Build typed six-face send/receive lists addressing padded storage.
 *
 * @param field Padded field with the decomposition's local owned box.
 * @param decomp Matching domain decomposition; it must outlive the exchanger.
 * @param rank Rank in that decomposition and the exchanger's communicator.
 * @param base_tag Shared tag offset on every communicating rank. Concurrent
 * exchanges need disjoint tag ranges (canonical direction tags are added).
 * @return RemoteHalo records for SparseExchange, with scatter_after_recv true.
 * Nonperiodic dead ends are omitted; periodic self-neighbors are included.
 * @throws std::invalid_argument For missing padding or incompatible local box.
 * @note Index lists are built once on the host. Only six face slabs are
 * exchanged; edges/corners need a separate connectivity contract. Physical
 * walls remain the reflection/boundary service's responsibility. The field
 * geometry and buffer must remain stable for the exchanger's lifetime.
 */
template <class T, class Space>
[[nodiscard]] std::vector<RemoteHalo<T>>
make_padded_sparse_halos(const data::Field<T, Space> &field,
                         const decomposition::Decomposition &decomp, int rank,
                         int base_tag = 0) {
  const auto box = decomposition::local_box(decomp, rank);
  if (field.storage_halo() < 1 || field.box().low != box.low ||
      field.box().size != box.size ||
      field.global_size() !=
          domain::get_size(decomposition::get_global_domain(decomp)) ||
      domain::get_periodic(field.domain()) !=
          domain::get_periodic(decomposition::get_global_domain(decomp)))
    throw std::invalid_argument(
        "make_padded_sparse_halos: incompatible padded field");
  const auto indices = [&](const HaloSlab &slab) {
    std::vector<std::size_t> out;
    out.reserve(slab.volume());
    const auto sy = std::size_t(field.padded_extent(0));
    const auto sz = sy * field.padded_extent(1);
    for (int k = 0; k < slab.count[2]; ++k)
      for (int j = 0; j < slab.count[1]; ++j)
        for (int i = 0; i < slab.count[0]; ++i)
          out.push_back(std::size_t(slab.start[0] + i) +
                        std::size_t(slab.start[1] + j) * sy +
                        std::size_t(slab.start[2] + k) * sz);
    return core::SparseVector<backend::CPUTag, T>(out);
  };
  std::vector<RemoteHalo<T>> halos;
  const auto directions = presets::Axes3D();
  for (const auto &direction : directions.dirs) {
    const int peer = decomposition::get_neighbor_rank(decomp, rank, direction);
    if (peer < 0) continue;
    RemoteHalo<T> halo;
    halo.peer_rank = peer;
    halo.direction = direction;
    halo.send_tag = base_tag + direction_to_canonical_tag(direction);
    halo.recv_tag = base_tag + direction_to_canonical_tag(
                                   {-direction[0], -direction[1], -direction[2]});
    halo.send_values =
        indices(padded_send_slab(box.size, field.storage_halo(), direction));
    halo.recv_values =
        indices(padded_recv_slab(box.size, field.storage_halo(), direction));
    halo.scatter_after_recv = true;
    halos.push_back(std::move(halo));
  }
  return halos;
}

} // namespace pfc::halo
