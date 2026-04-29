// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file separated_halo_exchange.hpp
 * @brief Face halo exchange into separate buffers; core stays FFT-safe
 *
 * @details
 * Same communication pattern as `HaloExchanger`, but received ghost values are
 * written to **contiguous per-face buffers** instead of boundary slabs of the
 * core. **Send** still uses MPI derived types into the core `nx×ny×nz` array.
 * Receives use the **pack path** (same ordering as `create_recv_halo` indices)
 * so face buffer layout matches `laplacian_7point_interior_separated`; a prior
 * `irecv_dense` + `isend_face` fast path could reorder elements vs that layout.
 *
 * Face buffer order matches `create_face_types_6` / `halo_face_layout.hpp`:
 * 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
 *
 * @see docs/halo_exchange.md
 * @see halo_exchange.hpp
 */

#pragma once

#include <array>
#include <cstring>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc {

template <typename T = double> class SeparatedFaceHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  SeparatedFaceHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                             int halo_width, MPI_Comm comm, int base_tag = 0)
      : m_decomp(decomp), m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag) {
    auto patterns = halo::create_halo_patterns<backend::CpuTag>(
        m_decomp, m_rank, halo::Connectivity::Faces, m_halo_width);

    const auto &local_world = decomposition::get_subworld(m_decomp, m_rank);
    auto local_size = world::get_size(local_world);
    int nx = local_size[0];
    int ny = local_size[1];
    int nz = local_size[2];

    m_face_types = halo::create_face_types_6(nx, ny, nz, m_halo_width,
                                             exchange::detail::get_mpi_type<T>());

    m_face_counts = halo::face_halo_counts(m_decomp, m_rank, m_halo_width);

    const std::vector<Int3> direction_order = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                               {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

    for (const auto &dir : direction_order) {
      auto it = patterns.find(dir);
      if (it == patterns.end()) {
        continue;
      }
      const auto &send_halo_idx = it->second.first;
      const auto &recv_halo_idx = it->second.second;
      int neighbor = decomposition::get_neighbor_rank(m_decomp, m_rank, dir);
      if (neighbor < 0) {
        continue;
      }

      std::vector<size_t> send_idx = sparsevector::get_index(send_halo_idx);
      std::vector<size_t> recv_idx = sparsevector::get_index(recv_halo_idx);
      m_directions.push_back(dir);
      m_neighbors.push_back(neighbor);
      m_dir_slot.push_back(direction_to_slot(dir));
      m_send_values.emplace_back(send_idx);
      m_recv_values.emplace_back(recv_idx);
    }

    m_requests.resize(2 * m_directions.size());
  }

  /** Per-face recv element counts (order +X,-X,+Y,-Y,+Z,-Z) */
  const std::array<size_t, 6> &face_counts() const { return m_face_counts.counts; }

  /**
   * @brief Exchange halos; recv into face_buffers[0..5], send from core.
   * @param core_ptr Local core field row-major [nx,ny,nz]
   * @param core_size nx*ny*nz
   * @param face_buffers six vectors, each sized >= face_counts()[k]
   */
  void exchange_halos(T *core_ptr, size_t core_size,
                      std::array<std::vector<T>, 6> &face_buffers) {
    start_halo_exchange(core_ptr, core_size, face_buffers);
    finish_halo_exchange(face_buffers);
  }

  void start_halo_exchange(T *core_ptr, size_t core_size,
                           std::array<std::vector<T>, 6> &face_buffers) {
    m_pending_core = core_ptr;
    validate_face_sizes(face_buffers);

    start_halo_exchange_pack(core_ptr, core_size, face_buffers);
  }

  void finish_halo_exchange(std::array<std::vector<T>, 6> &face_buffers) {
    const double _pfc_t0 = MPI_Wtime();
    exchange::wait_all(m_requests.data(), m_request_count);
    const size_t n = m_directions.size();
    if (m_pending_core != nullptr) {
      for (size_t i = 0; i < n; ++i) {
        int slot = m_dir_slot[i];
        const auto &recv_sv = m_recv_values[i];
        const T *src = recv_sv.data().data();
        T *dst = face_buffers[static_cast<size_t>(slot)].data();
        size_t nbytes = recv_sv.size() * sizeof(T);
        if (face_buffers[static_cast<size_t>(slot)].size() * sizeof(T) < nbytes) {
          throw std::runtime_error(
              "SeparatedFaceHaloExchanger: face buffer too small");
        }
        std::memcpy(dst, src, nbytes);
      }
    }
    m_pending_core = nullptr;
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - _pfc_t0);
  }

  size_t num_directions() const { return m_directions.size(); }

private:
  static int opposite_slot(int slot) {
    switch (slot) {
    case 0: return 1;
    case 1: return 0;
    case 2: return 3;
    case 3: return 2;
    case 4: return 5;
    case 5: return 4;
    default: return -1;
    }
  }

  static int direction_to_slot(const Int3 &d) {
    const std::array<Int3, 6> dirs = {
        {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    for (int i = 0; i < 6; ++i) {
      if (dirs[static_cast<size_t>(i)] == d) {
        return i;
      }
    }
    return -1;
  }

  void validate_face_sizes(const std::array<std::vector<T>, 6> &face_buffers) const {
    for (int s = 0; s < 6; ++s) {
      size_t need = m_face_counts.counts[static_cast<size_t>(s)];
      if (face_buffers[static_cast<size_t>(s)].size() < need) {
        throw std::runtime_error(
            "SeparatedFaceHaloExchanger: face buffer " + std::to_string(s) +
            " too small (need " + std::to_string(need) + ", have " +
            std::to_string(face_buffers[static_cast<size_t>(s)].size()) + ")");
      }
    }
  }

  void start_halo_exchange_pack(T *core_ptr, size_t core_size,
                                std::array<std::vector<T>, 6> &face_buffers) {
    (void)face_buffers;
    const size_t n = m_directions.size();
    size_t req_count = 0;
    for (size_t i = 0; i < n; ++i) {
      core::gather(m_send_values[i], core_ptr, core_size);
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + opposite_slot(m_dir_slot[i]);
      exchange::irecv_data(m_recv_values[i], m_neighbors[i], m_rank, m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + m_dir_slot[i];
      exchange::isend_data(m_send_values[i], m_rank, m_neighbors[i], m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    m_request_count = static_cast<int>(req_count);
  }

  const decomposition::Decomposition &m_decomp;
  int m_rank;
  int m_halo_width;
  MPI_Comm m_comm;
  int m_base_tag;

  std::array<halo::FaceTypes, 6> m_face_types;
  halo::FaceHaloCounts m_face_counts;
  std::vector<Int3> m_directions;
  std::vector<int> m_neighbors;
  std::vector<int> m_dir_slot;
  std::vector<core::SparseVector<backend::CpuTag, T>> m_send_values;
  std::vector<core::SparseVector<backend::CpuTag, T>> m_recv_values;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
  T *m_pending_core = nullptr;
};

} // namespace pfc
