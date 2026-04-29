// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_exchange.hpp
 * @brief Halo exchange driver: orchestrated, non-blocking face exchange
 *
 * @details
 * Builds halo patterns for all 6 face neighbors and runs exchange in a
 * deadlock-free way: post all Irecv, then all Isend, then Waitall.
 * With six face neighbors, uses zero-copy MPI derived types and non-blocking
 * Irecv/Isend; otherwise falls back to the pack path (gather/scatter).
 *
 * @see docs/halo_exchange.md for strategy and best practices
 * @see exchange.hpp for isend_data, irecv_data, wait_all
 * @see halo_pattern.hpp for create_halo_patterns
 */

#pragma once

#include <array>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc {

/**
 * @brief Orchestrates halo exchange for all face neighbors (CPU).
 *
 * Builds patterns once; each exchange_halos() posts all Irecv, then all Isend,
 * then wait_all. Six-face periodic bricks use zero-copy derived types; fewer
 * directions use gather/scatter buffers.
 */
template <typename T = double> class HaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct driver and build patterns (expensive, do once).
   * @param decomp Decomposition (must outlive this object)
   * @param rank Current MPI rank
   * @param halo_width Number of halo layers (e.g. 1 for 3-point stencil)
   * @param comm MPI communicator
   * @param base_tag Base tag for messages (direction index added)
   */
  HaloExchanger(const decomposition::Decomposition &decomp, int rank, int halo_width,
                MPI_Comm comm, int base_tag = 0)
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
      m_send_values.emplace_back(send_idx);
      m_recv_values.emplace_back(recv_idx);
    }

    m_requests.resize(2 * m_directions.size());
  }

  /**
   * @brief Run one halo exchange (zero-copy for faces: Irecv then Isend, wait).
   * @param field_ptr Local field (row-major [nx,ny,nz])
   * @param field_size Total number of elements (nx*ny*nz); unused when zero-copy
   */
  void exchange_halos(T *field_ptr, size_t field_size) {
    start_halo_exchange(field_ptr, field_size);
    finish_halo_exchange();
  }

  /**
   * @brief Run halo exchange on the pack path (gather / MPI / scatter).
   *
   * Ghost values are written in `create_recv_halo` index order, matching
   * `SeparatedFaceHaloExchanger` and `laplacian_7point_interior_separated`.
   * Default `exchange_halos()` uses zero-copy MPI subarrays for six face
   * neighbors, which can permute elements within a face relative to that order.
   */
  void exchange_halos_packed(T *field_ptr, size_t field_size) {
    exchange_halos_pack(field_ptr, field_size);
  }

  /**
   * @brief Post halo exchange (Irecv then Isend); return immediately for overlap.
   * Call finish_halo_exchange() after interior computation to wait and scatter.
   * @param field_ptr Local field (row-major [nx,ny,nz])
   * @param field_size Total number of elements (nx*ny*nz)
   */
  void start_halo_exchange(T *field_ptr, size_t field_size) {
    m_pending_field = field_ptr;
    m_pending_size = field_size;
    const size_t n = m_directions.size();
    if (n == 6) {
      size_t req_count = 0;
      void *buf = static_cast<void *>(field_ptr);
      for (size_t i = 0; i < n; ++i) {
        int tag = m_base_tag + opposite_slot(static_cast<int>(i));
        exchange::irecv_face(buf, m_face_types[i].recv_type.get(), m_neighbors[i],
                             m_comm, &m_requests[req_count], tag);
        req_count++;
      }
      for (size_t i = 0; i < n; ++i) {
        int tag = m_base_tag + static_cast<int>(i);
        exchange::isend_face(buf, m_face_types[i].send_type.get(), m_neighbors[i],
                             m_comm, &m_requests[req_count], tag);
        req_count++;
      }
      m_request_count = static_cast<int>(req_count);
    } else {
      start_halo_exchange_pack(field_ptr, field_size);
    }
  }

  /**
   * @brief Wait for halo exchange and scatter received data (pack path).
   * Must be called after start_halo_exchange(); can be delayed for overlap.
   */
  void finish_halo_exchange() {
    const double _pfc_t0 = MPI_Wtime();
    exchange::wait_all(m_requests.data(), m_request_count);
    const size_t n = m_directions.size();
    if (n != 6 && m_pending_field != nullptr) {
      for (size_t i = 0; i < n; ++i) {
        core::scatter(m_recv_values[i], m_pending_field, m_pending_size);
      }
    }
    m_pending_field = nullptr;
    m_pending_size = 0;
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - _pfc_t0);
  }

  /** @brief Number of active face directions (0--6) */
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

  void start_halo_exchange_pack(T *field_ptr, size_t field_size) {
    const size_t n = m_directions.size();
    size_t req_count = 0;
    for (size_t i = 0; i < n; ++i) {
      core::gather(m_send_values[i], field_ptr, field_size);
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      exchange::irecv_data(m_recv_values[i], m_neighbors[i], m_rank, m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + static_cast<int>(i);
      exchange::isend_data(m_send_values[i], m_rank, m_neighbors[i], m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    m_request_count = static_cast<int>(req_count);
  }
  void exchange_halos_pack(T *field_ptr, size_t field_size) {
    const double _pfc_t0 = MPI_Wtime();
    const size_t n = m_directions.size();
    size_t req_count = 0;
    for (size_t i = 0; i < n; ++i) {
      core::gather(m_send_values[i], field_ptr, field_size);
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      exchange::irecv_data(m_recv_values[i], m_neighbors[i], m_rank, m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    for (size_t i = 0; i < n; ++i) {
      int tag = m_base_tag + static_cast<int>(i);
      exchange::isend_data(m_send_values[i], m_rank, m_neighbors[i], m_comm,
                           &m_requests[req_count], tag);
      req_count++;
    }
    exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
    for (size_t i = 0; i < n; ++i) {
      core::scatter(m_recv_values[i], field_ptr, field_size);
    }
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - _pfc_t0);
  }

  const decomposition::Decomposition &m_decomp;
  int m_rank;
  int m_halo_width;
  MPI_Comm m_comm;
  int m_base_tag;

  std::array<halo::FaceTypes, 6> m_face_types;
  std::vector<Int3> m_directions;
  std::vector<int> m_neighbors;
  std::vector<core::SparseVector<backend::CpuTag, T>> m_send_values;
  std::vector<core::SparseVector<backend::CpuTag, T>> m_recv_values;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
  T *m_pending_field = nullptr;
  size_t m_pending_size = 0;
};

} // namespace pfc
