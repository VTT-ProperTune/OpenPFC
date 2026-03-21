// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file halo_persistent.hpp
 * @brief Optional persistent MPI requests for 6-face zero-copy halo exchange
 *
 * @details
 * Uses `MPI_Send_init` / `MPI_Recv_init` once, then `MPI_Startall` and
 * `MPI_Waitall` each step. Only valid when the decomposition exposes all six face
 * neighbors (same condition as the zero-copy path in `HaloExchanger`).
 *
 * The field buffer pointer passed to the constructor must remain the storage used
 * for every `start_exchange()` / `wait_exchange()` pair (MPI persistent operations
 * are bound to that address). Do not destroy this object while a request epoch is
 * in progress; call `wait_exchange()` before destruction.
 *
 * @see docs/halo_exchange.md §4
 * @see halo_exchange.hpp
 */

#pragma once

#include <array>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc {

/**
 * @brief Persistent 6-face halo exchange (CPU, MPI derived types, double default).
 */
template <typename T = double> class PersistentHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @param field_ptr Base pointer of the local field; must be stable for object lifetime
   */
  PersistentHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, T *field_ptr, int base_tag = 0)
      : m_comm(comm), m_base_tag(base_tag), m_buf(static_cast<void *>(field_ptr)) {
    auto patterns = halo::create_halo_patterns<backend::CpuTag>(
        decomp, rank, halo::Connectivity::Faces, halo_width);

    const auto &local_world = decomposition::get_subworld(decomp, rank);
    auto local_size = world::get_size(local_world);
    int nx = local_size[0], ny = local_size[1], nz = local_size[2];

    m_face_types = halo::create_face_types_6(nx, ny, nz, halo_width,
                                             exchange::detail::get_mpi_type<T>());

    const std::vector<Int3> direction_order = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                               {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

    for (const auto &dir : direction_order) {
      auto it = patterns.find(dir);
      if (it == patterns.end()) {
        continue;
      }
      int neighbor = decomposition::get_neighbor_rank(decomp, rank, dir);
      if (neighbor < 0) {
        continue;
      }
      m_neighbors.push_back(neighbor);
    }

    if (m_neighbors.size() != 6U) {
      throw std::runtime_error(
          "PersistentHaloExchanger: requires six face neighbors (full 3D periodic "
          "brick); use HaloExchanger for the general case.");
    }

    m_requests.assign(12, MPI_REQUEST_NULL);
    for (std::size_t i = 0; i < 6; ++i) {
      int tag = m_base_tag + static_cast<int>(i);
      int err = MPI_Recv_init(m_buf, 1, m_face_types[i].recv_type.get(),
                              m_neighbors[i], tag, m_comm, &m_requests[2 * i]);
      if (err != MPI_SUCCESS) {
        free_all_requests();
        throw std::runtime_error("MPI_Recv_init failed in PersistentHaloExchanger");
      }
      err = MPI_Send_init(m_buf, 1, m_face_types[i].send_type.get(), m_neighbors[i],
                          tag, m_comm, &m_requests[2 * i + 1]);
      if (err != MPI_SUCCESS) {
        free_all_requests();
        throw std::runtime_error("MPI_Send_init failed in PersistentHaloExchanger");
      }
    }
  }

  PersistentHaloExchanger(const PersistentHaloExchanger &) = delete;
  PersistentHaloExchanger &operator=(const PersistentHaloExchanger &) = delete;

  PersistentHaloExchanger(PersistentHaloExchanger &&other) noexcept
      : m_comm(other.m_comm), m_base_tag(other.m_base_tag), m_buf(other.m_buf),
        m_face_types(std::move(other.m_face_types)),
        m_neighbors(std::move(other.m_neighbors)), m_requests(std::move(other.m_requests)) {
    other.m_requests.clear();
    other.m_buf = nullptr;
  }

  PersistentHaloExchanger &operator=(PersistentHaloExchanger &&other) noexcept {
    if (this != &other) {
      free_all_requests();
      m_comm = other.m_comm;
      m_base_tag = other.m_base_tag;
      m_buf = other.m_buf;
      m_face_types = std::move(other.m_face_types);
      m_neighbors = std::move(other.m_neighbors);
      m_requests = std::move(other.m_requests);
      other.m_requests.clear();
      other.m_buf = nullptr;
    }
    return *this;
  }

  ~PersistentHaloExchanger() { free_all_requests(); }

  /** @brief Start one halo exchange (MPI_Startall on persistent requests). */
  void start_exchange() {
    MPI_Startall(static_cast<int>(m_requests.size()), m_requests.data());
  }

  /** @brief Complete the exchange started with start_exchange(). */
  void wait_exchange() {
    MPI_Waitall(static_cast<int>(m_requests.size()), m_requests.data(),
                MPI_STATUSES_IGNORE);
  }

  /** @brief Equivalent to start_exchange(); wait_exchange(); */
  void exchange_halos() {
    start_exchange();
    wait_exchange();
  }

private:
  void free_all_requests() {
    for (auto &r : m_requests) {
      if (r != MPI_REQUEST_NULL) {
        MPI_Request_free(&r);
        r = MPI_REQUEST_NULL;
      }
    }
    m_requests.clear();
  }

  MPI_Comm m_comm;
  int m_base_tag;
  void *m_buf;
  std::array<halo::FaceTypes, 6> m_face_types{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;
};

} // namespace pfc
