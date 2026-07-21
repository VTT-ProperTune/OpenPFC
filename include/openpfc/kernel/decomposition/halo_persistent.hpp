// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
#include <openpfc/kernel/decomposition/halo_direction_agreement.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc {

/**
 * @brief Persistent 6-face halo exchange (CPU, MPI derived types, double default).
 */
template <typename T = double> class PersistentHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct with the historical 6-face axis-aligned set (`Axes3D()`).
   *
   * @param field_ptr Base pointer of the local field; must be stable for object
   * lifetime.
   */
  PersistentHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, T *field_ptr,
                          int base_tag = 0)
      : PersistentHaloExchanger(decomp, rank, halo_width, comm, field_ptr,
                                halo::presets::Axes3D(), base_tag,
                                halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct a persistent exchange bound to the directions in `dirs`.
   *
   * Uses one persistent `MPI_Recv_init` / `MPI_Send_init` pair per active
   * face slot. The `field_ptr` and direction set must remain stable for the
   * lifetime of the exchanger (persistent requests are bound to that buffer
   * and tag layout).
   *
   * Non-face directions in `dirs` are tolerated but ignored — this is a
   * face-only persistent driver.
   *
   * @param dirs     Direction set (defaults to `Axes3D()` for back-compat).
   * @param selector Optional per-rank override of the direction set.
   */
  PersistentHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, T *field_ptr,
                          halo::HaloDirectionSet dirs, int base_tag = 0,
                          halo::HaloDirectionSelector selector = {})
      : m_comm(comm), m_base_tag(base_tag), m_buf(static_cast<void *>(field_ptr)),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
    halo::validate_neighbour_direction_agreement(comm, decomp, rank, m_dirs);

    auto patterns = halo::create_halo_patterns<backend::CpuTag>(
        decomp, rank, halo::Connectivity::Faces, halo_width);

    const auto &local_world = decomposition::get_subworld(decomp, rank);
    auto local_size = world::get_size(local_world);
    int nx = local_size[0];
    int ny = local_size[1];
    int nz = local_size[2];

    m_face_types = halo::create_face_types_6(nx, ny, nz, halo_width,
                                             exchange::detail::get_mpi_type<T>());

    static constexpr std::array<Int3, 6> kFaceDirs = {
        {Int3{1, 0, 0}, Int3{-1, 0, 0}, Int3{0, 1, 0}, Int3{0, -1, 0}, Int3{0, 0, 1},
         Int3{0, 0, -1}}};

    for (std::size_t i = 0; i < 6; ++i) {
      const Int3 &dir = kFaceDirs[i];
      m_active[i] = m_dirs.contains(dir);
      m_neighbors[i] = decomposition::get_neighbor_rank(decomp, rank, dir);
    }

    std::size_t n_active = 0;
    for (bool a : m_active) {
      if (a) ++n_active;
    }
    if (n_active == 0) {
      throw std::runtime_error("PersistentHaloExchanger: empty direction set "
                               "after filtering — nothing to exchange.");
    }

    m_requests.assign(2 * n_active, MPI_REQUEST_NULL);
    std::size_t r = 0;
    // Same ordering as `HaloExchanger::start_halo_exchange` zero-copy path: post
    // every `MPI_Recv_init` first, then every `MPI_Send_init`, so `MPI_Startall`
    // begins matching receives before sends (avoids MPI deadlock with standard
    // protocols).
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i]) {
        continue;
      }
      const int recv_tag = m_base_tag + opposite_face_slot(static_cast<int>(i));
      int err = MPI_Recv_init(m_buf, 1, m_face_types[i].recv_type.get(),
                              m_neighbors[i], recv_tag, m_comm, &m_requests[r]);
      if (err != MPI_SUCCESS) {
        free_all_requests();
        throw std::runtime_error("MPI_Recv_init failed in PersistentHaloExchanger");
      }
      ++r;
    }
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i]) {
        continue;
      }
      const int send_tag = m_base_tag + static_cast<int>(i);
      int err = MPI_Send_init(m_buf, 1, m_face_types[i].send_type.get(),
                              m_neighbors[i], send_tag, m_comm, &m_requests[r]);
      if (err != MPI_SUCCESS) {
        free_all_requests();
        throw std::runtime_error("MPI_Send_init failed in PersistentHaloExchanger");
      }
      ++r;
    }
  }

  PersistentHaloExchanger(const PersistentHaloExchanger &) = delete;
  PersistentHaloExchanger &operator=(const PersistentHaloExchanger &) = delete;

  PersistentHaloExchanger(PersistentHaloExchanger &&other) noexcept
      : m_comm(other.m_comm), m_base_tag(other.m_base_tag), m_buf(other.m_buf),
        m_dirs(std::move(other.m_dirs)), m_face_types(std::move(other.m_face_types)),
        m_active(other.m_active), m_neighbors(other.m_neighbors),
        m_requests(std::move(other.m_requests)) {
    other.m_requests.clear();
    other.m_buf = nullptr;
  }

  PersistentHaloExchanger &operator=(PersistentHaloExchanger &&other) noexcept {
    if (this != &other) {
      free_all_requests();
      m_comm = other.m_comm;
      m_base_tag = other.m_base_tag;
      m_buf = other.m_buf;
      m_dirs = std::move(other.m_dirs);
      m_face_types = std::move(other.m_face_types);
      m_active = other.m_active;
      m_neighbors = other.m_neighbors;
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
    const double _pfc_t0 = MPI_Wtime();
    MPI_Waitall(static_cast<int>(m_requests.size()), m_requests.data(),
                MPI_STATUSES_IGNORE);
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - _pfc_t0);
  }

  /** @brief Equivalent to start_exchange(); wait_exchange(); */
  void exchange_halos() {
    start_exchange();
    wait_exchange();
  }

private:
  static int opposite_face_slot(int slot) {
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
  halo::HaloDirectionSet m_dirs;
  std::array<halo::FaceTypes, 6> m_face_types{};
  std::array<bool, 6> m_active{};
  std::array<int, 6> m_neighbors{};
  std::vector<MPI_Request> m_requests;
};

} // namespace pfc
