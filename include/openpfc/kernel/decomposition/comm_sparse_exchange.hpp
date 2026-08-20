// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_sparse_exchange.hpp
 * @brief Unified host `pfc::comm::SparseExchange` (M4).
 *
 * @details
 * Index-set counterpart of `HaloExchange`. This increment is **host-only**:
 * it composes `SparseHaloExchanger` so callers can move onto the new name
 * before that class is deleted. Device `SparseExchange<CUDASpace/HIPSpace>`
 * lives in `runtime/gpu/comm_sparse_exchange_gpu.hpp` and keeps gather /
 * scatter on the device (no per-step full-field D2H).
 *
 * Structured construction uses `make_structured_halos` + `halo_geometry`
 * tag blocks. A custom `RemoteHalo` list is accepted for unstructured /
 * FEM patterns.
 *
 * @see sparse_halo_exchange.hpp
 * @see halo_face_layout.hpp
 */

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/sparse_halo_exchange.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>

namespace pfc::comm {

/// Construction knobs for `SparseExchange`.
struct SparseExchangeOptions {
  halo::HaloDirectionSet dirs = halo::presets::Axes3D();
  int exchange_base = 0;
  bool scatter_after_recv = false;
  /// Used when the field's `halo_width()` is 0. Must be > 0 for structured
  /// construction.
  int halo_width = 0;
};

/**
 * @brief Primary template: `HostSpace` is specialized below.
 *
 * Device spaces are specialized in `runtime/gpu/comm_sparse_exchange_gpu.hpp`.
 */
template <typename MemorySpace, typename T = double> class SparseExchange {
  static_assert(std::is_same_v<MemorySpace, HostSpace>,
                "pfc::comm::SparseExchange device specializations live in "
                "runtime/gpu/comm_sparse_exchange_gpu.hpp");
};

/**
 * @brief Host index-set halo exchange over a Field or a raw buffer.
 */
template <typename T> class SparseExchange<HostSpace, T> {
public:
  using FieldT = data::Field<T, HostSpace>;
  using halo_type = halo::RemoteHalo<T>;

  /**
   * @brief Structured exchange from a Field's subdomain geometry.
   *
   * Halo width is `field.halo_width()` if that is > 0, otherwise
   * `opt.halo_width`. The field pointer is bound for `exchange()`.
   */
  SparseExchange(FieldT &field, const decomposition::Decomposition &decomp,
                 int rank, MPI_Comm comm, SparseExchangeOptions opt = {})
      : SparseExchange(field.data(), field.size(), decomp, rank, comm,
                       resolve_hw_(field.halo_width(), opt), opt) {}

  /**
   * @brief Structured exchange over a raw buffer (unpadded `nx*ny*nz`).
   */
  SparseExchange(T *field, std::size_t field_size,
                 const decomposition::Decomposition &decomp, int rank,
                 MPI_Comm comm, int halo_width, SparseExchangeOptions opt = {})
      : m_field(field), m_field_size(field_size),
        m_impl(comm, rank,
               apply_scatter_flag_(
                   halo::make_structured_halos<T>(
                       decomp, rank, require_hw_(halo_width), opt.dirs,
                       opt.exchange_base),
                   opt.scatter_after_recv)) {
    if (field == nullptr) {
      throw std::invalid_argument(
          "pfc::comm::SparseExchange: field pointer must not be null");
    }
  }

  /**
   * @brief Custom `RemoteHalo` list (unstructured / FEM).
   *
   * The field pointer is bound later via `exchange(T*, size)` or
   * `bind()`.
   */
  SparseExchange(std::vector<halo_type> halos, int rank, MPI_Comm comm)
      : m_impl(comm, rank, std::move(halos)) {}

  SparseExchange(FieldT &field, std::vector<halo_type> halos, int rank,
                 MPI_Comm comm)
      : m_field(field.data()), m_field_size(field.size()),
        m_impl(comm, rank, std::move(halos)) {}

  void bind(T *field, std::size_t field_size) {
    if (field == nullptr) {
      throw std::invalid_argument(
          "pfc::comm::SparseExchange::bind: field pointer must not be null");
    }
    m_field = field;
    m_field_size = field_size;
  }

  /// Blocking gather → MPI → optional scatter of the bound field.
  void exchange() {
    require_bound_("exchange");
    m_impl.exchange_halos(m_field, m_field_size);
  }

  void exchange(T *field, std::size_t field_size) {
    bind(field, field_size);
    exchange();
  }

  void start() {
    require_bound_("start");
    m_impl.start_halo_exchange(m_field, m_field_size);
  }

  void finish() { m_impl.finish_halo_exchange(); }

  [[nodiscard]] std::size_t num_halos() const noexcept {
    return m_impl.num_halos();
  }
  [[nodiscard]] const std::vector<halo_type> &halos() const noexcept {
    return m_impl.halos();
  }
  [[nodiscard]] int rank() const noexcept { return m_impl.rank(); }

private:
  static int require_hw_(int hw) {
    if (hw <= 0) {
      throw std::invalid_argument(
          "pfc::comm::SparseExchange: structured construction requires "
          "halo_width > 0");
    }
    return hw;
  }

  static int resolve_hw_(int field_hw, const SparseExchangeOptions &opt) {
    if (field_hw > 0) {
      return field_hw;
    }
    return opt.halo_width;
  }

  static std::vector<halo_type>
  apply_scatter_flag_(std::vector<halo_type> halos, bool scatter) {
    if (scatter) {
      for (auto &h : halos) {
        h.scatter_after_recv = true;
      }
    }
    return halos;
  }

  void require_bound_(const char *op) const {
    if (m_field == nullptr) {
      throw std::logic_error(std::string("pfc::comm::SparseExchange::") + op +
                             ": no field bound; call bind() or exchange(ptr, n)");
    }
  }

  T *m_field = nullptr;
  std::size_t m_field_size = 0;
  SparseHaloExchanger<T> m_impl;
};

} // namespace pfc::comm
