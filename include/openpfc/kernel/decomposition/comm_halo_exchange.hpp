// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_halo_exchange.hpp
 * @brief Unified host `pfc::comm::HaloExchange` (M4).
 *
 * @details
 * M4 folds the structured halo zoo onto one class: 6-face or 26-direction
 * connectivity, blocking `exchange()`, split `start()`/`finish()`, optional
 * persistent requests, and multi-field tag blocks from `halo_geometry.hpp`.
 *
 * The host specialization composes the Faces backend (`HostFacesHalo`),
 * the Full backend (`HostFullHalo`), and `PersistentHaloExchanger` so
 * persistent mode still shares that implementation until it is inlined.
 * Device `HaloExchange<CUDASpace/HIPSpace>` lives in
 * `runtime/gpu/comm_halo_exchange_gpu.hpp`. CUDA execution of that half is
 * not available on LUMI.
 *
 * Tag layout: field `f` uses `halo::field_tag_base(exchange_base, f)` so two
 * exchangers (or six fields) with distinct bases cannot collide.
 *
 * @see halo_geometry.hpp
 * @see padded_halo_exchange.hpp
 * @see full_padded_halo_exchange.hpp
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/full_padded_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/kernel/decomposition/halo_persistent.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>

namespace pfc::comm {

/// Structured halo connectivity: 6 faces, or faces+edges+corners (26).
enum class HaloConnectivity { Faces, Full };

/// Construction knobs for `HaloExchange`.
struct HaloExchangeOptions {
  HaloConnectivity connectivity = HaloConnectivity::Faces;
  bool persistent = false;
  int exchange_base = 0;
  /// Empty: Faces → `Axes3D()`, Full → `Full3D()`. Set `Axes2D()` for 2D slabs.
  halo::HaloDirectionSet directions{};
};

/// Resolve the direction set actually used by a `HaloExchange` construction.
[[nodiscard]] inline halo::HaloDirectionSet
resolved_halo_directions(const HaloExchangeOptions &opt) {
  if (!opt.directions.empty()) {
    return opt.directions;
  }
  return opt.connectivity == HaloConnectivity::Full ? halo::presets::Full3D()
                                                    : halo::presets::Axes3D();
}

/**
 * @brief Primary template: `HostSpace` is specialized below.
 *
 * Device spaces are specialized in `runtime/gpu/comm_halo_exchange_gpu.hpp`.
 */
template <typename MemorySpace, typename T = double> class HaloExchange {
  static_assert(std::is_same_v<MemorySpace, HostSpace>,
                "pfc::comm::HaloExchange device specializations live in "
                "runtime/gpu/comm_halo_exchange_gpu.hpp");
};

/**
 * @brief Host structured halo exchange over one or more padded Fields.
 */
template <typename T> class HaloExchange<HostSpace, T> {
public:
  using FieldT = data::Field<T, HostSpace>;

  /**
   * @brief Bind a single padded field.
   *
   * @throws std::invalid_argument if the field is unpadded, or if
   *         `persistent` is requested with `HaloConnectivity::Full`.
   */
  HaloExchange(FieldT &field, const decomposition::Decomposition &decomp, int rank,
               MPI_Comm comm, HaloExchangeOptions opt = {})
      : HaloExchange(std::vector<FieldT *>{&field}, decomp, rank, comm, opt) {}

  /**
   * @brief Bind several padded fields; each gets its own tag block.
   *
   * Field `i` uses MPI tags
   * `[field_tag_base(exchange_base, i), field_tag_base(...) + 33)`.
   */
  HaloExchange(std::vector<FieldT *> fields,
               const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
               HaloExchangeOptions opt = {})
      : m_opt(opt), m_fields(std::move(fields)) {
    if (m_fields.empty()) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: at least one field is required");
    }
    if (m_opt.persistent && m_opt.connectivity == HaloConnectivity::Full) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: persistent requests are Faces-only "
          "(Full connectivity has no persistent path)");
    }
    m_faces.reserve(m_fields.size());
    m_full.reserve(m_fields.size());
    m_persist.reserve(m_fields.size());
    for (std::size_t i = 0; i < m_fields.size(); ++i) {
      FieldT *f = m_fields[i];
      if (f == nullptr) {
        throw std::invalid_argument(
            "pfc::comm::HaloExchange: field pointer must not be null");
      }
      if (f->storage_halo() <= 0) {
        throw std::invalid_argument(
            "pfc::comm::HaloExchange: Field binding requires storage_halo > 0");
      }
      const int tag0 =
          halo::field_tag_base(m_opt.exchange_base, static_cast<int>(i));
      const auto dirs = resolved_halo_directions(m_opt);
      if (m_opt.persistent) {
        m_persist.push_back(std::make_unique<PersistentHaloExchanger<T>>(
            f->box(), f->domain(), decomp, rank, f->storage_halo(), comm, f->data(),
            dirs, tag0));
      } else if (m_opt.connectivity == HaloConnectivity::Full) {
        m_full.push_back(std::make_unique<detail::HostFullHalo<T>>(
            f->box(), f->domain(), decomp, rank, f->storage_halo(), comm, dirs,
            tag0));
      } else {
        m_faces.push_back(std::make_unique<detail::HostFacesHalo<T>>(
            *f, decomp, rank, comm, dirs, tag0));
      }
    }
  }

  /// Blocking exchange of every bound field.
  void exchange() {
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->exchange_halos();
      }
      return;
    }
    if (!m_full.empty()) {
      for (std::size_t i = 0; i < m_full.size(); ++i) {
        m_full[i]->exchange_halos(m_fields[i]->data(), m_fields[i]->size());
      }
      return;
    }
    for (auto &h : m_faces) {
      h->start();
      h->finish();
    }
  }

  /**
   * @brief Post Irecv/Isend (or MPI_Startall) and return.
   *
   * @throws std::logic_error if connectivity is Full (no split-phase API).
   */
  void start() {
    if (!m_full.empty()) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::start: Full connectivity has no "
          "split-phase API; use exchange()");
    }
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->start_exchange();
      }
      return;
    }
    for (auto &h : m_faces) {
      h->start();
    }
  }

  /// Wait for `start()`.
  void finish() {
    if (!m_full.empty()) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::finish: Full connectivity has no "
          "split-phase API; use exchange()");
    }
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->wait_exchange();
      }
      return;
    }
    for (auto &h : m_faces) {
      h->finish();
    }
  }

  [[nodiscard]] HaloConnectivity connectivity() const noexcept {
    return m_opt.connectivity;
  }
  [[nodiscard]] bool persistent() const noexcept { return m_opt.persistent; }
  [[nodiscard]] std::size_t num_fields() const noexcept { return m_fields.size(); }

private:
  HaloExchangeOptions m_opt{};
  std::vector<FieldT *> m_fields;
  std::vector<std::unique_ptr<detail::HostFacesHalo<T>>> m_faces;
  std::vector<std::unique_ptr<detail::HostFullHalo<T>>> m_full;
  std::vector<std::unique_ptr<PersistentHaloExchanger<T>>> m_persist;
};

} // namespace pfc::comm
