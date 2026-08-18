// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_sparse_exchange_gpu.hpp
 * @brief Device `pfc::comm::SparseExchange` for `CudaSpace` / `HipSpace` (M4).
 *
 * @details
 * Builds device SparseVectors from the same structured index lists as the
 * host facade, then gather → device-pointer MPI → optional scatter. The
 * field never leaves the device. Non-blocking device MPI requires
 * GPU-aware MPI (see `gpu_aware_mpi.hpp`).
 *
 * CUDA execution is not available on LUMI; HIP can run here.
 *
 * @see kernel/decomposition/comm_sparse_exchange.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/exchange_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_aware_mpi.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <openpfc/runtime/gpu/sparse_vector_gpu.hpp>
#include <openpfc/runtime/gpu/sparse_vector_ops_gpu.hpp>

namespace pfc::comm {
namespace detail {

template <typename Tag, typename T> struct DeviceRemoteHalo {
  int peer_rank{-1};
  int send_tag{0};
  int recv_tag{0};
  bool scatter_after_recv{false};
  pfc::types::Int3 direction{0, 0, 0};
  core::SparseVector<Tag, T> send_values{static_cast<std::size_t>(0)};
  core::SparseVector<Tag, T> recv_values{static_cast<std::size_t>(0)};
};

template <typename Space, typename Tag, typename T> class DeviceSparseExchange {
  static_assert(std::is_same_v<T, double>,
                "pfc::comm::SparseExchange on device is double-only "
                "(existing gather/scatter kernels)");

public:
  using FieldT = data::Field<T, Space>;
  using halo_type = DeviceRemoteHalo<Tag, T>;

  DeviceSparseExchange(FieldT &field, const decomposition::Decomposition &decomp,
                       int rank, MPI_Comm comm, SparseExchangeOptions opt = {})
      : m_field(&field), m_comm(comm), m_rank(rank) {
    const int hw = field.halo_width() > 0 ? field.halo_width() : opt.halo_width;
    if (hw <= 0) {
      throw std::invalid_argument(
          "pfc::comm::SparseExchange: structured construction requires "
          "halo_width > 0");
    }
    auto host = halo::make_structured_halos<T>(decomp, rank, hw, opt.dirs,
                                               opt.exchange_base);
    m_halos.reserve(host.size());
    for (auto &h : host) {
      halo_type d;
      d.peer_rank = h.peer_rank;
      d.send_tag = h.send_tag;
      d.recv_tag = h.recv_tag;
      d.scatter_after_recv = opt.scatter_after_recv;
      d.direction = h.direction;
      d.send_values = core::SparseVector<Tag, T>(
          pfc::sparsevector::get_index(h.send_values));
      d.recv_values = core::SparseVector<Tag, T>(
          pfc::sparsevector::get_index(h.recv_values));
      m_halos.push_back(std::move(d));
    }
    m_requests.assign(2 * m_halos.size(), MPI_REQUEST_NULL);
  }

  DeviceSparseExchange(FieldT &field, std::vector<halo::RemoteHalo<T>> host_halos,
                       int rank, MPI_Comm comm)
      : m_field(&field), m_comm(comm), m_rank(rank) {
    m_halos.reserve(host_halos.size());
    for (auto &h : host_halos) {
      halo_type d;
      d.peer_rank = h.peer_rank;
      d.send_tag = h.send_tag;
      d.recv_tag = h.recv_tag;
      d.scatter_after_recv = h.scatter_after_recv;
      d.direction = h.direction;
      d.send_values = core::SparseVector<Tag, T>(
          pfc::sparsevector::get_index(h.send_values));
      d.recv_values = core::SparseVector<Tag, T>(
          pfc::sparsevector::get_index(h.recv_values));
      m_halos.push_back(std::move(d));
    }
    m_requests.assign(2 * m_halos.size(), MPI_REQUEST_NULL);
  }

  void exchange() {
    if (m_field == nullptr) {
      throw std::logic_error(
          "pfc::comm::SparseExchange::exchange: no field bound");
    }
    m_field->sync_to_device();
    start_device_(m_field->data(), m_field->size());
    finish_device_(m_field->data(), m_field->size());
    m_field->note_device_write();
  }

  void start() {
    throw std::logic_error(
        "pfc::comm::SparseExchange::start: device path is blocking-only; "
        "use exchange()");
  }

  void finish() {
    throw std::logic_error(
        "pfc::comm::SparseExchange::finish: device path is blocking-only; "
        "use exchange()");
  }

  [[nodiscard]] std::size_t num_halos() const noexcept { return m_halos.size(); }
  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] bool uses_gpu_aware_mpi() const noexcept {
    return pfc::gpu::runtime_mpi_gpu_aware();
  }

  [[nodiscard]] const std::vector<halo_type> &halos() const noexcept {
    return m_halos;
  }

  /// Device recv slabs for the 6 face slots (`+X,-X,+Y,-Y,+Z,-Z`).
  /// Unused slots (no structured entry, or empty recv) are `nullptr`.
  [[nodiscard]] std::array<T *, 6> face_recv_ptrs() {
    std::array<T *, 6> out{};
    for (auto &h : m_halos) {
      const int slot = halo::direction_to_face_slot(h.direction);
      if (slot < 0 || h.recv_values.empty()) {
        continue;
      }
      out[static_cast<std::size_t>(slot)] = h.recv_values.data().data();
    }
    return out;
  }

private:
  void start_device_(T *field_ptr, std::size_t field_size) {
    if (!pfc::gpu::runtime_mpi_gpu_aware()) {
      throw std::runtime_error(
          "pfc::comm::SparseExchange: GPU-aware MPI is required for device "
          "index-set exchange. Enable it or use the host SparseExchange.");
    }
    for (auto &h : m_halos) {
      if (!h.send_values.empty()) {
        core::gather(h.send_values, field_ptr, field_size);
      }
    }
    std::size_t req = 0;
    for (auto &h : m_halos) {
      exchange::irecv_data(h.recv_values, h.peer_rank, m_rank, m_comm,
                           &m_requests[req], h.recv_tag);
      ++req;
    }
    for (auto &h : m_halos) {
      exchange::isend_data(h.send_values, m_rank, h.peer_rank, m_comm,
                           &m_requests[req], h.send_tag);
      ++req;
    }
    m_request_count = static_cast<int>(req);
  }

  void finish_device_(T *field_ptr, std::size_t field_size) {
    exchange::wait_all(m_requests.data(), m_request_count);
    for (auto &h : m_halos) {
      if (h.scatter_after_recv && !h.recv_values.empty()) {
        core::scatter(h.recv_values, field_ptr, field_size);
      }
    }
    m_request_count = 0;
  }

  FieldT *m_field = nullptr;
  MPI_Comm m_comm = MPI_COMM_NULL;
  int m_rank = 0;
  std::vector<halo_type> m_halos;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
class SparseExchange<CudaSpace, T>
    : public detail::DeviceSparseExchange<CudaSpace, backend::CudaTag, T> {
  using Base = detail::DeviceSparseExchange<CudaSpace, backend::CudaTag, T>;

public:
  using Base::Base;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
class SparseExchange<HipSpace, T>
    : public detail::DeviceSparseExchange<HipSpace, backend::HipTag, T> {
  using Base = detail::DeviceSparseExchange<HipSpace, backend::HipTag, T>;

public:
  using Base::Base;
};
#endif

} // namespace pfc::comm

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
