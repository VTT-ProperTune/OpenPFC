// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_sparse_exchange_gpu.hpp
 * @brief Device `pfc::comm::SparseExchange` for `CUDASpace` / `HIPSpace` (M4).
 *
 * @details
 * Builds device SparseVectors from the same structured index lists as the
 * host facade, then gather → MPI → optional scatter. With GPU-aware MPI the
 * field never leaves the device. Without it, send/recv slabs host-stage
 * (D2H / MPI / H2D); gather and scatter stay on device.
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
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
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
      d.send_values = core::SparseVector<Tag, T>(h.send_values.indices().data(),
                                                 h.send_values.size());
      d.recv_values = core::SparseVector<Tag, T>(h.recv_values.indices().data(),
                                                 h.recv_values.size());
      m_halos.push_back(std::move(d));
    }
    m_requests.assign(2 * m_halos.size(), MPI_REQUEST_NULL);
    allocate_host_stage_();
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
      d.send_values = core::SparseVector<Tag, T>(h.send_values.indices().data(),
                                                 h.send_values.size());
      d.recv_values = core::SparseVector<Tag, T>(h.recv_values.indices().data(),
                                                 h.recv_values.size());
      m_halos.push_back(std::move(d));
    }
    m_requests.assign(2 * m_halos.size(), MPI_REQUEST_NULL);
    allocate_host_stage_();
  }

  void exchange() {
    if (m_field == nullptr) {
      throw std::logic_error("pfc::comm::SparseExchange::exchange: no field bound");
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
  void allocate_host_stage_() {
    if (pfc::gpu::runtime_mpi_gpu_aware()) {
      return;
    }
    m_host_send.resize(m_halos.size());
    m_host_recv.resize(m_halos.size());
    for (std::size_t i = 0; i < m_halos.size(); ++i) {
      m_host_send[i].resize(m_halos[i].send_values.size());
      m_host_recv[i].resize(m_halos[i].recv_values.size());
    }
  }

  void copy_d2h_(void *dst, const void *src, std::size_t bytes, const char *what) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Space, CUDASpace>) {
      pfc::exchange::detail::CUDAXchg::memcpy_d2h(dst, src, bytes, what);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Space, HIPSpace>) {
      pfc::exchange::detail::HIPXchg::memcpy_d2h(dst, src, bytes, what);
      return;
    }
#endif
    (void)dst;
    (void)src;
    (void)bytes;
    throw std::logic_error(std::string(what) + ": no GPU memcpy backend");
  }

  void copy_h2d_(void *dst, const void *src, std::size_t bytes, const char *what) {
#if defined(OpenPFC_ENABLE_CUDA)
    if constexpr (std::is_same_v<Space, CUDASpace>) {
      pfc::exchange::detail::CUDAXchg::memcpy_h2d(dst, src, bytes, what);
      return;
    }
#endif
#if defined(OpenPFC_ENABLE_HIP)
    if constexpr (std::is_same_v<Space, HIPSpace>) {
      pfc::exchange::detail::HIPXchg::memcpy_h2d(dst, src, bytes, what);
      return;
    }
#endif
    (void)dst;
    (void)src;
    (void)bytes;
    throw std::logic_error(std::string(what) + ": no GPU memcpy backend");
  }

  void start_device_(T *field_ptr, std::size_t field_size) {
    for (auto &h : m_halos) {
      if (!h.send_values.empty()) {
        core::gather(h.send_values, field_ptr, field_size);
      }
    }
    const bool aware = pfc::gpu::runtime_mpi_gpu_aware();
    if (!aware) {
      for (std::size_t i = 0; i < m_halos.size(); ++i) {
        auto &h = m_halos[i];
        if (h.send_values.empty()) {
          continue;
        }
        copy_d2h_(m_host_send[i].data(), h.send_values.data().data(),
                  h.send_values.size() * sizeof(T),
                  "SparseExchange host-stage send D2H");
      }
    }
    std::size_t req = 0;
    for (std::size_t i = 0; i < m_halos.size(); ++i) {
      auto &h = m_halos[i];
      if (aware) {
        exchange::irecv_data(h.recv_values, h.peer_rank, m_rank, m_comm,
                             &m_requests[req], h.recv_tag);
      } else if (h.recv_values.empty()) {
        m_requests[req] = MPI_REQUEST_NULL;
      } else {
        const int count = pfc::mpi::ensure_mpi_int_count(
            h.recv_values.size(), "SparseExchange host-stage recv");
        pfc::mpi::throw_on_mpi_error(MPI_Irecv(m_host_recv[i].data(), count,
                                               MPI_DOUBLE, h.peer_rank, h.recv_tag,
                                               m_comm, &m_requests[req]),
                                     "SparseExchange host-stage MPI_Irecv");
      }
      ++req;
    }
    for (std::size_t i = 0; i < m_halos.size(); ++i) {
      auto &h = m_halos[i];
      if (aware) {
        exchange::isend_data(h.send_values, m_rank, h.peer_rank, m_comm,
                             &m_requests[req], h.send_tag);
      } else if (h.send_values.empty()) {
        m_requests[req] = MPI_REQUEST_NULL;
      } else {
        const int count = pfc::mpi::ensure_mpi_int_count(
            h.send_values.size(), "SparseExchange host-stage send");
        pfc::mpi::throw_on_mpi_error(MPI_Isend(m_host_send[i].data(), count,
                                               MPI_DOUBLE, h.peer_rank, h.send_tag,
                                               m_comm, &m_requests[req]),
                                     "SparseExchange host-stage MPI_Isend");
      }
      ++req;
    }
    m_request_count = static_cast<int>(req);
  }

  void finish_device_(T *field_ptr, std::size_t field_size) {
    exchange::wait_all(m_requests.data(), m_request_count);
    const bool aware = pfc::gpu::runtime_mpi_gpu_aware();
    for (std::size_t i = 0; i < m_halos.size(); ++i) {
      auto &h = m_halos[i];
      if (!aware && !h.recv_values.empty()) {
        copy_h2d_(h.recv_values.data().data(), m_host_recv[i].data(),
                  h.recv_values.size() * sizeof(T),
                  "SparseExchange host-stage recv H2D");
      }
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
  std::vector<std::vector<T>> m_host_send;
  std::vector<std::vector<T>> m_host_recv;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
class SparseExchange<CUDASpace, T>
    : public detail::DeviceSparseExchange<CUDASpace, backend::CUDATag, T> {
  using Base = detail::DeviceSparseExchange<CUDASpace, backend::CUDATag, T>;

public:
  using Base::Base;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
class SparseExchange<HIPSpace, T>
    : public detail::DeviceSparseExchange<HIPSpace, backend::HIPTag, T> {
  using Base = detail::DeviceSparseExchange<HIPSpace, backend::HIPTag, T>;

public:
  using Base::Base;
};
#endif

} // namespace pfc::comm

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
