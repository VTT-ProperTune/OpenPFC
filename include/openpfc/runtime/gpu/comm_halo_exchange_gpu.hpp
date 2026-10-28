// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_halo_exchange_gpu.hpp
 * @brief Device `pfc::comm::HaloExchange` for `CUDASpace` / `HIPSpace` (M4).
 *
 * @details
 * Composes the Faces backend (`gpu::DeviceFacesHalo`) and the Full
 * backend (`gpu::DeviceFullHalo`) so the unified name matches the host
 * facade. Faces supports split-phase `start()` / `finish()` (and optional
 * `progress()` / `MPI_Testall`) so interior stencil work can run while
 * MPI is in flight. Full stays blocking because each widening pass must
 * complete before the next. `persistent` still fails closed. Elements are
 * `double`, `float` or `std::uint32_t` (`gpu::is_halo_element_v`); halos are
 * bitwise copies, so integer labels arrive exactly.
 *
 * Faces dependency graph (GPU-aware contiguous path):
 * 1. `start()`: stream-sync so `u` is ready; copy self-periodic faces;
 *    post Irecv; pack all remote send faces; stream-sync packs; post Isend;
 *    return (no `Waitall`).
 * 2. Caller computes halo-independent interior of `du` (reads owned `u`
 *    excluding an `hw` shell; does not write `u`). Launch that kernel on a
 *    *non-blocking* stream *before* `start()` if pack/MPI should overlap
 *    it; the default stream would drain a regular side stream.
 * 3. `finish()`: `Waitall`; unpack received faces into `u` halo; stream-sync.
 * 4. Caller computes the owned boundary shell of `du`, then updates `u`.
 *
 * `start()` is not a no-op wrapper around `exchange()`. Hidden waits after
 * the posts would serialize the interior kernel. Pack still has to finish
 * before `MPI_Isend` of those device buffers.
 *
 * Include this header for device fields. The host header stays free of
 * runtime/gpu includes (kernel must not depend on runtime).
 *
 * CUDA execution is not available on LUMI; HIP can run here.
 *
 * @see kernel/decomposition/comm_halo_exchange.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/full_padded_device_halo_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::comm {
namespace detail {

/**
 * @brief Shared device HaloExchange body, stamped per vendor space.
 */
template <typename Space, typename FaceEx, typename FullEx, typename T>
class DeviceHaloExchange {
  static_assert(gpu::is_halo_element_v<T>,
                "pfc::comm::HaloExchange on device supports double, float and "
                "std::uint32_t elements");

public:
  using FieldT = data::Field<T, Space>;

  DeviceHaloExchange(FieldT &field, const decomposition::Decomposition &decomp,
                     int rank, MPI_Comm comm, HaloExchangeOptions opt = {})
      : DeviceHaloExchange(std::vector<FieldT *>{&field}, decomp, rank, comm, opt) {}

  DeviceHaloExchange(std::vector<FieldT *> fields,
                     const decomposition::Decomposition &decomp, int rank,
                     MPI_Comm comm, HaloExchangeOptions opt = {})
      : m_opt(opt), m_fields(std::move(fields)) {
    if (m_fields.empty()) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: at least one field is required");
    }
    if (m_opt.persistent) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: persistent requests are host Faces-only "
          "(device exchangers have no persistent path)");
    }
    m_faces.reserve(m_fields.size());
    m_full.reserve(m_fields.size());
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
      const auto dirs = halo::resolve_direction_set(resolved_halo_directions(m_opt),
                                                    m_opt.selector, rank);
      if (m_opt.connectivity == HaloConnectivity::Full) {
        m_full.push_back(std::make_unique<FullEx>(decomp, rank, f->storage_halo(),
                                                  comm, /*n_fields=*/1, dirs, tag0));
      } else {
        m_faces.push_back(
            std::make_unique<FaceEx>(*f, decomp, rank, comm, dirs, tag0));
      }
    }
  }

  /// Blocking exchange of every bound field (default device stream).
  ///
  /// Faces is `start()` then `finish()`. Full stays sequential because each
  /// axis pass must complete before the next.
  void exchange() {
    if (!m_full.empty()) {
      if (m_phase != Phase::Idle) {
        throw std::logic_error(
            "pfc::comm::HaloExchange::exchange: Faces split-phase in flight");
      }
      for (auto *f : m_fields) {
        f->sync_to_device();
      }
      for (std::size_t i = 0; i < m_full.size(); ++i) {
        T *ptr = m_fields[i]->data();
        m_full[i]->exchange(&ptr, nullptr);
      }
      for (auto *f : m_fields) {
        f->note_device_write();
      }
      return;
    }
    start();
    finish();
  }

  /**
   * @brief Pack, post Irecv/Isend, and return without waiting.
   *
   * @throws std::logic_error if connectivity is Full, or if an exchange is
   *         already in flight.
   */
  void start() {
    require_faces_split_("start");
    if (m_phase != Phase::Idle) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::start: exchange already in flight");
    }
    for (auto *f : m_fields) {
      f->sync_to_device();
    }
    for (std::size_t i = 0; i < m_faces.size(); ++i) {
      m_faces[i]->start_halos_device(*m_fields[i]);
    }
    m_phase = Phase::Posted;
  }

  /**
   * @brief `MPI_Testall` on the in-flight Faces requests.
   *
   * Returns true when every request has completed (including the zero-request
   * self-periodic case). Does not unpack. Use this to pump Cray MPICH while
   * an interior kernel runs; do not add a progress thread from the caller.
   *
   * @throws std::logic_error if `start()` was not called.
   */
  bool progress() {
    require_faces_split_("progress");
    if (m_phase == Phase::Waited) {
      return true;
    }
    if (m_phase != Phase::Posted) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::progress: no in-flight exchange");
    }
    std::vector<MPI_Request> all = gather_outstanding_();
    int flag = 0;
    const int n = static_cast<int>(all.size());
    pfc::mpi::throw_on_mpi_error(
        MPI_Testall(n, n > 0 ? all.data() : nullptr, &flag, MPI_STATUSES_IGNORE),
        "HaloExchange::progress MPI_Testall");
    if (flag == 0) {
      return false;
    }
    mark_waited_(all);
    return true;
  }

  /// Wait (if still posted) and unpack received faces.
  void finish() {
    require_faces_split_("finish");
    if (m_phase == Phase::Idle) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::finish: no in-flight exchange");
    }
    if (m_phase == Phase::Posted) {
      wait_concatenated(m_faces);
      m_phase = Phase::Waited;
    }
    for (std::size_t i = 0; i < m_faces.size(); ++i) {
      m_faces[i]->complete_halos_device(*m_fields[i]);
    }
    for (auto *f : m_fields) {
      f->note_device_write();
    }
    m_phase = Phase::Idle;
  }

  [[nodiscard]] HaloConnectivity connectivity() const noexcept {
    return m_opt.connectivity;
  }
  [[nodiscard]] bool persistent() const noexcept { return m_opt.persistent; }
  [[nodiscard]] std::size_t num_fields() const noexcept { return m_fields.size(); }

  /// True when Faces mode selected GPU-aware MPI (Full does not expose this).
  [[nodiscard]] bool uses_gpu_aware_mpi() const noexcept {
    return !m_faces.empty() && m_faces.front()->uses_gpu_aware_mpi();
  }

  /// Pack-to-contiguous + device-pointer MPI (default GPU-aware transport).
  [[nodiscard]] bool uses_contiguous_device_mpi() const noexcept {
    if (!m_faces.empty()) {
      return m_faces.front()->uses_contiguous_device_mpi();
    }
    return !m_full.empty() && m_full.front()->uses_contiguous_device_mpi();
  }

private:
  enum class Phase { Idle, Posted, Waited };

  void require_faces_split_(const char *what) const {
    if (!m_full.empty()) {
      throw std::logic_error(std::string("pfc::comm::HaloExchange::") + what +
                             ": Full connectivity has no split-phase API; "
                             "use exchange()");
    }
  }

  std::vector<MPI_Request> gather_outstanding_() {
    std::vector<MPI_Request> all;
    for (auto &e : m_faces) {
      const int n = e->outstanding_count();
      if (n > 0) {
        MPI_Request *r = e->outstanding();
        all.insert(all.end(), r, r + n);
      }
    }
    return all;
  }

  void mark_waited_(const std::vector<MPI_Request> &all) {
    std::size_t off = 0;
    for (auto &e : m_faces) {
      const int n = e->outstanding_count();
      e->take_waitall_result(n > 0 ? all.data() + off : nullptr, n);
      off += static_cast<std::size_t>(n);
    }
    m_phase = Phase::Waited;
  }

  HaloExchangeOptions m_opt{};
  std::vector<FieldT *> m_fields;
  std::vector<std::unique_ptr<FaceEx>> m_faces;
  std::vector<std::unique_ptr<FullEx>> m_full;
  Phase m_phase = Phase::Idle;
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
class HaloExchange<CUDASpace, T>
    : public detail::DeviceHaloExchange<
          CUDASpace, gpu::DeviceFacesHalo<cuda::CUDAHaloOps, T>,
          gpu::DeviceFullHalo<cuda::CUDAHaloOps, T>, T> {
  using Base =
      detail::DeviceHaloExchange<CUDASpace,
                                 gpu::DeviceFacesHalo<cuda::CUDAHaloOps, T>,
                                 gpu::DeviceFullHalo<cuda::CUDAHaloOps, T>, T>;

public:
  using Base::Base;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
class HaloExchange<HIPSpace, T>
    : public detail::DeviceHaloExchange<HIPSpace,
                                        gpu::DeviceFacesHalo<hip::HIPHaloOps, T>,
                                        gpu::DeviceFullHalo<hip::HIPHaloOps, T>, T> {
  using Base =
      detail::DeviceHaloExchange<HIPSpace, gpu::DeviceFacesHalo<hip::HIPHaloOps, T>,
                                 gpu::DeviceFullHalo<hip::HIPHaloOps, T>, T>;

public:
  using Base::Base;
};
#endif

} // namespace pfc::comm

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
