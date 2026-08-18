// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_halo_exchange_gpu.hpp
 * @brief Device `pfc::comm::HaloExchange` for `CudaSpace` / `HipSpace` (M4).
 *
 * @details
 * Composes the existing `PaddedDeviceHaloExchanger` (Faces) and
 * `FullPaddedDeviceHalo` (Full) so the unified name matches the host facade.
 * Device exchangers are blocking-only: `start()` / `finish()` and
 * `persistent` fail closed. Pack kernels are double-only.
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
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
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
  static_assert(std::is_same_v<T, double>,
                "pfc::comm::HaloExchange on device is double-only "
                "(existing pack/unpack kernels)");

public:
  using FieldT = data::Field<T, Space>;

  DeviceHaloExchange(FieldT &field, const decomposition::Decomposition &decomp,
                     int rank, MPI_Comm comm, HaloExchangeOptions opt = {})
      : DeviceHaloExchange(std::vector<FieldT *>{&field}, decomp, rank, comm,
                           opt) {}

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
      const int tag0 = halo::field_tag_base(m_opt.exchange_base, static_cast<int>(i));
      if (m_opt.connectivity == HaloConnectivity::Full) {
        m_full.push_back(std::make_unique<FullEx>(decomp, rank, f->storage_halo(),
                                                  comm, /*n_fields=*/1, tag0));
      } else {
        m_faces.push_back(std::make_unique<FaceEx>(*f, decomp, rank, comm,
                                                   halo::presets::Axes3D(), tag0));
      }
    }
  }

  /// Blocking exchange of every bound field (default device stream).
  void exchange() {
    for (auto *f : m_fields) {
      f->sync_to_device();
    }
    if (!m_full.empty()) {
      for (std::size_t i = 0; i < m_full.size(); ++i) {
        T *ptr = m_fields[i]->data();
        m_full[i]->exchange(&ptr, nullptr);
      }
    } else {
      for (std::size_t i = 0; i < m_faces.size(); ++i) {
        m_faces[i]->exchange_halos_device(*m_fields[i]);
      }
    }
    for (auto *f : m_fields) {
      f->note_device_write();
    }
  }

  void start() {
    throw std::logic_error(
        "pfc::comm::HaloExchange::start: device exchangers are blocking-only; "
        "use exchange()");
  }

  void finish() {
    throw std::logic_error(
        "pfc::comm::HaloExchange::finish: device exchangers are blocking-only; "
        "use exchange()");
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
  HaloExchangeOptions m_opt{};
  std::vector<FieldT *> m_fields;
  std::vector<std::unique_ptr<FaceEx>> m_faces;
  std::vector<std::unique_ptr<FullEx>> m_full;
};

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
class HaloExchange<CudaSpace, T>
    : public detail::DeviceHaloExchange<CudaSpace, cuda::PaddedDeviceHaloExchanger,
                                        cuda::FullPaddedDeviceHalo, T> {
  using Base =
      detail::DeviceHaloExchange<CudaSpace, cuda::PaddedDeviceHaloExchanger,
                                 cuda::FullPaddedDeviceHalo, T>;

public:
  using Base::Base;
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
class HaloExchange<HipSpace, T>
    : public detail::DeviceHaloExchange<HipSpace, hip::PaddedDeviceHaloExchanger,
                                        hip::FullPaddedDeviceHalo, T> {
  using Base =
      detail::DeviceHaloExchange<HipSpace, hip::PaddedDeviceHaloExchanger,
                                 hip::FullPaddedDeviceHalo, T>;

public:
  using Base::Base;
};
#endif

} // namespace pfc::comm

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
