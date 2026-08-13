// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file exchange_gpu.hpp
 * @brief Single-source GPU SparseVector MPI exchange for CUDA and HIP (M3).
 *
 * Overloads `exchange::send` / `send_data` / `receive_data` / `isend_data` /
 * `irecv_data` for `SparseVector<CudaTag>` and/or `SparseVector<HipTag>`.
 * Vendor headers `exchange_cuda.hpp` / `exchange_hip.hpp` are thin includes
 * of this file so existing call sites keep compiling.
 *
 * Per-tag memcpy and MPI-aware probes call the native runtime (not
 * `gpu_api.hpp`) so a CUDA+HIP co-enabled translation unit can exchange
 * both tags. Non-blocking device MPI still requires GPU-aware MPI; blocking
 * helpers host-stage when unaware.
 *
 * @see kernel/decomposition/exchange.hpp
 * @see runtime/gpu/gpu_check.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_check.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#if (defined(OpenPFC_MPI_CUDA_AWARE) || defined(OpenPFC_MPI_HIP_AWARE)) &&          \
    defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#if defined(OpenPFC_MPI_CUDA_AWARE)
#ifndef OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT 1
#endif
#endif
#if defined(OpenPFC_MPI_HIP_AWARE)
#ifndef OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT 1
#endif
#endif
#endif

namespace pfc {
namespace exchange {
namespace detail {

#if defined(OpenPFC_ENABLE_CUDA)
/** Same conditions as `pfc::cuda::detail::runtime_mpi_cuda_aware()`. */
inline bool runtime_mpi_cuda_aware() {
#if defined(OpenPFC_MPI_CUDA_AWARE) && defined(OPEN_MPI) &&                         \
    defined(OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT)
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}

[[noreturn]] inline void throw_device_nb_requires_aware(const char *op) {
  throw std::runtime_error(
      std::string("exchange::") + op +
      " (SparseVector<CudaTag>): GPU-aware MPI is required for non-blocking "
      "device exchange (OpenPFC_MPI_CUDA_AWARE + MPIX_Query_cuda_support). "
      "Use blocking send_data/receive_data (host-staged) or enable "
      "device-aware MPI.");
}

struct CudaXchg {
  using tag = backend::CudaTag;
  static bool mpi_aware() { return runtime_mpi_cuda_aware(); }
  static void throw_nb(const char *op) { throw_device_nb_requires_aware(op); }
  static void memcpy_d2h(void *dst, const void *src, std::size_t bytes,
                         const char *what) {
    pfc::cuda::detail::cuda_check(
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), what);
  }
  static void memcpy_h2d(void *dst, const void *src, std::size_t bytes,
                         const char *what) {
    pfc::cuda::detail::cuda_check(
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), what);
  }
  static constexpr const char *indices_d2h = "cudaMemcpy indices D2H (exchange::send)";
  static constexpr const char *data_d2h = "cudaMemcpy data D2H (exchange::send)";
  static constexpr const char *send_data_d2h = "cudaMemcpy D2H (exchange::send_data)";
  static constexpr const char *recv_data_h2d =
      "cudaMemcpy H2D (exchange::receive_data)";
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
/** Same conditions as `pfc::hip::detail::runtime_mpi_hip_aware()`. */
inline bool runtime_mpi_hip_aware() {
#if defined(OpenPFC_MPI_HIP_AWARE) && defined(OPEN_MPI) &&                         \
    defined(OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT)
  return MPIX_Query_hip_support() == 1;
#else
  return false;
#endif
}

[[noreturn]] inline void throw_hip_nb_requires_aware(const char *op) {
  throw std::runtime_error(
      std::string("exchange::") + op +
      " (SparseVector<HipTag>): GPU-aware MPI is required for non-blocking "
      "device exchange (OpenPFC_MPI_HIP_AWARE + MPIX_Query_hip_support). "
      "Use blocking send_data/receive_data (host-staged) or enable "
      "device-aware MPI.");
}

struct HipXchg {
  using tag = backend::HipTag;
  static bool mpi_aware() { return runtime_mpi_hip_aware(); }
  static void throw_nb(const char *op) { throw_hip_nb_requires_aware(op); }
  static void memcpy_d2h(void *dst, const void *src, std::size_t bytes,
                         const char *what) {
    pfc::hip::detail::hip_check(
        hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), what);
  }
  static void memcpy_h2d(void *dst, const void *src, std::size_t bytes,
                         const char *what) {
    pfc::hip::detail::hip_check(
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), what);
  }
  static constexpr const char *indices_d2h = "hipMemcpy indices D2H (exchange::send)";
  static constexpr const char *data_d2h = "hipMemcpy data D2H (exchange::send)";
  static constexpr const char *send_data_d2h = "hipMemcpy D2H (exchange::send_data)";
  static constexpr const char *recv_data_h2d =
      "hipMemcpy H2D (exchange::receive_data)";
};
#endif

template <typename T, typename Ops>
void gpu_send(core::SparseVector<typename Ops::tag, T> &sparse_vector,
              int sender_rank, int receiver_rank, MPI_Comm comm, int tag) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::send");
  if (size == 0) {
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
        "MPI_Send");
    return;
  }

  std::vector<size_t> indices(size);
  std::vector<T> data(size);
  Ops::memcpy_d2h(indices.data(), sparse_vector.indices().data(),
                  size * sizeof(size_t), Ops::indices_d2h);
  Ops::memcpy_d2h(data.data(), sparse_vector.data().data(), size * sizeof(T),
                  Ops::data_d2h);

  pfc::mpi::throw_on_mpi_error(
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
      "MPI_Send");
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(indices.data(), count, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag + 1,
               comm),
      "MPI_Send");
  MPI_Datatype mpi_type = get_mpi_type<T>();
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(data.data(), count, mpi_type, receiver_rank, tag + 2, comm),
      "MPI_Send");
}

template <typename T, typename Ops>
void gpu_send_data(const core::SparseVector<typename Ops::tag, T> &sparse_vector,
                   int sender_rank, int receiver_rank, MPI_Comm comm, int tag) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = get_mpi_type<T>();
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::send_data");

  if (Ops::mpi_aware()) {
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag,
                 comm),
        "MPI_Send");
  } else {
    std::vector<T> data(size);
    Ops::memcpy_d2h(data.data(), sparse_vector.data().data(), size * sizeof(T),
                    Ops::send_data_d2h);
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(data.data(), count, mpi_type, receiver_rank, tag, comm),
        "MPI_Send");
  }
}

template <typename T, typename Ops>
void gpu_receive_data(core::SparseVector<typename Ops::tag, T> &sparse_vector,
                      int sender_rank, int receiver_rank, MPI_Comm comm, int tag) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != receiver_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = get_mpi_type<T>();
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::receive_data");

  if (Ops::mpi_aware()) {
    pfc::mpi::throw_on_mpi_error(
        MPI_Recv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag,
                 comm, MPI_STATUS_IGNORE),
        "MPI_Recv");
  } else {
    std::vector<T> data(size);
    pfc::mpi::throw_on_mpi_error(
        MPI_Recv(data.data(), count, mpi_type, sender_rank, tag, comm,
                 MPI_STATUS_IGNORE),
        "MPI_Recv");
    Ops::memcpy_h2d(sparse_vector.data().data(), data.data(), size * sizeof(T),
                    Ops::recv_data_h2d);
  }
}

template <typename T, typename Ops>
void gpu_isend_data(const core::SparseVector<typename Ops::tag, T> &sparse_vector,
                    int sender_rank, int receiver_rank, MPI_Comm comm,
                    MPI_Request *request, int tag) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != sender_rank) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  if (!Ops::mpi_aware()) {
    *request = MPI_REQUEST_NULL;
    Ops::throw_nb("isend_data");
  }

  MPI_Datatype mpi_type = get_mpi_type<T>();
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::isend_data");
  pfc::mpi::throw_on_mpi_error(
      MPI_Isend(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag,
                comm, request),
      "MPI_Isend");
}

template <typename T, typename Ops>
void gpu_irecv_data(core::SparseVector<typename Ops::tag, T> &sparse_vector,
                    int sender_rank, int receiver_rank, MPI_Comm comm,
                    MPI_Request *request, int tag) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != receiver_rank) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  if (!Ops::mpi_aware()) {
    *request = MPI_REQUEST_NULL;
    Ops::throw_nb("irecv_data");
  }

  MPI_Datatype mpi_type = get_mpi_type<T>();
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::irecv_data");
  pfc::mpi::throw_on_mpi_error(
      MPI_Irecv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag, comm,
                request),
      "MPI_Irecv");
}

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
void send(core::SparseVector<backend::CudaTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_send<T, detail::CudaXchg>(sparse_vector, sender_rank, receiver_rank,
                                        comm, tag);
}

template <typename T>
void send_data(const core::SparseVector<backend::CudaTag, T> &sparse_vector,
               int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_send_data<T, detail::CudaXchg>(sparse_vector, sender_rank,
                                             receiver_rank, comm, tag);
}

template <typename T>
void receive_data(core::SparseVector<backend::CudaTag, T> &sparse_vector,
                  int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_receive_data<T, detail::CudaXchg>(sparse_vector, sender_rank,
                                                receiver_rank, comm, tag);
}

template <typename T>
void isend_data(const core::SparseVector<backend::CudaTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  detail::gpu_isend_data<T, detail::CudaXchg>(sparse_vector, sender_rank,
                                              receiver_rank, comm, request, tag);
}

template <typename T>
void irecv_data(core::SparseVector<backend::CudaTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  detail::gpu_irecv_data<T, detail::CudaXchg>(sparse_vector, sender_rank,
                                              receiver_rank, comm, request, tag);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
void send(core::SparseVector<backend::HipTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_send<T, detail::HipXchg>(sparse_vector, sender_rank, receiver_rank,
                                       comm, tag);
}

template <typename T>
void send_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
               int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_send_data<T, detail::HipXchg>(sparse_vector, sender_rank,
                                            receiver_rank, comm, tag);
}

template <typename T>
void receive_data(core::SparseVector<backend::HipTag, T> &sparse_vector,
                  int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  detail::gpu_receive_data<T, detail::HipXchg>(sparse_vector, sender_rank,
                                               receiver_rank, comm, tag);
}

template <typename T>
void isend_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  detail::gpu_isend_data<T, detail::HipXchg>(sparse_vector, sender_rank,
                                             receiver_rank, comm, request, tag);
}

template <typename T>
void irecv_data(core::SparseVector<backend::HipTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  detail::gpu_irecv_data<T, detail::HipXchg>(sparse_vector, sender_rank,
                                             receiver_rank, comm, request, tag);
}
#endif

} // namespace exchange
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
