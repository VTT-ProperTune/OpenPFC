// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file exchange_cuda.hpp
 * @brief CUDA overloads for exchange (runtime/cuda only)
 *
 * Include this header when using exchange::send, send_data, receive_data,
 * isend_data, irecv_data with SparseVector<CudaTag, T>.
 *
 * Device-pointer MPI (Isend/Irecv/Send/Recv into CudaTag buffers) requires
 * GPU-aware MPI: compile-time `OpenPFC_MPI_CUDA_AWARE` and runtime
 * `MPIX_Query_cuda_support() == 1` (same probe as
 * `pfc::cuda::detail::runtime_mpi_cuda_aware` in
 * `padded_device_halo_exchange.hpp`). When unaware, blocking
 * `send_data`/`receive_data`/`send` host-stage via `cudaMemcpy`; non-blocking
 * `isend_data`/`irecv_data` fail closed with `std::runtime_error` (never a
 * silent no-op / `MPI_REQUEST_NULL` success).
 *
 * @see kernel/decomposition/exchange.hpp for CPU and interface
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#include <openpfc/runtime/cuda/cuda_check.hpp>

#if defined(OpenPFC_MPI_CUDA_AWARE) && defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#ifndef OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT 1
#endif
#endif

namespace pfc {
namespace exchange {
namespace detail {

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

} // namespace detail

template <typename T>
void send(core::SparseVector<backend::CudaTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  // Match CPU exchange.hpp: always post one size word (including 0).
  if (size == 0) {
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
        "MPI_Send");
    return;
  }

  std::vector<size_t> indices(size);
  std::vector<T> data(size);
  pfc::cuda::detail::cuda_check(
      cudaMemcpy(indices.data(), sparse_vector.indices().data(),
                 size * sizeof(size_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy indices D2H (exchange::send)");
  pfc::cuda::detail::cuda_check(
      cudaMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy data D2H (exchange::send)");

  pfc::mpi::throw_on_mpi_error(
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
      "MPI_Send");
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(indices.data(), static_cast<int>(size), MPI_UNSIGNED_LONG_LONG,
               receiver_rank, tag + 1, comm),
      "MPI_Send");
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(data.data(), static_cast<int>(size), mpi_type, receiver_rank,
               tag + 2, comm),
      "MPI_Send");
}

template <typename T>
void send_data(const core::SparseVector<backend::CudaTag, T> &sparse_vector,
               int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

  if (detail::runtime_mpi_cuda_aware()) {
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag,
                 comm),
        "MPI_Send");
  } else {
    std::vector<T> data(size);
    pfc::cuda::detail::cuda_check(
        cudaMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H (exchange::send_data)");
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(data.data(), count, mpi_type, receiver_rank, tag, comm),
        "MPI_Send");
  }
}

template <typename T>
void receive_data(core::SparseVector<backend::CudaTag, T> &sparse_vector,
                  int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &my_rank), "MPI_Comm_rank");

  if (my_rank != receiver_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

  if (detail::runtime_mpi_cuda_aware()) {
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
    pfc::cuda::detail::cuda_check(
        cudaMemcpy(sparse_vector.data().data(), data.data(), size * sizeof(T),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D (exchange::receive_data)");
  }
}

template <typename T>
void isend_data(const core::SparseVector<backend::CudaTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
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

  if (!detail::runtime_mpi_cuda_aware()) {
    *request = MPI_REQUEST_NULL;
    detail::throw_device_nb_requires_aware("isend_data");
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);
  pfc::mpi::throw_on_mpi_error(
      MPI_Isend(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag,
                comm, request),
      "MPI_Isend");
}

template <typename T>
void irecv_data(core::SparseVector<backend::CudaTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
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

  if (!detail::runtime_mpi_cuda_aware()) {
    *request = MPI_REQUEST_NULL;
    detail::throw_device_nb_requires_aware("irecv_data");
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);
  pfc::mpi::throw_on_mpi_error(
      MPI_Irecv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag,
                comm, request),
      "MPI_Irecv");
}

} // namespace exchange
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
