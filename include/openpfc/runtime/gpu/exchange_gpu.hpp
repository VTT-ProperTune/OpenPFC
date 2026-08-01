// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file exchange_gpu.hpp
 * @brief Single-source GPU exchange operations for CUDA and HIP backends (M3)
 *
 * Unified implementation for CUDA and HIP backends using the GPU vendor shim.
 * Provides exchange operations for SparseVector with GPU memory backends.
 *
 * Include this header when using exchange::send, send_data, receive_data,
 * isend_data, irecv_data with GPU-enabled SparseVector types.
 *
 * @see kernel/decomposition/exchange.hpp for CPU and interface
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <openpfc/runtime/cuda/backend_tags_cuda.hpp>
#elif defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#endif

#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
namespace {
bool runtime_mpi_gpu_aware() { return MPIX_Query_cuda_support() == 1; }
} // namespace
#elif defined(OpenPFC_ENABLE_HIP) && defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
namespace {
bool runtime_mpi_gpu_aware() { return MPIX_Query_hip_support() == 1; }
} // namespace
#else
namespace {
bool runtime_mpi_gpu_aware() { return false; }
} // namespace
#endif

namespace pfc {
namespace exchange {
namespace detail {

[[noreturn]] inline void throw_device_nb_requires_aware(const char *op) {
  throw std::runtime_error(
      std::string("exchange::") + op +
      " (GPU-tagged SparseVector): GPU-aware MPI is required for non-blocking "
      "device exchange (OpenPFC_MPI_*_AWARE + MPIX_Query_*_support). "
      "Use blocking send_data/receive_data (host-staged) or enable "
      "device-aware MPI.");
}

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)

template <typename T>
void send(core::SparseVector<backend::CudaTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
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
  GPU_CHECK(gpuMemcpyAsync(indices.data(), sparse_vector.indices().data(),
                           size * sizeof(size_t), cudaMemcpyDeviceToHost, nullptr),
            "gpuMemcpyAsync indices D2H (exchange::send)");
  GPU_CHECK(gpuMemcpyAsync(data.data(), sparse_vector.data().data(),
                           size * sizeof(T), cudaMemcpyDeviceToHost, nullptr),
            "gpuMemcpyAsync data D2H (exchange::send)");
  gpuDeviceSynchronize();

  pfc::mpi::throw_on_mpi_error(
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
      "MPI_Send");
  pfc::mpi::throw_on_mpi_error(MPI_Send(indices.data(), count,
                                        MPI_UNSIGNED_LONG_LONG, receiver_rank,
                                        tag + 1, comm),
                               "MPI_Send");
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(data.data(), count, mpi_type, receiver_rank, tag + 2, comm),
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
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::send_data");

  if (runtime_mpi_gpu_aware()) {
    pfc::mpi::throw_on_mpi_error(MPI_Send(sparse_vector.data().data(), count,
                                          mpi_type, receiver_rank, tag, comm),
                                 "MPI_Send");
  } else {
    std::vector<T> data(size);
    GPU_CHECK(gpuMemcpyAsync(data.data(), sparse_vector.data().data(),
                             size * sizeof(T), cudaMemcpyDeviceToHost, nullptr),
              "gpuMemcpyAsync D2H (exchange::send_data)");
    gpuDeviceSynchronize();
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
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::receive_data");

  if (runtime_mpi_gpu_aware()) {
    pfc::mpi::throw_on_mpi_error(MPI_Recv(sparse_vector.data().data(), count,
                                          mpi_type, sender_rank, tag, comm,
                                          MPI_STATUS_IGNORE),
                                 "MPI_Recv");
  } else {
    std::vector<T> data(size);
    pfc::mpi::throw_on_mpi_error(MPI_Recv(data.data(), count, mpi_type, sender_rank,
                                          tag, comm, MPI_STATUS_IGNORE),
                                 "MPI_Recv");
    GPU_CHECK(gpuMemcpyAsync(sparse_vector.data().data(), data.data(),
                             size * sizeof(T), cudaMemcpyHostToDevice, nullptr),
              "gpuMemcpyAsync H2D (exchange::receive_data)");
    gpuDeviceSynchronize();
  }
}

template <typename T>
void isend_data(const core::SparseVector<backend::CudaTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  if (!runtime_mpi_gpu_aware()) {
    detail::throw_device_nb_requires_aware("isend_data");
  }

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

  *request = MPI_REQUEST_NULL;
}

#elif defined(OpenPFC_ENABLE_HIP)

template <typename T>
void send(core::SparseVector<backend::HipTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
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
  GPU_CHECK(gpuMemcpyAsync(indices.data(), sparse_vector.indices().data(),
                           size * sizeof(size_t), cudaMemcpyDeviceToHost, nullptr),
            "gpuMemcpyAsync indices D2H (exchange::send)");
  GPU_CHECK(gpuMemcpyAsync(data.data(), sparse_vector.data().data(),
                           size * sizeof(T), cudaMemcpyDeviceToHost, nullptr),
            "gpuMemcpyAsync data D2H (exchange::send)");
  gpuDeviceSynchronize();

  pfc::mpi::throw_on_mpi_error(
      MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm),
      "MPI_Send");
  pfc::mpi::throw_on_mpi_error(MPI_Send(indices.data(), count,
                                        MPI_UNSIGNED_LONG_LONG, receiver_rank,
                                        tag + 1, comm),
                               "MPI_Send");
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  pfc::mpi::throw_on_mpi_error(
      MPI_Send(data.data(), count, mpi_type, receiver_rank, tag + 2, comm),
      "MPI_Send");
}

template <typename T>
void send_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
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
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::send_data");

  if (runtime_mpi_gpu_aware()) {
    pfc::mpi::throw_on_mpi_error(MPI_Send(sparse_vector.data().data(), count,
                                          mpi_type, receiver_rank, tag, comm),
                                 "MPI_Send");
  } else {
    std::vector<T> data(size);
    GPU_CHECK(gpuMemcpyAsync(data.data(), sparse_vector.data().data(),
                             size * sizeof(T), cudaMemcpyDeviceToHost, nullptr),
              "gpuMemcpyAsync D2H (exchange::send_data)");
    gpuDeviceSynchronize();
    pfc::mpi::throw_on_mpi_error(
        MPI_Send(data.data(), count, mpi_type, receiver_rank, tag, comm),
        "MPI_Send");
  }
}

template <typename T>
void receive_data(core::SparseVector<backend::HipTag, T> &sparse_vector,
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
  const int count = pfc::mpi::ensure_mpi_int_count(size, "exchange::receive_data");

  if (runtime_mpi_gpu_aware()) {
    pfc::mpi::throw_on_mpi_error(MPI_Recv(sparse_vector.data().data(), count,
                                          mpi_type, sender_rank, tag, comm,
                                          MPI_STATUS_IGNORE),
                                 "MPI_Recv");
  } else {
    std::vector<T> data(size);
    pfc::mpi::throw_on_mpi_error(MPI_Recv(data.data(), count, mpi_type, sender_rank,
                                          tag, comm, MPI_STATUS_IGNORE),
                                 "MPI_Recv");
    GPU_CHECK(gpuMemcpyAsync(sparse_vector.data().data(), data.data(),
                             size * sizeof(T), cudaMemcpyHostToDevice, nullptr),
              "gpuMemcpyAsync H2D (exchange::receive_data)");
    gpuDeviceSynchronize();
  }
}

template <typename T>
void isend_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  if (!runtime_mpi_gpu_aware()) {
    detail::throw_device_nb_requires_aware("isend_data");
  }

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

  *request = MPI_REQUEST_NULL;
}

#endif

} // namespace exchange
} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP