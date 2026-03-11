// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file exchange_hip.hpp
 * @brief HIP overloads for exchange (runtime/hip only)
 *
 * Include this header when using exchange::send, send_data, receive_data,
 * isend_data, irecv_data with SparseVector<HipTag, T>.
 *
 * @see kernel/decomposition/exchange.hpp for CPU and interface
 * @see runtime/cuda/exchange_cuda.hpp for CUDA
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/runtime/hip/backend_tags_hip.hpp>
#include <vector>

namespace pfc {
namespace exchange {

template <typename T>
void send(core::SparseVector<backend::HipTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    MPI_Send(nullptr, 0, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm);
    return;
  }

  std::vector<size_t> indices(size);
  std::vector<T> data(size);
  hipMemcpy(indices.data(), sparse_vector.indices().data(), size * sizeof(size_t),
            hipMemcpyDeviceToHost);
  hipMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
            hipMemcpyDeviceToHost);

  MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm);
  MPI_Send(indices.data(), static_cast<int>(size), MPI_UNSIGNED_LONG_LONG,
           receiver_rank, tag + 1, comm);
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  MPI_Send(data.data(), static_cast<int>(size), mpi_type, receiver_rank, tag + 2,
           comm);
}

template <typename T>
void send_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
               int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != sender_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

#if defined(OpenPFC_MPI_HIP_AWARE)
  MPI_Send(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag, comm);
#else
  std::vector<T> data(size);
  hipMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
            hipMemcpyDeviceToHost);
  MPI_Send(data.data(), count, mpi_type, receiver_rank, tag, comm);
#endif
}

template <typename T>
void receive_data(core::SparseVector<backend::HipTag, T> &sparse_vector,
                  int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != receiver_rank) {
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

#if defined(OpenPFC_MPI_HIP_AWARE)
  MPI_Recv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag, comm,
           MPI_STATUS_IGNORE);
#else
  std::vector<T> data(size);
  MPI_Recv(data.data(), count, mpi_type, sender_rank, tag, comm, MPI_STATUS_IGNORE);
  hipMemcpy(sparse_vector.data().data(), data.data(), size * sizeof(T),
            hipMemcpyHostToDevice);
#endif
}

template <typename T>
void isend_data(const core::SparseVector<backend::HipTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != sender_rank) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

#if defined(OpenPFC_MPI_HIP_AWARE)
  MPI_Isend(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag, comm,
            request);
#else
  (void)sparse_vector;
  (void)receiver_rank;
  (void)tag;
  (void)comm;
  *request = MPI_REQUEST_NULL;
#endif
}

template <typename T>
void irecv_data(core::SparseVector<backend::HipTag, T> &sparse_vector,
                int sender_rank, int receiver_rank, MPI_Comm comm,
                MPI_Request *request, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != receiver_rank) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  size_t size = sparse_vector.size();
  if (size == 0) {
    *request = MPI_REQUEST_NULL;
    return;
  }

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

#if defined(OpenPFC_MPI_HIP_AWARE)
  MPI_Irecv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag, comm,
            request);
#else
  (void)sparse_vector;
  (void)sender_rank;
  (void)tag;
  (void)comm;
  *request = MPI_REQUEST_NULL;
#endif
}

} // namespace exchange
} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
