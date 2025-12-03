// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file exchange.hpp
 * @brief MPI exchange operations for SparseVector
 *
 * @details
 * Provides two-phase exchange for SparseVector:
 *
 * 1. **Setup phase** (expensive, done once):
 *    - `send(sparse_vec, sender_rank, receiver_rank, comm)`: Sends both indices and
 * data
 *    - `receive(sparse_vec, sender_rank, receiver_rank, comm)`: Receives both
 * indices and data
 *
 * 2. **Runtime phase** (cheap, done every step):
 *    - `send_data(sparse_vec, sender_rank, receiver_rank, comm)`: Sends only data
 * values
 *    - `receive_data(sparse_vec, sender_rank, receiver_rank, comm)`: Receives only
 * data values
 *
 * Key optimization: Indices are exchanged once, then only values are transferred.
 *
 * @code
 * // Setup: Exchange indices once
 * auto sparse = sparsevector::create<double>({0, 2, 4});
 * exchange::send(sparse, my_rank, neighbor_rank, MPI_COMM_WORLD);
 * exchange::receive(sparse, neighbor_rank, my_rank, MPI_COMM_WORLD);
 *
 * // Runtime: Exchange only values (repeatedly)
 * for (int step = 0; step < 1000; ++step) {
 *   exchange::send_data(sparse, my_rank, neighbor_rank, MPI_COMM_WORLD);
 *   exchange::receive_data(sparse, neighbor_rank, my_rank, MPI_COMM_WORLD);
 * }
 * @endcode
 *
 * @see core/sparse_vector.hpp for SparseVector definition
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <mpi.h>
#include <vector>

#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/sparse_vector.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace pfc {
namespace exchange {
namespace detail {

/**
 * @brief Get MPI datatype for type T
 */
template <typename T> MPI_Datatype get_mpi_type();

template <> inline MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }

template <> inline MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }

template <> inline MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template <> inline MPI_Datatype get_mpi_type<size_t>() {
  return MPI_UNSIGNED_LONG_LONG;
}

} // namespace detail

/**
 * @brief Send SparseVector (both indices and data) - setup phase
 *
 * Sends both indices and data to receiver. Used during initialization
 * to establish communication pattern.
 *
 * @param sparse_vector SparseVector to send
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void send(const core::SparseVector<BackendTag, T> &sparse_vector, int sender_rank,
          int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != sender_rank) {
    return; // Not the sender
  }

  size_t size = sparse_vector.size();

  // Send size first
  MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, receiver_rank, tag, comm);

  if (size == 0) {
    return;
  }

  // Get indices and data (may need to copy from device for CUDA)
  std::vector<size_t> indices;
  std::vector<T> data;

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    // CPU: Direct access
    indices.resize(size);
    data.resize(size);
    std::copy(sparse_vector.indices().data(), sparse_vector.indices().data() + size,
              indices.begin());
    std::copy(sparse_vector.data().data(), sparse_vector.data().data() + size,
              data.begin());
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
    // CUDA: Copy to host first
    indices.resize(size);
    data.resize(size);
    cudaMemcpy(indices.data(), sparse_vector.indices().data(), size * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
               cudaMemcpyDeviceToHost);
  }
#endif

  // Send indices
  MPI_Send(indices.data(), static_cast<int>(size), MPI_UNSIGNED_LONG_LONG,
           receiver_rank, tag + 1, comm);

  // Send data
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  MPI_Send(data.data(), static_cast<int>(size), mpi_type, receiver_rank, tag + 2,
           comm);
}

/**
 * @brief Receive SparseVector (both indices and data) - setup phase
 *
 * Receives both indices and data from sender. Used during initialization.
 *
 * @param sparse_vector SparseVector to receive into (will be resized)
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void receive(core::SparseVector<BackendTag, T> &sparse_vector, int sender_rank,
             int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != receiver_rank) {
    return; // Not the receiver
  }

  // Receive size first
  size_t size;
  MPI_Recv(&size, 1, MPI_UNSIGNED_LONG_LONG, sender_rank, tag, comm,
           MPI_STATUS_IGNORE);

  if (size == 0) {
    sparse_vector = core::SparseVector<BackendTag, T>(0);
    return;
  }

  // Receive indices and data
  std::vector<size_t> indices(size);
  std::vector<T> data(size);

  int count = static_cast<int>(size);
  MPI_Recv(indices.data(), count, MPI_UNSIGNED_LONG_LONG, sender_rank, tag + 1, comm,
           MPI_STATUS_IGNORE);

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  MPI_Recv(data.data(), count, mpi_type, sender_rank, tag + 2, comm,
           MPI_STATUS_IGNORE);

  // Create SparseVector from received data (will sort indices)
  sparse_vector = core::SparseVector<BackendTag, T>(indices, data);
}

/**
 * @brief Send only data values (indices already known) - runtime phase
 *
 * Sends only the data values, assuming receiver already knows the indices.
 * Much cheaper than full send.
 *
 * @param sparse_vector SparseVector to send data from
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void send_data(const core::SparseVector<BackendTag, T> &sparse_vector,
               int sender_rank, int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != sender_rank) {
    return; // Not the sender
  }

  size_t size = sparse_vector.size();

  if (size == 0) {
    return;
  }

  // Get data (may need to copy from device for CUDA)
  std::vector<T> data;

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    // CPU: Direct access
    data.resize(size);
    std::copy(sparse_vector.data().data(), sparse_vector.data().data() + size,
              data.begin());
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
    // CUDA: Copy to host first
    data.resize(size);
    cudaMemcpy(data.data(), sparse_vector.data().data(), size * sizeof(T),
               cudaMemcpyDeviceToHost);
  }
#endif

  // Send data only
  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  MPI_Send(data.data(), static_cast<int>(size), mpi_type, receiver_rank, tag, comm);
}

/**
 * @brief Receive only data values (indices already known) - runtime phase
 *
 * Receives only the data values, assuming indices are already set.
 * Much cheaper than full receive.
 *
 * @param sparse_vector SparseVector to receive data into
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void receive_data(core::SparseVector<BackendTag, T> &sparse_vector, int sender_rank,
                  int receiver_rank, MPI_Comm comm, int tag = 0) {
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);

  if (my_rank != receiver_rank) {
    return; // Not the receiver
  }

  size_t size = sparse_vector.size();

  if (size == 0) {
    return;
  }

  // Receive data
  std::vector<T> data(size);

  MPI_Datatype mpi_type = exchange::detail::get_mpi_type<T>();
  int count = static_cast<int>(size);
  MPI_Recv(data.data(), count, mpi_type, sender_rank, tag, comm, MPI_STATUS_IGNORE);

  // Copy to device if needed
  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    // CPU: Direct copy
    std::copy(data.begin(), data.end(), sparse_vector.data().data());
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<BackendTag, backend::CudaTag>) {
    // CUDA: Copy to device
    cudaMemcpy(sparse_vector.data().data(), data.data(), size * sizeof(T),
               cudaMemcpyHostToDevice);
  }
#endif
}

} // namespace exchange
} // namespace pfc
