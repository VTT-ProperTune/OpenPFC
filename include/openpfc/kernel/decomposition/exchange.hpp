// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
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
 * @see kernel/decomposition/sparse_vector.hpp for SparseVector definition
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <mpi.h>
#include <vector>

#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>

namespace pfc {
namespace exchange {

template <typename T> constexpr bool dependent_false_exchange = false;

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
 * @brief Zero-copy face exchange: send one face, receive one face (blocking)
 *
 * Uses MPI derived types so MPI reads/writes directly from buf. No gather/scatter.
 * @param buf Base pointer of field (row-major [nx,ny,nz])
 * @param send_type MPI_Datatype for send face (from halo::create_face_type)
 * @param recv_type MPI_Datatype for recv face
 * @param send_to_rank Destination rank for send
 * @param recv_from_rank Source rank for recv
 * @param comm MPI communicator
 * @param tag Message tag
 */
inline void sendrecv_face(void *buf, MPI_Datatype send_type, MPI_Datatype recv_type,
                          int send_to_rank, int recv_from_rank, MPI_Comm comm,
                          int tag = 0) {
  MPI_Sendrecv(buf, 1, send_type, send_to_rank, tag, buf, 1, recv_type,
               recv_from_rank, tag, comm, MPI_STATUS_IGNORE);
}

/**
 * @brief Zero-copy non-blocking send face (post Isend)
 */
inline void isend_face(void *buf, MPI_Datatype send_type, int send_to_rank,
                       MPI_Comm comm, MPI_Request *request, int tag = 0) {
  MPI_Isend(buf, 1, send_type, send_to_rank, tag, comm, request);
}

/**
 * @brief Zero-copy non-blocking receive face (post Irecv)
 */
inline void irecv_face(void *buf, MPI_Datatype recv_type, int recv_from_rank,
                       MPI_Comm comm, MPI_Request *request, int tag = 0) {
  MPI_Irecv(buf, 1, recv_type, recv_from_rank, tag, comm, request);
}

/**
 * @brief Non-blocking receive of contiguous elements (e.g. separated face halo)
 */
template <typename T>
inline void irecv_dense(T *buf, int count, int recv_from_rank, MPI_Comm comm,
                        MPI_Request *request, int tag = 0) {
  MPI_Irecv(buf, count, detail::get_mpi_type<T>(), recv_from_rank, tag, comm,
            request);
}

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
  } else {
    static_assert(
        dependent_false_exchange<BackendTag>,
        "CudaTag/HipTag: include openpfc/runtime/cuda/exchange_cuda.hpp or "
        "openpfc/runtime/hip/exchange_hip.hpp");
  }

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

  MPI_Datatype mpi_type = detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    MPI_Send(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag, comm);
  } else {
    static_assert(
        dependent_false_exchange<BackendTag>,
        "CudaTag/HipTag: include openpfc/runtime/cuda/exchange_cuda.hpp or "
        "openpfc/runtime/hip/exchange_hip.hpp");
  }
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

  MPI_Datatype mpi_type = exchange::detail::get_mpi_type<T>();
  int count = static_cast<int>(size);

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    MPI_Recv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag, comm,
             MPI_STATUS_IGNORE);
  } else {
    static_assert(
        dependent_false_exchange<BackendTag>,
        "CudaTag/HipTag: include openpfc/runtime/cuda/exchange_cuda.hpp or "
        "openpfc/runtime/hip/exchange_hip.hpp");
  }
}

/**
 * @brief Non-blocking send of data only - runtime phase
 *
 * Posts MPI_Isend; buffer must remain valid until wait_all() is called.
 * Callers should post all Irecv first, then all Isend, then wait_all (see
 * docs/halo_exchange.md).
 *
 * @param sparse_vector SparseVector to send data from (buffer must stay valid)
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param request Output: MPI_Request (must be passed to wait_all later)
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void isend_data(const core::SparseVector<BackendTag, T> &sparse_vector,
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

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    MPI_Isend(sparse_vector.data().data(), count, mpi_type, receiver_rank, tag, comm,
              request);
  } else {
    static_assert(
        dependent_false_exchange<BackendTag>,
        "CudaTag/HipTag: include openpfc/runtime/cuda/exchange_cuda.hpp or "
        "openpfc/runtime/hip/exchange_hip.hpp");
  }
}

/**
 * @brief Non-blocking receive of data only - runtime phase
 *
 * Posts MPI_Irecv into sparse_vector.data(); buffer must remain valid until
 * wait_all() is called.
 *
 * @param sparse_vector SparseVector to receive into (buffer must stay valid)
 * @param sender_rank MPI rank of sender
 * @param receiver_rank MPI rank of receiver
 * @param comm MPI communicator
 * @param request Output: MPI_Request (must be passed to wait_all later)
 * @param tag MPI message tag (default: 0)
 */
template <typename BackendTag, typename T>
void irecv_data(core::SparseVector<BackendTag, T> &sparse_vector, int sender_rank,
                int receiver_rank, MPI_Comm comm, MPI_Request *request,
                int tag = 0) {
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

  if constexpr (std::is_same_v<BackendTag, backend::CpuTag>) {
    MPI_Irecv(sparse_vector.data().data(), count, mpi_type, sender_rank, tag, comm,
              request);
  } else {
    static_assert(
        dependent_false_exchange<BackendTag>,
        "CudaTag/HipTag: include openpfc/runtime/cuda/exchange_cuda.hpp or "
        "openpfc/runtime/hip/exchange_hip.hpp");
  }
}

/**
 * @brief Wait for all non-blocking requests to complete
 *
 * Call after posting all Irecv then all Isend. Frees the requests (MPI standard).
 *
 * @param requests Array of MPI_Request (may contain MPI_REQUEST_NULL)
 * @param count Number of requests
 */
inline void wait_all(MPI_Request *requests, int count) {
  std::vector<MPI_Request> non_null;
  non_null.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (requests[i] != MPI_REQUEST_NULL) {
      non_null.push_back(requests[i]);
    }
  }
  if (!non_null.empty()) {
    MPI_Waitall(static_cast<int>(non_null.size()), non_null.data(),
                MPI_STATUSES_IGNORE);
  }
}

/** @brief Overload for vector of requests */
inline void wait_all(std::vector<MPI_Request> &requests) {
  wait_all(requests.data(), static_cast<int>(requests.size()));
}

} // namespace exchange
} // namespace pfc
