// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file bind_local_device.hpp
 * @brief Bind this MPI rank to one GPU on the node (`local_rank % n_devices`).
 *
 * Without this, every rank on a multi-GPU node uses device 0. FD CUDA/HIP
 * apps already bind in `main`; GPU spectral stacks call this from the
 * constructor so tungsten/aluminum ETD sessions pick up the same mapping.
 *
 * Leave every GCD on the node visible (`ROCR_VISIBLE_DEVICES` unset). Pinning
 * one device per rank with a visibility mask hides the others from GPU-aware
 * MPI, so intra-node HeFFTe reshapes cannot use HIP IPC / xGMI.
 */

#include <stdexcept>
#include <string>

#include <openpfc/runtime/common/launch_rank.hpp>

#include <mpi.h>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc::runtime::gpu {

/// Pin the local GPU before MPI_Init so Cray MPICH GPU NIC policy sees it.
inline void bind_local_device_before_mpi() {
  const int local_rank = pfc::runtime::local_rank_from_launch_env();
  if (local_rank < 0) {
    return;
  }
#if defined(OpenPFC_ENABLE_CUDA)
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
    return;
  }
  (void)cudaSetDevice(local_rank % n_dev);
#elif defined(OpenPFC_ENABLE_HIP)
  int n_dev = 0;
  if (hipGetDeviceCount(&n_dev) != hipSuccess || n_dev < 1) {
    return;
  }
  (void)hipSetDevice(local_rank % n_dev);
#else
  (void)local_rank;
#endif
}

inline void bind_local_device(MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  int local_rank = 0;
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
  }

#if defined(OpenPFC_ENABLE_CUDA)
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
    throw std::runtime_error("bind_local_device: no CUDA devices visible");
  }
  if (cudaSetDevice(local_rank % n_dev) != cudaSuccess) {
    throw std::runtime_error("bind_local_device: cudaSetDevice(" +
                             std::to_string(local_rank % n_dev) + ") failed");
  }
#elif defined(OpenPFC_ENABLE_HIP)
  int n_dev = 0;
  if (hipGetDeviceCount(&n_dev) != hipSuccess || n_dev < 1) {
    throw std::runtime_error("bind_local_device: no HIP devices visible");
  }
  if (hipSetDevice(local_rank % n_dev) != hipSuccess) {
    throw std::runtime_error("bind_local_device: hipSetDevice(" +
                             std::to_string(local_rank % n_dev) + ") failed");
  }
#else
  (void)local_rank;
#endif
}

} // namespace pfc::runtime::gpu
