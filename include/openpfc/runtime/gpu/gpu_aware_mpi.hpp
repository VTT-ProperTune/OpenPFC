// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gpu_aware_mpi.hpp
 * @brief How OpenPFC decides whether MPI accepts device pointers (M4).
 *
 * @details
 * Order:
 *   1. `OPENPFC_ASSUME_GPU_AWARE_MPI=0|1` — hard override.
 *   2. Open MPI `MPIX_Query_cuda_support` / `MPIX_Query_hip_support` when
 *      the compile was GPU-aware and `<mpi-ext.h>` is present.
 *   3. Cray MPICH: `MPICH_GPU_SUPPORT_ENABLED=1` plus a HIP or CUDA-aware
 *      compile (the LUMI contract; Open MPI's query is absent here).
 *   4. Optional `OPENPFC_PROBE_GPU_AWARE_MPI=1` — device-pointer Sendrecv
 *      smoke test (can crash if MPI is not actually device-aware).
 *
 * CUDA execution of the probe is not available on LUMI; HIP is.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <mpi.h>

#if (defined(OpenPFC_MPI_CUDA_AWARE) || defined(OpenPFC_MPI_HIP_AWARE)) &&          \
    defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#ifndef OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT 1
#endif
#ifndef OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT 1
#endif
#endif

#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
#include <hip/hip_runtime.h>
#endif

namespace pfc::gpu {

/// How `decide_gpu_aware_mpi` reached its answer (for tests and the log).
enum class GpuAwareMpiHow {
  AssumeOff,
  AssumeOn,
  OpenMpiQueryOn,
  OpenMpiQueryOff,
  CrayMpichEnv,
  ProbeOn,
  ProbeOff,
  CompileTimeOff,
};

struct GpuAwareMpiDecision {
  bool enabled = false;
  GpuAwareMpiHow how = GpuAwareMpiHow::CompileTimeOff;
};

inline bool env_first_char(const char *name, char expected) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] == expected;
}

inline const char *gpu_aware_how_cstr(GpuAwareMpiHow how) {
  switch (how) {
  case GpuAwareMpiHow::AssumeOff: return "OPENPFC_ASSUME_GPU_AWARE_MPI=0";
  case GpuAwareMpiHow::AssumeOn: return "OPENPFC_ASSUME_GPU_AWARE_MPI=1";
  case GpuAwareMpiHow::OpenMpiQueryOn: return "MPIX_Query=1";
  case GpuAwareMpiHow::OpenMpiQueryOff: return "MPIX_Query=0";
  case GpuAwareMpiHow::CrayMpichEnv: return "MPICH_GPU_SUPPORT_ENABLED=1";
  case GpuAwareMpiHow::ProbeOn: return "OPENPFC_PROBE_GPU_AWARE_MPI ok";
  case GpuAwareMpiHow::ProbeOff: return "OPENPFC_PROBE_GPU_AWARE_MPI failed";
  case GpuAwareMpiHow::CompileTimeOff: return "not compiled GPU-aware";
  }
  return "unknown";
}

#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
inline bool probe_hip_device_mpi() {
  int n_dev = 0;
  if (hipGetDeviceCount(&n_dev) != hipSuccess || n_dev <= 0) {
    return false;
  }
  double *d = nullptr;
  if (hipMalloc(reinterpret_cast<void **>(&d), sizeof(double)) != hipSuccess) {
    return false;
  }
  const double send = 42.0;
  if (hipMemcpy(d, &send, sizeof(double), hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(d);
    return false;
  }
  MPI_Status st{};
  const int err =
      MPI_Sendrecv(d, 1, MPI_DOUBLE, 0, 701, d, 1, MPI_DOUBLE, 0, 701,
                   MPI_COMM_SELF, &st);
  double got = 0.0;
  const bool ok_copy =
      hipMemcpy(&got, d, sizeof(double), hipMemcpyDeviceToHost) == hipSuccess;
  hipFree(d);
  return err == MPI_SUCCESS && ok_copy && got == send;
}
#endif

#if defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
inline bool probe_cuda_device_mpi() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
    return false;
  }
  double *d = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&d), sizeof(double)) != cudaSuccess) {
    return false;
  }
  const double send = 42.0;
  if (cudaMemcpy(d, &send, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d);
    return false;
  }
  MPI_Status st{};
  const int err =
      MPI_Sendrecv(d, 1, MPI_DOUBLE, 0, 701, d, 1, MPI_DOUBLE, 0, 701,
                   MPI_COMM_SELF, &st);
  double got = 0.0;
  const bool ok_copy =
      cudaMemcpy(&got, d, sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d);
  return err == MPI_SUCCESS && ok_copy && got == send;
}
#endif

/// Pure decision (no cache). Safe to call after setenv in tests.
inline GpuAwareMpiDecision decide_gpu_aware_mpi() {
  GpuAwareMpiDecision d{};
#if !defined(OpenPFC_MPI_CUDA_AWARE) && !defined(OpenPFC_MPI_HIP_AWARE)
  d.how = GpuAwareMpiHow::CompileTimeOff;
  return d;
#endif
  if (env_first_char("OPENPFC_ASSUME_GPU_AWARE_MPI", '0')) {
    d.how = GpuAwareMpiHow::AssumeOff;
    return d;
  }
  if (env_first_char("OPENPFC_ASSUME_GPU_AWARE_MPI", '1')) {
    d.enabled = true;
    d.how = GpuAwareMpiHow::AssumeOn;
    return d;
  }
#if defined(OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT) && defined(OpenPFC_MPI_CUDA_AWARE)
  d.enabled = MPIX_Query_cuda_support() == 1;
  d.how = d.enabled ? GpuAwareMpiHow::OpenMpiQueryOn : GpuAwareMpiHow::OpenMpiQueryOff;
  return d;
#endif
#if defined(OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT) && defined(OpenPFC_MPI_HIP_AWARE)
  d.enabled = MPIX_Query_hip_support() == 1;
  d.how = d.enabled ? GpuAwareMpiHow::OpenMpiQueryOn : GpuAwareMpiHow::OpenMpiQueryOff;
  return d;
#endif
  if (env_first_char("MPICH_GPU_SUPPORT_ENABLED", '1')) {
    d.enabled = true;
    d.how = GpuAwareMpiHow::CrayMpichEnv;
    return d;
  }
  if (env_first_char("OPENPFC_PROBE_GPU_AWARE_MPI", '1')) {
#if defined(OpenPFC_ENABLE_HIP) && defined(OpenPFC_MPI_HIP_AWARE)
    d.enabled = probe_hip_device_mpi();
#elif defined(OpenPFC_ENABLE_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)
    d.enabled = probe_cuda_device_mpi();
#endif
    d.how = d.enabled ? GpuAwareMpiHow::ProbeOn : GpuAwareMpiHow::ProbeOff;
    return d;
  }
  d.how = GpuAwareMpiHow::CompileTimeOff;
  return d;
}

/// Cached decision + one-shot rank-0 log to stderr.
inline bool runtime_mpi_gpu_aware() {
  static const GpuAwareMpiDecision d = []() {
    const GpuAwareMpiDecision r = decide_gpu_aware_mpi();
    int rank = 0;
    int mpi_ready = 0;
    MPI_Initialized(&mpi_ready);
    if (mpi_ready != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      std::fprintf(stderr, "[openpfc] GPU-aware MPI: %s (%s)\n",
                   r.enabled ? "on" : "off", gpu_aware_how_cstr(r.how));
    }
    return r;
  }();
  return d.enabled;
}

} // namespace pfc::gpu
