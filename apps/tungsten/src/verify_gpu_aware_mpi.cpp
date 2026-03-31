// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Minimal check: MPI_Send/MPI_Recv with device pointers (GPU-aware MPI).
// Requires MPICH_GPU_SUPPORT_ENABLED=1 on Cray MPICH (LUMI-G).

#include <mpi.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

static void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(e));
    std::abort();
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    if (rank == 0) {
      std::printf(
          "[verify_gpu_aware_mpi] SKIP: need at least 2 MPI ranks "
          "(e.g. srun -n2 ...)\n");
    }
    MPI_Finalize();
    return 0;
  }

  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &node_comm);
  int local_rank = 0;
  if (node_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
  }

  int n_dev = 0;
  hip_check(hipGetDeviceCount(&n_dev), "hipGetDeviceCount");
  if (n_dev <= 0) {
    if (rank == 0) {
      std::fprintf(stderr, "[verify_gpu_aware_mpi] FAIL: no HIP devices\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  hip_check(hipSetDevice(local_rank % n_dev), "hipSetDevice");

  double *d_buf = nullptr;
  hip_check(hipMalloc(&d_buf, sizeof(double)), "hipMalloc");

  const double send_val = 1000.0 + static_cast<double>(rank);
  hip_check(hipMemcpy(d_buf, &send_val, sizeof(double), hipMemcpyHostToDevice),
            "hipMemcpy H2D");

  MPI_Status st{};
  if (rank == 0) {
    MPI_Send(d_buf, 1, MPI_DOUBLE, 1, 42, MPI_COMM_WORLD);
  } else if (rank == 1) {
    MPI_Recv(d_buf, 1, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD, &st);
  }

  double host = 0.0;
  if (rank == 1) {
    hip_check(hipMemcpy(&host, d_buf, sizeof(double), hipMemcpyDeviceToHost),
              "hipMemcpy D2H");
  }

  hipFree(d_buf);

  int ok_local = (rank != 1) ? 1 : (host == 1000.0) ? 1 : 0;
  int ok = 0;
  MPI_Allreduce(&ok_local, &ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

  if (rank == 0) {
    const char *env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
    std::printf("[verify_gpu_aware_mpi] MPICH_GPU_SUPPORT_ENABLED=%s\n",
                env && env[0] ? env : "(unset)");
    if (ok) {
      std::printf(
          "[verify_gpu_aware_mpi] OK: device-buffer MPI_Send/Recv succeeded.\n");
    } else {
      std::fprintf(
          stderr,
          "[verify_gpu_aware_mpi] FAIL: wrong payload on rank 1 (GPU-aware "
          "MPI likely off or broken).\n");
    }
  }

  MPI_Finalize();
  return ok ? 0 : 1;
}
