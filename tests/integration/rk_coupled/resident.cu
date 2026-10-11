// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "resident.hpp"
#include <cuda_runtime.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    int devices = 0;
    coupled_rk::require(cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0,
                        "actual CUDA hardware required");
    coupled_rk::require(cudaSetDevice(coupled_rk::rank() % devices) == cudaSuccess,
                        "select actual CUDA device");
    cudaDeviceProp prop{};
    coupled_rk::require(
        cudaGetDeviceProperties(&prop, coupled_rk::rank() % devices) == cudaSuccess,
        "device metadata");
    std::fprintf(stderr, "resident RK rank=%d visible=%d selected=%d device=%s\n",
                 coupled_rk::rank(), devices, coupled_rk::rank() % devices,
                 prop.name);
    result = coupled_rk::run_resident();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
