// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "resident.hpp"
#include <hip/hip_runtime.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    int devices = 0;
    coupled_rk::require(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
                        "actual HIP hardware required");
    coupled_rk::require(hipSetDevice(coupled_rk::rank() % devices) == hipSuccess,
                        "select actual HIP device");
    hipDeviceProp_t prop{};
    coupled_rk::require(
        hipGetDeviceProperties(&prop, coupled_rk::rank() % devices) == hipSuccess,
        "device metadata");
    std::fprintf(
        stderr,
        "resident RK rank=%d visible=%d selected=%d device=%s architecture=%s\n",
        coupled_rk::rank(), devices, coupled_rk::rank() % devices, prop.name,
        prop.gcnArchName);
    result = coupled_rk::run_resident();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
