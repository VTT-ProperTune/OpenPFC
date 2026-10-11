// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "resident.hpp"
#include <hip/hip_runtime.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    int devices = 0;
    nfield_rk::require(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
                       "actual HIP hardware required");
    nfield_rk::require(hipSetDevice(nfield_rk::rank() % devices) == hipSuccess,
                       "select actual HIP device");
    result = nfield_rk::run_resident();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
