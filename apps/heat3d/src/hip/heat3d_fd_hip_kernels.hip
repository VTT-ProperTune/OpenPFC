// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "heat3d_fd_hip kernels require HIP"
#endif

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include <heat3d/device_step.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>

namespace {

void hip_check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

struct HeatDeviceModel {
  OPENPFC_HD double rhs(double /*t*/, const heat3d::HeatGrads &g) const noexcept {
    return heat3d::kD * (g.xx + g.yy + g.zz);
  }
};

__global__ void euler_axpy_kernel(double *u, const double *du, double dt,
                                  std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    u[i] += dt * du[i];
  }
}

} // namespace

namespace heat3d {

void fd_rhs_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval,
                double *du_padded, double t, int nx, int ny, int nz) {
  HeatDeviceModel model;
  pfc::sim::gpu::for_each_interior_device<HeatDeviceModel, HeatGrads>(
      model, eval, du_padded, t, nx, ny, nz);
}

void euler_axpy_hip(double *u, const double *du, double dt, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((n + static_cast<std::size_t>(block) - 1) /
                                    static_cast<std::size_t>(block));
  euler_axpy_kernel<<<grid, block>>>(u, du, dt, n);
  hip_check(hipGetLastError(), "heat3d euler_axpy launch");
  hip_check(hipDeviceSynchronize(), "heat3d euler_axpy sync");
}

} // namespace heat3d
