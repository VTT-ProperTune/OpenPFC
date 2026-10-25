// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "heat3d_fd_hip kernels require HIP"
#endif

#include <hip/hip_runtime.h>
#include <openpfc/runtime/gpu/check_error.hpp>

#include <stdexcept>
#include <string>

#include <heat3d/device_step.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/runtime/hip/for_each_interior_device.hpp>

namespace {

using pfc::runtime::gpu::hip_check;

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

void fd_rhs_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval, double *du_padded,
                double t, int nx, int ny, int nz) {
  HeatDeviceModel model;
  pfc::sim::gpu::for_each_interior_device<HeatDeviceModel, HeatGrads>(
      model, eval, du_padded, t, nx, ny, nz);
}

void fd_rhs_inner_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval,
                      double *du_padded, double t, int nx, int ny, int nz, int hw,
                      bool sync, hipStream_t stream) {
  HeatDeviceModel model;
  pfc::sim::gpu::for_each_inner_device<HeatDeviceModel, HeatGrads>(
      model, eval, du_padded, t, nx, ny, nz, hw, stream, sync);
}

void fd_rhs_border_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval,
                       double *du_padded, double t, int nx, int ny, int nz, int hw,
                       bool sync) {
  HeatDeviceModel model;
  pfc::sim::gpu::for_each_border_device<HeatDeviceModel, HeatGrads>(
      model, eval, du_padded, t, nx, ny, nz, hw, nullptr, sync);
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
