// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <wave2d/device_step.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace wave2d {
namespace {

__global__ void wave2d_step_kernel(double *u_core, double *v_core, const double *hpx,
                                   const double *hnx, const double *hpy,
                                   const double *hny, int nx, int ny, int nz, int hw,
                                   double inv_dx2, double inv_dy2, double dt,
                                   double wave_c) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= nx || iy >= ny || nz != 1) {
    return;
  }
  constexpr int iz = 0;
  const int sxy = nx * ny;
  const std::size_t c = static_cast<std::size_t>(ix) +
                        static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                        static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
  const double uc = u_core[c];
  const double vc = v_core[c];
  const double uxm =
      (ix > 0)
          ? u_core[c - 1]
          : hnx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                static_cast<std::size_t>(hw - 1)];
  const double uxp =
      (ix + 1 < nx)
          ? u_core[c + 1]
          : hpx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw)];
  const double uym =
      (iy > 0)
          ? u_core[c - static_cast<std::size_t>(nx)]
          : hny[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                static_cast<std::size_t>(hw - 1) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
  const double uyp =
      (iy + 1 < ny)
          ? u_core[c + static_cast<std::size_t>(nx)]
          : hpy[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                static_cast<std::size_t>(ix)];
  const double lap =
      inv_dx2 * (uxp + uxm - 2.0 * uc) + inv_dy2 * (uyp + uym - 2.0 * uc);
  u_core[c] = uc + dt * vc;
  v_core[c] = vc + dt * wave_c * wave_c * lap;
}

} // namespace

void wave2d_step_cuda(double *u_dev, double *v_dev, const double *hpx_dev,
                      const double *hnx_dev, const double *hpy_dev,
                      const double *hny_dev, const double *hpz_dev,
                      const double *hnz_dev, int nx, int ny, int nz, int halo_width,
                      double inv_dx2, double inv_dy2, double dt, double wave_c) {
  (void)hpz_dev;
  (void)hnz_dev;
  const int hw = halo_width;
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(ny) + block.y - 1) / block.y);
  wave2d_step_kernel<<<grid, block>>>(u_dev, v_dev, hpx_dev, hnx_dev, hpy_dev,
                                      hny_dev, nx, ny, nz, hw, inv_dx2, inv_dy2, dt,
                                      wave_c);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("wave2d_step_cuda: ") +
                             cudaGetErrorString(err));
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("wave2d_step_cuda launch: ") +
                             cudaGetErrorString(err));
  }
}

} // namespace wave2d

#endif // OpenPFC_ENABLE_CUDA
