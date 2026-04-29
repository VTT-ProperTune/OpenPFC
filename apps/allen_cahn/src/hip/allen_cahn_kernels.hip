// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_HIP)

#include <allen_cahn/device_step.hpp>
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace allen_cahn {
namespace {

__global__ void
allen_cahn_step_kernel(double *core, const double *hpx, const double *hnx,
                       const double *hpy, const double *hny, const double * /*hpz*/,
                       const double * /*hnz*/, int nx, int ny, int nz, int hw,
                       double inv_dx2, double inv_dy2, double dt, double M,
                       double inv_eps2, int imin, int imax, int jmin, int jmax) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= nx || iy >= ny) {
    return;
  }
  constexpr int iz = 0;
  if (nz != 1) {
    return;
  }
  const int sxy = nx * ny;
  const std::size_t c = static_cast<std::size_t>(ix) +
                        static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
                        static_cast<std::size_t>(iz) * static_cast<std::size_t>(sxy);
  const double p = core[c];
  double lap = 0.0;
  if (ix >= imin && ix < imax && iy >= jmin && iy < jmax) {
    const double uc = p;
    const double uxm =
        (ix > imin)
            ? core[c - 1]
            : hpx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                  static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                  static_cast<std::size_t>(hw - 1)];
    const double uxp =
        (ix + 1 < imax)
            ? core[c + 1]
            : hnx[static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny * hw) +
                  static_cast<std::size_t>(iy) * static_cast<std::size_t>(hw) +
                  static_cast<std::size_t>(ix + 1 - (nx - hw))];
    const double uym =
        (iy > jmin)
            ? core[c - static_cast<std::size_t>(nx)]
            : hpy[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                  static_cast<std::size_t>(hw - 1) * static_cast<std::size_t>(nx) +
                  static_cast<std::size_t>(ix)];
    const double uyp =
        (iy + 1 < jmax)
            ? core[c + static_cast<std::size_t>(nx)]
            : hny[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * hw) +
                  static_cast<std::size_t>(iy + 1 - (ny - hw)) *
                      static_cast<std::size_t>(nx) +
                  static_cast<std::size_t>(ix)];
    const double dxx = uxp + uxm - 2.0 * uc;
    const double dyy = uyp + uym - 2.0 * uc;
    lap = inv_dx2 * dxx + inv_dy2 * dyy;
  }
  core[c] = p + dt * (M * lap - inv_eps2 * (p * p * p - p));
}

} // namespace

void allen_cahn_step_hip(double *core_dev, const double *hpx_dev,
                         const double *hnx_dev, const double *hpy_dev,
                         const double *hny_dev, const double *hpz_dev,
                         const double *hnz_dev, int nx, int ny, int nz,
                         int halo_width, double inv_dx2, double inv_dy2, double dt,
                         double M, double inv_eps2) {
  const int hw = halo_width;
  const int imin = hw;
  const int imax = nx - hw;
  const int jmin = hw;
  const int jmax = ny - hw;
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(ny) + block.y - 1) / block.y);
  hipLaunchKernelGGL(allen_cahn_step_kernel, grid, block, 0, 0, core_dev, hpx_dev,
                     hnx_dev, hpy_dev, hny_dev, hpz_dev, hnz_dev, nx, ny, nz, hw,
                     inv_dx2, inv_dy2, dt, M, inv_eps2, imin, imax, jmin, jmax);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("allen_cahn_step_hip: ") +
                             hipGetErrorString(err));
  }
  err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("allen_cahn_step_hip launch: ") +
                             hipGetErrorString(err));
  }
}

} // namespace allen_cahn

#endif // OpenPFC_ENABLE_HIP
