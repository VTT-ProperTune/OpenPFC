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

__global__ void wave2d_patch_y_faces_kernel(const double *u, double *hpy,
                                            double *hny, int nx, int ny,
                                            int lower_y, int Ny, int dirichlet,
                                            double u_wall) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ix >= nx) {
    return;
  }
  constexpr int hw = 1;
  constexpr int iz = 0;
  const std::size_t c = static_cast<std::size_t>(ix) +
                        static_cast<std::size_t>(iz) *
                            static_cast<std::size_t>(nx) * static_cast<std::size_t>(hw);
  if (lower_y == 0 && hny != nullptr) {
    if (dirichlet != 0) {
      const std::size_t uc = static_cast<std::size_t>(ix) +
                             static_cast<std::size_t>(iz) *
                                 static_cast<std::size_t>(nx * ny);
      hny[c] = 2.0 * u_wall - u[uc];
    } else if (ny >= 2) {
      const std::size_t uc1 =
          static_cast<std::size_t>(ix) + static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hny[c] = u[uc1];
    }
  }
  if (lower_y + ny - 1 == Ny - 1 && hpy != nullptr) {
    if (dirichlet != 0) {
      const std::size_t uc =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(ny - 1) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hpy[c] = 2.0 * u_wall - u[uc];
    } else if (ny >= 2) {
      const std::size_t ucm =
          static_cast<std::size_t>(ix) +
          static_cast<std::size_t>(ny - 2) * static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
      hpy[c] = u[ucm];
    }
  }
}

__global__ void wave2d_enforce_dirichlet_walls_kernel(double *u, double *v, int nx,
                                                     int ny, int lower_y, int Ny,
                                                     double u_wall) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= nx || iy >= ny) {
    return;
  }
  const int gy = lower_y + iy;
  if (gy != 0 && gy != Ny - 1) {
    return;
  }
  const std::size_t idx = static_cast<std::size_t>(ix) +
                          static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx);
  u[idx] = u_wall;
  v[idx] = 0.0;
}

void cuda_check_k(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
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

void wave2d_patch_y_faces_cuda(const double *u_dev, double *hpy_dev, double *hny_dev,
                               int nx, int ny, int lower_y, int Ny_global,
                               bool dirichlet, double u_wall) {
  if (nx < 1 || ny < 1) {
    return;
  }
  dim3 block(256);
  dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x);
  wave2d_patch_y_faces_kernel<<<grid, block>>>(u_dev, hpy_dev, hny_dev, nx, ny,
                                               lower_y, Ny_global, dirichlet ? 1 : 0,
                                               u_wall);
  cuda_check_k(cudaDeviceSynchronize(), "wave2d_patch_y_faces_cuda sync");
  cuda_check_k(cudaGetLastError(), "wave2d_patch_y_faces_cuda launch");
}

void wave2d_enforce_dirichlet_walls_cuda(double *u_dev, double *v_dev, int nx,
                                         int ny, int lower_y, int Ny_global,
                                         double u_wall) {
  if (nx < 1 || ny < 1) {
    return;
  }
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(ny) + block.y - 1) / block.y);
  wave2d_enforce_dirichlet_walls_kernel<<<grid, block>>>(u_dev, v_dev, nx, ny, lower_y,
                                                         Ny_global, u_wall);
  cuda_check_k(cudaDeviceSynchronize(), "wave2d_enforce_dirichlet_walls_cuda sync");
  cuda_check_k(cudaGetLastError(), "wave2d_enforce_dirichlet_walls_cuda launch");
}

} // namespace wave2d

#endif // OpenPFC_ENABLE_CUDA
