// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <kobayashi/defaults.hpp>
#include <kobayashi/device_step_cuda.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace kobayashi {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

__device__ inline std::size_t brick_lin(int i, int j, int k, int hw, int nx_pad,
                                        int ny_pad) noexcept {
  const auto h = static_cast<std::size_t>(hw);
  return (static_cast<std::size_t>(i) + h) +
         (static_cast<std::size_t>(j) + h) * static_cast<std::size_t>(nx_pad) +
         (static_cast<std::size_t>(k) + h) * static_cast<std::size_t>(nx_pad) *
             static_cast<std::size_t>(ny_pad);
}

/** Periodic x/y slab halos (`hw == 1`, `nz == 1`): one thread per `j` in `[0, ny)`. */
__global__ void kobayashi_periodic_halos_x_edges_kernel(double *pad, int nx, int ny, int hw,
                                                        int nx_pad, int ny_pad) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < 0 || j >= ny) {
    return;
  }
  constexpr int k = 0;
  pad[brick_lin(-1, j, k, hw, nx_pad, ny_pad)] =
      pad[brick_lin(nx - 1, j, k, hw, nx_pad, ny_pad)];
  pad[brick_lin(nx, j, k, hw, nx_pad, ny_pad)] = pad[brick_lin(0, j, k, hw, nx_pad, ny_pad)];
}

/**
 * Periodic y edges including corners; `i` runs `-1 … nx` inclusive (`nx+2` values).
 * Must run after `kobayashi_periodic_halos_x_edges_kernel`.
 */
__global__ void kobayashi_periodic_halos_y_edges_kernel(double *pad, int nx, int ny, int hw,
                                                        int nx_pad, int ny_pad) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int count = nx + 2;
  if (t < 0 || t >= count) {
    return;
  }
  constexpr int k = 0;
  const int i = -1 + t;
  pad[brick_lin(i, -1, k, hw, nx_pad, ny_pad)] =
      pad[brick_lin(i, ny - 1, k, hw, nx_pad, ny_pad)];
  pad[brick_lin(i, ny, k, hw, nx_pad, ny_pad)] = pad[brick_lin(i, 0, k, hw, nx_pad, ny_pad)];
}

/** Thin slab: periodic z halos (`k = -1` and `k = nz` mirror owned `k = 0 … nz-1`). */
__global__ void kobayashi_periodic_halos_z_edges_hw1_kernel(double *pad, int nx, int ny, int nz,
                                                             int hw, int nx_pad, int ny_pad) {
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int nline = (nx + 2) * (ny + 2);
  if (tid < 0 || tid >= nline) {
    return;
  }
  const int py = tid / (nx + 2);
  const int px = tid % (nx + 2);
  const int i = -1 + px;
  const int j = -1 + py;
  pad[brick_lin(i, j, -1, hw, nx_pad, ny_pad)] =
      pad[brick_lin(i, j, nz - 1, hw, nx_pad, ny_pad)];
  pad[brick_lin(i, j, nz, hw, nx_pad, ny_pad)] =
      pad[brick_lin(i, j, 0, hw, nx_pad, ny_pad)];
}

__global__ void kobayashi_stage_a_kernel(const double *phi, const double *tempr,
                                         double *lap_phi, double *lap_t,
                                         double *phidx, double *phidy,
                                         double *epsilon, double *epsilon_deriv,
                                         int nx, int ny, int /*nz*/, int hw,
                                         double inv_dx, double inv_dy,
                                         double inv_lap_den) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= nx || iy >= ny) {
    return;
  }
  constexpr int iz = 0;
  const int nx_pad = nx + 2 * hw;
  const int ny_pad = ny + 2 * hw;

  const std::size_t c = brick_lin(ix, iy, iz, hw, nx_pad, ny_pad);

  const double hnc = phi[c];
  const double hne = phi[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)];
  const double hnw = phi[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)];
  const double hns = phi[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)];
  const double hnn = phi[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)];
  lap_phi[c] = (hne + hnw + hns + hnn - 4.0 * hnc) * inv_lap_den;

  const double Tnc = tempr[c];
  const double Tne = tempr[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)];
  const double Tnw = tempr[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)];
  const double Tns = tempr[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)];
  const double Tnn = tempr[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)];
  lap_t[c] = (Tne + Tnw + Tns + Tnn - 4.0 * Tnc) * inv_lap_den;

  const double dpx =
      (phi[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)] -
       phi[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)]) *
      inv_dx;
  const double dpy =
      (phi[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)] -
       phi[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)]) *
      inv_dy;
  phidx[c] = dpx;
  phidy[c] = dpy;

  const double theta = atan2(dpy, dpx);
  epsilon[c] = kEpsilonb * (1.0 + kDelta * cos(kAniso * (theta - kTheta0)));
  epsilon_deriv[c] =
      -kEpsilonb * kAniso * kDelta * sin(kAniso * (theta - kTheta0));
}

__global__ void kobayashi_stage_b_kernel(double *phi, double *tempr,
                                         const double *lap_phi, const double *lap_t,
                                         const double *epsilon,
                                         const double *epsilon_deriv,
                                         const double *phidx, const double *phidy,
                                         int nx, int ny, int /*nz*/, int hw,
                                         double inv_dx, double inv_dy, double dt) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= nx || iy >= ny) {
    return;
  }
  constexpr int iz = 0;
  const int nx_pad = nx + 2 * hw;
  const int ny_pad = ny + 2 * hw;

  const std::size_t c = brick_lin(ix, iy, iz, hw, nx_pad, ny_pad);

  const double phiold = phi[c];

  const double term1 =
      (epsilon[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)] *
           epsilon_deriv[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)] *
           phidx[brick_lin(ix, iy + 1, iz, hw, nx_pad, ny_pad)] -
       epsilon[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)] *
           epsilon_deriv[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)] *
           phidx[brick_lin(ix, iy - 1, iz, hw, nx_pad, ny_pad)]) *
      inv_dy;

  const double term2 =
      -(epsilon[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)] *
            epsilon_deriv[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)] *
            phidy[brick_lin(ix + 1, iy, iz, hw, nx_pad, ny_pad)] -
        epsilon[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)] *
            epsilon_deriv[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)] *
            phidy[brick_lin(ix - 1, iy, iz, hw, nx_pad, ny_pad)]) *
      inv_dx;

  const double ep = epsilon[c];
  const double term3 = ep * ep * lap_phi[c];

  const double m =
      kAlpha / kPi * atan(kGamma * (kTeq - tempr[c]));
  const double term4 = phiold * (1.0 - phiold) * (phiold - 0.5 + m);

  phi[c] = phiold + (dt / kTau) * (term1 + term2 + term3 + term4);

  tempr[c] = tempr[c] + dt * lap_t[c] + kKappa * (phi[c] - phiold);
}

void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

void launch_rect(int nx, int ny, dim3 *grid, dim3 *block) {
  // Default 32² matches large GPUs (e.g. H100); override with OPENPFC_KOBAYASHI_CUDA_BLOCK=16 etc.
  int bs = 32;
  if (const char *env = std::getenv("OPENPFC_KOBAYASHI_CUDA_BLOCK")) {
    const int v = std::atoi(env);
    if (v >= 8 && v <= 32) {
      bs = v;
    }
  }
  block->x = static_cast<unsigned>(bs);
  block->y = static_cast<unsigned>(bs);
  block->z = 1;
  grid->x = (static_cast<unsigned>(nx) + block->x - 1) / block->x;
  grid->y = (static_cast<unsigned>(ny) + block->y - 1) / block->y;
  grid->z = 1;
}

} // namespace

void kobayashi_stage_a_cuda(const double *phi_dev, const double *tempr_dev,
                            double *lap_phi_dev, double *lap_t_dev, double *phidx_dev,
                            double *phidy_dev, double *epsilon_dev,
                            double *epsilon_deriv_dev, int nx, int ny, int nz, int hw,
                            double inv_dx, double inv_dy, double inv_lap_den) {
  if (nz != 1) {
    throw std::runtime_error("kobayashi_stage_a_cuda: nz must be 1 (2D slab)");
  }
  dim3 grid{};
  dim3 block{};
  launch_rect(nx, ny, &grid, &block);
  kobayashi_stage_a_kernel<<<grid, block>>>(
      phi_dev, tempr_dev, lap_phi_dev, lap_t_dev, phidx_dev, phidy_dev, epsilon_dev,
      epsilon_deriv_dev, nx, ny, nz, hw, inv_dx, inv_dy, inv_lap_den);
  cuda_check(cudaGetLastError(), "kobayashi_stage_a_cuda launch");
}

void kobayashi_stage_b_cuda(double *phi_dev, double *tempr_dev,
                            const double *lap_phi_dev, const double *lap_t_dev,
                            const double *epsilon_dev, const double *epsilon_deriv_dev,
                            const double *phidx_dev, const double *phidy_dev, int nx, int ny,
                            int nz, int hw, double inv_dx, double inv_dy, double dt) {
  if (nz != 1) {
    throw std::runtime_error("kobayashi_stage_b_cuda: nz must be 1 (2D slab)");
  }
  dim3 grid{};
  dim3 block{};
  launch_rect(nx, ny, &grid, &block);
  kobayashi_stage_b_kernel<<<grid, block>>>(
      phi_dev, tempr_dev, lap_phi_dev, lap_t_dev, epsilon_dev, epsilon_deriv_dev,
      phidx_dev, phidy_dev, nx, ny, nz, hw, inv_dx, inv_dy, dt);
  cuda_check(cudaGetLastError(), "kobayashi_stage_b_cuda launch");
}

void kobayashi_periodic_halos_xy_cuda(double *pad_dev, int nx, int ny, int nz, int hw) {
  if (hw != 1) {
    throw std::runtime_error("kobayashi_periodic_halos_xy_cuda: only hw==1 is implemented");
  }
  const int nx_pad = nx + 2 * hw;
  const int ny_pad = ny + 2 * hw;
  constexpr int threads = 256;
  {
    const int blocks = (ny + threads - 1) / threads;
    kobayashi_periodic_halos_x_edges_kernel<<<blocks, threads>>>(
        pad_dev, nx, ny, hw, nx_pad, ny_pad);
    cuda_check(cudaGetLastError(), "kobayashi_periodic_halos_x_edges_kernel");
  }
  {
    const int count = nx + 2;
    const int blocks = (count + threads - 1) / threads;
    kobayashi_periodic_halos_y_edges_kernel<<<blocks, threads>>>(
        pad_dev, nx, ny, hw, nx_pad, ny_pad);
    cuda_check(cudaGetLastError(), "kobayashi_periodic_halos_y_edges_kernel");
  }
  {
    const int nline = (nx + 2) * (ny + 2);
    const int blocks = (nline + threads - 1) / threads;
    kobayashi_periodic_halos_z_edges_hw1_kernel<<<blocks, threads>>>(
        pad_dev, nx, ny, nz, hw, nx_pad, ny_pad);
    cuda_check(cudaGetLastError(), "kobayashi_periodic_halos_z_edges_hw1_kernel");
  }
  // No `cudaDeviceSynchronize` here: same default stream as `kobayashi_stage_*` orders work.
}

} // namespace kobayashi

#endif // OpenPFC_ENABLE_CUDA
