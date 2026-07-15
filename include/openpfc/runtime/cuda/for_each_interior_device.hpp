// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file for_each_interior_device.hpp
 * @brief GPU twin of the canonical CPU driver loop
 *        `pfc::sim::for_each_interior` — point-wise apply a model's
 *        `rhs(t, g)` over every owned cell of a padded device buffer.
 *
 * @details
 * On the CPU side, [`pfc::sim::for_each_interior`](
 * ../../kernel/simulation/for_each_interior.hpp) is the four-line driver
 * that walks every interior cell, calls a per-point evaluator
 * (`FdGradient<G>` / `SpectralGradient<G>`) to materialise the model's
 * `G` aggregate, calls `model.rhs(t, g)` to compute the increment, and
 * scatters the result into a single (or multi-field tuple) output
 * buffer. This header is the GPU counterpart.
 *
 * **Single-field only**, for now. The CPU driver ships a multi-field
 * "tuple-protocol" overload too; the GPU equivalent is a small follow-up
 * (the kernel template just gains a tuple of pointers and a structured
 * scatter). Heat- and Kobayashi-style apps are single-field, so the
 * single-field path is what unlocks the SymPy-codegen kernel pipeline.
 *
 * **Output layout**: for the device path we keep the output increment
 * `du` in **the same padded layout** as the source field. Owned cells
 * are written; halo cells are left untouched. This matches the existing
 * Kobayashi `stage_a` / `stage_b` device kernels and lets the driver
 * loop hand the same pointer to the next halo exchange without any
 * pad ↔ unpad bookkeeping.
 *
 * **Usage** (inside a `.cu` translation unit that has access to the
 * model, the grads aggregate `G`, and the device evaluator POD):
 *
 * @code
 * pfc::cuda::FdGradientDevice<MyGrads> eval(d_padded_u, nx, ny, nz, dx, dy, dz,
 *                                           hw, order);
 * pfc::sim::cuda::for_each_interior_device(model, eval.pod(), d_padded_du, t,
 *                                          nx, ny, nz, hw, stream);
 * @endcode
 *
 * @see openpfc/runtime/cuda/fd_gradient_device.hpp — per-point evaluator
 *      consumed here
 * @see openpfc/kernel/simulation/for_each_interior.hpp — CPU twin
 * @see openpfc/runtime/cuda/full_padded_device_halo.hpp — corner-filled
 *      halo policy that should run before this driver when the model's
 *      grads aggregate uses any mixed second derivative
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <openpfc/runtime/cuda/fd_gradient_device.hpp>

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
#include "openpfc/runtime/common/gpu_autotune.hpp"
#endif

namespace pfc::sim::cuda {

namespace detail {

/**
 * @brief Single-field padded-output kernel.
 *
 * Each thread handles one owned-cell `(ix, iy, iz)` and:
 *   1. Builds `G g = evaluate_fd_grad<G>(eval, ix, iy, iz)`.
 *   2. Computes `inc = model.rhs(t, g)`.
 *   3. Writes `du_padded[c_padded] = inc`, where `c_padded` is the
 *      same linear index used by `evaluate_fd_grad` to read the field.
 *
 * The kernel takes both `Model` and `EvalPOD` by value, so they must be
 * trivially copyable. The model is **not** required to expose any
 * device-specific machinery beyond `rhs(t, g)` annotated `OPENPFC_HD`.
 */
template <class Model, class G>
__global__ void
for_each_interior_device_kernel(Model model, ::pfc::cuda::FdGradientDevicePOD eval,
                                double *du_padded, double t, int nx, int ny,
                                int nz) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    return;
  }
  G g = ::pfc::cuda::evaluate_fd_grad<G>(eval, ix, iy, iz);
  const double inc = model.rhs(t, g);

  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix + eval.hw) +
                           static_cast<std::ptrdiff_t>(iy + eval.hw) * eval.sy +
                           static_cast<std::ptrdiff_t>(iz + eval.hw) * eval.sz;
  du_padded[c] = inc;
}

} // namespace detail

/**
 * @brief Launch the per-point increment kernel over `(nx, ny, nz)` owned
 *        cells.
 *
 * @tparam Model   User-defined model type with an `OPENPFC_HD double
 *                 rhs(double t, const G &g) const noexcept` method.
 *                 Must be trivially copyable so it can be passed to the
 *                 kernel by value.
 * @tparam G       Per-point grads aggregate (`HasXx`, `HeatGrads`, ...).
 *                 Must be trivially constructible and have only POD
 *                 members for the fields detected by `pfc::field::has_*`.
 *
 * @param model       Model instance. Captured by value into the kernel.
 * @param eval        `FdGradientDevicePOD` returned by
 *                    `pfc::cuda::FdGradientDevice<G>::pod()`.
 * @param du_padded   Output buffer in the **padded** layout (same
 *                    extents as the source field's padded box, same
 *                    strides). Owned cells are written; halo cells are
 *                    left untouched.
 * @param t           Current simulation time forwarded to `model.rhs`.
 * @param nx,ny,nz    Owned-region extents of the local subdomain.
 * @param stream      CUDA stream to launch on (0 for the default stream).
 *
 * @throws std::runtime_error if the kernel launch fails. The host code
 *         path is otherwise free of CUDA error checking — a follow-up
 *         can wire `cudaGetLastError()` into the launcher.
 */
template <class Model, class G>
inline void for_each_interior_device(const Model &model,
                                     const ::pfc::cuda::FdGradientDevicePOD &eval,
                                     double *du_padded, double t, int nx, int ny,
                                     int nz, cudaStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  size_t total_elements = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  auto config = pfc::gpu::AutoTuner::instance().get_config("for_each_interior_3d", total_elements);
  dim3 block(config.block_size_x, config.block_size_y, config.block_size_z);
#else
  constexpr int Tx = 32;
  constexpr int Ty = 4;
  constexpr int Tz = 1;
  dim3 block(Tx, Ty, Tz);
#endif
  dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(ny) + block.y - 1) / block.y,
            (static_cast<unsigned>(nz) + block.z - 1) / block.z);
  detail::for_each_interior_device_kernel<Model, G>
      <<<grid, block, 0, stream>>>(model, eval, du_padded, t, nx, ny, nz);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("for_each_interior_device: kernel "
                                         "launch failed: ") +
                             cudaGetErrorString(e));
  }
}

} // namespace pfc::sim::cuda

#endif // OpenPFC_ENABLE_CUDA
