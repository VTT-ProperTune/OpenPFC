// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file for_each_interior_device.hpp
 * @brief HIP twin of the canonical CPU driver loop
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
 * buffer. This header is the HIP counterpart.
 *
 * **Single-field only**, for now. The CPU driver ships a multi-field
 * "tuple-protocol" overload too; the HIP equivalent is a small follow-up
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
 * **Usage** (inside a `.hip` translation unit that has access to the
 * model, the grads aggregate `G`, and the device evaluator POD):
 *
 * @code
 * pfc::hip::FdGradientDevice<MyGrads> eval(d_padded_u, nx, ny, nz, dx, dy, dz,
 *                                           hw, order);
 * pfc::sim::hip::for_each_interior_device(model, eval.pod(), d_padded_du, t,
 *                                          nx, ny, nz, hw, stream);
 * @endcode
 *
 * @see openpfc/runtime/hip/fd_gradient_device.hpp — per-point evaluator
 *      consumed here
 * @see openpfc/kernel/simulation/for_each_interior.hpp — CPU twin
 * @see openpfc/runtime/cuda/for_each_interior_device.hpp — CUDA twin
 */

#if defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include <openpfc/runtime/hip/fd_gradient_device.hpp>

namespace pfc::sim::hip {

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
for_each_interior_device_kernel(Model model, ::pfc::hip::FdGradientDevicePOD eval,
                                double *du_padded, double t, int nx, int ny,
                                int nz) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    return;
  }
  G g = ::pfc::hip::evaluate_fd_grad<G>(eval, ix, iy, iz);
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
 *                    `pfc::hip::FdGradientDevice<G>::pod()`.
 * @param du_padded   Output buffer in the **padded** layout (same
 *                    extents as the source field's padded box, same
 *                    strides). Owned cells are written; halo cells are
 *                    left untouched.
 * @param t           Current simulation time forwarded to `model.rhs`.
 * @param nx,ny,nz    Owned-region extents of the local subdomain.
 * @param stream      HIP stream to launch on (0 for the default stream).
 *
 * @throws std::runtime_error if the kernel launch fails. The host code
 *         path is otherwise free of HIP error checking — a follow-up
 *         can wire `hipGetLastError()` into the launcher.
 */
template <class Model, class G>
inline void for_each_interior_device(const Model &model,
                                     const ::pfc::hip::FdGradientDevicePOD &eval,
                                     double *du_padded, double t, int nx, int ny,
                                     int nz, hipStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  constexpr int Tx = 32;
  constexpr int Ty = 4;
  constexpr int Tz = 1;
  const dim3 block(Tx, Ty, Tz);
  const dim3 grid((static_cast<unsigned>(nx) + Tx - 1) / Tx,
                  (static_cast<unsigned>(ny) + Ty - 1) / Ty,
                  (static_cast<unsigned>(nz) + Tz - 1) / Tz);
  detail::for_each_interior_device_kernel<Model, G>
      <<<grid, block, 0, stream>>>(model, eval, du_padded, t, nx, ny, nz);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    throw std::runtime_error(std::string("for_each_interior_device: kernel "
                                         "launch failed: ") +
                             hipGetErrorString(e));
  }
}

/**
 * @brief Launch the per-point increment kernel over `(nx, ny, nz)` owned
 *        cells (typed overload).
 *
 * Convenience overload that extracts the POD from `FdGradientDevice<G>`
 * and delegates. Provides API parity with the CUDA backend.
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
 * @param eval        `FdGradientDevice<G>` instance.
 * @param du_padded   Output buffer in the **padded** layout (same
 *                    extents as the source field's padded box, same
 *                    strides). Owned cells are written; halo cells are
 *                    left untouched.
 * @param t           Current simulation time forwarded to `model.rhs`.
 * @param nx,ny,nz    Owned-region extents of the local subdomain.
 * @param stream      HIP stream to launch on (0 for the default stream).
 *
 * @throws std::runtime_error if the kernel launch fails.
 */
template <class Model, class G>
inline void for_each_interior_device(const Model &model,
                                     const ::pfc::hip::FdGradientDevice<G> &eval,
                                     double *du_padded, double t, int nx, int ny,
                                     int nz, hipStream_t stream = nullptr) {
  for_each_interior_device<Model, G>(model, eval.pod(), du_padded, t, nx, ny, nz,
                                     stream);
}

} // namespace pfc::sim::hip

#endif // OpenPFC_ENABLE_HIP
