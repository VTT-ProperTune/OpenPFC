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
 * **Single-field and multi-field (N=2..4).** The single-field path takes a
 * `double *du_padded`. The multi-field path takes a `DevicePtrPackN` of
 * padded device pointers and a `CompositeGradientDevicePOD`, launching
 * `for_each_interior_device_kernel_multi` with explicit `PerFieldGrads...`.
 * Device scatter uses named members on `DevicePtrPackN` / `DeviceIncN`
 * (`scatter_device`) — it does **not** call `std::get`, `std::apply`,
 * `std::forward_as_tuple`, or `to_tuple` under `__device__`.
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
 *                                          nx, ny, nz, stream);
 * @endcode
 *
 * Multi-field (catalog per-field grads `UGrads` / `VGrads`):
 *
 * @code
 * auto composite = pfc::cuda::create_composite_device<WaveLocal>(eval_u, eval_v);
 * auto du = pfc::sim::cuda::make_device_ptr_pack(d_du, d_dv);
 * pfc::sim::cuda::for_each_interior_device<WaveModel, WaveLocal, UGrads, VGrads>(
 *     model, composite.pod(), du, t, nx, ny, nz);
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
#include <tuple>
#include <type_traits>

#include <cuda_runtime.h>

#include <openpfc/runtime/cuda/fd_gradient_device.hpp>

#if defined(__CUDACC__) && defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
#error "OpenPFC CUDA device drivers require CUDA toolkit 11.0 or higher"
#endif

#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
#include "openpfc/runtime/common/gpu_autotune.hpp"
#endif

namespace pfc::sim::cuda {

/** Fixed-arity device pointer packs (no `std::tuple` / `std::get` on device). */
struct DevicePtrPack2 {
  double *p0{nullptr};
  double *p1{nullptr};
};
struct DevicePtrPack3 {
  double *p0{nullptr};
  double *p1{nullptr};
  double *p2{nullptr};
};
struct DevicePtrPack4 {
  double *p0{nullptr};
  double *p1{nullptr};
  double *p2{nullptr};
  double *p3{nullptr};
};

/**
 * Fixed-arity increment packs returned by multi-field `Model::rhs`.
 *
 * Named members are written by device `scatter_device`. Host-only
 * `as_tuple()` (no `OPENPFC_HD`) lets CPU `pfc::sim::detail::scatter` /
 * `for_each_interior` normalize via `tuple_protocol` — never call
 * `as_tuple` from `__device__` code.
 */
struct DeviceInc2 {
  double v0{};
  double v1{};
  auto as_tuple() { return std::tie(v0, v1); }
  auto as_tuple() const { return std::tie(v0, v1); }
};
struct DeviceInc3 {
  double v0{};
  double v1{};
  double v2{};
  auto as_tuple() { return std::tie(v0, v1, v2); }
  auto as_tuple() const { return std::tie(v0, v1, v2); }
};
struct DeviceInc4 {
  double v0{};
  double v1{};
  double v2{};
  double v3{};
  auto as_tuple() { return std::tie(v0, v1, v2, v3); }
  auto as_tuple() const { return std::tie(v0, v1, v2, v3); }
};

[[nodiscard]] inline DevicePtrPack2 make_device_ptr_pack(double *a, double *b) {
  return DevicePtrPack2{a, b};
}
[[nodiscard]] inline DevicePtrPack3 make_device_ptr_pack(double *a, double *b,
                                                        double *c) {
  return DevicePtrPack3{a, b, c};
}
[[nodiscard]] inline DevicePtrPack4 make_device_ptr_pack(double *a, double *b,
                                                        double *c, double *d) {
  return DevicePtrPack4{a, b, c, d};
}

template <class T> struct is_device_ptr_pack : std::false_type {};
template <> struct is_device_ptr_pack<DevicePtrPack2> : std::true_type {};
template <> struct is_device_ptr_pack<DevicePtrPack3> : std::true_type {};
template <> struct is_device_ptr_pack<DevicePtrPack4> : std::true_type {};

namespace detail {

/**
 * @brief Device scatter via named members (no `std::get` / `std::apply`).
 */
__device__ inline void scatter_device(DevicePtrPack2 du, DeviceInc2 inc,
                                     std::ptrdiff_t c) {
  du.p0[c] = inc.v0;
  du.p1[c] = inc.v1;
}
__device__ inline void scatter_device(DevicePtrPack3 du, DeviceInc3 inc,
                                     std::ptrdiff_t c) {
  du.p0[c] = inc.v0;
  du.p1[c] = inc.v1;
  du.p2[c] = inc.v2;
}
__device__ inline void scatter_device(DevicePtrPack4 du, DeviceInc4 inc,
                                     std::ptrdiff_t c) {
  du.p0[c] = inc.v0;
  du.p1[c] = inc.v1;
  du.p2[c] = inc.v2;
  du.p3[c] = inc.v3;
}

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

/**
 * @brief Multi-field padded-output kernel (N=2..4 via `DuPack` / `DeviceIncN`).
 *
 * Carries the same `PerFieldGrads...` pack as
 * `evaluate_fd_grad_composite<Composite, PerFieldGrads...>`.
 */
template <class Model, class Composite, class DuPack, class... PerFieldGrads>
__global__ void for_each_interior_device_kernel_multi(
    Model model, ::pfc::cuda::CompositeGradientDevicePOD eval, DuPack du,
    double t, int nx, int ny, int nz) {
  const int ix = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int iy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int iz = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    return;
  }
  const Composite grad =
      ::pfc::cuda::evaluate_fd_grad_composite<Composite, PerFieldGrads...>(
          eval, ix, iy, iz);
  const auto inc = model.rhs(t, grad);
  const auto &f0 = eval.fields[0];
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(ix + f0.hw) +
                           static_cast<std::ptrdiff_t>(iy + f0.hw) * f0.sy +
                           static_cast<std::ptrdiff_t>(iz + f0.hw) * f0.sz;
  scatter_device(du, inc, c);
}

inline dim3 for_each_interior_grid(int nx, int ny, int nz, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y,
              (static_cast<unsigned>(nz) + block.z - 1) / block.z);
}

inline dim3 for_each_interior_block(int nx, int ny, int nz) {
  (void)nx;
  (void)ny;
  (void)nz;
#ifdef OpenPFC_ENABLE_GPU_AUTOTUNING
  size_t total_elements = static_cast<size_t>(nx) * static_cast<size_t>(ny) *
                          static_cast<size_t>(nz);
  auto config = pfc::gpu::AutoTuner::instance().get_config("for_each_interior_3d",
                                                           total_elements);
  return dim3(config.block_size_x, config.block_size_y, config.block_size_z);
#else
  constexpr int Tx = 32;
  constexpr int Ty = 4;
  constexpr int Tz = 1;
  return dim3(Tx, Ty, Tz);
#endif
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
 * @throws std::runtime_error if the kernel launch or synchronization fails.
 *         Both kernel launch and stream synchronization are checked.
 */
template <class Model, class G>
inline void for_each_interior_device(const Model &model,
                                     const ::pfc::cuda::FdGradientDevicePOD &eval,
                                     double *du_padded, double t, int nx, int ny,
                                     int nz, cudaStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const dim3 block = detail::for_each_interior_block(nx, ny, nz);
  const dim3 grid = detail::for_each_interior_grid(nx, ny, nz, block);
  detail::for_each_interior_device_kernel<Model, G>
      <<<grid, block, 0, stream>>>(model, eval, du_padded, t, nx, ny, nz);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("for_each_interior_device: kernel "
                                         "launch failed: ") +
                             cudaGetErrorString(e));
  }
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("for_each_interior_device: synchronize "
                                         "failed: ") +
                             cudaGetErrorString(e));
  }
}

/**
 * @brief Multi-field launch for `DevicePtrPack2` (SFINAE-free fixed arity).
 *
 * Call with explicit `PerFieldGrads...`, e.g.
 * `for_each_interior_device<Model, Composite, UGrads, VGrads>(...)`.
 */
template <class Model, class Composite, class... PerFieldGrads>
inline void
for_each_interior_device(const Model &model,
                         const ::pfc::cuda::CompositeGradientDevicePOD &eval,
                         DevicePtrPack2 du, double t, int nx, int ny, int nz,
                         cudaStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const dim3 block = detail::for_each_interior_block(nx, ny, nz);
  const dim3 grid = detail::for_each_interior_grid(nx, ny, nz, block);
  detail::for_each_interior_device_kernel_multi<Model, Composite, DevicePtrPack2,
                                               PerFieldGrads...>
      <<<grid, block, 0, stream>>>(model, eval, du, t, nx, ny, nz);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): kernel launch "
                    "failed: ") +
        cudaGetErrorString(e));
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): synchronize "
                    "failed: ") +
        cudaGetErrorString(e));
  }
}

template <class Model, class Composite, class... PerFieldGrads>
inline void
for_each_interior_device(const Model &model,
                         const ::pfc::cuda::CompositeGradientDevicePOD &eval,
                         DevicePtrPack3 du, double t, int nx, int ny, int nz,
                         cudaStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const dim3 block = detail::for_each_interior_block(nx, ny, nz);
  const dim3 grid = detail::for_each_interior_grid(nx, ny, nz, block);
  detail::for_each_interior_device_kernel_multi<Model, Composite, DevicePtrPack3,
                                               PerFieldGrads...>
      <<<grid, block, 0, stream>>>(model, eval, du, t, nx, ny, nz);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): kernel launch "
                    "failed: ") +
        cudaGetErrorString(e));
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): synchronize "
                    "failed: ") +
        cudaGetErrorString(e));
  }
}

template <class Model, class Composite, class... PerFieldGrads>
inline void
for_each_interior_device(const Model &model,
                         const ::pfc::cuda::CompositeGradientDevicePOD &eval,
                         DevicePtrPack4 du, double t, int nx, int ny, int nz,
                         cudaStream_t stream = nullptr) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const dim3 block = detail::for_each_interior_block(nx, ny, nz);
  const dim3 grid = detail::for_each_interior_grid(nx, ny, nz, block);
  detail::for_each_interior_device_kernel_multi<Model, Composite, DevicePtrPack4,
                                               PerFieldGrads...>
      <<<grid, block, 0, stream>>>(model, eval, du, t, nx, ny, nz);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): kernel launch "
                    "failed: ") +
        cudaGetErrorString(e));
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("for_each_interior_device(multi-field): synchronize "
                    "failed: ") +
        cudaGetErrorString(e));
  }
}

/**
 * @brief Convenience overload: deduce `PerFieldGrads...` from a
 *        `CompositeGradientDevice` host wrapper.
 */
template <class Model, class Composite, class... PerFieldGrads>
inline void for_each_interior_device(
    const Model &model,
    const ::pfc::cuda::CompositeGradientDevice<Composite, PerFieldGrads...> &eval,
    DevicePtrPack2 du, double t, int nx, int ny, int nz,
    cudaStream_t stream = nullptr) {
  for_each_interior_device<Model, Composite, PerFieldGrads...>(
      model, eval.pod(), du, t, nx, ny, nz, stream);
}

template <class Model, class Composite, class... PerFieldGrads>
inline void for_each_interior_device(
    const Model &model,
    const ::pfc::cuda::CompositeGradientDevice<Composite, PerFieldGrads...> &eval,
    DevicePtrPack3 du, double t, int nx, int ny, int nz,
    cudaStream_t stream = nullptr) {
  for_each_interior_device<Model, Composite, PerFieldGrads...>(
      model, eval.pod(), du, t, nx, ny, nz, stream);
}

template <class Model, class Composite, class... PerFieldGrads>
inline void for_each_interior_device(
    const Model &model,
    const ::pfc::cuda::CompositeGradientDevice<Composite, PerFieldGrads...> &eval,
    DevicePtrPack4 du, double t, int nx, int ny, int nz,
    cudaStream_t stream = nullptr) {
  for_each_interior_device<Model, Composite, PerFieldGrads...>(
      model, eval.pod(), du, t, nx, ny, nz, stream);
}

} // namespace pfc::sim::cuda

#endif // OpenPFC_ENABLE_CUDA
