// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file indexed_accumulate_gpu.inc
 * @brief Single-source scatter-add / gather-add kernels (CUDA and HIP).
 *
 * Stamped into `indexed_accumulate.cu` / `.hip`; do not include from a host
 * TU. Extents are validated by the public overloads before these launch.
 * Complex input is read as interleaved real/imag (`std::complex` layout).
 *
 * @see runtime/gpu/indexed_accumulate_gpu.hpp
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/indexed_accumulate_gpu.hpp>

namespace pfc::core::detail {

#if defined(OPENPFC_GPU_API_CUDA)
#define OPENPFC_INDEXED_BACKEND cuda
#elif defined(OPENPFC_GPU_API_HIP)
#define OPENPFC_INDEXED_BACKEND hip
#endif

namespace OPENPFC_INDEXED_BACKEND {

namespace {

constexpr int kThreads = 256;

unsigned blocks_for(std::size_t n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void check_launch(const char *op) {
  const ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(op) +
                             " kernel failed: " + ::pfc::gpuGetErrorString(err));
  }
}

template <typename T>
__global__ void scatter_add_kernel(T *out, const T *in, const std::size_t *source,
                                   const std::size_t *bin, const T *w,
                                   std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out + bin[i], w[i] * in[source[i]]);
  }
}

template <typename T>
__global__ void
scatter_add_abs2_kernel(T *out, const T *in_interleaved, const std::size_t *source,
                        const std::size_t *bin, const T *w, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const std::size_t k = source[i];
    const T re = in_interleaved[2 * k];
    const T im = in_interleaved[2 * k + 1];
    atomicAdd(out + bin[i], w[i] * (re * re + im * im));
  }
}

template <typename T>
__global__ void gather_add_kernel(T *out, const T *g, const std::size_t *source,
                                  const std::size_t *bin, const T *w,
                                  std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out + source[i], w[i] * g[bin[i]]);
  }
}

template <typename T>
void launch_scatter_add(T *out, const T *in, const std::size_t *source,
                        const std::size_t *bin, const T *w, std::size_t n) {
  if (n == 0) return;
  scatter_add_kernel<<<blocks_for(n), kThreads>>>(out, in, source, bin, w, n);
  check_launch("scatter_add");
}

template <typename T>
void launch_scatter_add_abs2(T *out, const std::complex<T> *in,
                             const std::size_t *source, const std::size_t *bin,
                             const T *w, std::size_t n) {
  if (n == 0) return;
  scatter_add_abs2_kernel<<<blocks_for(n), kThreads>>>(
      out, reinterpret_cast<const T *>(in), source, bin, w, n);
  check_launch("scatter_add_abs2");
}

template <typename T>
void launch_gather_add(T *out, const T *g, const std::size_t *source,
                       const std::size_t *bin, const T *w, std::size_t n) {
  if (n == 0) return;
  gather_add_kernel<<<blocks_for(n), kThreads>>>(out, g, source, bin, w, n);
  check_launch("gather_add");
}

} // namespace

#define OPENPFC_DEFINE_INDEXED_ACCUMULATE(T)                                        \
  void scatter_add(T *out, const T *in, const std::size_t *source,                  \
                   const std::size_t *bin, const T *w, std::size_t n) {             \
    launch_scatter_add(out, in, source, bin, w, n);                                 \
  }                                                                                 \
  void scatter_add_abs2(T *out, const std::complex<T> *in,                          \
                        const std::size_t *source, const std::size_t *bin,          \
                        const T *w, std::size_t n) {                                \
    launch_scatter_add_abs2(out, in, source, bin, w, n);                            \
  }                                                                                 \
  void gather_add(T *out, const T *g, const std::size_t *source,                    \
                  const std::size_t *bin, const T *w, std::size_t n) {              \
    launch_gather_add(out, g, source, bin, w, n);                                   \
  }

OPENPFC_DEFINE_INDEXED_ACCUMULATE(float)
OPENPFC_DEFINE_INDEXED_ACCUMULATE(double)

#undef OPENPFC_DEFINE_INDEXED_ACCUMULATE

} // namespace OPENPFC_INDEXED_BACKEND

#undef OPENPFC_INDEXED_BACKEND

} // namespace pfc::core::detail
