// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file transform_gpu.hpp
 * @brief Device `out[i] = f(a[i], b[i], ...)` with a user `OPENPFC_HD`
 *        functor over one to four buffers (CUDA + HIP).
 *
 * @details
 * The `DataBuffer<CUDATag | HIPTag>` overload of `pfc::core::transform` has
 * the semantics of the host path in `kernel/execution/transform.hpp`. It calls
 * the launcher `pfc::gpu::transform<F, Out, In...>`, which is *declared* for
 * every translation unit and *defined* only where a device compiler is active
 * (`__CUDACC__` / `__HIPCC__`). A host TU that calls `transform` on device
 * buffers therefore links against an explicit instantiation that must live in
 * exactly one `.cu` / `.hip` TU of the consumer:
 *
 * @code
 * // src/gpu/reaction_transform.inc (stamped into reaction_transform.cu/.hip)
 * #include <app/reaction.hpp>                // the functor only
 * #include <openpfc/runtime/gpu/transform_gpu.hpp>
 * // functor, output element type, input element types (one to four)
 * OPENPFC_INSTANTIATE_TRANSFORM(app::Reaction, double, double, double, double)
 * @endcode
 *
 * Forgetting the instantiation fails closed at link time with the functor's
 * name in the missing symbol. Only the functor header is device-compiled, so
 * the consumer's session and I/O headers stay out of the device compiler. A
 * functor type whose name contains a comma needs an alias first
 * (`using Fn = Map<int, double>;`).
 *
 * Complex buffers hold `std::complex<T>`, and the functor receives that type.
 * Its members are host-only constexpr functions, so under nvcc the
 * instantiating TU must be compiled with `--expt-relaxed-constexpr` (as
 * `openpfc_gpu_kernels` is); the launcher refuses to compile a complex
 * instantiation without it. HIP needs no flag.
 *
 * Kernels run on the default stream (or the stream passed to the launcher)
 * and do not synchronize.
 *
 * @see kernel/execution/transform.hpp
 * @see runtime/gpu/spectral_pointwise_gpu.hpp (the same instantiation pattern)
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <concepts>
#include <cstddef>

#include <openpfc/kernel/execution/transform.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::gpu {

/// Fixed-arity pack of const device pointers, one per `transform` input.
template <typename... In> struct DeviceInputs;
template <typename A> struct DeviceInputs<A> {
  const A *p0;
};
template <typename A, typename B> struct DeviceInputs<A, B> {
  const A *p0;
  const B *p1;
};
template <typename A, typename B, typename C> struct DeviceInputs<A, B, C> {
  const A *p0;
  const B *p1;
  const C *p2;
};
template <typename A, typename B, typename C, typename D>
struct DeviceInputs<A, B, C, D> {
  const A *p0;
  const B *p1;
  const C *p2;
  const D *p3;
};

/**
 * @brief Launch `out[i] = f(in[i]...)` for `i` in `[0, n)` on device memory.
 *
 * Does not synchronize @p stream. Defined only in a device TU; instantiate it
 * with `OPENPFC_INSTANTIATE_TRANSFORM`.
 */
template <typename F, typename Out, typename... In>
void transform(Out *out, std::size_t n, F f, DeviceInputs<In...> in,
               gpuStream_t stream);

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

namespace detail {
template <typename T> inline constexpr bool is_complex_v = false;
template <typename T> inline constexpr bool is_complex_v<std::complex<T>> = true;
} // namespace detail

template <typename F, typename Out, typename... In>
__global__ void transform_kernel(Out *out, std::size_t n, F f,
                                 DeviceInputs<In...> in) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  constexpr std::size_t arity = sizeof...(In);
  if constexpr (arity == 1) {
    out[i] = static_cast<Out>(f(in.p0[i]));
  } else if constexpr (arity == 2) {
    out[i] = static_cast<Out>(f(in.p0[i], in.p1[i]));
  } else if constexpr (arity == 3) {
    out[i] = static_cast<Out>(f(in.p0[i], in.p1[i], in.p2[i]));
  } else {
    out[i] = static_cast<Out>(f(in.p0[i], in.p1[i], in.p2[i], in.p3[i]));
  }
}

template <typename F, typename Out, typename... In>
void transform(Out *out, std::size_t n, F f, DeviceInputs<In...> in,
               gpuStream_t stream) {
  static_assert(::pfc::core::ElementwiseFunctor<F, Out, In...>,
                "pfc::gpu::transform: F must be trivially copyable and callable "
                "as OPENPFC_HD Out operator()(const In&...) with 1..4 inputs");
#if defined(__CUDACC__) && !defined(__CUDACC_RELAXED_CONSTEXPR__)
  static_assert(!(detail::is_complex_v<Out> || (detail::is_complex_v<In> || ...)),
                "pfc::gpu::transform: std::complex elements need nvcc "
                "--expt-relaxed-constexpr on the instantiating TU; without it "
                "std::complex members are host-only and the results are undefined");
#endif
  if (n == 0) {
    return;
  }
  constexpr unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((n + threads - 1) / static_cast<std::size_t>(threads));
  GPU_LAUNCH_KERNEL((transform_kernel<F, Out, In...>), blocks, threads,
                    (out, n, f, in), stream);
}

/// Explicitly instantiate the device launcher: functor, output element type,
/// then the one to four input element types.
#define OPENPFC_INSTANTIATE_TRANSFORM(F, Out, ...)                                  \
  template void ::pfc::gpu::transform<F, Out, __VA_ARGS__>(                         \
      Out *, std::size_t, F, ::pfc::gpu::DeviceInputs<__VA_ARGS__>,                 \
      ::pfc::gpuStream_t);

#endif // device compiler

} // namespace pfc::gpu

namespace pfc::core {

namespace detail {

template <typename Tag>
concept TransformDeviceTag =
#if defined(OpenPFC_ENABLE_CUDA)
    std::same_as<Tag, backend::CUDATag> ||
#endif
#if defined(OpenPFC_ENABLE_HIP)
    std::same_as<Tag, backend::HIPTag> ||
#endif
    false;

} // namespace detail

/// Device `out[i] = f(in[i]...)`; links against the consumer's
/// `OPENPFC_INSTANTIATE_TRANSFORM(F, Out, In...)`.
template <typename Tag, typename Out, typename F, typename... In>
  requires detail::TransformDeviceTag<Tag> && ElementwiseFunctor<F, Out, In...>
void transform(DataBuffer<Tag, Out> &out, F f, const DataBuffer<Tag, In> &...in) {
  detail::check_transform_sizes(out.size(), {in.size()...});
  if (out.empty()) {
    return;
  }
  ::pfc::gpu::transform<F, Out, In...>(out.data(), out.size(), f,
                                       ::pfc::gpu::DeviceInputs<In...>{in.data()...},
                                       ::pfc::gpuStream_t{});
}

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
