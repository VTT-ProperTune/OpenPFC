// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file indexed_accumulate_gpu.hpp
 * @brief CUDA and HIP overloads of `scatter_add`, `scatter_add_abs2` and
 *        `gather_add` over a `BinMap`.
 *
 * @details
 * Same contract as the host path in `kernel/execution/indexed_accumulate.hpp`:
 * the call accumulates into `out`, checks extents in O(1) against the bounds
 * the map recorded on the host, and throws before launching anything. The
 * kernels use atomic adds, so duplicate bins (scatter) or duplicate sources
 * (gather) sum in an unspecified order.
 *
 * Kernels run on the default stream and do not synchronize. A following
 * `copy_to_host` of the small `out` array waits for them; for a distributed
 * total, reduce that host copy with the existing span `allreduce_sum`.
 *
 * Compiled from `src/openpfc/runtime/gpu/indexed_accumulate.cu` / `.hip`.
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <complex>
#include <cstddef>

#include <openpfc/kernel/execution/indexed_accumulate.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>

namespace pfc::core {

namespace detail {

#define OPENPFC_DECLARE_INDEXED_ACCUMULATE(T)                                       \
  void scatter_add(T *out, const T *in, const std::size_t *source,                  \
                   const std::size_t *bin, const T *w, std::size_t n);              \
  void scatter_add_abs2(T *out, const std::complex<T> *in,                          \
                        const std::size_t *source, const std::size_t *bin,          \
                        const T *w, std::size_t n);                                 \
  void gather_add(T *out, const T *g, const std::size_t *source,                    \
                  const std::size_t *bin, const T *w, std::size_t n);

#if defined(OpenPFC_ENABLE_CUDA)
namespace cuda {
OPENPFC_DECLARE_INDEXED_ACCUMULATE(float)
OPENPFC_DECLARE_INDEXED_ACCUMULATE(double)
} // namespace cuda
#endif

#if defined(OpenPFC_ENABLE_HIP)
namespace hip {
OPENPFC_DECLARE_INDEXED_ACCUMULATE(float)
OPENPFC_DECLARE_INDEXED_ACCUMULATE(double)
} // namespace hip
#endif

#undef OPENPFC_DECLARE_INDEXED_ACCUMULATE

/// Binds a device tag to its vendor kernels; unspecialized for other tags.
template <typename Tag> struct IndexedKernels;

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct IndexedKernels<backend::CUDATag> {
  template <typename... A> static void scatter_add(A... a) {
    cuda::scatter_add(a...);
  }
  template <typename... A> static void scatter_add_abs2(A... a) {
    cuda::scatter_add_abs2(a...);
  }
  template <typename... A> static void gather_add(A... a) { cuda::gather_add(a...); }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct IndexedKernels<backend::HIPTag> {
  template <typename... A> static void scatter_add(A... a) {
    hip::scatter_add(a...);
  }
  template <typename... A> static void scatter_add_abs2(A... a) {
    hip::scatter_add_abs2(a...);
  }
  template <typename... A> static void gather_add(A... a) { hip::gather_add(a...); }
};
#endif

template <typename Tag>
concept IndexedDeviceTag = requires { sizeof(IndexedKernels<Tag>); };

} // namespace detail

/// Device `out[bin[i]] += w[i] * in[source[i]]`.
template <typename Tag, AccumulateScalar T>
  requires detail::IndexedDeviceTag<Tag>
void scatter_add(DataBuffer<Tag, T> &out, const DataBuffer<Tag, T> &in,
                 const BinMap<Tag> &map, const DataBuffer<Tag, T> &w) {
  detail::check_indexed_extents("scatter_add", map.size(), w.size(),
                                map.source_end(), in.size(), map.bin_end(),
                                out.size());
  if (map.empty()) return;
  detail::IndexedKernels<Tag>::scatter_add(out.data(), in.data(),
                                           map.source().data(), map.bin().data(),
                                           w.data(), map.size());
}

/// Device `out[bin[i]] += w[i] * |in[source[i]]|^2` for complex @p in.
template <typename Tag, AccumulateScalar T>
  requires detail::IndexedDeviceTag<Tag>
void scatter_add_abs2(DataBuffer<Tag, T> &out,
                      const DataBuffer<Tag, std::complex<T>> &in,
                      const BinMap<Tag> &map, const DataBuffer<Tag, T> &w) {
  detail::check_indexed_extents("scatter_add_abs2", map.size(), w.size(),
                                map.source_end(), in.size(), map.bin_end(),
                                out.size());
  if (map.empty()) return;
  detail::IndexedKernels<Tag>::scatter_add_abs2(
      out.data(), in.data(), map.source().data(), map.bin().data(), w.data(),
      map.size());
}

/// Device `out[source[i]] += w[i] * g[bin[i]]` (adjoint of `scatter_add`).
template <typename Tag, AccumulateScalar T>
  requires detail::IndexedDeviceTag<Tag>
void gather_add(DataBuffer<Tag, T> &out, const DataBuffer<Tag, T> &g,
                const BinMap<Tag> &map, const DataBuffer<Tag, T> &w) {
  detail::check_indexed_extents("gather_add", map.size(), w.size(), map.source_end(),
                                out.size(), map.bin_end(), g.size());
  if (map.empty()) return;
  detail::IndexedKernels<Tag>::gather_add(out.data(), g.data(), map.source().data(),
                                          map.bin().data(), w.data(), map.size());
}

} // namespace pfc::core

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
