// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file padded_halo_faces_gpu.inc
 * @brief Single-source padded face pack/unpack kernels (included from vendor .cu/.hip).
 *
 * Do not include this from a host TU. Error strings keep the vendor kind
 * (`"cuda padded_pack_face kernel launch failed: …"` /
 * `"hip padded_pack_face kernel launch failed: …"`).
 *
 * Symbols stay `pfc::cuda::detail::launch_padded_pack_face` /
 * `pfc::hip::detail::launch_padded_pack_face` (and unpack twins), plus
 * `launch_padded_self_copy_faces` (fused self-neighbor copy, both faces of one axis).
 *
 * @see runtime/gpu/padded_device_halo_exchange_gpu.hpp
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/padded_self_copy.hpp>

#if defined(OPENPFC_GPU_API_CUDA)
namespace pfc::cuda::detail {
constexpr const char *kGpuKind = "cuda";
using ::pfc::gpu::detail::PaddedSelfCopy;
#elif defined(OPENPFC_GPU_API_HIP)
namespace pfc::hip::detail {
constexpr const char *kGpuKind = "hip";
using ::pfc::gpu::detail::PaddedSelfCopy;
#endif

__global__ void padded_pack_face_kernel(double *dst_contig, const double *pad,
                                        int ox, int oy, int oz, int sx, int sy,
                                        int sz, int nxp, int nyp, int nzp) {
  (void)nzp;
  const int n = sx * sy * sz;
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) {
    return;
  }
  const int ix = tid % sx;
  const int t2 = tid / sx;
  const int iy = t2 % sy;
  const int iz = t2 / sy;
  const int pi = ox + ix;
  const int pj = oy + iy;
  const int pk = oz + iz;
  const std::size_t lin =
      static_cast<std::size_t>(pi) +
      static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
      static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
          static_cast<std::size_t>(nyp);
  dst_contig[tid] = pad[lin];
}

__global__ void padded_unpack_face_kernel(double *pad, const double *src_contig,
                                          int ox, int oy, int oz, int sx, int sy,
                                          int sz, int nxp, int nyp, int nzp) {
  (void)nzp;
  const int n = sx * sy * sz;
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) {
    return;
  }
  const int ix = tid % sx;
  const int t2 = tid / sx;
  const int iy = t2 % sy;
  const int iz = t2 / sy;
  const int pi = ox + ix;
  const int pj = oy + iy;
  const int pk = oz + iz;
  const std::size_t lin =
      static_cast<std::size_t>(pi) +
      static_cast<std::size_t>(pj) * static_cast<std::size_t>(nxp) +
      static_cast<std::size_t>(pk) * static_cast<std::size_t>(nxp) *
          static_cast<std::size_t>(nyp);
  pad[lin] = src_contig[tid];
}

void launch_padded_pack_face(double *d_dst_contig, const double *d_pad, int ox,
                             int oy, int oz, int sx, int sy, int sz, int nxp,
                             int nyp, int nzp, ::pfc::gpuStream_t stream) {
  const int n = sx * sy * sz;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  padded_pack_face_kernel<<<blocks, threads, 0, stream>>>(
      d_dst_contig, d_pad, ox, oy, oz, sx, sy, sz, nxp, nyp, nzp);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) +
                             " padded_pack_face kernel launch failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

void launch_padded_unpack_face(double *d_pad, const double *d_src_contig, int ox,
                               int oy, int oz, int sx, int sy, int sz, int nxp,
                               int nyp, int nzp, ::pfc::gpuStream_t stream) {
  const int n = sx * sy * sz;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  padded_unpack_face_kernel<<<blocks, threads, 0, stream>>>(
      d_pad, d_src_contig, ox, oy, oz, sx, sy, sz, nxp, nyp, nzp);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) +
                             " padded_unpack_face kernel launch failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

/// Self-neighbor copy of up to `kMaxSelfCopyFields` padded buffers and two
/// slab pairs (one axis, both faces) in one launch: pad[dst] = pad[src].
/// Source (interior strip) and destination (opposite halo strip) never overlap.
__global__ void padded_self_copy_faces_kernel(PaddedSelfCopy c) {
  const long long v0 = static_cast<long long>(c.ext[0][0]) * c.ext[0][1] * c.ext[0][2];
  const long long v1 =
      c.nslab > 1 ? static_cast<long long>(c.ext[1][0]) * c.ext[1][1] * c.ext[1][2] : 0;
  const long long per_field = v0 + v1;
  const long long n = per_field * c.nfields;
  for (long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < n; tid += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int f = static_cast<int>(tid / per_field);
    long long r = tid % per_field;
    const int s = r < v0 ? 0 : 1;
    if (s == 1) r -= v0;
    const int sx = c.ext[s][0], sy = c.ext[s][1];
    const int ix = static_cast<int>(r % sx);
    const int iy = static_cast<int>((r / sx) % sy);
    const int iz = static_cast<int>(r / (static_cast<long long>(sx) * sy));
    const std::size_t nxp = static_cast<std::size_t>(c.nxp);
    const std::size_t nxy = nxp * static_cast<std::size_t>(c.nyp);
    const std::size_t src = static_cast<std::size_t>(c.src[s][0] + ix) +
                            static_cast<std::size_t>(c.src[s][1] + iy) * nxp +
                            static_cast<std::size_t>(c.src[s][2] + iz) * nxy;
    const std::size_t dst = static_cast<std::size_t>(c.dst[s][0] + ix) +
                            static_cast<std::size_t>(c.dst[s][1] + iy) * nxp +
                            static_cast<std::size_t>(c.dst[s][2] + iz) * nxy;
    double *pad = c.pads[f];
    pad[dst] = pad[src];
  }
}

void launch_padded_self_copy_faces(const PaddedSelfCopy &c, ::pfc::gpuStream_t stream) {
  long long n = 0;
  for (int s = 0; s < c.nslab; ++s) {
    n += static_cast<long long>(c.ext[s][0]) * c.ext[s][1] * c.ext[s][2];
  }
  n *= c.nfields;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const long long want = (n + threads - 1) / threads;
  const int blocks = static_cast<int>(want < 65535LL * 32 ? want : 65535LL * 32);
  padded_self_copy_faces_kernel<<<blocks, threads, 0, stream>>>(c);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kGpuKind) +
                             " padded_self_copy_faces kernel launch failed: " +
                             ::pfc::gpuGetErrorString(err));
  }
}

} // namespace pfc::{cuda,hip}::detail
