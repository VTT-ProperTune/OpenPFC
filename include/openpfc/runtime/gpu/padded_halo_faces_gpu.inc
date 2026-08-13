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
 * `pfc::hip::detail::launch_padded_pack_face` (and unpack twins).
 *
 * @see runtime/gpu/padded_device_halo_exchange_gpu.hpp
 */

#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>

#if defined(OPENPFC_GPU_API_CUDA)
namespace pfc::cuda::detail {
constexpr const char *kGpuKind = "cuda";
#elif defined(OPENPFC_GPU_API_HIP)
namespace pfc::hip::detail {
constexpr const char *kGpuKind = "hip";
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

} // namespace pfc::{cuda,hip}::detail
