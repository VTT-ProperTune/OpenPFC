// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>

#include <cstddef>

namespace pfc::cuda::detail {

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
                             int nyp, int nzp, cudaStream_t stream) {
  const int n = sx * sy * sz;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  padded_pack_face_kernel<<<blocks, threads, 0, stream>>>(
      d_dst_contig, d_pad, ox, oy, oz, sx, sy, sz, nxp, nyp, nzp);
}

void launch_padded_unpack_face(double *d_pad, const double *d_src_contig, int ox,
                               int oy, int oz, int sx, int sy, int sz, int nxp,
                               int nyp, int nzp, cudaStream_t stream) {
  const int n = sx * sy * sz;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  padded_unpack_face_kernel<<<blocks, threads, 0, stream>>>(
      d_pad, d_src_contig, ox, oy, oz, sx, sy, sz, nxp, nyp, nzp);
}

} // namespace pfc::cuda::detail

#endif // OpenPFC_ENABLE_CUDA
