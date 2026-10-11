// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/solvers/finite_strain_fft/gpu/projector.hpp>
namespace pfc::finite_strain::gpu {
__global__ void project_tensor_kernel(ComplexComponents fields, ProjectionGrid grid,
                                      std::size_t n) {
  auto p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= n) return;
  int position[3] = {int(p % grid.extent[0]),
                     int((p / grid.extent[0]) % grid.extent[1]),
                     int(p / (grid.extent[0] * grid.extent[1]))};
  double q[3], q2 = 0;
  for (int d = 0; d < 3; ++d) {
    int k = position[d] + grid.low[d];
    if (k > grid.shape[d] / 2) k -= grid.shape[d];
    q[d] = double(k) / (grid.shape[d] * grid.spacing[d]);
    q2 += q[d] * q[d];
  }
  for (int row = 0; row < 3; ++row) {
    double re = 0, im = 0;
    for (int c = 0; c < 3; ++c) {
      re += fields.values[3 * row + c][2 * p] * q[c];
      im += fields.values[3 * row + c][2 * p + 1] * q[c];
    }
    for (int c = 0; c < 3; ++c) {
      fields.values[3 * row + c][2 * p] = q2 ? re * q[c] / q2 : 0;
      fields.values[3 * row + c][2 * p + 1] = q2 ? im * q[c] / q2 : 0;
    }
  }
}
__global__ void tangent_kernel(const double *K, const double *v, double *w,
                               std::size_t n) {
  auto p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= n) return;
  double out[9] = {};
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j) out[i] += K[(i * 9 + j) * n + p] * v[j * n + p];
  for (int i = 0; i < 9; ++i) w[i * n + p] = out[i];
}
#if defined(OPENPFC_GPU_API_CUDA)
void project_tensor_cuda(ComplexComponents fields, ProjectionGrid grid,
                         std::size_t n) {
#else
void project_tensor_hip(ComplexComponents fields, ProjectionGrid grid,
                        std::size_t n) {
#endif
  if (n) project_tensor_kernel<<<(n + 255) / 256, 256>>>(fields, grid, n);
  solvers::gpu::detail::check(pfc::gpuGetLastError());
}
#if defined(OPENPFC_GPU_API_CUDA)
void apply_tangents_cuda(const double *K, const double *v, double *w,
                         std::size_t n) {
#else
void apply_tangents_hip(const double *K, const double *v, double *w, std::size_t n) {
#endif
  if (n) tangent_kernel<<<(n + 255) / 256, 256>>>(K, v, w, n);
  solvers::gpu::detail::check(pfc::gpuGetLastError());
}
} // namespace pfc::finite_strain::gpu
