// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/runtime/gpu/constitutive_point.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
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

namespace pfc::finite_strain {
__global__ void j2_point_kernel(J2Moduli moduli, J2State state, Tensor2 deformation,
                                J2EvaluationResult *out) {
  *out = try_evaluate_j2(moduli, state, deformation);
}
__global__ void fcc_point_kernel(FccSlipParameters parameters, Tensor2 orientation,
                                 FccSlipState state, Tensor2 deformation,
                                 FccEvaluationResult *out) {
  *out = try_evaluate_fcc(parameters, orientation, state, deformation);
}

#if defined(OPENPFC_GPU_API_CUDA)
void evaluate_j2_point_cuda(J2Moduli moduli, const J2State &state,
                            const Tensor2 &deformation, J2EvaluationResult *out) {
#else
void evaluate_j2_point_hip(J2Moduli moduli, const J2State &state,
                           const Tensor2 &deformation, J2EvaluationResult *out) {
#endif
  J2EvaluationResult *device = nullptr;
  solvers::gpu::detail::check(
      pfc::gpuMalloc(reinterpret_cast<void **>(&device), sizeof(*device)));
  j2_point_kernel<<<1, 1>>>(moduli, state, deformation, device);
  solvers::gpu::detail::check(pfc::gpuGetLastError());
  solvers::gpu::detail::check(
      pfc::gpuMemcpy(out, device, sizeof(*device), pfc::gpuMemcpyDeviceToHost));
  solvers::gpu::detail::check(pfc::gpuFree(device));
}

#if defined(OPENPFC_GPU_API_CUDA)
void evaluate_fcc_point_cuda(const FccSlipParameters &parameters,
                             const Tensor2 &orientation, const FccSlipState &state,
                             const Tensor2 &deformation, FccEvaluationResult *out) {
#else
void evaluate_fcc_point_hip(const FccSlipParameters &parameters,
                            const Tensor2 &orientation, const FccSlipState &state,
                            const Tensor2 &deformation, FccEvaluationResult *out) {
#endif
  FccEvaluationResult *device = nullptr;
  solvers::gpu::detail::check(
      pfc::gpuMalloc(reinterpret_cast<void **>(&device), sizeof(*device)));
  fcc_point_kernel<<<1, 1>>>(parameters, orientation, state, deformation, device);
  solvers::gpu::detail::check(pfc::gpuGetLastError());
  solvers::gpu::detail::check(
      pfc::gpuMemcpy(out, device, sizeof(*device), pfc::gpuMemcpyDeviceToHost));
  solvers::gpu::detail::check(pfc::gpuFree(device));
}
} // namespace pfc::finite_strain
