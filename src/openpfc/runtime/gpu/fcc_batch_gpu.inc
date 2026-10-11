// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/runtime/gpu/fcc_batch.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <stdexcept>
#include <string>

namespace pfc::finite_strain {
__global__ void fcc_batch_kernel(FccSlipParameters parameters, FccBatchView view) {
  const std::size_t p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t n = view.count;
  if (p >= n) return;
  Tensor2 orientation, deformation;
  FccSlipState state;
  for (int c = 0; c < 9; ++c) {
    orientation(c / 3, c % 3) = view.orientation[c * n + p];
    deformation(c / 3, c % 3) = view.deformation[c * n + p];
    state.plastic(c / 3, c % 3) = view.committed[c * n + p];
    state.velocity(c / 3, c % 3) = view.committed[(9 + c) * n + p];
  }
  for (int c = 0; c < 12; ++c) {
    state.resistance[c] = view.committed[(18 + c) * n + p];
    state.shear[c] = view.committed[(30 + c) * n + p];
  }
  const auto result = try_evaluate_fcc(parameters, orientation, state, deformation);
  view.status[p] = result.status;
  // Failed points never publish partially evaluated output.
  if (result.status != FccStatus::Success) return;
  const auto &out = result.value;
  for (int c = 0; c < 9; ++c) {
    view.piola[c * n + p] = out.update.piola(c / 3, c % 3);
    view.trial[c * n + p] = out.update.state.plastic(c / 3, c % 3);
    view.trial[(9 + c) * n + p] = out.update.state.velocity(c / 3, c % 3);
  }
  for (int c = 0; c < 81; ++c)
    view.tangent[c * n + p] = out.tangent(c / 27, (c / 9) % 3, (c / 3) % 3, c % 3);
  for (int c = 0; c < 12; ++c) {
    view.trial[(18 + c) * n + p] = out.update.state.resistance[c];
    view.trial[(30 + c) * n + p] = out.update.state.shear[c];
    if (view.resolved_shear) view.resolved_shear[c * n + p] = out.update.tau[c];
    if (view.shear_rate) view.shear_rate[c * n + p] = out.update.shear_rate[c];
  }
}
#if defined(OPENPFC_GPU_API_CUDA)
void evaluate_fcc_batch_cuda(FccSlipParameters parameters, FccBatchView view) {
#else
void evaluate_fcc_batch_hip(FccSlipParameters parameters, FccBatchView view) {
#endif
  if (!view.count) return;
  if (!view.orientation || !view.deformation || !view.committed || !view.trial ||
      !view.piola || !view.tangent || !view.status || view.committed == view.trial)
    throw std::invalid_argument(
        "FCC batch requires distinct committed/trial storage");
  constexpr int threads = 64;
  fcc_batch_kernel<<<(view.count + threads - 1) / threads, threads>>>(parameters,
                                                                      view);
  const auto error = gpuGetLastError();
  if (error != gpuSuccess)
    throw std::runtime_error(std::string("FCC batch launch: ") +
                             gpuGetErrorString(error));
}
__global__ void fcc_status_kernel(const FccStatus *status, std::size_t n,
                                  unsigned long long *out) {
  __shared__ unsigned long long values[256];
  unsigned long long value = ~0ULL;
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    if (status[i] != FccStatus::Success) {
      const auto encoded = (static_cast<unsigned long long>(i) << 8) |
                           static_cast<unsigned long long>(status[i]);
      value = value < encoded ? value : encoded;
    }
  values[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride && values[threadIdx.x + stride] < values[threadIdx.x])
      values[threadIdx.x] = values[threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) *out = values[0];
}
#if defined(OPENPFC_GPU_API_CUDA)
FccBatchFailure check_fcc_batch_cuda(const FccStatus *status, std::size_t n,
                                     unsigned long long *workspace) {
#else
FccBatchFailure check_fcc_batch_hip(const FccStatus *status, std::size_t n,
                                    unsigned long long *workspace) {
#endif
  if (!n) return {};
  if (!status || !workspace || n > (~0ULL >> 8))
    throw std::invalid_argument("invalid FCC status workspace");
  fcc_status_kernel<<<1, 256>>>(status, n, workspace);
  unsigned long long result;
  auto error = gpuMemcpy(&result, workspace, sizeof(result), gpuMemcpyDeviceToHost);
  if (error != gpuSuccess)
    throw std::runtime_error(std::string("FCC status reduction: ") +
                             gpuGetErrorString(error));
  if (result == ~0ULL) return {};
  return {static_cast<FccStatus>(result & 255),
          static_cast<std::size_t>(result >> 8)};
}
} // namespace pfc::finite_strain
