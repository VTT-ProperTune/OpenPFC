// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file implicit_geometry_gpu.inc
 * @brief Single-source CUDA/HIP launch for Cartesian implicit geometry.
 */

#include <openpfc/kernel/geometry/cut_cell.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/implicit_geometry.hpp>

#include <chrono>
#include <vector>

namespace pfc::geometry {
namespace {

__global__ void classify_cells(PackedImplicit packed, CellLayout layout,
                               Policy policy, unsigned char *mask) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= layout.size()) return;
  const int i = id % layout.count[0];
  const int jk = id / layout.count[0];
  const int j = jk % layout.count[1];
  const int k = jk / layout.count[1];
  const Verdict verdict = classify_box(unpack(packed), layout.box(i, j, k), policy);
  unsigned char code = 2;
  if (verdict.kind == CellKind::Interior)
    code = 0;
  else if (verdict.kind == CellKind::Exterior)
    code = 1;
  mask[id] = code;
}

__global__ void measure_cuts(PackedImplicit packed, CellLayout layout, Policy policy,
                             const int *index, int ncut, double *inside,
                             double *outside, double *area) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ncut) return;
  const int id = index[t];
  const int i = id % layout.count[0];
  const int jk = id / layout.count[0];
  const int j = jk % layout.count[1];
  const int k = jk / layout.count[1];
  const CutMeasure measured =
      measure_box(unpack(packed), layout.box(i, j, k), layout.dimension(), policy);
  inside[t] = measured.inside_volume;
  outside[t] = measured.outside_volume;
  area[t] = measured.interface_area;
}

double elapsed_ms(gpuEvent_t start, gpuEvent_t stop) {
  float ms = 0;
  GPU_CHECK(gpuEventSynchronize(stop));
  GPU_CHECK(gpuEventElapsedTime(&ms, start, stop));
  return static_cast<double>(ms) * 1e-3;
}

void launch_classify(const CellLayout &layout, const Implicit &shape,
                     const Policy &policy, std::vector<std::uint8_t> &mask,
                     DeviceProfile &profile, unsigned char *&device_mask) {
  const int cells = layout.size();
  profile.cells = cells;
  mask.assign(static_cast<std::size_t>(cells), 0);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&device_mask),
                      static_cast<std::size_t>(cells)));
#if defined(OPENPFC_GPU_API_CUDA)
  GPU_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024));
#elif defined(OPENPFC_GPU_API_HIP)
  GPU_CHECK(hipDeviceSetLimit(hipLimitStackSize, 128 * 1024));
#endif
  gpuEvent_t start, stop;
  GPU_CHECK(gpuEventCreate(&start));
  GPU_CHECK(gpuEventCreate(&stop));
  const int threads = 128;
  const int blocks = (cells + threads - 1) / threads;
  GPU_CHECK(gpuEventRecord(start, nullptr));
  GPU_LAUNCH_KERNEL(classify_cells, blocks, threads,
                    (pack(shape), layout, policy, device_mask), nullptr);
  GPU_CHECK(gpuEventRecord(stop, nullptr));
  profile.classify_seconds = elapsed_ms(start, stop);
  const auto copy_begin = std::chrono::steady_clock::now();
  GPU_CHECK(gpuMemcpy(mask.data(), device_mask, static_cast<std::size_t>(cells),
                      gpuMemcpyDeviceToHost));
  profile.copy_to_host_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_begin)
          .count();
  profile.bytes_to_host += static_cast<std::size_t>(cells);
  profile.device_bytes += static_cast<std::size_t>(cells);
  for (unsigned char code : mask) {
    if (code == 0)
      ++profile.interior;
    else if (code == 1)
      ++profile.exterior;
    else
      ++profile.cut_cells;
  }
  GPU_CHECK(gpuEventDestroy(start));
  GPU_CHECK(gpuEventDestroy(stop));
}

} // namespace

void classify_on_device(const CellLayout &layout, const Implicit &shape,
                        const Policy &policy, std::vector<std::uint8_t> &mask,
                        DeviceProfile &profile) {
  profile = {};
  unsigned char *device_mask = nullptr;
  launch_classify(layout, shape, policy, mask, profile, device_mask);
  GPU_CHECK(gpuFree(device_mask));
}

void evaluate_on_device(const CellLayout &layout, const Implicit &shape,
                        const Policy &policy, std::vector<std::uint8_t> &mask,
                        std::vector<double> &inside_volume,
                        std::vector<double> &outside_volume,
                        std::vector<double> &interface_area,
                        DeviceProfile &profile) {
  profile = {};
  unsigned char *device_mask = nullptr;
  launch_classify(layout, shape, policy, mask, profile, device_mask);
  std::vector<int> cut;
  cut.reserve(static_cast<std::size_t>(profile.cut_cells));
  for (int id = 0; id < layout.size(); ++id)
    if (mask[static_cast<std::size_t>(id)] == 2) cut.push_back(id);
  const std::size_t ncut = cut.size();
  inside_volume.assign(ncut, 0);
  outside_volume.assign(ncut, 0);
  interface_area.assign(ncut, 0);
  int *device_index = nullptr;
  double *device_in = nullptr;
  double *device_out = nullptr;
  double *device_area = nullptr;
  if (ncut > 0) {
    GPU_CHECK(
        gpuMalloc(reinterpret_cast<void **>(&device_index), ncut * sizeof(int)));
    GPU_CHECK(
        gpuMalloc(reinterpret_cast<void **>(&device_in), ncut * sizeof(double)));
    GPU_CHECK(
        gpuMalloc(reinterpret_cast<void **>(&device_out), ncut * sizeof(double)));
    GPU_CHECK(
        gpuMalloc(reinterpret_cast<void **>(&device_area), ncut * sizeof(double)));
    const auto copy_begin = std::chrono::steady_clock::now();
    GPU_CHECK(gpuMemcpy(device_index, cut.data(), ncut * sizeof(int),
                        gpuMemcpyHostToDevice));
    profile.copy_to_device_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - copy_begin)
            .count();
    profile.bytes_to_device += ncut * sizeof(int);
    gpuEvent_t start, stop;
    GPU_CHECK(gpuEventCreate(&start));
    GPU_CHECK(gpuEventCreate(&stop));
    const int threads = 128;
    const int blocks = static_cast<int>((ncut + threads - 1) / threads);
    GPU_CHECK(gpuEventRecord(start, nullptr));
    GPU_LAUNCH_KERNEL(measure_cuts, blocks, threads,
                      (pack(shape), layout, policy, device_index,
                       static_cast<int>(ncut), device_in, device_out, device_area),
                      nullptr);
    GPU_CHECK(gpuEventRecord(stop, nullptr));
    profile.measure_seconds = elapsed_ms(start, stop);
    GPU_CHECK(gpuEventDestroy(start));
    GPU_CHECK(gpuEventDestroy(stop));
    const auto back = std::chrono::steady_clock::now();
    GPU_CHECK(gpuMemcpy(inside_volume.data(), device_in, ncut * sizeof(double),
                        gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(outside_volume.data(), device_out, ncut * sizeof(double),
                        gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(interface_area.data(), device_area, ncut * sizeof(double),
                        gpuMemcpyDeviceToHost));
    profile.copy_to_host_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - back)
            .count();
    profile.bytes_to_host += 3 * ncut * sizeof(double);
    profile.device_bytes += ncut * sizeof(int) + 3 * ncut * sizeof(double);
  }
  GPU_CHECK(gpuFree(device_mask));
  GPU_CHECK(gpuFree(device_index));
  GPU_CHECK(gpuFree(device_in));
  GPU_CHECK(gpuFree(device_out));
  GPU_CHECK(gpuFree(device_area));
}

} // namespace pfc::geometry
