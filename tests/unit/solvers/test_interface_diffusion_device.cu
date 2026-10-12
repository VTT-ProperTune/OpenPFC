// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Device apply for the Cartesian transmission operator. The unknown stays
// on the device; only the final field is copied back.

#include <openpfc/solvers/interface_diffusion.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct DevRow {
  int node, n, col[7];
  double val[7];
};

__global__ void apply_interface(const DevRow *rows, int nrows, const double *u,
                                double *y) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nrows) return;
  double acc = 0;
  for (int s = 0; s < rows[t].n; ++s) acc += rows[t].val[s] * u[rows[t].col[s]];
  y[rows[t].node] = acc;
}

__global__ void apply_regular(pfc::geometry::PackedImplicit shape, int cells,
                              double h, double kappa_minus, double kappa_plus,
                              const int *skip, const double *u, double *y) {
  const int n = cells + 1;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n * n * n) return;
  if (skip[id]) return;
  const int i = id % n;
  const int jk = id / n;
  const int j = jk % n;
  const int k = jk / n;
  if (i == 0 || j == 0 || k == 0 || i == cells || j == cells || k == cells) {
    y[id] = u[id];
    return;
  }
  const auto phi = pfc::geometry::unpack(shape).value({i * h, j * h, k * h});
  const double kappa = phi < 0 ? kappa_minus : kappa_plus;
  const double invh2 = kappa / (h * h);
  const int sx = 1, sy = n, sz = n * n;
  y[id] = invh2 * (6 * u[id] - u[id - sx] - u[id + sx] - u[id - sy] - u[id + sy] -
                   u[id - sz] - u[id + sz]);
}

int fail(const char *why) {
  std::cerr << why << "\n";
  return 1;
}

} // namespace

int main() {
  const auto shape = pfc::geometry::make_plane({0.46, 0.52, 0.48}, {1.0, 1.3, 0.4});
  pfc::solvers::TransmissionRequest request;
  request.cells = 16;
  request.closure = pfc::solvers::TransmissionClosure::Calibrated;
  request.kappa_minus = 1;
  request.kappa_plus = 10;
  const auto op = pfc::solvers::build_transmission(shape, request);
  std::vector<double> u(static_cast<std::size_t>(op.node_count)), host(u.size());
  for (int i = 0; i < op.node_count; ++i)
    u[static_cast<std::size_t>(i)] = 0.1 * std::sin(0.2 * i);
  op.apply(u, host);
  std::vector<DevRow> rows(op.interface_row.size());
  std::vector<int> skip(static_cast<std::size_t>(op.node_count), 0);
  for (std::size_t r = 0; r < op.interface_row.size(); ++r) {
    rows[r].node = op.interface_row[r].node;
    rows[r].n = op.interface_row[r].n;
    skip[static_cast<std::size_t>(rows[r].node)] = 1;
    for (int s = 0; s < rows[r].n; ++s) {
      rows[r].col[s] = op.interface_row[r].column[s];
      rows[r].val[s] = op.interface_row[r].value[s];
    }
  }
  double *du = nullptr, *dy = nullptr;
  DevRow *drows = nullptr;
  int *dskip = nullptr;
  cudaMalloc(&du, u.size() * sizeof(double));
  cudaMalloc(&dy, u.size() * sizeof(double));
  cudaMalloc(&drows, rows.size() * sizeof(DevRow));
  cudaMalloc(&dskip, skip.size() * sizeof(int));
  cudaMemcpy(du, u.data(), u.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(drows, rows.data(), rows.size() * sizeof(DevRow),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dskip, skip.data(), skip.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dy, 0, u.size() * sizeof(double));
  const int threads = 128;
  const int nblocks = (op.node_count + threads - 1) / threads;
  const int rblocks = static_cast<int>((rows.size() + threads - 1) / threads);
  apply_regular<<<nblocks, threads>>>(pfc::geometry::pack(shape), op.cells,
                                      op.spacing, op.kappa_minus, op.kappa_plus,
                                      dskip, du, dy);
  if (!rows.empty())
    apply_interface<<<std::max(rblocks, 1), threads>>>(
        drows, static_cast<int>(rows.size()), du, dy);
  cudaDeviceSynchronize();
  std::vector<double> device(u.size());
  cudaMemcpy(device.data(), dy, device.size() * sizeof(double),
             cudaMemcpyDeviceToHost);
  double max_diff = 0;
  for (std::size_t i = 0; i < device.size(); ++i)
    max_diff = std::max(max_diff, std::abs(device[i] - host[i]));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int repeat = 0; repeat < 50; ++repeat) {
    apply_regular<<<nblocks, threads>>>(pfc::geometry::pack(shape), op.cells,
                                        op.spacing, op.kappa_minus, op.kappa_plus,
                                        dskip, du, dy);
    if (!rows.empty())
      apply_interface<<<std::max(rblocks, 1), threads>>>(
          drows, static_cast<int>(rows.size()), du, dy);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "device apply max diff " << max_diff << " rows " << rows.size()
            << " apply_s " << (ms / 50.0) * 1e-3 << "\n";
  cudaFree(du);
  cudaFree(dy);
  cudaFree(drows);
  cudaFree(dskip);
  if (!(max_diff < 1e-10)) return fail("device apply does not match the host");
  std::cout << "device transmission apply passed\n";
  return 0;
}
