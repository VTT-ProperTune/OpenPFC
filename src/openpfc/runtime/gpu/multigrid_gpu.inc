// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/runtime/gpu/multigrid.hpp>
namespace pfc::solvers::gpu::detail {
__global__ void sparse_kernel(CsrView a, const double *x, double *y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows) return;
  double sum = 0;
  for (int p = a.offsets[row]; p < a.offsets[row + 1]; ++p)
    sum += a.values[p] * x[a.columns[p]];
  y[row] = sum;
}
__global__ void dense_kernel(const double *a, const double *x, double *y, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  double sum = 0;
  for (int j = 0; j < n; ++j) sum += a[row * n + j] * x[j];
  y[row] = sum;
}
__global__ void smooth_kernel(double *x, const double *b, const double *ax,
                              const double *diag, double omega, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += omega * (b[i] - ax[i]) / diag[i];
}
__global__ void boundary_kernel(double *x, int nx, int ny, int nz, int components) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= nx * ny * nz) return;
  int i = node % nx, j = (node / nx) % ny, k = node / (nx * ny);
  if (!i || !j || !k || i + 1 == nx || j + 1 == ny || k + 1 == nz)
    for (int c = 0; c < components; ++c) x[node * components + c] = 0;
}
#if defined(OPENPFC_GPU_API_CUDA)
#define NAME(n) n##_cuda
#else
#define NAME(n) n##_hip
#endif
void NAME(sparse)(CsrView a, const double *x, double *y) {
  if (a.rows) sparse_kernel<<<(a.rows + 255) / 256, 256>>>(a, x, y);
  check(pfc::gpuGetLastError());
}
void NAME(dense)(const double *a, const double *x, double *y, int n) {
  if (n) dense_kernel<<<(n + 255) / 256, 256>>>(a, x, y, n);
  check(pfc::gpuGetLastError());
}
void NAME(smooth)(double *x, const double *b, const double *ax, const double *diag,
                  double omega, int n) {
  if (n) smooth_kernel<<<(n + 255) / 256, 256>>>(x, b, ax, diag, omega, n);
  check(pfc::gpuGetLastError());
}
void NAME(boundary)(double *x, StructuredGrid grid) {
  boundary_kernel<<<(grid.nodes() + 255) / 256, 256>>>(
      x, grid.shape[0], grid.shape[1], grid.shape[2], grid.components);
  check(pfc::gpuGetLastError());
}
#undef NAME
} // namespace pfc::solvers::gpu::detail
