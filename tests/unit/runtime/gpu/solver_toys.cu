// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cuda_runtime.h>
#include <stdexcept>
__global__ void toy_action(int n, const double *x, double *y, int kind,
                           const double *state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (kind == 0)
    y[i] = (4 + .01 * i) * x[i] + .7 * x[(i + 1) % n] - .2 * x[(i + n - 1) % n];
  if (kind == 1) y[i] = x[i] * x[i] - (1 + .002 * i) * (1 + .002 * i);
  if (kind == 2) y[i] = 2 * state[i] * x[i];
}
void solver_toy(int n, const double *x, double *y, int kind,
                const double *state = nullptr) {
  toy_action<<<(n + 255) / 256, 256>>>(n, x, y, kind, state);
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("toy launch failed");
}
