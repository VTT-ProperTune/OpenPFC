// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_operations_gpu.hpp
 * @brief Owned-cell reductions for `pfc::solvers::FieldOperations`.
 */

#include <cstddef>

namespace pfc::solvers::gpu::fops {

/// Owned extents, padded pitch and storage halo of one field.
struct Geometry {
  int n[3] = {1, 1, 1};
  int pitch[3] = {1, 1, 1};
  int hw = 0;
};

/// One partial sum per block, capped so the follow-up sum fits one launch.
inline int dot_blocks(std::size_t owned) {
  const std::size_t b = (owned + 255) / 256;
  return static_cast<int>(b < 1 ? 1 : (b > 1024 ? 1024 : b));
}

#if defined(OpenPFC_ENABLE_CUDA)
void owned_dot_cuda(const Geometry &g, const double *x, const double *y,
                    const double *m, double *out, int blocks);
double sum_cuda(double *p, int n);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void owned_dot_hip(const Geometry &g, const double *x, const double *y,
                   const double *m, double *out, int blocks);
double sum_hip(double *p, int n);
#endif

} // namespace pfc::solvers::gpu::fops
