// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_step.hpp
 * @brief HIP kernels for the 3D heat-equation FD driver (`heat3d_fd_hip`).
 *
 * Host wrappers live in `src/hip/heat3d_fd_hip_kernels.hip`. The RHS uses
 * `pfc::sim::gpu::for_each_interior_device` with `heat3d::HeatGrads`; the
 * Euler update is a padded axpy (`u += dt * du`).
 */

#include <cstddef>

#include <hip/hip_runtime.h>

#include <heat3d/heat_model.hpp>
#include <openpfc/runtime/gpu/fd_gradient_device_gpu.hpp>

namespace heat3d {

void fd_rhs_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval, double *du_padded,
                double t, int nx, int ny, int nz);

/// Halo-independent owned interior. @p sync false leaves the kernel in flight.
/// @p stream defaults to the null (legacy default) stream.
void fd_rhs_inner_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval,
                      double *du_padded, double t, int nx, int ny, int nz, int hw,
                      bool sync = true, hipStream_t stream = nullptr);

/// Owned boundary shell after halo completion.
void fd_rhs_border_hip(const pfc::gpu::FDGradientDevice<HeatGrads> &eval,
                       double *du_padded, double t, int nx, int ny, int nz, int hw,
                       bool sync = true);

void euler_axpy_hip(double *u, const double *du, double dt, std::size_t n);

} // namespace heat3d
