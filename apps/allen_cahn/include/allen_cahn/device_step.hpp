// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

namespace allen_cahn {

/// One explicit Euler step on device (CUDA): updates `core` in place; face buffers
/// must already hold post-exchange ghost values on device.
void allen_cahn_step_cuda(double *core_dev, const double *hpx_dev,
                          const double *hnx_dev, const double *hpy_dev,
                          const double *hny_dev, const double *hpz_dev,
                          const double *hnz_dev, int nx, int ny, int nz,
                          int halo_width, double inv_dx2, double inv_dy2, double dt,
                          double M, double inv_eps2);

void allen_cahn_step_hip(double *core_dev, const double *hpx_dev,
                         const double *hnx_dev, const double *hpy_dev,
                         const double *hny_dev, const double *hpz_dev,
                         const double *hnz_dev, int nx, int ny, int nz,
                         int halo_width, double inv_dx2, double inv_dy2, double dt,
                         double M, double inv_eps2);

} // namespace allen_cahn
