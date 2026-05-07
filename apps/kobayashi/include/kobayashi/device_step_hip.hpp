// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if !defined(OpenPFC_ENABLE_HIP)
#error "kobayashi/device_step_hip.hpp requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

namespace kobayashi {

/// Stage A: Laplacians, gradients, and anisotropic \(\epsilon(\theta)\) (matches CPU ordering).
void kobayashi_stage_a_hip(const double *phi_dev, const double *tempr_dev,
                           double *lap_phi_dev, double *lap_t_dev, double *phidx_dev,
                           double *phidy_dev, double *epsilon_dev,
                           double *epsilon_deriv_dev, int nx, int ny, int nz, int hw,
                           double inv_dx, double inv_dy, double inv_lap_den);

/// Stage B: Euler update for \(\phi\) and \(T\) (reads exchanged \(\epsilon\) halos).
void kobayashi_stage_b_hip(double *phi_dev, double *tempr_dev,
                           const double *lap_phi_dev, const double *lap_t_dev,
                           const double *epsilon_dev, const double *epsilon_deriv_dev,
                           const double *phidx_dev, const double *phidy_dev, int nx, int ny,
                           int nz, int hw, double inv_dx, double inv_dy, double dt);

} // namespace kobayashi
