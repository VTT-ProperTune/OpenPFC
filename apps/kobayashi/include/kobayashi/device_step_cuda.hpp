// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if !defined(OpenPFC_ENABLE_CUDA)
#error                                                                              \
    "kobayashi/device_step_cuda.hpp requires CUDA (configure with -DOpenPFC_ENABLE_CUDA=ON)"
#endif

namespace kobayashi {

/// Stage A: Laplacians, gradients, and anisotropic \(\epsilon(\theta)\) (matches
/// CPU/HIP ordering).
///
/// When `extend > 0`, the kernel additionally writes the **`extend`-cell ring**
/// outside the owned interior [0, nx) x [0, ny), inside the brick's halo region.
/// This lets the caller skip the pre-stage-B halo exchange of the stage-A outputs
/// (eps, eps_d, phidx, phidy) by trading it for a wider phi/tempr halo
/// (`hw >= extend + 1`). See `KOBAYASHI_HALO_EXTENDED` in `kobayashi_fd_cuda.cpp`.
void kobayashi_stage_a_cuda(const double *phi_dev, const double *tempr_dev,
                            double *lap_phi_dev, double *lap_t_dev,
                            double *phidx_dev, double *phidy_dev,
                            double *epsilon_dev, double *epsilon_deriv_dev, int nx,
                            int ny, int nz, int hw, double inv_dx, double inv_dy,
                            double inv_lap_den, int extend = 0);

/// Stage B: Euler update for \(\phi\) and \(T\) (reads exchanged \(\epsilon\)
/// halos).
void kobayashi_stage_b_cuda(double *phi_dev, double *tempr_dev,
                            const double *lap_phi_dev, const double *lap_t_dev,
                            const double *epsilon_dev,
                            const double *epsilon_deriv_dev, const double *phidx_dev,
                            const double *phidy_dev, int nx, int ny, int nz, int hw,
                            double inv_dx, double inv_dy, double dt);

/**
 * @brief Apply **periodic** x/y (and z for `nz==1`) halos on device for `hw == 1`
 * only.
 *
 * For **single-rank** Kobayashi runs this avoids **MPI + global CUDA sync** on every
 * exchange (see `PaddedDeviceHaloExchanger::exchange_halos_device`). Multi-rank jobs
 * must keep using MPI halos.
 */
void kobayashi_periodic_halos_xy_cuda(double *pad_dev, int nx, int ny, int nz,
                                      int hw);

} // namespace kobayashi
