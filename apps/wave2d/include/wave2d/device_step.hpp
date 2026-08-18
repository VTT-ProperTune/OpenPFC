// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

namespace wave2d {

void wave2d_step_cuda(double *u_dev, double *v_dev, const double *hpx_dev,
                      const double *hnx_dev, const double *hpy_dev,
                      const double *hny_dev, const double * /*hpz_dev*/,
                      const double * /*hnz_dev*/, int nx, int ny, int nz,
                      int halo_width, double inv_dx2, double inv_dy2, double dt,
                      double wave_c);

void wave2d_patch_y_faces_cuda(const double *u_dev, double *hpy_dev, double *hny_dev,
                               int nx, int ny, int lower_y, int Ny_global,
                               bool dirichlet, double u_wall);

void wave2d_enforce_dirichlet_walls_cuda(double *u_dev, double *v_dev, int nx,
                                         int ny, int lower_y, int Ny_global,
                                         double u_wall);

} // namespace wave2d

#endif

#if defined(OpenPFC_ENABLE_HIP)

namespace wave2d {

void wave2d_step_hip(double *u_dev, double *v_dev, const double *hpx_dev,
                     const double *hnx_dev, const double *hpy_dev,
                     const double *hny_dev, const double * /*hpz_dev*/,
                     const double * /*hnz_dev*/, int nx, int ny, int nz,
                     int halo_width, double inv_dx2, double inv_dy2, double dt,
                     double wave_c);

/// Overwrite ±Y recv slabs on device (same math as the host patch helpers).
void wave2d_patch_y_faces_hip(const double *u_dev, double *hpy_dev, double *hny_dev,
                              int nx, int ny, int lower_y, int Ny_global,
                              bool dirichlet, double u_wall);

/// Set owned y-wall cells: u = u_wall, v = 0.
void wave2d_enforce_dirichlet_walls_hip(double *u_dev, double *v_dev, int nx,
                                        int ny, int lower_y, int Ny_global,
                                        double u_wall);

} // namespace wave2d

#endif
