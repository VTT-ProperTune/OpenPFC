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

} // namespace wave2d

#endif
