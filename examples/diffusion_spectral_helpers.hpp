// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

#include <mpi.h>
#include <openpfc/kernel/fft/fft.hpp>

namespace diffusion_example {

/// One implicit-Euler spectral diffusion step: forward FFT, apply `opL`, backward
/// FFT.
inline void spectral_diffusion_step(pfc::fft::IFFT &fft, std::vector<double> &psi,
                                    std::vector<std::complex<double>> &psi_F,
                                    const std::vector<double> &opL) {
  fft.forward(psi, psi_F);
  for (size_t k = 0, N = psi_F.size(); k < N; k++) {
    psi_F[k] = opL[k] * psi_F[k];
  }
  fft.backward(psi_F, psi);
}

/// Local min/max of `psi`, then `MPI_Reduce` to rank 0 (`MPI_COMM_WORLD`).
inline void reduce_psi_min_max_mpi(const std::vector<double> &psi, double &psi_min,
                                   double &psi_max) {
  double local_min = std::numeric_limits<double>::max();
  double local_max = std::numeric_limits<double>::lowest();
  for (double v : psi) {
    local_min = std::min(local_min, v);
    local_max = std::max(local_max, v);
  }
  MPI_Reduce(&local_min, &psi_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_max, &psi_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

} // namespace diffusion_example
