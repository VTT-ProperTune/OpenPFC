// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spinodal_generator.hpp
 * @brief Process-parameter Cahn–Hilliard microstructures (issue #161 Stage 6).
 *
 * @details
 * Low-dimensional family: volume fraction \(c_0\), interfacial \(\kappa\),
 * evolution time, and optional anisotropy \((a_x,a_y,a_z)\) in the
 * Laplacian (spinodoid-like). Semi-implicit spectral CH on the same
 * HeFFTe plan as the homogenizer — not a neural surrogate, not free
 * voxel TO. Mean composition is conserved.
 *
 * Bicontinuous CH morphologies are not expected to be auxetic; that is
 * the comparison against `--init=rotating-squares`.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <inverse_homogenization/phase_field_inverse.hpp>

namespace pfc::apps::inverse {

struct SpinodalSpec {
  double c0{0.5};
  double kappa{1.0};
  double mobility{1.0};
  double dt{0.2};
  int steps{200};
  double noise{0.12};
  unsigned seed{1};
  double ax{1.0};
  double ay{1.0};
  double az{1.0};
};

/// Seed \(h=c_0\) plus a few Fourier modes (mean-preserving).
inline void seed_spinodal_noise(RealField &h, int nx, int ny, int nz,
                                const SpinodalSpec &spec) {
  const auto n = h.local_size();
  const double twopi = 2.0 * 3.141592653589793;
  const double s = static_cast<double>(spec.seed);
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = h.global(i, j, k);
        const double n1 = std::sin(twopi * (g[0] + s) / std::max(nx, 1));
        const double n2 = std::sin(twopi * (2.0 * g[1] + 2.0 * s) / std::max(ny, 1));
        const double n3 = std::sin(twopi * (g[2] + 3.0 * s) / std::max(nz, 1));
        const double n4 =
            std::sin(2.0 * twopi * g[0] / std::max(nx, 1)) *
            std::sin(twopi * g[1] / std::max(ny, 1));
        const double fl = spec.noise * (0.5 * n1 * n2 + 0.3 * n3 + 0.4 * n4);
        h(i, j, k) = std::min(1.0, std::max(0.0, spec.c0 + fl));
      }
    }
  }
  h.note_host_write();
}

/**
 * @brief Semi-implicit Cahn–Hilliard: \(\hat h \leftarrow
 * (\hat h - \Delta t M k^2 \widehat{W'(h)}) / (1 + \Delta t M \kappa k^4)\).
 */
inline void generate_spinodal(const pfc::Domain &domain, pfc::fft::IHostFFT &fft,
                              RealField &h, const SpinodalSpec &spec) {
  if (spec.steps < 0 || spec.dt <= 0.0 || spec.kappa <= 0.0 ||
      spec.mobility <= 0.0) {
    throw std::invalid_argument("generate_spinodal: kappa, M, dt > 0");
  }
  ComplexField hat(domain, fft.get_outbox_bounds(), 0);
  ComplexField hat_nl(domain, fft.get_outbox_bounds(), 0);
  RealField nl =
      pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds());
  const double M = spec.mobility;
  const double kap = spec.kappa;
  const double dt = spec.dt;
  for (int s = 0; s < spec.steps; ++s) {
    double *nld = nl.data();
    const double *hd = h.data();
    for (std::size_t i = 0; i < h.size(); ++i)
      nld[i] = double_well_prime(hd[i]);
    nl.note_host_write();
    std::vector<double> tmp_h(h.vec());
    std::vector<double> tmp_nl(nl.vec());
    fft.forward(tmp_h, hat.vec());
    fft.forward(tmp_nl, hat_nl.vec());
    pfc::fft::kspace::for_each_kpoint(
        fft.get_outbox_bounds(), domain,
        [&](std::size_t idx, double kx, double ky, double kz, int, int, int) {
          const double k2 =
              spec.ax * kx * kx + spec.ay * ky * ky + spec.az * kz * kz;
          const std::complex<double> rhs =
              hat.data()[idx] - dt * M * k2 * hat_nl.data()[idx];
          const double den = 1.0 + dt * M * kap * k2 * k2;
          hat.data()[idx] = rhs / den;
        });
    hat.note_host_write();
    fft.backward(hat.vec(), h.vec());
    h.note_host_write();
  }
}

} // namespace pfc::apps::inverse
