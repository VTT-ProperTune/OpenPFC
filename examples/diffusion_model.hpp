// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

namespace diffusion_example {

inline void fill_gaussian(pfc::data::Field<double> &psi, double D) {
  psi.apply([&](double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z) / (4.0 * D));
  });
}

[[nodiscard]] inline int find_midpoint_idx(const pfc::data::Field<double> &psi) {
  int idx = 0;
  int found = -1;
  const auto &box = psi.box();
  const auto &origin = psi.origin();
  const auto &spacing = psi.spacing();
  for (int k = 0; k < box.size[2]; ++k) {
    for (int j = 0; j < box.size[1]; ++j) {
      for (int i = 0; i < box.size[0]; ++i) {
        const double x = origin[0] + (box.low[0] + i) * spacing[0];
        const double y = origin[1] + (box.low[1] + j) * spacing[1];
        const double z = origin[2] + (box.low[2] + k) * spacing[2];
        if (std::abs(x) < 1.0e-9 && std::abs(y) < 1.0e-9 && std::abs(z) < 1.0e-9) {
          found = idx;
        }
        ++idx;
      }
    }
  }
  return found;
}

inline void prepare_implicit_euler_opL(pfc::fft::IHostFFT &fft,
                                       const pfc::Domain &domain, double dt,
                                       std::vector<double> &opL) {
  opL.resize(fft.size_outbox());
  pfc::fft::kspace::for_each_kpoint(
      get_outbox(fft), domain,
      [&](std::size_t idx, double ki, double kj, double kk, int, int, int) {
        const double kLap = -(ki * ki + kj * kj + kk * kk);
        opL[idx] = 1.0 / (1.0 - dt * kLap);
      });
}

} // namespace diffusion_example
