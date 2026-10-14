// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Installed-header consumer for `CoupledSpectralETD2`. Two fields, one
// Fourier mode, lower-triangular symbol L = [[-k^2, 0], [c, -2 k^2]] and no
// nonlinearity. ETD2 is exact for a linear equation, so the oracle is the
// closed form of exp(t L): diagonal decays and the off-diagonal entry
// c (e^{a t} - e^{b t}) / (a - b).
#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/coupled_spectral_etd2.hpp>

#include <cmath>
#include <cstdio>
#include <span>

// Returns 0 when one installed coupled ETD2 step matches the closed form.
int run_coupled_etd2_consumer() {
  constexpr int n = 8;
  constexpr double c = 0.75;
  constexpr double dt = 0.2;
  auto domain = pfc::domain::create(pfc::GridSize({n, n, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  auto symbol = pfc::sim::coupled_linear_symbol(
      fft, fft.get_outbox_bounds(), domain, [](double kx, double ky, double) {
        const double k2 = kx * kx + ky * ky;
        return pfc::integrator::Matrix2x2{-k2, 0.0, c, -2.0 * k2};
      });
  auto none = [](double, std::span<const double>, std::span<const double>,
                 std::span<double> f0, std::span<double> f1) {
    for (auto &v : f0) v = 0.0;
    for (auto &v : f1) v = 0.0;
  };
  pfc::sim::CoupledSpectralETD2 system(fft, symbol, none,
                                       pfc::sim::CoupledStageOutput::nonlinear);
  const double k = 2.0 * pfc::pi / static_cast<double>(n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      system.field(0)[static_cast<std::size_t>(i + j * n)] = std::cos(k * i);
    }
  }
  system.load_state();
  system.step(0.0, dt);
  const double a = -k * k;
  const double b = -2.0 * k * k;
  const double e0 = std::exp(a * dt);
  const double e1 = c * (std::exp(a * dt) - std::exp(b * dt)) / (a - b);
  double worst = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const auto q = static_cast<std::size_t>(i + j * n);
      worst = std::fmax(worst, std::fabs(system.field(0)[q] - e0 * std::cos(k * i)));
      worst = std::fmax(worst, std::fabs(system.field(1)[q] - e1 * std::cos(k * i)));
    }
  }
  if (!(worst < 1e-13)) {
    std::printf("CoupledSpectralETD2: closed-form error %.3e\n", worst);
    return 1;
  }
  return 0;
}
