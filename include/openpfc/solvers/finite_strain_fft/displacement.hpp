// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <openpfc/solvers/finite_strain_fft/projector.hpp>

namespace pfc::finite_strain {
struct DisplacementReconstruction {
  std::vector<std::array<double, 3>> fluctuation;
  /// Local squared norms; sum across the FFT communicator for a global ratio.
  double gradient_error_squared = 0, gradient_squared = 0;
};

/// Reconstruct zero-mean periodic w from F = macro + grad(w), on an odd grid.
/// Uses the supplied FFT's normalization, layout and communicator. All ranks
/// must call in the same order. This is accepted-field host postprocessing.
/// Incompatible fields return their least-squares compatible displacement;
/// the gradient diagnostic measures that discrepancy rather than hiding it.
inline DisplacementReconstruction
reconstruct_displacement(fft::IHostFFT &fft, const Domain &domain,
                         const std::vector<Tensor2> &deformation,
                         const Tensor2 &macro) {
  require_odd_grid(domain);
  if (deformation.size() != fft.size_inbox())
    throw std::invalid_argument("displacement field does not match FFT inbox");
  for (bool periodic : domain::get_periodic(domain))
    if (!periodic)
      throw std::invalid_argument(
          "displacement reconstruction requires periodic domain");
  for (const auto &f : deformation)
    for (int c = 0; c < 9; ++c)
      if (!std::isfinite(f(c / 3, c % 3)) || !std::isfinite(macro(c / 3, c % 3)))
        throw std::invalid_argument("nonfinite displacement gradient");
  DisplacementReconstruction result;
  result.fluctuation.resize(deformation.size());
  std::vector<double> real(fft.size_inbox());
  std::vector<std::complex<double>> hat(fft.size_outbox()), w(hat.size()),
      derivative(hat.size());
  const auto box = fft.get_outbox_bounds();
  const std::complex<double> imaginary(0, 1);
  for (int i = 0; i < 3; ++i) {
    std::fill(w.begin(), w.end(), 0);
    for (int j = 0; j < 3; ++j) {
      for (std::size_t p = 0; p < real.size(); ++p)
        real[p] = deformation[p](i, j) - macro(i, j);
      fft.forward(real, hat);
      fft::kspace::for_each_kpoint(
          box, domain,
          [&](std::size_t p, double x, double y, double z, int, int, int) {
            const double q[3] = {x, y, z}, q2 = x * x + y * y + z * z;
            if (q2 > 0) w[p] += -imaginary * q[j] * hat[p] / q2;
          });
    }
    fft.backward(w, real);
    for (std::size_t p = 0; p < real.size(); ++p) result.fluctuation[p][i] = real[p];
    for (int j = 0; j < 3; ++j) {
      fft::kspace::for_each_kpoint(
          box, domain,
          [&](std::size_t p, double x, double y, double z, int, int, int) {
            const double q[3] = {x, y, z};
            derivative[p] = imaginary * q[j] * w[p];
          });
      fft.backward(derivative, real);
      for (std::size_t p = 0; p < real.size(); ++p) {
        const double target = deformation[p](i, j) - macro(i, j);
        result.gradient_error_squared += (real[p] - target) * (real[p] - target);
        result.gradient_squared += target * target;
      }
    }
  }
  return result;
}
} // namespace pfc::finite_strain
