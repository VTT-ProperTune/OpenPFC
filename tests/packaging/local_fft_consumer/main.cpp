// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// No vendor or HeFFTe header/link target is supplied by this consumer.
#include <cmath>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <vector>

int main() {
  pfc::fft::LocalFFT transform({13, 7, 3});
  std::vector<double> original(transform.size_inbox()), result(original.size());
  for (std::size_t i = 0; i < original.size(); ++i)
    original[i] = std::sin(0.17 * static_cast<double>(i));
  const auto saved = original;
  std::vector<std::complex<double>> spectrum(transform.size_outbox());
  transform.forward(original, spectrum);
  const auto saved_spectrum = spectrum;
  transform.backward(spectrum, result);
  if (original != saved || spectrum != saved_spectrum) return 1;
  for (std::size_t i = 0; i < original.size(); ++i)
    if (!std::isfinite(result[i]) || std::abs(result[i] - original[i]) > 1e-12)
      return 2;
  return 0;
}
