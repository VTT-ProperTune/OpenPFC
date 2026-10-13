// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <concepts>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#include <utility>

using Stack = pfc::sim::stacks::SpectralCPUStack;
static_assert(
    std::same_as<decltype(std::declval<Stack &>().fft()), pfc::fft::IHostFFT &>);
static_assert(std::same_as<decltype(std::declval<const Stack &>().fft()),
                           const pfc::fft::IHostFFT &>);

bool round_trip(pfc::fft::IHostFFT &fft) {
  std::vector<double> original(fft.size_inbox()), restored(original.size());
  for (std::size_t i = 0; i < original.size(); ++i)
    original[i] = std::sin(0.17 * static_cast<double>(i));
  const auto saved = original;
  std::vector<std::complex<double>> spectrum(fft.size_outbox());
  fft.forward(original, spectrum);
  const auto saved_spectrum = spectrum;
  fft.backward(spectrum, restored);
  if (original != saved || spectrum != saved_spectrum) return false;
  for (std::size_t i = 0; i < original.size(); ++i)
    if (!std::isfinite(restored[i]) || std::abs(restored[i] - original[i]) > 1e-12)
      return false;
  return true;
}

int main() {
  MPI_Init(nullptr, nullptr);
  bool ok = true;
  {
    auto domain = pfc::domain::create({13, 7, 3});
    Stack reference(domain, 0, 1, MPI_COMM_SELF);
    Stack local(domain, 0, 1, MPI_COMM_SELF, pfc::fft::Implementation::Local);
    ok = round_trip(reference.fft()) && round_trip(local.fft());
    // Explicit concrete consumers keep the existing factory and overloads.
    auto concrete = pfc::fft::create(reference.decomposition(), 0, MPI_COMM_SELF);
    pfc::fft::CPUFFT &legacy = concrete;
    ok = ok && round_trip(legacy);
  }
  MPI_Finalize();
  return ok ? 0 : 1;
}
