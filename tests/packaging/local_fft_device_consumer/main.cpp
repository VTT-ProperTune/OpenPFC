// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <openpfc/runtime/common/mpi_main.hpp>
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
#if defined(OPENPFC_TEST_HIP)
  using Space = pfc::HIPSpace;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return 77;
#else
  using Space = pfc::CUDASpace;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
#endif
  return pfc::runtime::mpi_main(argc, argv, [](int, char **, int rank, int ranks) {
    auto domain = pfc::domain::create({13, 7, 3});
    pfc::sim::stacks::GPUSpectralStack<Space> stack(
        domain, rank, ranks, MPI_COMM_WORLD,
        pfc::sim::stacks::DeviceFFTChoice::single_device);
    auto &fft = stack.fft();
    if (stack.fft_choice() != pfc::sim::stacks::DeviceFFTChoice::single_device ||
        fft.get_inbox_bounds() != pfc::domain::index_box(domain))
      throw std::runtime_error("local FFT selection or geometry mismatch");
    using Real = typename pfc::fft::IDeviceFFT<Space>::RealBuffer;
    using Complex = typename pfc::fft::IDeviceFFT<Space>::ComplexBuffer;
    std::vector<double> original(fft.size_inbox());
    for (std::size_t i = 0; i < original.size(); ++i)
      original[i] = std::sin(0.17 * static_cast<double>(i));
    Real input(original.size()), result(original.size());
    Complex spectrum(fft.size_outbox());
    input.copy_from_host(original);
    fft.forward(input, spectrum);
    const auto saved = spectrum.to_host();
    fft.backward(spectrum, result);
    if (input.to_host() != original || spectrum.to_host() != saved)
      throw std::runtime_error("local FFT modified an input");
    const auto restored = result.to_host();
    for (std::size_t i = 0; i < original.size(); ++i)
      if (!std::isfinite(restored[i]) || std::abs(restored[i] - original[i]) > 1e-12)
        throw std::runtime_error("local FFT round trip mismatch");
    return 0;
  });
}
