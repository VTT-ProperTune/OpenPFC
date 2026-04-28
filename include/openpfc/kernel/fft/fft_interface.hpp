// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_interface.hpp
 * @brief FFT interface types (IFFT, backends, buffers) without HeFFTe headers
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

#include <complex>
#include <cstdint>
#include <vector>

namespace pfc {
namespace fft {

using RealVector = std::vector<double>;
using ComplexVector = std::vector<std::complex<double>>;

using RealDataBuffer = core::DataBuffer<backend::CpuTag, double>;
using ComplexDataBuffer = core::DataBuffer<backend::CpuTag, std::complex<double>>;

/**
 * @brief FFT backend selection
 *
 * FFTW is in kernel; CUDA backend is selected via runtime (include
 * openpfc/runtime/cuda/fft_cuda.hpp for RealDataBufferCUDA, create_cuda, etc.)
 */
enum class Backend : std::uint8_t {
  FFTW, ///< CPU-based FFT using FFTW (default, always available)
  CUDA  ///< GPU-based FFT using cuFFT (include runtime/cuda/fft_cuda.hpp)
};

struct IFFT {
  virtual ~IFFT() = default;

  virtual void forward(const RealVector &in, ComplexVector &out) = 0;
  virtual void backward(const ComplexVector &in, RealVector &out) = 0;

  virtual void reset_fft_time() = 0;
  virtual double get_fft_time() const = 0;

  virtual size_t size_inbox() const = 0;
  virtual size_t size_outbox() const = 0;
  virtual size_t size_workspace() const = 0;

  virtual size_t get_allocated_memory_bytes() const = 0;

  /**
   * @brief Local real-space index box (inclusive corners) for this rank
   */
  [[nodiscard]] virtual Box3i get_inbox_bounds() const = 0;

  /**
   * @brief Local Fourier-space index box for this rank
   */
  [[nodiscard]] virtual Box3i get_outbox_bounds() const = 0;
};

[[nodiscard]] inline Box3i get_inbox(const IFFT &fft) noexcept {
  return fft.get_inbox_bounds();
}

[[nodiscard]] inline Box3i get_outbox(const IFFT &fft) noexcept {
  return fft.get_outbox_bounds();
}

} // namespace fft
} // namespace pfc
