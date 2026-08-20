// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_interface.hpp
 * @brief FFT interface types (`IHostFFT` / `IDeviceFFT` / `IFFT`) without
 *        HeFFTe headers.
 *
 * ADR 0005: host factories return `IHostFFT` only. Device factories
 * (`create_cuda` / `create_hip`) return objects that implement
 * `IDeviceFFT<MemorySpace>`.
 */

#pragma once

#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/fft/box3i.hpp>

#include <complex>
#include <cstdint>
#include <vector>

namespace pfc {
namespace fft {

using RealVector = std::vector<double>;
using ComplexVector = std::vector<std::complex<double>>;

using RealDataBuffer = core::DataBuffer<backend::CPUTag, double>;
using ComplexDataBuffer = core::DataBuffer<backend::CPUTag, std::complex<double>>;

/**
 * @brief FFT backend selection
 *
 * FFTW is in kernel; GPU backends are selected via runtime (include
 * `openpfc/runtime/cuda/fft_cuda.hpp` or `openpfc/runtime/hip/fft_hip.hpp`).
 */
enum class Backend : std::uint8_t {
  FFTW, ///< CPU-based FFT using FFTW (default, always available)
  CUDA, ///< GPU-based FFT using cuFFT (include runtime/cuda/fft_cuda.hpp)
  HIP   ///< GPU-based FFT using rocFFT (include runtime/hip/fft_hip.hpp)
};

/// Size, bounds, and timing queries shared by host and device FFT interfaces.
struct IFFTQueries {
  virtual ~IFFTQueries() = default;

  virtual void reset_fft_time() = 0;
  [[nodiscard]] virtual double get_fft_time() const = 0;

  /** @brief Local real-buffer element count required by `forward`/`backward`. */
  [[nodiscard]] virtual size_t size_inbox() const = 0;
  /** @brief Local complex-buffer element count required by `forward`/`backward`. */
  [[nodiscard]] virtual size_t size_outbox() const = 0;
  [[nodiscard]] virtual size_t size_workspace() const = 0;

  [[nodiscard]] virtual size_t get_allocated_memory_bytes() const = 0;

  /// Local real-space index box (inclusive corners) for this rank.
  [[nodiscard]] virtual Box3i get_inbox_bounds() const = 0;
  /// Local Fourier-space index box for this rank.
  [[nodiscard]] virtual Box3i get_outbox_bounds() const = 0;
};

/// Host-container FFT (`std::vector` / host Field). Device backends are not this.
struct IHostFFT : IFFTQueries {
  /**
   * @brief Forward real-to-complex transform on this rank's local boxes.
   *
   * @param in Real buffer; must satisfy `in.size() == size_inbox()`
   * @param out Complex buffer; must satisfy `out.size() == size_outbox()`
   * @throws std::invalid_argument when either buffer size mismatches the
   *         inbox/outbox contract (implementations report expected and actual).
   */
  virtual void forward(const RealVector &in, ComplexVector &out) = 0;

  /**
   * @brief Backward complex-to-real transform on this rank's local boxes.
   *
   * @param in Complex buffer; must satisfy `in.size() == size_outbox()`
   * @param out Real buffer; must satisfy `out.size() == size_inbox()`
   * @throws std::invalid_argument when either buffer size mismatches the
   *         outbox/inbox contract (implementations report expected and actual).
   */
  virtual void backward(const ComplexVector &in, RealVector &out) = 0;
};

/**
 * @brief Device-buffer FFT for one memory space (`DataBuffer` transforms).
 *
 * Instantiating `IDeviceFFT<CUDASpace>` / `IDeviceFFT<HIPSpace>` requires the
 * matching runtime memory-space header so `memory_space_to_backend_t` is
 * defined. Double is the interface precision; float overloads stay on the
 * concrete `FFT_Impl`.
 */
template <typename MemorySpace> struct IDeviceFFT : IFFTQueries {
  using backend_tag = memory_space_to_backend_t<MemorySpace>;
  using RealBuffer = core::DataBuffer<backend_tag, double>;
  using ComplexBuffer = core::DataBuffer<backend_tag, std::complex<double>>;

  virtual void forward(const RealBuffer &in, ComplexBuffer &out) = 0;
  virtual void backward(const ComplexBuffer &in, RealBuffer &out) = 0;
};

/// Temporary alias until remaining `IFFT` call sites migrate to `IHostFFT`.
using IFFT = IHostFFT;

[[nodiscard]] inline Box3i get_inbox(const IFFTQueries &fft) noexcept {
  return fft.get_inbox_bounds();
}

[[nodiscard]] inline Box3i get_outbox(const IFFTQueries &fft) noexcept {
  return fft.get_outbox_bounds();
}

/** @brief Clear accumulated FFT timing (preferred over `reset_fft_time()`). */
inline void reset_fft_time(IFFTQueries &fft) noexcept { fft.reset_fft_time(); }

/** @brief Accumulated FFT time since last reset (preferred over `get_fft_time()`). */
[[nodiscard]] inline double get_fft_time(const IFFTQueries &fft) noexcept {
  return fft.get_fft_time();
}

} // namespace fft
} // namespace pfc
