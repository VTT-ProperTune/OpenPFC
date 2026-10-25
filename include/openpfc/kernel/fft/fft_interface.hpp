// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file fft_interface.hpp
 * @brief FFT interface types (`IHostFFT` / `IDeviceFFT` / `IFFTQueries`) without
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
 * `openpfc/runtime/gpu/fft_gpu.hpp`; vendor headers re-export it).
 */
enum class Backend : std::uint8_t {
  FFTW, ///< CPU-based FFT using FFTW (default, always available)
  CUDA, ///< GPU-based FFT using cuFFT (include runtime/gpu/fft_gpu.hpp)
  HIP   ///< GPU-based FFT using rocFFT (include runtime/gpu/fft_gpu.hpp)
};

/// Size, bounds, and timing queries shared by host and device FFT interfaces.
struct IFFTQueries {
  virtual ~IFFTQueries() = default;

  virtual void reset_fft_time() = 0;
  /// Rank-local accumulated elapsed seconds since reset. HeFFTe uses
  /// MPI_Wtime(); direct local backends use a steady clock.
  /// Device backends need not synchronize solely for timing; this is not
  /// necessarily elapsed GPU kernel time. No MPI reduction is performed.
  [[nodiscard]] virtual double get_fft_time() const = 0;

  /** @brief Local real-buffer element count required by `forward`/`backward`. */
  [[nodiscard]] virtual size_t size_inbox() const = 0;
  /** @brief Local complex-buffer element count required by `forward`/`backward`. */
  [[nodiscard]] virtual size_t size_outbox() const = 0;
  [[nodiscard]] virtual size_t size_workspace() const = 0;

  [[nodiscard]] virtual size_t get_allocated_memory_bytes() const = 0;

  /// Local real-space index box (inclusive corners) for this rank.
  [[nodiscard]] virtual Box3i get_inbox_bounds() const = 0;
  /// Local r2c Fourier-space index box (inclusive corners).
  /// Values use x-fastest storage within this box; the reduced axis stores
  /// only indices 0..N_axis/2. Ownership may differ from the real inbox.
  [[nodiscard]] virtual Box3i get_outbox_bounds() const = 0;
};

/// Host-container FFT (`std::vector` / host Field). Device backends are not this.
struct IHostFFT : IFFTQueries {
  /**
   * @brief Collective, unscaled real-to-complex transform on the plan communicator.
   *
   * Every rank must call the same transform in the same order with valid
   * local buffers. The output uses get_outbox_bounds(), with x-fastest
   * ordering; its partition can differ from the input box.
   *
   * @param in Real buffer; must satisfy `in.size() == size_inbox()`
   * @param out Complex buffer; must satisfy `out.size() == size_outbox()`
   * @throws std::invalid_argument when either buffer size mismatches the
   *         inbox/outbox contract (implementations report expected and actual).
   */
  virtual void forward(const RealVector &in, ComplexVector &out) = 0;

  /**
   * @brief Collective complex-to-real transform, scaled by 1/N globally.
   *
   * N is the product of the full real-grid dimensions (not the local size).
   * This normalization makes backward(forward(x)) recover x to roundoff.
   * All plan ranks must participate in the same transform order.
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
 *
 * Both transforms are collective on the plan communicator. forward() is
 * unscaled; backward() applies 1/N for the full real grid. Buffers use the
 * rank-local inbox/outbox shapes and x-fastest storage. Size mismatches throw
 * std::invalid_argument locally; this is not an MPI vote.
 *
 * This interface has no stream parameter or asynchronous completion token.
 * The HeFFTe implementation uses its plan's default stream and adds no explicit
 * device-wide synchronization here. Order external producers/consumers on that
 * stream or synchronize explicitly before cross-stream access. Keep buffers
 * alive until the backend has completed their device work.
 */
template <typename MemorySpace> struct IDeviceFFT : IFFTQueries {
  using backend_tag = memory_space_to_backend_t<MemorySpace>;
  using RealBuffer = core::DataBuffer<backend_tag, double>;
  using ComplexBuffer = core::DataBuffer<backend_tag, std::complex<double>>;

  /// Collective unscaled r2c; input/output sizes are size_inbox()/size_outbox().
  virtual void forward(const RealBuffer &in, ComplexBuffer &out) = 0;
  /// Collective c2r with full-grid 1/N scaling; sizes reverse the forward pair.
  virtual void backward(const ComplexBuffer &in, RealBuffer &out) = 0;
};

[[nodiscard]] inline Box3i get_inbox(const IFFTQueries &fft) noexcept {
  return fft.get_inbox_bounds();
}

[[nodiscard]] inline Box3i get_outbox(const IFFTQueries &fft) noexcept {
  return fft.get_outbox_bounds();
}

/** @brief Clear accumulated FFT timing (preferred over `reset_fft_time()`). */
inline void reset_fft_time(IFFTQueries &fft) noexcept { fft.reset_fft_time(); }

/** @brief Rank-local accumulated elapsed seconds since reset.
 */
[[nodiscard]] inline double get_fft_time(const IFFTQueries &fft) noexcept {
  return fft.get_fft_time();
}

} // namespace fft
} // namespace pfc
