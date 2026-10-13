// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <openpfc/kernel/fft/fft_interface.hpp>

namespace pfc::fft {

/// Transform implementation, independently of the hardware backend.
enum class Implementation { Heffte, Local };

/**
 * @brief Complete-domain double-precision r2c/c2r transforms through FFTW.
 *
 * Stores x fastest and reduces x to nx/2+1 complex modes. Forward is
 * unscaled; backward applies 1/(nx*ny*nz). Both caller inputs are preserved.
 * Plans execute on owned FFTW-aligned buffers, so caller vectors may change
 * allocation/alignment between calls. The public header needs no FFTW or
 * HeFFTe headers; the compiled implementation requires the FFTW SDK.
 *
 * Construction/destruction must be serialized with other FFTW planners,
 * as for RealTransform. One instance is not reentrant. This class does not
 * initialize FFTW threads, change planner thread counts or call global
 * cleanup. The application owns those process-wide settings.
 */
class LocalFFT final : public IHostFFT {
public:
  explicit LocalFFT(const std::array<int, 3> &size);
  ~LocalFFT() override;
  LocalFFT(const LocalFFT &) = delete;
  LocalFFT &operator=(const LocalFFT &) = delete;
  LocalFFT(LocalFFT &&) = delete;
  LocalFFT &operator=(LocalFFT &&) = delete;

  void forward(const RealVector &in, ComplexVector &out) override;
  void backward(const ComplexVector &in, RealVector &out) override;
  void reset_fft_time() override;
  [[nodiscard]] double get_fft_time() const override;
  [[nodiscard]] std::size_t size_inbox() const override;
  [[nodiscard]] std::size_t size_outbox() const override;
  /// Owned scratch expressed in complex-sized slots; excludes FFTW plan internals.
  [[nodiscard]] std::size_t size_workspace() const override;
  /// Owned aligned-buffer bytes; excludes FFTW plan internals.
  [[nodiscard]] std::size_t get_allocated_memory_bytes() const override;
  [[nodiscard]] Box3i get_inbox_bounds() const override;
  [[nodiscard]] Box3i get_outbox_bounds() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft
