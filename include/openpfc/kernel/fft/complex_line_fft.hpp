// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file complex_line_fft.hpp
 * @brief Unnormalized one-dimensional complex DFT.
 *
 * This is the line transform used by the Chebyshev even extension. It is
 * not `IHostFFT`: that contract, and the cuFFT and rocFFT pencil plans,
 * are complete-domain real-to-complex volume transforms. The matching
 * one-rank device plans are `CudaComplexLineFFT` and `HipComplexLineFFT`.
 */

#include <complex>
#include <memory>
#include <span>

namespace pfc::fft {

/** @brief Rank-local unnormalized complex DFT of one line.
 *
 * Forward and backward are the ordinary FFTW complex transforms of
 * `length` samples. `backward(forward(x))` equals `length * x`. Both
 * caller spans are preserved. Plans execute on owned FFTW-aligned
 * buffers. The public header needs no FFTW headers.
 *
 * Construction and destruction must be serialized with other FFTW
 * planners. One instance is not reentrant. This class does not
 * initialize FFTW threads or call global cleanup. The application owns
 * those process-wide settings. Planning uses `FFTW_ESTIMATE`. Copying
 * and moving are disabled.
 */
class ComplexLineFFT {
public:
  explicit ComplexLineFFT(int length);
  ~ComplexLineFFT();
  ComplexLineFFT(const ComplexLineFFT &) = delete;
  ComplexLineFFT &operator=(const ComplexLineFFT &) = delete;
  ComplexLineFFT(ComplexLineFFT &&) = delete;
  ComplexLineFFT &operator=(ComplexLineFFT &&) = delete;

  void forward(std::span<const std::complex<double>> in,
               std::span<std::complex<double>> out);
  void backward(std::span<const std::complex<double>> in,
                std::span<std::complex<double>> out);
  [[nodiscard]] int length() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace pfc::fft
