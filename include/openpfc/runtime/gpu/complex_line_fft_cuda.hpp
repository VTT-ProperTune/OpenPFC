// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file complex_line_fft_cuda.hpp
 * @brief One-rank cuFFT form of the unnormalized complex line DFT.
 *
 * Same contract as `pfc::fft::ComplexLineFFT`: one complex line,
 * `backward(forward(x))` equals `length * x`. This is `cufftPlan1d`
 * Z2Z, not the real-to-complex pencil `IDeviceFFT`.
 */

#include <complex>
#include <memory>
#include <span>

#if defined(OpenPFC_ENABLE_CUDA)

namespace pfc::fft {

/** @brief Rank-local unnormalized complex DFT of one line on one CUDA device.
 *
 * Forward and backward share one Z2Z plan. cuFFT scaling applies:
 * `backward(forward(x))` equals `length * x`. Both caller spans are
 * preserved. The plan executes on owned device buffers through the
 * legacy default stream.
 *
 * Planning must be serialized with other cuFFT planners. One instance
 * is not reentrant. Copying and moving are disabled. rocFFT and a
 * distributed line are a different step.
 */
class CudaComplexLineFFT {
public:
  explicit CudaComplexLineFFT(int length);
  ~CudaComplexLineFFT();
  CudaComplexLineFFT(const CudaComplexLineFFT &) = delete;
  CudaComplexLineFFT &operator=(const CudaComplexLineFFT &) = delete;
  CudaComplexLineFFT(CudaComplexLineFFT &&) = delete;
  CudaComplexLineFFT &operator=(CudaComplexLineFFT &&) = delete;

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

#endif
