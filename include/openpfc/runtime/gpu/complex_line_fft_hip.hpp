// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/** @file complex_line_fft_hip.hpp
 * @brief One-rank rocFFT form of the unnormalized complex line DFT.
 *
 * Same contract as `pfc::fft::ComplexLineFFT`: one complex line,
 * `backward(forward(x))` equals `length * x`. This is a complex-to-complex
 * rocFFT plan, not the real-to-complex pencil `IDeviceFFT`.
 */

#include <complex>
#include <memory>
#include <span>

#if defined(OpenPFC_ENABLE_HIP)

namespace pfc::fft {

/** @brief Rank-local unnormalized complex DFT of one line on one HIP device.
 *
 * Forward and backward use separate complex rocFFT plans. rocFFT does not
 * normalize the inverse: `backward(forward(x))` equals `length * x`. Both
 * caller spans are preserved. The plans execute on owned device buffers.
 *
 * One instance is not reentrant. Copying and moving are disabled. A
 * distributed line is a different step.
 */
class HipComplexLineFFT {
public:
  explicit HipComplexLineFFT(int length);
  ~HipComplexLineFFT();
  HipComplexLineFFT(const HipComplexLineFFT &) = delete;
  HipComplexLineFFT &operator=(const HipComplexLineFFT &) = delete;
  HipComplexLineFFT(HipComplexLineFFT &&) = delete;
  HipComplexLineFFT &operator=(HipComplexLineFFT &&) = delete;

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
