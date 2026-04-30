// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file backend_from_string.hpp
 * @brief Parse FFT backend name to fft::Backend (runtime; keeps #ifdef out of
 * frontend)
 *
 * Frontend (e.g. from_json) should use this instead of branching on
 * OpenPFC_ENABLE_CUDA.
 */

#pragma once

#include <openpfc/kernel/fft/fft_interface.hpp>
#include <optional>
#include <string>

namespace pfc::runtime {

/**
 * @brief Map backend string (e.g. "fftw", "cuda") to fft::Backend
 * @param s Lowercase backend name
 * @return Backend if supported and compiled in, else std::nullopt
 */
inline std::optional<fft::Backend> backend_from_string(const std::string &s) {
  if (s == "fftw") {
    return fft::Backend::FFTW;
  }
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
  if (s == "cuda") {
    return fft::Backend::CUDA;
  }
#endif
  return std::nullopt;
}

} // namespace pfc::runtime
