// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_fft_backend.hpp
 * @brief `from_json` specialization for `fft::Backend`
 */

#ifndef PFC_UI_FROM_JSON_FFT_BACKEND_HPP
#define PFC_UI_FROM_JSON_FFT_BACKEND_HPP

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/common/backend_from_string.hpp>

namespace pfc::ui {

/**
 * @brief Converts a JSON string to fft::Backend enum
 *
 * Parses backend selection from configuration. Supported values:
 * - "fftw" or "FFTW": CPU-based FFT (always available)
 * - "cuda" or "CUDA": GPU-based FFT using cuFFT (requires OpenPFC_ENABLE_CUDA)
 *
 * @param j The JSON object to parse (looks for "backend" field)
 * @return The fft::Backend enum value
 * @throws std::runtime_error if backend is not supported or not compiled in
 */
template <> inline fft::Backend from_json<fft::Backend>(const json &j) {
  if (!j.contains("backend") || !j["backend"].is_string()) {
    // Default to FFTW if not specified
    pfc::log_info(from_json_info_logger(),
                  "No FFT backend specified, defaulting to FFTW");
    return fft::Backend::FFTW;
  }

  std::string backend_str = j["backend"];
  std::transform(backend_str.begin(), backend_str.end(), backend_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  pfc::log_info(from_json_info_logger(),
                std::string("Selected FFT backend: ") + backend_str);

  std::optional<fft::Backend> backend = runtime::backend_from_string(backend_str);
  if (backend) {
    return *backend;
  }
  if (backend_str == "cuda") {
    throw std::runtime_error(
        "CUDA backend requested but OpenPFC was not compiled with CUDA support. "
        "Rebuild with -DOpenPFC_ENABLE_CUDA=ON");
  }
  throw std::runtime_error(
      "Unknown FFT backend: " + j["backend"].get<std::string>() +
      ". Supported: fftw, cuda");
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_FFT_BACKEND_HPP
