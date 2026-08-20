// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file from_json_session_selection.hpp
 * @brief JSON `method` / `backend` / `fd_order` → `SessionSelection`.
 */

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/simulation/session_selection.hpp>

namespace pfc::ui {

namespace detail {

inline std::string ascii_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline std::vector<std::string> compiled_session_backend_tokens() {
  std::vector<std::string> v{"cpu", "fftw"};
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
  v.emplace_back("cuda");
#endif
#if defined(OpenPFC_ENABLE_HIP) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
  v.emplace_back("hip");
  v.emplace_back("rocm");
#endif
  return v;
}

} // namespace detail

template <>
[[nodiscard]] inline pfc::sim::SessionSelection
from_json<pfc::sim::SessionSelection>(const json &j) {
  pfc::sim::SessionSelection s{};

  if (j.contains("method")) {
    if (!j["method"].is_string()) {
      throw std::invalid_argument(format_config_error(
          "method", "discretization method", "string (spectral or fd)",
          get_json_value_string(j, "method"), {"spectral", "fd"},
          "\"method\": \"spectral\""));
    }
    const std::string raw = j["method"].get<std::string>();
    const auto method =
        pfc::sim::simulation_method_from_string(detail::ascii_lower(raw));
    if (!method) {
      throw std::invalid_argument(format_config_error(
          "method", "discretization method", "string (spectral or fd)", raw,
          {"spectral", "fd"}, "\"method\": \"spectral\""));
    }
    s.method = *method;
  }

  if (j.contains("backend")) {
    if (!j["backend"].is_string()) {
      throw std::invalid_argument(format_config_error(
          "backend", "execution backend", "string (cpu, cuda, or hip)",
          get_json_value_string(j, "backend"),
          detail::compiled_session_backend_tokens(), "\"backend\": \"cpu\""));
    }
    const std::string raw = j["backend"].get<std::string>();
    const auto backend =
        pfc::sim::simulation_backend_from_string(detail::ascii_lower(raw));
    if (!backend) {
      throw std::invalid_argument(format_config_error(
          "backend", "execution backend", "string (cpu, cuda, or hip)", raw,
          detail::compiled_session_backend_tokens(), "\"backend\": \"cpu\""));
    }
    s.backend = *backend;
    if (!pfc::sim::session_backend_compiled(s.backend, s.method)) {
      throw std::invalid_argument(format_config_error(
          "backend", "execution backend compiled into this OpenPFC build",
          "one of the valid options for this build", raw,
          detail::compiled_session_backend_tokens(), "\"backend\": \"cpu\""));
    }
  }

  if (j.contains("fd_order")) {
    if (!j["fd_order"].is_number_integer()) {
      throw std::invalid_argument(format_config_error(
          "fd_order", "even finite-difference spatial order", "even integer 2..20",
          get_json_value_string(j, "fd_order"), {}, "\"fd_order\": 2"));
    }
    s.fd_order = j["fd_order"].get<int>();
    if (!pfc::sim::even_fd_order(s.fd_order)) {
      throw std::invalid_argument(format_config_error(
          "fd_order", "even finite-difference spatial order", "even integer 2..20",
          std::to_string(s.fd_order),
          {"2", "4", "6", "8", "10", "12", "14", "16", "18", "20"},
          "\"fd_order\": 2"));
    }
  }

  return s;
}

} // namespace pfc::ui
