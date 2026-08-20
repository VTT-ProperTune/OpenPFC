// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file session_selection.hpp
 * @brief Method × backend selection for `SimulationSession` (M10).
 *
 * @details
 * JSON `method` is `spectral` | `fd`. JSON `backend` is `cpu` | `cuda` |
 * `hip` (`fftw` aliases cpu; `rocm` aliases hip so existing FFT JSON still
 * selects a CPU or GPU stack). Optional `fd_order` is even in `[2, 20]`;
 * halo width is `fd_order / 2`.
 */

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pfc::sim {

enum class SimulationMethod { Spectral, Fd };
enum class SimulationBackend { Cpu, Cuda, Hip };

struct SessionSelection {
  SimulationMethod method{SimulationMethod::Spectral};
  SimulationBackend backend{SimulationBackend::Cpu};
  int fd_order{2};
};

[[nodiscard]] inline bool even_fd_order(int order) noexcept {
  return order >= 2 && order <= 20 && (order % 2) == 0;
}

[[nodiscard]] inline int halo_width_from_fd_order(int fd_order) noexcept {
  return fd_order / 2;
}

[[nodiscard]] inline std::optional<SimulationMethod>
simulation_method_from_string(std::string_view s) noexcept {
  if (s == "spectral") {
    return SimulationMethod::Spectral;
  }
  if (s == "fd") {
    return SimulationMethod::Fd;
  }
  return std::nullopt;
}

[[nodiscard]] inline std::optional<SimulationBackend>
simulation_backend_from_string(std::string_view s) noexcept {
  if (s == "cpu" || s == "fftw") {
    return SimulationBackend::Cpu;
  }
  if (s == "cuda") {
    return SimulationBackend::Cuda;
  }
  if (s == "hip" || s == "rocm") {
    return SimulationBackend::Hip;
  }
  return std::nullopt;
}

[[nodiscard]] inline const char *to_cstring(SimulationMethod method) noexcept {
  switch (method) {
  case SimulationMethod::Fd: return "fd";
  case SimulationMethod::Spectral: return "spectral";
  }
  return "spectral";
}

[[nodiscard]] inline const char *to_cstring(SimulationBackend backend) noexcept {
  switch (backend) {
  case SimulationBackend::Cuda: return "cuda";
  case SimulationBackend::Hip: return "hip";
  case SimulationBackend::Cpu: return "cpu";
  }
  return "cpu";
}

[[nodiscard]] inline bool
session_backend_compiled(SimulationBackend backend,
                         SimulationMethod method) noexcept {
  switch (backend) {
  case SimulationBackend::Cpu: return true;
  case SimulationBackend::Cuda:
    if (method == SimulationMethod::Spectral) {
#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
      return true;
#else
      return false;
#endif
    }
#if defined(OpenPFC_ENABLE_CUDA)
    return true;
#else
    return false;
#endif
  case SimulationBackend::Hip:
    if (method == SimulationMethod::Spectral) {
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
      return true;
#else
      return false;
#endif
    }
#if defined(OpenPFC_ENABLE_HIP)
    return true;
#else
    return false;
#endif
  }
  return false;
}

[[nodiscard]] inline bool
session_backend_compiled(const SessionSelection &s) noexcept {
  return session_backend_compiled(s.backend, s.method);
}

/// Intended stack type name for logs and session-matrix asserts.
[[nodiscard]] inline const char *
intended_stack_name(const SessionSelection &s) noexcept {
  if (s.method == SimulationMethod::Spectral) {
    switch (s.backend) {
    case SimulationBackend::Cuda: return "GPUSpectralStack<CUDASpace>";
    case SimulationBackend::Hip: return "GPUSpectralStack<HIPSpace>";
    case SimulationBackend::Cpu: return "SpectralCPUStack";
    }
  }
  switch (s.backend) {
  case SimulationBackend::Cuda: return "FDGPUStack<CUDASpace>";
  case SimulationBackend::Hip: return "FDGPUStack<HIPSpace>";
  case SimulationBackend::Cpu: return "FDCPUStack";
  }
  return "SpectralCPUStack";
}

/// Fail closed when @p s does not match the stack this factory is about to
/// construct, or when the backend is not compiled into this build.
inline void require_session_for_stack(const SessionSelection &s,
                                      SimulationMethod method,
                                      SimulationBackend backend) {
  if (!session_backend_compiled(s)) {
    throw std::invalid_argument(std::string("SessionSelection backend '") +
                                to_cstring(s.backend) +
                                "' is not compiled into this OpenPFC build");
  }
  if (s.method != method || s.backend != backend) {
    throw std::invalid_argument(std::string("SessionSelection maps to ") +
                                intended_stack_name(s) +
                                ", not the requested stack");
  }
  if (method == SimulationMethod::Fd && !even_fd_order(s.fd_order)) {
    throw std::invalid_argument("SessionSelection fd_order must be even in [2, 20]");
  }
}

/// Per-stack factory used by `SimulationSession<Stack>`. Specializations live
/// with the CPU/GPU stack factories.
template <class Stack> struct stack_builder;

} // namespace pfc::sim
