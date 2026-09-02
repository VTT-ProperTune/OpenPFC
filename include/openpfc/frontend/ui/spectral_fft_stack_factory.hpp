// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_fft_stack_factory.hpp
 * @brief JSON helpers for HeFFTe plan options on CPU vs GPU spectral driver paths
 *
 * @details
 * `merged_spectral_plan_options_json` centralizes merging root `backend` into the
 * `plan_options` object (same rules as `cpu_spectral_plan_options_from_json`).
 * GPU entry points start from cuFFT / ROCm HeFFTe defaults and overlay the same
 * reshape / pencil / GPU-aware keys as the CPU `from_json<heffte::plan_options>`
 * path (`detail::apply_heffte_plan_options_json_overrides` in
 * `from_json_heffte.hpp`). `make_simulation_session<GPUSpectralStack<…>>` and
 * GPU ETD sessions pass those options into `create_cuda` / `create_hip`.
 *
 * @see from_json_heffte.hpp for `detail::apply_heffte_plan_options_json_overrides`
 */

#ifndef PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP
#define PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP

#include <cctype>
#include <heffte.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <openpfc/frontend/ui/from_json_heffte.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#endif

namespace pfc::ui {

/**
 * @brief Merge root `plan_options` with optional root-level `backend` string
 *
 * If `plan_options.backend` is absent but `settings.backend` is a string, the
 * root value is copied in (same convention as `from_json<fft::Backend>` on the
 * whole document).
 */
[[nodiscard]] inline nlohmann::json
merged_spectral_plan_options_json(const nlohmann::json &settings) {
  nlohmann::json plan_opts = nlohmann::json::object();
  if (settings.contains("plan_options") && !settings["plan_options"].is_null() &&
      settings["plan_options"].is_object()) {
    plan_opts = settings["plan_options"];
  }
  if (!plan_opts.contains("backend") && settings.contains("backend") &&
      settings["backend"].is_string()) {
    plan_opts["backend"] = settings["backend"];
  }
  return plan_opts;
}

namespace detail {

inline std::string lowercase_ascii(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

/** Reject GPU backend on the JSON → `SpectralCPUStack` / `CPUFFT` path. */
inline void reject_cuda_backend_for_cpu_spectral_stack(const nlohmann::json &plan) {
  if (!plan.contains("backend") || !plan["backend"].is_string()) {
    return;
  }
  const std::string b = lowercase_ascii(plan["backend"].get<std::string>());
  if (b == "cuda") {
    throw std::invalid_argument(
        "SpectralCPUStack builds fft::CPUFFT (FFTW). plan_options.backend "
        "\"cuda\" is not supported on this path. Use \"fftw\", omit backend, or "
        "use a GPU-specific application driver.");
  }
}

} // namespace detail

/**
 * @brief HeFFTe plan options for CPU FFTW from JSON or project defaults
 *
 * If `settings` contains `"plan_options"`, that object is parsed via
 * `from_json<heffte::plan_options>`. If it does **not** specify `"backend"` but
 * the root `settings` has `"backend"` (same convention as
 * `from_json<fft::Backend>`), the root value is copied into the plan slice so one
 * JSON file can drive both helpers consistently.
 *
 * `backend: \"cuda\"` is rejected here: this path always constructs CPU HeFFTe
 * (`fft::create` → `CPUFFT`).
 */
[[nodiscard]] inline heffte::plan_options
cpu_spectral_plan_options_from_json(const nlohmann::json &settings) {
  const nlohmann::json plan_opts = merged_spectral_plan_options_json(settings);
  if (plan_opts.empty()) {
    return heffte::default_options<heffte::backend::fftw>();
  }
  detail::reject_cuda_backend_for_cpu_spectral_stack(plan_opts);
  return ui::from_json<heffte::plan_options>(plan_opts);
}

/**
 * @brief Construct `fft::CPUFFT` for a decomposition using JSON plan options
 *
 * Centralizes `fft::layout::create` + `fft::create` for the CPU spectral path.
 */
[[nodiscard]] inline fft::CPUFFT
cpu_fft_from_json_and_decomposition(const nlohmann::json &settings,
                                    const decomposition::Decomposition &decomp,
                                    int rank_id, MPI_Comm comm) {
  return fft::create(fft::layout::create(decomp, 0), rank_id,
                     cpu_spectral_plan_options_from_json(settings), comm);
}

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)

/**
 * @brief HeFFTe plan options for a cuFFT-backed spectral driver from app JSON
 *
 * Starts from `heffte::default_options<heffte::backend::cufft>()` and overlays
 * keys from `merged_spectral_plan_options_json(settings)`. Used by
 * `make_simulation_session<GPUSpectralStack<CUDASpace>>` and GPU ETD sessions.
 */
[[nodiscard]] inline heffte::plan_options
cuda_spectral_plan_options_from_json(const nlohmann::json &settings) {
  const nlohmann::json merged = merged_spectral_plan_options_json(settings);
  if (merged.empty()) {
    return heffte::default_options<heffte::backend::cufft>();
  }
  heffte::plan_options options = heffte::default_options<heffte::backend::cufft>();
  detail::apply_heffte_plan_options_json_overrides(merged, options);
  return options;
}

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)

/**
 * @brief HeFFTe plan options for a ROCm-backed spectral driver from app JSON
 *
 * Same overlay pattern as @ref cuda_spectral_plan_options_from_json but with
 * `heffte::backend::rocfft` defaults.
 */
[[nodiscard]] inline heffte::plan_options
hip_spectral_plan_options_from_json(const nlohmann::json &settings) {
  const nlohmann::json merged = merged_spectral_plan_options_json(settings);
  if (merged.empty()) {
    return heffte::default_options<heffte::backend::rocfft>();
  }
  heffte::plan_options options = heffte::default_options<heffte::backend::rocfft>();
  detail::apply_heffte_plan_options_json_overrides(merged, options);
  return options;
}

#endif // OpenPFC_ENABLE_HIP_SPECTRAL

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

/**
 * @brief HeFFTe plan options for `GPUSpectralStack<MemorySpace>` from app JSON.
 *
 * Specializations dispatch to `cuda_spectral_plan_options_from_json` /
 * `hip_spectral_plan_options_from_json`.
 */
template <class MemorySpace>
[[nodiscard]] heffte::plan_options
gpu_spectral_plan_options_from_json(const nlohmann::json &settings);

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
template <>
[[nodiscard]] inline heffte::plan_options
gpu_spectral_plan_options_from_json<pfc::CUDASpace>(const nlohmann::json &settings) {
  return cuda_spectral_plan_options_from_json(settings);
}
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <>
[[nodiscard]] inline heffte::plan_options
gpu_spectral_plan_options_from_json<pfc::HIPSpace>(const nlohmann::json &settings) {
  return hip_spectral_plan_options_from_json(settings);
}
#endif

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL || OpenPFC_ENABLE_HIP_SPECTRAL

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP
