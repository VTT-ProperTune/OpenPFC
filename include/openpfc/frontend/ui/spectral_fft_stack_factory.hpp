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
 * `from_json_heffte.hpp`).
 *
 * **Avoid a second dummy CPU FFT:** GPU models still take `pfc::FFT&` from the
 * base `Model` constructor. Reuse the single `fft::CpuFft` owned by
 * `SpectralCpuStack` / `SpectralSimulationSession::fft()` for that reference, and
 * build cuFFT / ROCm HeFFTe (`cuda_spectral_plan_options_from_json`, etc.) only
 * for the device path—do not construct another throwaway `CpuFft` in app code
 * solely to satisfy the `Model` wiring.
 *
 * @see spectral_cpu_stack_detail.hpp for the CPU `SpectralCpuStack` factory
 * @see from_json_heffte.hpp for `detail::apply_heffte_plan_options_json_overrides`
 */

#ifndef PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP
#define PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP

#include <heffte.h>
#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json_heffte.hpp>

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

#if defined(OpenPFC_ENABLE_CUDA)

/**
 * @brief HeFFTe plan options for a cuFFT-backed spectral driver from app JSON
 *
 * Starts from `heffte::default_options<heffte::backend::cufft>()` and overlays
 * keys from `merged_spectral_plan_options_json(settings)`. Use when constructing
 * `fft::create(...)` for GPU models that still take a host `pfc::FFT` reference
 * for the base `Model` (e.g. Tungsten CUDA integration tests).
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

#endif // OpenPFC_ENABLE_CUDA

#if defined(OpenPFC_ENABLE_HIP)

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

#endif // OpenPFC_ENABLE_HIP

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_FFT_STACK_FACTORY_HPP
