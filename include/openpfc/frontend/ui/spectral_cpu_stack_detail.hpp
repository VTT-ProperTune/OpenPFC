// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_cpu_stack_detail.hpp
 * @brief Shared JSON-driven helpers for building the CPU HeFFTe spectral stack
 *
 * @details
 * `SpectralCpuStack` uses these functions so plan options and FFT construction
 * live in one place. Future GPU-backed stack builders can mirror the same JSON
 * surface (`plan_options`, `world`, `time`) while swapping the FFT factory
 * (see `docs/refactoring_roadmap.md`, Phase C).
 */

#ifndef PFC_UI_SPECTRAL_CPU_STACK_DETAIL_HPP
#define PFC_UI_SPECTRAL_CPU_STACK_DETAIL_HPP

#include <cctype>
#include <mpi.h>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <stdexcept>
#include <string>

namespace pfc::ui {
namespace detail {

inline std::string lowercase_ascii(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

/** Reject GPU backend on the JSON → `SpectralCpuStack` / `CpuFft` path. */
inline void reject_cuda_backend_for_cpu_spectral_stack(const nlohmann::json &plan) {
  if (!plan.contains("backend") || !plan["backend"].is_string()) {
    return;
  }
  const std::string b = lowercase_ascii(plan["backend"].get<std::string>());
  if (b == "cuda") {
    throw std::invalid_argument(
        "SpectralCpuStack builds fft::CpuFft (FFTW). plan_options.backend "
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
 * (`fft::create` → `CpuFft`).
 */
[[nodiscard]] inline heffte::plan_options
cpu_spectral_plan_options_from_json(const nlohmann::json &settings) {
  nlohmann::json plan_opts = nlohmann::json::object();
  if (settings.contains("plan_options") && !settings["plan_options"].is_null() &&
      settings["plan_options"].is_object()) {
    plan_opts = settings["plan_options"];
  }
  if (!plan_opts.contains("backend") && settings.contains("backend") &&
      settings["backend"].is_string()) {
    plan_opts["backend"] = settings["backend"];
  }
  if (plan_opts.empty()) {
    return heffte::default_options<heffte::backend::fftw>();
  }
  detail::reject_cuda_backend_for_cpu_spectral_stack(plan_opts);
  return ui::from_json<heffte::plan_options>(plan_opts);
}

/**
 * @brief Construct `fft::CpuFft` for a decomposition using JSON plan options
 *
 * Centralizes `fft::layout::create` + `fft::create` for the CPU spectral path.
 */
[[nodiscard]] inline fft::CpuFft
cpu_fft_from_json_and_decomposition(const nlohmann::json &settings,
                                    const decomposition::Decomposition &decomp,
                                    int rank_id, MPI_Comm comm) {
  return fft::create(fft::layout::create(decomp, 0), rank_id,
                     cpu_spectral_plan_options_from_json(settings), comm);
}

} // namespace pfc::ui

#endif // PFC_UI_SPECTRAL_CPU_STACK_DETAIL_HPP
