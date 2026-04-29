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

#include <mpi.h>

#include <nlohmann/json.hpp>

#include <openpfc/frontend/ui/from_json.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

namespace pfc::ui {

/**
 * @brief HeFFTe plan options for CPU FFTW from JSON or project defaults
 *
 * If `settings` contains `"plan_options"`, parses via `from_json`; otherwise
 * returns `heffte::default_options<heffte::backend::fftw>()`.
 */
[[nodiscard]] inline heffte::plan_options
cpu_spectral_plan_options_from_json(const nlohmann::json &settings) {
  if (settings.contains("plan_options")) {
    return ui::from_json<heffte::plan_options>(settings["plan_options"]);
  }
  return heffte::default_options<heffte::backend::fftw>();
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
