// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_heffte.hpp
 * @brief HeFFTe `plan_options` JSON overlay and `from_json` specialization
 */

#ifndef PFC_UI_FROM_JSON_HEFFTE_HPP
#define PFC_UI_FROM_JSON_HEFFTE_HPP

#include <heffte.h>
#include <sstream>
#include <stdexcept>

#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/frontend/ui/from_json_log.hpp>

namespace pfc::ui {

namespace detail {

/**
 * @brief Overlay JSON keys onto an existing `heffte::plan_options` value
 *
 * Used by `from_json<heffte::plan_options>` (FFTW defaults) and by
 * `spectral_fft_stack_factory.hpp` (cuFFT / ROCm defaults) so GPU and CPU paths
 * share the same reshape / pencil / GPU-aware parsing.
 */
inline void apply_heffte_plan_options_json_overrides(const json &j,
                                                     heffte::plan_options &options) {
  pfc::log_debug(from_json_debug_logger(), "Parsing HeFFTe plan options (overlay)");
  if (j.contains("use_reorder")) {
    pfc::log_debug(from_json_debug_logger(), "Using strided 1d fft operations");
    options.use_reorder = j["use_reorder"];
  }
  if (j.contains("reshape_algorithm")) {
    if (j["reshape_algorithm"] == "alltoall") {
      pfc::log_debug(from_json_debug_logger(), "Using alltoall reshape algorithm");
      options.algorithm = heffte::reshape_algorithm::alltoall;
    } else if (j["reshape_algorithm"] == "alltoallv") {
      pfc::log_debug(from_json_debug_logger(), "Using alltoallv reshape algorithm");
      options.algorithm = heffte::reshape_algorithm::alltoallv;
    } else if (j["reshape_algorithm"] == "p2p") {
      pfc::log_debug(from_json_debug_logger(), "Using p2p reshape algorithm");
      options.algorithm = heffte::reshape_algorithm::p2p;
    } else if (j["reshape_algorithm"] == "p2p_plined") {
      pfc::log_debug(from_json_debug_logger(), "Using p2p_plined reshape algorithm");
      options.algorithm = heffte::reshape_algorithm::p2p_plined;
    } else {
      throw std::invalid_argument(
          "Unknown HeFFTe reshape_algorithm: " + j["reshape_algorithm"].dump() +
          ". Supported: alltoall, alltoallv, p2p, "
          "p2p_plined");
    }
  }
  if (j.contains("use_pencils")) {
    pfc::log_debug(from_json_debug_logger(), "Using pencil decomposition");
    options.use_pencils = j["use_pencils"];
  }
  if (j.contains("use_gpu_aware")) {
    pfc::log_debug(from_json_debug_logger(), "Using gpu aware fft");
    options.use_gpu_aware = j["use_gpu_aware"];
  }
  std::ostringstream options_ss;
  options_ss << "Backend options: " << options;
  pfc::log_debug(from_json_debug_logger(), options_ss.str());
}

} // namespace detail

/**
 * @brief Converts a JSON object to heffte::plan_options.
 *
 * This function parses the provided JSON object and constructs a
 * heffte::plan_options object based on the values found in the JSON. The
 * function prints debug information to the console regarding the options being
 * parsed.
 *
 * @param j The JSON object to parse.
 * @return The heffte::plan_options object constructed from the JSON.
 */
template <>
[[nodiscard]] inline heffte::plan_options
from_json<heffte::plan_options>(const json &j) {
  heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
  detail::apply_heffte_plan_options_json_overrides(j, options);
  return options;
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_HEFFTE_HPP
