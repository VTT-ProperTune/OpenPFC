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

#include <openpfc/kernel/decomposition/brick_split.hpp>

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
  if (j.contains("num_subranks")) {
    const int nsub = j["num_subranks"].get<int>();
    pfc::log_debug(from_json_debug_logger(),
                   "Using HeFFTe num_subranks=" + std::to_string(nsub));
    options.use_num_subranks(nsub);
  }
  std::ostringstream options_ss;
  options_ss << "Backend options: " << options;
  pfc::log_debug(from_json_debug_logger(), options_ss.str());
}

} // namespace detail

/**
 * @brief One LUMI-G node has 8 GCDs (one MPI rank per GCD).
 *
 * Off-node, `spectral_fft_proc_grid` uses a 1×8×nnodes layout so consecutive
 * ranks stay on one node. HeFFTe pencils match that 2D grid: the y-pencil
 * reshape is intra-node, the z-pencil reshape is pairwise between nodes.
 * Rank counts that are not a multiple of 8 still drop to 1D slabs.
 */
inline constexpr int kHeffteAlltoallMinRanks = 9;
inline constexpr int kHeffteNodeGcds = 8;

/**
 * @brief Match HeFFTe pencils / slabs to `spectral_fft_proc_grid`.
 *
 * Multiples of one LUMI-G node (16, 24, 32, …) keep pencils. Other off-node
 * counts use slabs. Does not change `reshape_algorithm`.
 */
inline void apply_heffte_comm_scale(heffte::plan_options &options, int nproc) {
  if (nproc < kHeffteAlltoallMinRanks) {
    return;
  }
  if (pfc::decomposition::fft_node_grid_override() && nproc % kHeffteNodeGcds == 0) {
    if (!options.use_pencils) {
      pfc::log_debug(from_json_debug_logger(),
                     "HeFFTe use_pencils enabled for node-aware 1x8xN grid");
      options.use_pencils = true;
    }
    return;
  }
  if (options.use_pencils) {
    pfc::log_debug(from_json_debug_logger(),
                   "HeFFTe use_pencils disabled (slabs) for nproc exceeding "
                   "one LUMI-G node");
    options.use_pencils = false;
  }
}

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
