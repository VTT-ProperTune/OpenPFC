// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file session_profiling_hdf5.hpp
 * @brief HDF5 export for ProfilingSession (only when @c OPENPFC_HAS_HDF5)
 */

#ifndef PFC_KERNEL_PROFILING_DETAIL_SESSION_PROFILING_HDF5_HPP
#define PFC_KERNEL_PROFILING_DETAIL_SESSION_PROFILING_HDF5_HPP

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace pfc::profiling::detail {

void write_profiling_hdf5_file(
    const std::string &path, int size, const std::vector<int> &row_counts,
    const std::vector<std::size_t> &row_offset, int stride, int nmeta, int kpaths,
    const std::vector<std::string> &frame_metric_names,
    const std::vector<std::string> &path_names, const std::vector<double> &all_flat,
    const std::string &run_id, const nlohmann::json &export_metadata);

} // namespace pfc::profiling::detail

#endif // PFC_KERNEL_PROFILING_DETAIL_SESSION_PROFILING_HDF5_HPP
