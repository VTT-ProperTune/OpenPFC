// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file session_merge_json.hpp
 * @brief Nested JSON merge for profiling region paths (JSON export)
 */

#ifndef PFC_KERNEL_PROFILING_DETAIL_SESSION_MERGE_JSON_HPP
#define PFC_KERNEL_PROFILING_DETAIL_SESSION_MERGE_JSON_HPP

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace pfc::profiling::detail {

[[nodiscard]] std::vector<std::string> split_path_segments(const std::string &path);

void merge_region_json(nlohmann::json &root, const std::string &path, double inc,
                       double exc);

} // namespace pfc::profiling::detail

#endif // PFC_KERNEL_PROFILING_DETAIL_SESSION_MERGE_JSON_HPP
