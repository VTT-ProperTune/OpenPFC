// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/detail/session_merge_json.hpp>

namespace pfc::profiling::detail {

std::vector<std::string> split_path_segments(const std::string &path) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= path.size()) {
    std::size_t end = path.find('/', start);
    if (end == std::string::npos) {
      end = path.size();
    }
    if (end > start) {
      out.push_back(path.substr(start, end - start));
    }
    start = end + 1;
  }
  return out;
}

void merge_region_json(nlohmann::json &root, const std::string &path, double inc,
                       double exc) {
  const auto segs = split_path_segments(path);
  if (segs.empty()) {
    return;
  }
  nlohmann::json *cur = &root;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    nlohmann::json &slot = (*cur)[segs[i]];
    if (i + 1 == segs.size()) {
      if (!slot.is_object()) {
        slot = nlohmann::json::object();
      }
      slot["inclusive"] = inc;
      slot["exclusive"] = exc;
    } else {
      if (!slot.is_object()) {
        slot = nlohmann::json::object();
      }
      cur = &slot;
    }
  }
}

} // namespace pfc::profiling::detail
