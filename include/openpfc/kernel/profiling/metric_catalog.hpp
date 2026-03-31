// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file metric_catalog.hpp
 * @brief Ordered region paths for MPI-dense profiling (defaults + config extras)
 */

#ifndef PFC_KERNEL_PROFILING_METRIC_CATALOG_HPP
#define PFC_KERNEL_PROFILING_METRIC_CATALOG_HPP

#include <openpfc/kernel/profiling/names.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pfc::profiling {

/**
 * @brief Immutable ordered list of `/`-separated region paths.
 *
 * All MPI ranks must use an identical catalog (same config). Paths include
 * implicit parent prefixes (e.g. `a/b` adds `a` and `a/b`).
 */
class ProfilingMetricCatalog {
public:
  /// Default regions plus @p extra_paths (each may contain `/`; parents inserted).
  static ProfilingMetricCatalog
  with_defaults_and_extras(const std::vector<std::string> &extra_paths) {
    std::set<std::string> ordered;
    ordered.insert(std::string{kProfilingRegionCommunication});
    ordered.insert(std::string{kProfilingRegionFft});
    ordered.insert(std::string{kProfilingRegionGradient});
    for (const std::string &p : extra_paths) {
      insert_path_with_parents(ordered, p);
    }
    return ProfilingMetricCatalog{std::move(ordered)};
  }

  /// User-defined paths only (no built-in fft/communication/gradient). Each entry
  /// may contain `/`; parent prefixes are inserted automatically.
  static ProfilingMetricCatalog
  from_paths_only(const std::vector<std::string> &paths) {
    std::set<std::string> ordered;
    for (const std::string &p : paths) {
      insert_path_with_parents(ordered, p);
    }
    return ProfilingMetricCatalog{std::move(ordered)};
  }

  /// @p base plus @p path and any parent prefixes (same rules as
  /// insert_path_with_parents).
  static ProfilingMetricCatalog merge_one_path(const ProfilingMetricCatalog &base,
                                               std::string_view path) {
    std::set<std::string> ordered(base.paths().begin(), base.paths().end());
    insert_path_with_parents(ordered, std::string(path));
    return ProfilingMetricCatalog{std::move(ordered)};
  }

  ProfilingMetricCatalog() = default;

  const std::vector<std::string> &paths() const noexcept { return paths_; }
  std::size_t size() const noexcept { return paths_.size(); }

  /// @return false if @p path is not in the catalog.
  bool try_index(std::string_view path, std::size_t &out_index) const noexcept {
    std::string key(path);
    auto it = index_.find(key);
    if (it == index_.end()) {
      return false;
    }
    out_index = it->second;
    return true;
  }

private:
  explicit ProfilingMetricCatalog(std::set<std::string> &&ordered) {
    paths_.assign(ordered.begin(), ordered.end());
    for (std::size_t i = 0; i < paths_.size(); ++i) {
      index_[paths_[i]] = i;
    }
  }

  static void insert_path_with_parents(std::set<std::string> &out,
                                       const std::string &path) {
    std::string p = normalize_path(path);
    if (p.empty()) {
      return;
    }
    std::string accum;
    std::size_t start = 0;
    while (start <= p.size()) {
      std::size_t end = p.find('/', start);
      if (end == std::string::npos) {
        end = p.size();
      }
      if (end > start) {
        std::string seg = p.substr(start, end - start);
        if (accum.empty()) {
          accum = std::move(seg);
        } else {
          accum += "/" + seg;
        }
        out.insert(accum);
      }
      start = end + 1;
    }
  }

  static std::string normalize_path(const std::string &path) {
    std::string p = path;
    while (!p.empty() && p.front() == '/') {
      p.erase(p.begin());
    }
    while (!p.empty() && p.back() == '/') {
      p.pop_back();
    }
    return p;
  }

  std::vector<std::string> paths_;
  std::unordered_map<std::string, std::size_t> index_;
};

} // namespace pfc::profiling

#endif // PFC_KERNEL_PROFILING_METRIC_CATALOG_HPP
