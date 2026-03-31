// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/timer_report.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace pfc::profiling {

namespace {

constexpr double kEps = 1e-15;

std::string parent_prefix(const std::string &path) {
  const auto pos = path.rfind('/');
  if (pos == std::string::npos) return std::string{};
  return path.substr(0, pos);
}

std::string last_segment(const std::string &path) {
  const auto pos = path.rfind('/');
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

std::string format_seconds(double s) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  if (s >= 1.0 || s == 0.0) {
    oss << s << " s";
    return oss.str();
  }
  if (s >= 1e-3) {
    oss << (s * 1e3) << " ms";
    return oss.str();
  }
  if (s >= 1e-6) {
    oss << (s * 1e6) << " us";
    return oss.str();
  }
  oss << (s * 1e9) << " ns";
  return oss.str();
}

} // namespace

void print_profiling_timer(std::ostream &os, const ProfilingSession &session,
                           const ProfilingPrintOptions &opts) {
  const std::size_t n = session.num_frames();
  const std::size_t K = session.catalog().size();
  if (n == 0 || K == 0) {
    os << opts.title << " (no frames)\n";
    return;
  }

  std::vector<double> total_inc(K, 0.0);
  std::vector<double> total_exc(K, 0.0);
  std::vector<std::size_t> ncalls(K, 0);
  double denom = 0.0;

  for (std::size_t f = 0; f < n; ++f) {
    denom += session.wall_step_[f];
    for (std::size_t i = 0; i < K; ++i) {
      const std::size_t idx = f * K + i;
      const double inc = session.timer_inclusive_[idx];
      const double exc = session.timer_exclusive_[idx];
      total_inc[i] += inc;
      total_exc[i] += exc;
      if (inc > kEps) ++ncalls[i];
    }
  }

  if (denom <= 0.0) denom = 1.0;

  const auto &paths = session.catalog().paths();
  std::unordered_map<std::string, std::vector<std::string>> children;
  children.reserve(paths.size());
  for (const std::string &p : paths) {
    children[parent_prefix(p)].push_back(p);
  }

  // Index path -> catalog index
  std::unordered_map<std::string, std::size_t> path_index;
  for (std::size_t i = 0; i < K; ++i) path_index[paths[i]] = i;

  for (auto &kv : children) {
    auto &vec = kv.second;
    if (opts.sort_by_time) {
      std::sort(vec.begin(), vec.end(),
                [&](const std::string &a, const std::string &b) {
                  return total_inc[path_index.at(a)] > total_inc[path_index.at(b)];
                });
    } else {
      std::sort(vec.begin(), vec.end());
    }
  }

  const bool ascii = opts.ascii_lines;

  auto line = [&](std::size_t w) {
    if (ascii) {
      os << std::string(w, '-');
    } else {
      for (std::size_t i = 0; i < w; ++i) os << "\u2500";
    }
    os << '\n';
  };

  const int sec_w = 42;
  const int n_w = 8;
  const int t_w = 14;
  const int p_w = 8;
  const int a_w = 14;
  const int ex_w = 14;
  const int extra = opts.show_exclusive_column ? ex_w + 1 : 0;
  const auto table_w =
      2u + static_cast<std::size_t>(sec_w) + static_cast<std::size_t>(n_w) +
      static_cast<std::size_t>(t_w) + static_cast<std::size_t>(p_w) +
      static_cast<std::size_t>(a_w) + static_cast<std::size_t>(extra);

  os << '\n' << opts.title << '\n';
  if (session.report_clock_valid_) {
    const double wall_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      session.report_clock_origin_)
            .count();
    os << "  Wall clock since reset_report_clock(): " << format_seconds(wall_elapsed)
       << '\n';
  }
  line(table_w);
  os << std::left << std::setw(sec_w) << "Section" << std::right << std::setw(n_w)
     << "ncalls" << std::setw(t_w) << "time" << std::setw(p_w) << "%tot"
     << std::setw(a_w) << "avg";
  if (opts.show_exclusive_column) os << std::setw(ex_w) << "exclusive";
  os << '\n';
  line(table_w);

  std::function<void(const std::string &, int)> walk;
  walk = [&](const std::string &parent, int depth) {
    auto it = children.find(parent);
    if (it == children.end()) return;
    for (const std::string &p : it->second) {
      const std::size_t pi = path_index.at(p);
      const std::size_t nc = ncalls[pi] == 0 ? 1 : ncalls[pi];
      const double ttot = total_inc[pi];
      const double pct = 100.0 * ttot / denom;
      const double av = ttot / static_cast<double>(nc);
      std::ostringstream pct_col;
      pct_col << std::fixed << std::setprecision(1) << pct << '%';
      std::ostringstream sec;
      sec << std::string(static_cast<std::size_t>(depth) * 2, ' ')
          << last_segment(p);
      std::string sec_str = sec.str();
      if (sec_str.size() > static_cast<std::size_t>(sec_w))
        sec_str = sec_str.substr(0, static_cast<std::size_t>(sec_w - 3)) + "...";
      os << std::left << std::setw(sec_w) << sec_str << std::right << std::setw(n_w)
         << nc << std::setw(t_w) << format_seconds(ttot) << std::setw(p_w)
         << pct_col.str() << std::setw(a_w) << format_seconds(av);
      if (opts.show_exclusive_column)
        os << std::setw(ex_w) << format_seconds(total_exc[pi]);
      os << '\n';
      walk(p, depth + 1);
    }
  };

  walk(std::string{}, 0);
  line(table_w);
  os << "  Tot wall (sum steps): " << format_seconds(denom) << '\n';
}

void print_profiling_timer(std::ostream &os, const ProfilingPrintOptions &opts) {
  ProfilingSession *s = current_session();
  if (s && s->num_frames() > 0) print_profiling_timer(os, *s, opts);
}

} // namespace pfc::profiling
