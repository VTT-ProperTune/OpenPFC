// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/session.hpp>
#include <openpfc/kernel/profiling/timer_report.hpp>

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

double combine_stat(const std::vector<double> &v, const std::string &stat_raw) {
  if (v.empty()) return 0.0;
  std::string s = stat_raw;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  double sum = 0.0;
  for (double x : v) sum += x;
  if (s == "sum") return sum;
  if (s == "mean") return sum / static_cast<double>(v.size());
  if (s == "min") return *std::min_element(v.begin(), v.end());
  if (s == "max") return *std::max_element(v.begin(), v.end());
  if (s == "median") {
    std::vector<double> w = v;
    std::sort(w.begin(), w.end());
    const std::size_t n = w.size();
    if (n % 2 == 1) return w[n / 2];
    return 0.5 * (w[n / 2 - 1] + w[n / 2]);
  }
  return sum / static_cast<double>(v.size());
}

void print_profiling_table(
    std::ostream &os, const ProfilingMetricCatalog &catalog,
    const std::vector<double> &total_inc, const std::vector<double> &total_exc,
    const std::vector<std::size_t> &ncalls, double denom,
    const ProfilingPrintOptions &opts, bool report_clock_valid,
    std::chrono::steady_clock::time_point report_clock_origin) {
  const std::size_t K = catalog.size();
  if (K == 0 || total_inc.size() != K || total_exc.size() != K ||
      ncalls.size() != K) {
    os << opts.title << " (no frames)\n";
    return;
  }

  if (denom <= 0.0) denom = 1.0;

  const auto &paths = catalog.paths();
  std::unordered_map<std::string, std::vector<std::string>> children;
  children.reserve(paths.size());
  for (const std::string &p : paths) {
    children[parent_prefix(p)].push_back(p);
  }

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
  if (report_clock_valid) {
    const double wall_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      report_clock_origin)
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
  const auto &fm_names = session.frame_metric_names();
  const std::size_t nmeta = fm_names.size();
  std::size_t wall_ix = static_cast<std::size_t>(-1);
  if (!opts.wall_denominator_metric.empty()) {
    for (std::size_t i = 0; i < fm_names.size(); ++i) {
      if (fm_names[i] == opts.wall_denominator_metric) {
        wall_ix = i;
        break;
      }
    }
  }

  for (std::size_t f = 0; f < n; ++f) {
    if (wall_ix != static_cast<std::size_t>(-1) && nmeta > 0)
      denom += session.frame_metric_values_[f * nmeta + wall_ix];
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

  print_profiling_table(os, session.catalog(), total_inc, total_exc, ncalls, denom,
                        opts, session.report_clock_valid_,
                        session.report_clock_origin_);
}

void print_profiling_timer(std::ostream &os, MPI_Comm comm,
                           const ProfilingSession &session,
                           const ProfilingPrintOptions &opts) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (!opts.mpi_aggregate_stdout || size == 1) {
    if (rank == 0) print_profiling_timer(os, session, opts);
    return;
  }

  std::vector<int> row_counts;
  std::vector<double> all_flat;
  std::vector<std::size_t> row_offset;
  session.mpi_gather_packed_frames(comm, row_counts, all_flat, row_offset);
  if (rank != 0) return;

  const int kpaths = static_cast<int>(session.catalog().size());
  const int nmeta = static_cast<int>(session.frame_metric_names().size());
  const int stride = static_cast<int>(session.frame_metric_names().size() +
                                      2u * session.catalog().size());
  const int mpi_size = static_cast<int>(row_counts.size());

  if (kpaths == 0 || stride <= 0) {
    os << opts.title << " (no frames)\n";
    return;
  }

  std::vector<double> metrics_buf, inc_buf, exc_buf;

  std::vector<std::vector<double>> sum_inc(static_cast<std::size_t>(mpi_size));
  std::vector<std::vector<double>> sum_exc(static_cast<std::size_t>(mpi_size));
  std::vector<std::vector<std::size_t>> cnt(static_cast<std::size_t>(mpi_size));
  for (int r = 0; r < mpi_size; ++r) {
    sum_inc[static_cast<std::size_t>(r)].assign(static_cast<std::size_t>(kpaths),
                                                0.0);
    sum_exc[static_cast<std::size_t>(r)].assign(static_cast<std::size_t>(kpaths),
                                                0.0);
    cnt[static_cast<std::size_t>(r)].assign(static_cast<std::size_t>(kpaths), 0u);
  }

  double denom = 0.0;
  const auto &fm_names = session.frame_metric_names();
  const auto nmeta_u = static_cast<std::size_t>(nmeta);
  auto wall_ix = static_cast<std::size_t>(-1);
  if (!opts.wall_denominator_metric.empty()) {
    for (std::size_t i = 0; i < fm_names.size(); ++i) {
      if (fm_names[i] == opts.wall_denominator_metric) {
        wall_ix = i;
        break;
      }
    }
  }

  for (int r = 0; r < mpi_size; ++r) {
    const int nf = row_counts[static_cast<std::size_t>(r)];
    for (int f = 0; f < nf; ++f) {
      const std::size_t global_row =
          row_offset[static_cast<std::size_t>(r)] + static_cast<std::size_t>(f);
      detail::unpack_gathered_profiling_row(global_row, stride, nmeta, kpaths,
                                            all_flat, metrics_buf, inc_buf, exc_buf);
      if (wall_ix != static_cast<std::size_t>(-1) && nmeta_u > 0)
        denom += metrics_buf[wall_ix];
      for (int ki = 0; ki < kpaths; ++ki) {
        const double inc = inc_buf[static_cast<std::size_t>(ki)];
        const double exc = exc_buf[static_cast<std::size_t>(ki)];
        sum_inc[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)] += inc;
        sum_exc[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)] += exc;
        if (inc > kEps)
          ++cnt[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)];
      }
    }
  }

  if (denom <= 0.0) denom = 1.0;

  std::vector<double> total_inc(static_cast<std::size_t>(kpaths), 0.0);
  std::vector<double> total_exc(static_cast<std::size_t>(kpaths), 0.0);
  std::vector<std::size_t> ncalls(static_cast<std::size_t>(kpaths), 0u);

  std::vector<double> col(static_cast<std::size_t>(mpi_size));
  for (int ki = 0; ki < kpaths; ++ki) {
    for (int r = 0; r < mpi_size; ++r)
      col[static_cast<std::size_t>(r)] =
          sum_inc[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)];
    total_inc[static_cast<std::size_t>(ki)] =
        combine_stat(col, opts.mpi_aggregate_stat);
    for (int r = 0; r < mpi_size; ++r)
      col[static_cast<std::size_t>(r)] =
          sum_exc[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)];
    total_exc[static_cast<std::size_t>(ki)] =
        combine_stat(col, opts.mpi_aggregate_stat);
    std::size_t ncall_sum = 0;
    for (int r = 0; r < mpi_size; ++r)
      ncall_sum += cnt[static_cast<std::size_t>(r)][static_cast<std::size_t>(ki)];
    ncalls[static_cast<std::size_t>(ki)] = ncall_sum;
  }

  print_profiling_table(os, session.catalog(), total_inc, total_exc, ncalls, denom,
                        opts, session.report_clock_valid_,
                        session.report_clock_origin_);
}

void print_profiling_timer(std::ostream &os, const ProfilingPrintOptions &opts) {
  ProfilingSession *s = current_session();
  if (s && s->num_frames() > 0) print_profiling_timer(os, *s, opts);
}

} // namespace pfc::profiling
