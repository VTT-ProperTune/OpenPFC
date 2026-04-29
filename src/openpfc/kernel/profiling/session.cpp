// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/detail/session_merge_json.hpp>
#include <openpfc/kernel/profiling/openpfc_frame_metrics.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#ifdef OPENPFC_HAS_HDF5
#include <openpfc/kernel/profiling/detail/session_profiling_hdf5.hpp>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef OPENPFC_PROFILING_BUILD_VERSION
#define OPENPFC_PROFILING_BUILD_VERSION "unknown"
#endif

namespace pfc::profiling {

std::string sanitize_profiling_run_id_for_hdf5(std::string_view run_id) {
  std::string out;
  out.reserve(run_id.size());
  for (unsigned char c : run_id) {
    if (std::isalnum(c) != 0 || c == '_' || c == '-') {
      out += static_cast<char>(c);
    } else {
      out += '_';
    }
  }
  if (out.empty()) {
    return "run";
  }
  return out;
}

void record_time(std::string_view path, double seconds) noexcept {
  ProfilingSession *s = current_session();
  if (s != nullptr) {
    s->add_recorded_time(path, seconds);
  }
}

std::vector<std::string> ProfilingSession::openpfc_default_frame_metrics() {
  return openpfc_default_frame_metric_names();
}

ProfilingSession::ProfilingSession(ProfilingMetricCatalog catalog,
                                   std::vector<std::string> frame_metric_names)
    : catalog_(std::move(catalog)),
      frame_metric_names_(std::move(frame_metric_names)) {
  frame_scratch_inc_.assign(catalog_.size(), 0.0);
  frame_scratch_exc_.assign(catalog_.size(), 0.0);
  frame_metric_scratch_.assign(frame_metric_names_.size(), 0.0);
  for (std::size_t i = 0; i < frame_metric_names_.size(); ++i) {
    frame_metric_ix_[frame_metric_names_[i]] = i;
  }
}

std::size_t ProfilingSession::num_frames() const noexcept {
  const std::size_t K = catalog_.size();
  const std::size_t nmeta = frame_metric_names_.size();
  if (nmeta > 0) {
    return frame_metric_values_.size() / nmeta;
  }
  if (K > 0) {
    return timer_inclusive_.size() / K;
  }
  return 0;
}

void ProfilingSession::begin_frame() noexcept {
  scope_stack_.clear();
  std::fill(frame_scratch_inc_.begin(), frame_scratch_inc_.end(), 0.0);
  std::fill(frame_scratch_exc_.begin(), frame_scratch_exc_.end(), 0.0);
  std::fill(frame_metric_scratch_.begin(), frame_metric_scratch_.end(), 0.0);
  frame_open_ = true;
  frame_wall_t0_ = std::chrono::steady_clock::now();
}

void ProfilingSession::set_frame_metric(std::string_view name,
                                        double value) noexcept {
  if (!frame_open_) {
    return;
  }
  if (!std::isfinite(value)) {
    return;
  }
  auto it = frame_metric_ix_.find(std::string(name));
  if (it == frame_metric_ix_.end()) {
    return;
  }
  frame_metric_scratch_[it->second] = value;
}

void ProfilingSession::set_frame_metric_elapsed_since_begin(
    std::string_view name) noexcept {
  if (!frame_open_) {
    return;
  }
  auto it = frame_metric_ix_.find(std::string(name));
  if (it == frame_metric_ix_.end()) {
    return;
  }
  const auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - frame_wall_t0_).count();
  if (!std::isfinite(sec) || sec < 0.0) {
    sec = 0.0;
  }
  frame_metric_scratch_[it->second] = sec;
}

void ProfilingSession::end_frame() noexcept {
  if (!frame_open_) {
    return;
  }
  while (!scope_stack_.empty()) {
    pop_timed_scope();
  }

  const std::size_t nmeta = frame_metric_names_.size();
  for (std::size_t m = 0; m < nmeta; ++m) {
    frame_metric_values_.push_back(frame_metric_scratch_[m]);
  }

  const std::size_t K = catalog_.size();
  for (std::size_t i = 0; i < K; ++i) {
    timer_inclusive_.push_back(frame_scratch_inc_[i]);
    timer_exclusive_.push_back(frame_scratch_exc_[i]);
  }

  frame_open_ = false;
}

void ProfilingSession::migrate_to_catalog(
    ProfilingMetricCatalog &&new_cat) noexcept {
  const ProfilingMetricCatalog old_cat = catalog_;
  const auto &op = old_cat.paths();
  const auto &np = new_cat.paths();
  if (op.size() == np.size()) {
    bool same = true;
    for (std::size_t i = 0; i < op.size(); ++i) {
      if (op[i] != np[i]) {
        same = false;
        break;
      }
    }
    if (same) {
      return;
    }
  }

  std::unordered_map<std::string, std::size_t> old_ix;
  old_ix.reserve(op.size());
  for (std::size_t i = 0; i < op.size(); ++i) {
    old_ix[op[i]] = i;
  }

  const std::size_t n = num_frames();
  const std::size_t Ko = op.size();
  const std::size_t Kn = np.size();

  std::vector<double> ni;
  std::vector<double> ne;
  ni.assign(n * Kn, 0.0);
  ne.assign(n * Kn, 0.0);
  for (std::size_t f = 0; f < n; ++f) {
    for (std::size_t j = 0; j < Kn; ++j) {
      auto it = old_ix.find(np[j]);
      if (it != old_ix.end()) {
        const std::size_t oi = it->second;
        if (Ko > 0 && ((f * Ko) + oi) < timer_inclusive_.size()) {
          ni[(f * Kn) + j] = timer_inclusive_[(f * Ko) + oi];
          ne[(f * Kn) + j] = timer_exclusive_[(f * Ko) + oi];
        }
      }
    }
  }
  timer_inclusive_ = std::move(ni);
  timer_exclusive_ = std::move(ne);
  catalog_ = std::move(new_cat);

  std::vector<double> new_sinc(Kn, 0.0);
  std::vector<double> new_sexc(Kn, 0.0);
  for (std::size_t j = 0; j < Kn; ++j) {
    auto it = old_ix.find(catalog_.paths()[j]);
    if (it != old_ix.end()) {
      const std::size_t oi = it->second;
      if (oi < frame_scratch_inc_.size()) {
        new_sinc[j] = frame_scratch_inc_[oi];
        new_sexc[j] = frame_scratch_exc_[oi];
      }
    }
  }
  frame_scratch_inc_ = std::move(new_sinc);
  frame_scratch_exc_ = std::move(new_sexc);

  for (auto &sf : scope_stack_) {
    if (sf.path_index < op.size()) {
      const std::string &pname = op[sf.path_index];
      std::size_t out = 0;
      if (catalog_.try_index(pname, out)) {
        sf.path_index = out;
      }
    }
  }
}

void ProfilingSession::ensure_path(std::string_view path) noexcept {
  if (path.empty()) {
    return;
  }
  std::size_t idx = 0;
  if (catalog_.try_index(path, idx)) {
    return;
  }
  ProfilingMetricCatalog new_cat =
      ProfilingMetricCatalog::merge_one_path(catalog_, path);
  if (new_cat.size() == catalog_.size()) {
    return;
  }
  migrate_to_catalog(std::move(new_cat));
}

void ProfilingSession::add_recorded_time(std::string_view path,
                                         double seconds) noexcept {
  ensure_path(path);
  if (!frame_open_ || !std::isfinite(seconds)) {
    return;
  }
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) {
    return;
  }
  frame_scratch_inc_[idx] += seconds;
  frame_scratch_exc_[idx] += seconds;
}

void ProfilingSession::assign_recorded_time(std::string_view path,
                                            double seconds) noexcept {
  ensure_path(path);
  if (!frame_open_ || !std::isfinite(seconds)) {
    return;
  }
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) {
    return;
  }
  frame_scratch_inc_[idx] = seconds;
  frame_scratch_exc_[idx] = seconds;
}

void ProfilingSession::push_timed_scope(std::string_view path) noexcept {
  if (!frame_open_) {
    return;
  }
  ensure_path(path);
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) {
    return;
  }
  scope_stack_.push_back(ScopeFrame{idx, std::chrono::steady_clock::now(), 0.0});
}

void ProfilingSession::pop_timed_scope() noexcept {
  if (!frame_open_ || scope_stack_.empty()) {
    return;
  }
  ScopeFrame top = scope_stack_.back();
  scope_stack_.pop_back();
  const auto t1 = std::chrono::steady_clock::now();
  const double inc = std::chrono::duration<double>(t1 - top.t0).count();
  if (!std::isfinite(inc)) {
    return;
  }
  const double child_sum = top.children_inclusive_sum;
  const double exc = inc - child_sum;
  frame_scratch_inc_[top.path_index] += inc;
  frame_scratch_exc_[top.path_index] += exc;
  if (!scope_stack_.empty()) {
    scope_stack_.back().children_inclusive_sum += inc;
  }
}

void ProfilingSession::reset_report_clock() noexcept {
  report_clock_valid_ = true;
  report_clock_origin_ = std::chrono::steady_clock::now();
}

void ProfilingSession::pack_frames_flat(std::vector<double> &out) const {
  const std::size_t n = num_frames();
  const int stride = stride_doubles();
  if (stride <= 0) {
    out.clear();
    return;
  }
  const std::size_t K = catalog_.size();
  const std::size_t nmeta = frame_metric_names_.size();
  out.resize(n * static_cast<std::size_t>(stride));
  for (std::size_t f = 0; f < n; ++f) {
    const std::size_t b = f * static_cast<std::size_t>(stride);
    for (std::size_t m = 0; m < nmeta; ++m) {
      out[b + m] = frame_metric_values_[(f * nmeta) + m];
    }
    for (std::size_t i = 0; i < K; ++i) {
      const std::size_t ti = (f * K) + i;
      out[b + nmeta + (2 * i)] = timer_inclusive_[ti];
      out[b + nmeta + (2 * i) + 1] = timer_exclusive_[ti];
    }
  }
}

void ProfilingSession::mpi_gather_packed_frames(
    MPI_Comm comm, std::vector<int> &row_counts, std::vector<double> &all_flat,
    std::vector<std::size_t> &row_offset) const {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int n_local = static_cast<int>(num_frames());
  const int stride = stride_doubles();

  if (rank == 0) {
    row_counts.resize(static_cast<std::size_t>(size));
  } else {
    row_counts.clear();
  }
  MPI_Gather(&n_local, 1, MPI_INT, rank == 0 ? row_counts.data() : nullptr, 1,
             MPI_INT, 0, comm);

  std::vector<double> local_flat;
  pack_frames_flat(local_flat);
  const int send_doubles = n_local * stride;

  std::vector<int> recvcounts_d;
  std::vector<int> displs_d;
  int total_doubles = 0;
  if (rank == 0) {
    recvcounts_d.resize(static_cast<std::size_t>(size));
    displs_d.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
      recvcounts_d[static_cast<std::size_t>(r)] =
          row_counts[static_cast<std::size_t>(r)] * stride;
      displs_d[static_cast<std::size_t>(r)] = total_doubles;
      total_doubles += recvcounts_d[static_cast<std::size_t>(r)];
    }
  }

  if (rank == 0) {
    all_flat.resize(static_cast<std::size_t>(total_doubles));
  } else {
    all_flat.clear();
  }

  MPI_Gatherv(local_flat.data(), send_doubles, MPI_DOUBLE,
              rank == 0 ? all_flat.data() : nullptr,
              rank == 0 ? recvcounts_d.data() : nullptr,
              rank == 0 ? displs_d.data() : nullptr, MPI_DOUBLE, 0, comm);

  if (rank == 0) {
    row_offset.resize(static_cast<std::size_t>(size) + 1U);
    row_offset[0] = 0;
    for (int r = 0; r < size; ++r) {
      row_offset[static_cast<std::size_t>(r) + 1U] =
          row_offset[static_cast<std::size_t>(r)] +
          static_cast<std::size_t>(row_counts[static_cast<std::size_t>(r)]);
    }
  } else {
    row_offset.clear();
  }
}

void ProfilingSession::finalize_and_export(
    MPI_Comm comm, const ProfilingExportOptions &options) const {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::vector<int> row_counts;
  std::vector<double> all_flat;
  std::vector<std::size_t> row_offset;
  mpi_gather_packed_frames(comm, row_counts, all_flat, row_offset);

  if (rank != 0) {
    return;
  }

  const int kpaths = static_cast<int>(catalog_.size());
  const int nmeta = static_cast<int>(frame_metric_names_.size());
  const int stride = stride_doubles();
  const int size = static_cast<int>(row_counts.size());
  const std::size_t total_rows = row_offset.back();

  std::vector<double> inc_buf;
  std::vector<double> exc_buf;
  std::vector<double> metrics_buf;

  nlohmann::json ranks_json = nlohmann::json::array();
  for (int mpi_r = 0; mpi_r < size; ++mpi_r) {
    const int nf = row_counts[static_cast<std::size_t>(mpi_r)];
    nlohmann::json rank_obj;
    rank_obj["mpi_rank"] = mpi_r;
    rank_obj["n_frames"] = nf;
    nlohmann::json frames = nlohmann::json::array();
    if (stride > 0) {
      for (int local_f = 0; local_f < nf; ++local_f) {
        const std::size_t global_row = row_offset[static_cast<std::size_t>(mpi_r)] +
                                       static_cast<std::size_t>(local_f);
        detail::unpack_gathered_profiling_row(global_row, stride, nmeta, kpaths,
                                              all_flat, metrics_buf, inc_buf,
                                              exc_buf);
        nlohmann::json fr;
        nlohmann::json scalars = nlohmann::json::array();
        for (double v : metrics_buf) {
          scalars.push_back(v);
        }
        fr["scalars"] = std::move(scalars);
        nlohmann::json regions = nlohmann::json::object();
        for (int i = 0; i < kpaths; ++i) {
          detail::merge_region_json(regions,
                                    catalog_.paths()[static_cast<std::size_t>(i)],
                                    inc_buf[static_cast<std::size_t>(i)],
                                    exc_buf[static_cast<std::size_t>(i)]);
        }
        fr["regions"] = std::move(regions);
        frames.push_back(std::move(fr));
      }
    }
    rank_obj["frames"] = std::move(frames);
    ranks_json.push_back(std::move(rank_obj));
  }

  nlohmann::json j;
  if (options.run_id.empty()) {
    j["schema_version"] = 2;
  } else {
    j["schema_version"] = 3;
    j["run_id"] = options.run_id;
    j["metadata"] = options.export_metadata.is_object() ? options.export_metadata
                                                        : nlohmann::json::object();
  }
  j["openpfc_version"] = OPENPFC_PROFILING_BUILD_VERSION;
  j["n_mpi_ranks"] = size;
  j["total_frames"] = total_rows;
  j["frame_metric_names"] = nlohmann::json::array();
  for (const auto &nm : frame_metric_names_) {
    j["frame_metric_names"].push_back(nm);
  }
  j["region_paths"] = nlohmann::json::array();
  for (const auto &p : catalog_.paths()) {
    j["region_paths"].push_back(p);
  }
  j["ranks"] = std::move(ranks_json);

  if (options.write_json && !options.json_path.empty()) {
    std::ofstream ofs(options.json_path);
    if (!ofs) {
      throw std::runtime_error("ProfilingSession: cannot open JSON path " +
                               options.json_path);
    }
    ofs << j.dump(2);
  }

#ifdef OPENPFC_HAS_HDF5
  if (options.write_hdf5 && !options.hdf5_path.empty()) {
    detail::write_profiling_hdf5_file(
        options.hdf5_path, size, row_counts, row_offset, stride, nmeta, kpaths,
        frame_metric_names_, catalog_.paths(), all_flat, options.run_id,
        options.export_metadata);
  }
#else
  if (options.write_hdf5 && !options.hdf5_path.empty()) {
    throw std::runtime_error(
        "ProfilingSession: HDF5 export requested but OpenPFC was built "
        "without OpenPFC_ENABLE_HDF5=ON");
  }
#endif
}

} // namespace pfc::profiling
