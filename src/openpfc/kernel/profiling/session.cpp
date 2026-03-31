// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/openpfc_frame_metrics.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef OPENPFC_HAS_HDF5
#include <hdf5.h>
#endif

#ifndef OPENPFC_PROFILING_BUILD_VERSION
#define OPENPFC_PROFILING_BUILD_VERSION "unknown"
#endif

namespace pfc::profiling {

namespace {

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

#ifdef OPENPFC_HAS_HDF5
hid_t open_or_create_group(hid_t parent, const char *name) {
  if (H5Lexists(parent, name, H5P_DEFAULT) > 0) {
    hid_t g = H5Gopen2(parent, name, H5P_DEFAULT);
    if (g >= 0) {
      return g;
    }
  }
  return H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

hid_t ensure_group_chain(hid_t prof_root, const std::string &path,
                         std::vector<hid_t> &opened) {
  opened.clear();
  const auto segs = split_path_segments(path);
  hid_t cur = prof_root;
  for (const auto &seg : segs) {
    hid_t next = open_or_create_group(cur, seg.c_str());
    if (next < 0) {
      for (auto it = opened.rbegin(); it != opened.rend(); ++it) {
        H5Gclose(*it);
      }
      opened.clear();
      return -1;
    }
    opened.push_back(next);
    cur = next;
  }
  return cur;
}

void write_h5_int_attr(hid_t loc, const char *name, int v) {
  hid_t s = H5Screate(H5S_SCALAR);
  hid_t a = H5Acreate2(loc, name, H5T_NATIVE_INT, s, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, H5T_NATIVE_INT, &v);
  H5Aclose(a);
  H5Sclose(s);
}

/// Schema 2: `openpfc/profiling/ranks/<mpi_rank>/…` with per-rank frame counts.
void write_profiling_hdf5_v2(const std::string &path, int size,
                             const std::vector<int> &row_counts,
                             const std::vector<std::size_t> &row_offset, int stride,
                             int nmeta, int kpaths,
                             const std::vector<std::string> &frame_metric_names,
                             const std::vector<std::string> &path_names,
                             const std::vector<double> &all_flat) {
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("ProfilingSession: H5Fcreate failed for " + path);
  }

  hid_t root = H5Gcreate2(file, "openpfc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (root < 0) {
    H5Fclose(file);
    throw std::runtime_error("ProfilingSession: H5Gcreate2 openpfc failed");
  }

  hid_t prof = H5Gcreate2(root, "profiling", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (prof < 0) {
    H5Gclose(root);
    H5Fclose(file);
    throw std::runtime_error("ProfilingSession: H5Gcreate2 profiling failed");
  }

  hid_t scalar = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(prof, "schema_version", H5T_NATIVE_INT, scalar,
                          H5P_DEFAULT, H5P_DEFAULT);
  int sv = 2;
  H5Awrite(attr, H5T_NATIVE_INT, &sv);
  H5Aclose(attr);

  hid_t vtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(vtype, H5T_VARIABLE);
  hid_t aspace = H5Screate(H5S_SCALAR);
  attr =
      H5Acreate2(prof, "openpfc_version", vtype, aspace, H5P_DEFAULT, H5P_DEFAULT);
  const char *vptr = OPENPFC_PROFILING_BUILD_VERSION;
  H5Awrite(attr, vtype, &vptr);
  H5Aclose(attr);
  H5Sclose(aspace);
  H5Tclose(vtype);
  H5Sclose(scalar);

  hid_t vltype = H5Tcopy(H5T_C_S1);
  H5Tset_size(vltype, H5T_VARIABLE);

  if (nmeta > 0) {
    hsize_t nm = static_cast<hsize_t>(nmeta);
    hid_t space_nm = H5Screate_simple(1, &nm, nullptr);
    hid_t ds_names = H5Dcreate2(prof, "frame_metric_names", vltype, space_nm,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (ds_names < 0) {
      H5Sclose(space_nm);
      H5Tclose(vltype);
      H5Gclose(prof);
      H5Gclose(root);
      H5Fclose(file);
      throw std::runtime_error(
          "ProfilingSession: H5Dcreate2 frame_metric_names failed");
    }
    std::vector<const char *> name_ptrs(static_cast<std::size_t>(nmeta));
    for (int i = 0; i < nmeta; ++i) {
      name_ptrs[static_cast<std::size_t>(i)] =
          frame_metric_names[static_cast<std::size_t>(i)].c_str();
    }
    H5Dwrite(ds_names, vltype, H5S_ALL, H5S_ALL, H5P_DEFAULT, name_ptrs.data());
    H5Dclose(ds_names);
    H5Sclose(space_nm);
  }

  if (kpaths > 0) {
    hsize_t kp = static_cast<hsize_t>(kpaths);
    hid_t space_kp = H5Screate_simple(1, &kp, nullptr);
    hid_t ds_rp = H5Dcreate2(prof, "region_paths", vltype, space_kp, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
    if (ds_rp < 0) {
      H5Sclose(space_kp);
      H5Tclose(vltype);
      H5Gclose(prof);
      H5Gclose(root);
      H5Fclose(file);
      throw std::runtime_error("ProfilingSession: H5Dcreate2 region_paths failed");
    }
    std::vector<const char *> path_ptrs(static_cast<std::size_t>(kpaths));
    for (int i = 0; i < kpaths; ++i) {
      path_ptrs[static_cast<std::size_t>(i)] =
          path_names[static_cast<std::size_t>(i)].c_str();
    }
    H5Dwrite(ds_rp, vltype, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_ptrs.data());
    H5Dclose(ds_rp);
    H5Sclose(space_kp);
  }

  H5Tclose(vltype);

  hid_t ranks_root =
      H5Gcreate2(prof, "ranks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (ranks_root < 0) {
    H5Gclose(prof);
    H5Gclose(root);
    H5Fclose(file);
    throw std::runtime_error("ProfilingSession: H5Gcreate2 ranks failed");
  }

  std::vector<double> metrics_buf;
  std::vector<double> inc_buf;
  std::vector<double> exc_buf;

  for (int mpi_r = 0; mpi_r < size; ++mpi_r) {
    const int nf = row_counts[static_cast<std::size_t>(mpi_r)];
    const std::string rname = std::to_string(mpi_r);
    hid_t gr =
        H5Gcreate2(ranks_root, rname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (gr < 0) {
      H5Gclose(ranks_root);
      H5Gclose(prof);
      H5Gclose(root);
      H5Fclose(file);
      throw std::runtime_error("ProfilingSession: H5Gcreate2 rank group failed");
    }
    write_h5_int_attr(gr, "n_frames", nf);

    if (nf > 0 && nmeta > 0) {
      std::vector<double> fs_r(static_cast<std::size_t>(nf) *
                               static_cast<std::size_t>(nmeta));
      for (int f = 0; f < nf; ++f) {
        const std::size_t global_row = row_offset[static_cast<std::size_t>(mpi_r)] +
                                       static_cast<std::size_t>(f);
        detail::unpack_gathered_profiling_row(global_row, stride, nmeta, kpaths,
                                              all_flat, metrics_buf, inc_buf,
                                              exc_buf);
        for (int m = 0; m < nmeta; ++m) {
          fs_r[static_cast<std::size_t>(f) * static_cast<std::size_t>(nmeta) +
               static_cast<std::size_t>(m)] =
              metrics_buf[static_cast<std::size_t>(m)];
        }
      }
      hsize_t dims_2d[2] = {static_cast<hsize_t>(nf), static_cast<hsize_t>(nmeta)};
      hid_t space_2d = H5Screate_simple(2, dims_2d, nullptr);
      hid_t ds_fs = H5Dcreate2(gr, "frame_scalars", H5T_NATIVE_DOUBLE, space_2d,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_fs, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, fs_r.data());
      H5Dclose(ds_fs);
      H5Sclose(space_2d);
    }

    if (nf > 0 && kpaths > 0) {
      std::vector<double> timer_inc_r(static_cast<std::size_t>(nf) *
                                      static_cast<std::size_t>(kpaths));
      std::vector<double> timer_exc_r(static_cast<std::size_t>(nf) *
                                      static_cast<std::size_t>(kpaths));
      for (int f = 0; f < nf; ++f) {
        const std::size_t global_row = row_offset[static_cast<std::size_t>(mpi_r)] +
                                       static_cast<std::size_t>(f);
        detail::unpack_gathered_profiling_row(global_row, stride, nmeta, kpaths,
                                              all_flat, metrics_buf, inc_buf,
                                              exc_buf);
        for (int i = 0; i < kpaths; ++i) {
          timer_inc_r[static_cast<std::size_t>(f) *
                          static_cast<std::size_t>(kpaths) +
                      static_cast<std::size_t>(i)] =
              inc_buf[static_cast<std::size_t>(i)];
          timer_exc_r[static_cast<std::size_t>(f) *
                          static_cast<std::size_t>(kpaths) +
                      static_cast<std::size_t>(i)] =
              exc_buf[static_cast<std::size_t>(i)];
        }
      }

      hsize_t dims_1[1] = {static_cast<hsize_t>(nf)};
      hid_t space = H5Screate_simple(1, dims_1, nullptr);

      for (int pi = 0; pi < kpaths; ++pi) {
        std::vector<hid_t> grp_chain;
        hid_t leaf = ensure_group_chain(gr, path_names[static_cast<std::size_t>(pi)],
                                        grp_chain);
        if (leaf < 0) {
          H5Sclose(space);
          H5Gclose(gr);
          H5Gclose(ranks_root);
          H5Gclose(prof);
          H5Gclose(root);
          H5Fclose(file);
          throw std::runtime_error("ProfilingSession: HDF5 group chain failed");
        }
        std::vector<double> col_inc(static_cast<std::size_t>(nf));
        std::vector<double> col_exc(static_cast<std::size_t>(nf));
        for (int f = 0; f < nf; ++f) {
          col_inc[static_cast<std::size_t>(f)] =
              timer_inc_r[static_cast<std::size_t>(f) *
                              static_cast<std::size_t>(kpaths) +
                          static_cast<std::size_t>(pi)];
          col_exc[static_cast<std::size_t>(f)] =
              timer_exc_r[static_cast<std::size_t>(f) *
                              static_cast<std::size_t>(kpaths) +
                          static_cast<std::size_t>(pi)];
        }
        hid_t ds_i = H5Dcreate2(leaf, "inclusive", H5T_NATIVE_DOUBLE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_i, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 col_inc.data());
        H5Dclose(ds_i);
        hid_t ds_e = H5Dcreate2(leaf, "exclusive", H5T_NATIVE_DOUBLE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_e, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 col_exc.data());
        H5Dclose(ds_e);
        for (auto it = grp_chain.rbegin(); it != grp_chain.rend(); ++it) {
          H5Gclose(*it);
        }
      }
      H5Sclose(space);
    }

    H5Gclose(gr);
  }

  H5Gclose(ranks_root);
  H5Gclose(prof);
  H5Gclose(root);
  H5Fclose(file);
}
#endif

} // namespace

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
          merge_region_json(regions, catalog_.paths()[static_cast<std::size_t>(i)],
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
  j["schema_version"] = 2;
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
    write_profiling_hdf5_v2(options.hdf5_path, size, row_counts, row_offset, stride,
                            nmeta, kpaths, frame_metric_names_, catalog_.paths(),
                            all_flat);
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
