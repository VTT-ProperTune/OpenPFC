// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
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

namespace pfc {
namespace profiling {

namespace {

std::vector<std::string> split_path_segments(const std::string &path) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= path.size()) {
    std::size_t end = path.find('/', start);
    if (end == std::string::npos) end = path.size();
    if (end > start) out.push_back(path.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

void merge_region_json(nlohmann::json &root, const std::string &path, double inc,
                       double exc) {
  const auto segs = split_path_segments(path);
  if (segs.empty()) return;
  nlohmann::json *cur = &root;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    nlohmann::json &slot = (*cur)[segs[i]];
    if (i + 1 == segs.size()) {
      if (!slot.is_object()) slot = nlohmann::json::object();
      slot["inclusive"] = inc;
      slot["exclusive"] = exc;
    } else {
      if (!slot.is_object()) slot = nlohmann::json::object();
      cur = &slot;
    }
  }
}

#ifdef OPENPFC_HAS_HDF5
hid_t open_or_create_group(hid_t parent, const char *name) {
  hid_t g = H5Gopen2(parent, name, H5P_DEFAULT);
  if (g >= 0) return g;
  return H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

/// Opens nested groups under @p prof_root; appends each new group id to @p opened
/// (for H5Gclose in reverse order after writing datasets).
hid_t ensure_group_chain(hid_t prof_root, const std::string &path,
                         std::vector<hid_t> &opened) {
  opened.clear();
  const auto segs = split_path_segments(path);
  hid_t cur = prof_root;
  for (const auto &seg : segs) {
    hid_t next = open_or_create_group(cur, seg.c_str());
    if (next < 0) {
      for (auto it = opened.rbegin(); it != opened.rend(); ++it) H5Gclose(*it);
      opened.clear();
      return -1;
    }
    opened.push_back(next);
    cur = next;
  }
  return cur;
}

void write_hdf5_v2(const std::string &path, hsize_t nrows, int npaths,
                   const std::vector<double> &step, const std::vector<double> &rank,
                   const std::vector<double> &wall_step,
                   const std::vector<double> &rss_bytes,
                   const std::vector<double> &model_heap,
                   const std::vector<double> &fft_heap,
                   const std::vector<std::string> &path_names,
                   const std::vector<double> &timer_inc,
                   const std::vector<double> &timer_exc) {
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0)
    throw std::runtime_error("ProfilingSession: H5Fcreate failed for " + path);

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

  hsize_t dims[1] = {nrows};
  hid_t space = H5Screate_simple(1, dims, nullptr);

  auto write_top_ds = [&](const char *name, const std::vector<double> &col) {
    hid_t ds = H5Dcreate2(prof, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
    if (ds < 0) throw std::runtime_error(std::string("H5Dcreate2 failed: ") + name);
    const double *ptr = col.empty() ? nullptr : col.data();
    herr_t st = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
    H5Dclose(ds);
    if (st < 0) throw std::runtime_error(std::string("H5Dwrite failed: ") + name);
  };

  try {
    write_top_ds("step", step);
    write_top_ds("mpi_rank", rank);
    write_top_ds("wall_step", wall_step);
    write_top_ds("rss_bytes", rss_bytes);
    write_top_ds("model_heap_bytes", model_heap);
    write_top_ds("fft_heap_bytes", fft_heap);

    for (int pi = 0; pi < npaths; ++pi) {
      std::vector<hid_t> grp_chain;
      hid_t leaf = ensure_group_chain(prof, path_names[static_cast<std::size_t>(pi)],
                                      grp_chain);
      if (leaf < 0)
        throw std::runtime_error("ProfilingSession: HDF5 group chain failed");
      std::vector<double> col_inc(nrows);
      std::vector<double> col_exc(nrows);
      for (hsize_t r = 0; r < nrows; ++r) {
        const std::size_t idx =
            static_cast<std::size_t>(r) * static_cast<std::size_t>(npaths) +
            static_cast<std::size_t>(pi);
        col_inc[r] = timer_inc[idx];
        col_exc[r] = timer_exc[idx];
      }
      hid_t ds_i = H5Dcreate2(leaf, "inclusive", H5T_NATIVE_DOUBLE, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (ds_i < 0) {
        for (auto it = grp_chain.rbegin(); it != grp_chain.rend(); ++it)
          H5Gclose(*it);
        throw std::runtime_error("H5Dcreate2 inclusive failed");
      }
      H5Dwrite(ds_i, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               col_inc.data());
      H5Dclose(ds_i);
      hid_t ds_e = H5Dcreate2(leaf, "exclusive", H5T_NATIVE_DOUBLE, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (ds_e < 0) {
        for (auto it = grp_chain.rbegin(); it != grp_chain.rend(); ++it)
          H5Gclose(*it);
        throw std::runtime_error("H5Dcreate2 exclusive failed");
      }
      H5Dwrite(ds_e, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               col_exc.data());
      H5Dclose(ds_e);
      for (auto it = grp_chain.rbegin(); it != grp_chain.rend(); ++it) H5Gclose(*it);
    }
  } catch (...) {
    H5Sclose(space);
    H5Gclose(prof);
    H5Gclose(root);
    H5Fclose(file);
    throw;
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
  H5Sclose(space);
  H5Gclose(prof);
  H5Gclose(root);
  H5Fclose(file);
}
#endif

std::string csv_escape_path(const std::string &path) {
  std::string s;
  s.reserve(path.size());
  for (char c : path) {
    if (c == '/')
      s += '_';
    else
      s += c;
  }
  return s;
}

void write_csv_v2(const std::string &path, std::size_t nrows, int npaths,
                  const std::vector<double> &step, const std::vector<double> &rank,
                  const std::vector<double> &wall_step,
                  const std::vector<double> &rss_bytes,
                  const std::vector<double> &model_heap,
                  const std::vector<double> &fft_heap,
                  const std::vector<std::string> &path_names,
                  const std::vector<double> &timer_inc,
                  const std::vector<double> &timer_exc) {
  std::ofstream ofs(path);
  if (!ofs)
    throw std::runtime_error("ProfilingSession: cannot open CSV path " + path);
  ofs << std::scientific << std::setprecision(17);
  ofs << "step,mpi_rank,wall_step,rss_bytes,model_heap_bytes,fft_heap_bytes";
  for (int pi = 0; pi < npaths; ++pi) {
    const std::string esc =
        csv_escape_path(path_names[static_cast<std::size_t>(pi)]);
    ofs << ',' << esc << "_inclusive," << esc << "_exclusive";
  }
  ofs << '\n';
  for (std::size_t r = 0; r < nrows; ++r) {
    ofs << step[r] << ',' << rank[r] << ',' << wall_step[r] << ',' << rss_bytes[r]
        << ',' << model_heap[r] << ',' << fft_heap[r];
    for (int pi = 0; pi < npaths; ++pi) {
      const std::size_t idx =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(npaths) +
          static_cast<std::size_t>(pi);
      ofs << ',' << timer_inc[idx] << ',' << timer_exc[idx];
    }
    ofs << '\n';
  }
}

void unpack_row_v2(std::size_t row, int stride, int kpaths,
                   const std::vector<double> &flat, double &step, double &rank,
                   double &wall, double &rss, double &model, double &fftmem,
                   std::vector<double> &inc, std::vector<double> &exc) {
  const std::size_t b =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
  step = flat[b + 0];
  rank = flat[b + 1];
  wall = flat[b + 2];
  rss = flat[b + 3];
  model = flat[b + 4];
  fftmem = flat[b + 5];
  inc.resize(static_cast<std::size_t>(kpaths));
  exc.resize(static_cast<std::size_t>(kpaths));
  for (int i = 0; i < kpaths; ++i) {
    inc[static_cast<std::size_t>(i)] = flat[b + 6 + 2 * i];
    exc[static_cast<std::size_t>(i)] = flat[b + 6 + 2 * i + 1];
  }
}

} // namespace

void record_time(std::string_view path, double seconds) noexcept {
  ProfilingSession *s = current_session();
  if (s) s->add_recorded_time(path, seconds);
}

ProfilingSession::ProfilingSession(ProfilingMetricCatalog catalog)
    : catalog_(std::move(catalog)) {
  frame_scratch_inc_.assign(catalog_.size(), 0.0);
  frame_scratch_exc_.assign(catalog_.size(), 0.0);
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
    if (same) return;
  }

  std::unordered_map<std::string, std::size_t> old_ix;
  old_ix.reserve(op.size());
  for (std::size_t i = 0; i < op.size(); ++i) old_ix[op[i]] = i;

  const std::size_t n = step_index_.size();
  const std::size_t Ko = op.size();
  const std::size_t Kn = np.size();

  std::vector<double> ni, ne;
  ni.assign(n * Kn, 0.0);
  ne.assign(n * Kn, 0.0);
  for (std::size_t f = 0; f < n; ++f) {
    for (std::size_t j = 0; j < Kn; ++j) {
      auto it = old_ix.find(np[j]);
      if (it != old_ix.end()) {
        const std::size_t oi = it->second;
        if (Ko > 0 && f * Ko + oi < timer_inclusive_.size()) {
          ni[f * Kn + j] = timer_inclusive_[f * Ko + oi];
          ne[f * Kn + j] = timer_exclusive_[f * Ko + oi];
        }
      }
    }
  }
  timer_inclusive_ = std::move(ni);
  timer_exclusive_ = std::move(ne);
  catalog_ = std::move(new_cat);

  std::vector<double> new_sinc(Kn, 0.0), new_sexc(Kn, 0.0);
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
      if (catalog_.try_index(pname, out)) sf.path_index = out;
    }
  }
}

void ProfilingSession::ensure_path(std::string_view path) noexcept {
  if (path.empty()) return;
  std::size_t idx = 0;
  if (catalog_.try_index(path, idx)) return;
  ProfilingMetricCatalog new_cat =
      ProfilingMetricCatalog::merge_one_path(catalog_, path);
  if (new_cat.size() == catalog_.size()) return;
  migrate_to_catalog(std::move(new_cat));
}

void ProfilingSession::begin_step_frame(int step_index, int mpi_rank) noexcept {
  scope_stack_.clear();
  std::fill(frame_scratch_inc_.begin(), frame_scratch_inc_.end(), 0.0);
  std::fill(frame_scratch_exc_.begin(), frame_scratch_exc_.end(), 0.0);
  pending_step_ = step_index;
  pending_rank_ = mpi_rank;
  wall_override_set_ = false;
  frame_wall_t0_ = std::chrono::steady_clock::now();
}

void ProfilingSession::add_recorded_time(std::string_view path,
                                         double seconds) noexcept {
  ensure_path(path);
  if (pending_step_ < 0 || !std::isfinite(seconds)) return;
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) return;
  frame_scratch_inc_[idx] += seconds;
  frame_scratch_exc_[idx] += seconds;
}

void ProfilingSession::assign_recorded_time(std::string_view path,
                                            double seconds) noexcept {
  ensure_path(path);
  if (pending_step_ < 0 || !std::isfinite(seconds)) return;
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) return;
  frame_scratch_inc_[idx] = seconds;
  frame_scratch_exc_[idx] = seconds;
}

void ProfilingSession::set_frame_wall_step(double seconds) noexcept {
  if (pending_step_ < 0 || !std::isfinite(seconds) || seconds < 0.0) return;
  wall_override_set_ = true;
  wall_override_value_ = seconds;
}

void ProfilingSession::push_timed_scope(std::string_view path) noexcept {
  if (pending_step_ < 0) return;
  ensure_path(path);
  std::size_t idx = 0;
  if (!catalog_.try_index(path, idx)) return;
  scope_stack_.push_back(ScopeFrame{idx, std::chrono::steady_clock::now(), 0.0});
}

void ProfilingSession::pop_timed_scope() noexcept {
  if (pending_step_ < 0 || scope_stack_.empty()) return;
  ScopeFrame top = scope_stack_.back();
  scope_stack_.pop_back();
  const auto t1 = std::chrono::steady_clock::now();
  const double inc = std::chrono::duration<double>(t1 - top.t0).count();
  if (!std::isfinite(inc)) return;
  const double child_sum = top.children_inclusive_sum;
  const double exc = inc - child_sum;
  frame_scratch_inc_[top.path_index] += inc;
  frame_scratch_exc_[top.path_index] += exc;
  if (!scope_stack_.empty()) scope_stack_.back().children_inclusive_sum += inc;
}

void ProfilingSession::commit_frame(double wall_step_seconds, bool inject_fft_region,
                                    double fft_seconds, std::uint64_t rss_bytes,
                                    std::uint64_t model_heap_bytes,
                                    std::uint64_t fft_heap_bytes) noexcept {
  if (pending_step_ < 0) return;
  while (!scope_stack_.empty()) pop_timed_scope();

  if (inject_fft_region) {
    std::size_t fft_i = 0;
    if (catalog_.try_index(kProfilingRegionFft, fft_i)) {
      frame_scratch_inc_[fft_i] = fft_seconds;
      frame_scratch_exc_[fft_i] = fft_seconds;
    }
  }

  step_index_.push_back(pending_step_);
  mpi_rank_.push_back(pending_rank_);
  wall_step_.push_back(wall_step_seconds);
  rss_bytes_.push_back(static_cast<double>(rss_bytes));
  model_heap_bytes_.push_back(static_cast<double>(model_heap_bytes));
  fft_heap_bytes_.push_back(static_cast<double>(fft_heap_bytes));

  const std::size_t K = catalog_.size();
  for (std::size_t i = 0; i < K; ++i) {
    timer_inclusive_.push_back(frame_scratch_inc_[i]);
    timer_exclusive_.push_back(frame_scratch_exc_[i]);
  }

  pending_step_ = -1;
}

void ProfilingSession::end_step_frame(std::uint64_t rss_bytes,
                                      std::uint64_t model_heap_bytes,
                                      std::uint64_t fft_heap_bytes) noexcept {
  if (pending_step_ < 0) return;
  double wall = 0.0;
  if (wall_override_set_) {
    wall = wall_override_value_;
    wall_override_set_ = false;
  } else {
    const auto t1 = std::chrono::steady_clock::now();
    wall = std::chrono::duration<double>(t1 - frame_wall_t0_).count();
    if (!std::isfinite(wall) || wall < 0.0) wall = 0.0;
  }
  commit_frame(wall, false, 0.0, rss_bytes, model_heap_bytes, fft_heap_bytes);
}

void ProfilingSession::end_step_frame(double wall_step_seconds, double fft_seconds,
                                      std::uint64_t rss_bytes,
                                      std::uint64_t model_heap_bytes,
                                      std::uint64_t fft_heap_bytes) noexcept {
  wall_override_set_ = false;
  commit_frame(wall_step_seconds, true, fft_seconds, rss_bytes, model_heap_bytes,
               fft_heap_bytes);
}

void ProfilingSession::reset_report_clock() noexcept {
  report_clock_valid_ = true;
  report_clock_origin_ = std::chrono::steady_clock::now();
}

void ProfilingSession::pack_frames_flat(std::vector<double> &out) const {
  const std::size_t n = step_index_.size();
  const int stride = stride_doubles();
  const std::size_t K = catalog_.size();
  out.resize(n * static_cast<std::size_t>(stride));
  for (std::size_t f = 0; f < n; ++f) {
    const std::size_t b = f * static_cast<std::size_t>(stride);
    out[b + 0] = static_cast<double>(step_index_[f]);
    out[b + 1] = static_cast<double>(mpi_rank_[f]);
    out[b + 2] = wall_step_[f];
    out[b + 3] = rss_bytes_[f];
    out[b + 4] = model_heap_bytes_[f];
    out[b + 5] = fft_heap_bytes_[f];
    for (std::size_t i = 0; i < K; ++i) {
      const std::size_t ti = f * K + i;
      out[b + 6 + 2 * i] = timer_inclusive_[ti];
      out[b + 6 + 2 * i + 1] = timer_exclusive_[ti];
    }
  }
}

void ProfilingSession::finalize_and_export(
    MPI_Comm comm, const ProfilingExportOptions &options) const {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int n_local = static_cast<int>(num_frames());
  const int stride = stride_doubles();

  std::vector<int> row_counts;
  if (rank == 0) row_counts.resize(static_cast<std::size_t>(size));
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

  std::vector<double> all_flat;
  if (rank == 0) all_flat.resize(static_cast<std::size_t>(total_doubles));

  MPI_Gatherv(local_flat.data(), send_doubles, MPI_DOUBLE,
              rank == 0 ? all_flat.data() : nullptr, recvcounts_d.data(),
              displs_d.data(), MPI_DOUBLE, 0, comm);

  if (rank != 0) return;

  const std::size_t total_rows = static_cast<std::size_t>(
      stride > 0 && total_doubles > 0 ? total_doubles / stride : 0);
  const int kpaths = static_cast<int>(catalog_.size());

  std::vector<double> col_step(total_rows), col_rank(total_rows),
      col_wall(total_rows), col_rss(total_rows), col_model(total_rows),
      col_fftmem(total_rows);
  std::vector<double> timer_inc(total_rows * static_cast<std::size_t>(kpaths));
  std::vector<double> timer_exc(total_rows * static_cast<std::size_t>(kpaths));

  std::vector<double> inc_buf, exc_buf;
  nlohmann::json frames = nlohmann::json::array();
  for (std::size_t r = 0; r < total_rows; ++r) {
    unpack_row_v2(r, stride, kpaths, all_flat, col_step[r], col_rank[r], col_wall[r],
                  col_rss[r], col_model[r], col_fftmem[r], inc_buf, exc_buf);
    for (int i = 0; i < kpaths; ++i) {
      const std::size_t idx =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(kpaths) +
          static_cast<std::size_t>(i);
      timer_inc[idx] = inc_buf[static_cast<std::size_t>(i)];
      timer_exc[idx] = exc_buf[static_cast<std::size_t>(i)];
    }
    nlohmann::json fr;
    fr["step"] = col_step[r];
    fr["mpi_rank"] = col_rank[r];
    fr["wall_step"] = col_wall[r];
    fr["rss_bytes"] = col_rss[r];
    fr["model_heap_bytes"] = col_model[r];
    fr["fft_heap_bytes"] = col_fftmem[r];
    nlohmann::json regions = nlohmann::json::object();
    for (int i = 0; i < kpaths; ++i) {
      merge_region_json(regions, catalog_.paths()[static_cast<std::size_t>(i)],
                        inc_buf[static_cast<std::size_t>(i)],
                        exc_buf[static_cast<std::size_t>(i)]);
    }
    fr["regions"] = std::move(regions);
    frames.push_back(std::move(fr));
  }

  nlohmann::json j;
  j["schema_version"] = 2;
  j["openpfc_version"] = OPENPFC_PROFILING_BUILD_VERSION;
  j["n_frames"] = total_rows;
  j["n_ranks"] = size;
  j["catalog"] = nlohmann::json::array();
  for (const auto &p : catalog_.paths()) j["catalog"].push_back(p);
  j["frames"] = std::move(frames);

  if (options.write_json && !options.json_path.empty()) {
    std::ofstream ofs(options.json_path);
    if (!ofs)
      throw std::runtime_error("ProfilingSession: cannot open JSON path " +
                               options.json_path);
    ofs << j.dump(2);
  }

  if (options.write_csv && !options.csv_path.empty()) {
    write_csv_v2(options.csv_path, total_rows, kpaths, col_step, col_rank, col_wall,
                 col_rss, col_model, col_fftmem, catalog_.paths(), timer_inc,
                 timer_exc);
  }

#ifdef OPENPFC_HAS_HDF5
  if (options.write_hdf5 && !options.hdf5_path.empty()) {
    write_hdf5_v2(options.hdf5_path, total_rows, kpaths, col_step, col_rank,
                  col_wall, col_rss, col_model, col_fftmem, catalog_.paths(),
                  timer_inc, timer_exc);
  }
#else
  if (options.write_hdf5 && !options.hdf5_path.empty()) {
    throw std::runtime_error(
        "ProfilingSession: HDF5 export requested but OpenPFC was built "
        "without OpenPFC_ENABLE_HDF5=ON");
  }
#endif
}

} // namespace profiling
} // namespace pfc
