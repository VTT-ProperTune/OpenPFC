// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/profiling/detail/session_merge_json.hpp>
#include <openpfc/kernel/profiling/detail/session_profiling_hdf5.hpp>
#include <openpfc/kernel/profiling/session.hpp>

#include <hdf5.h>

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OPENPFC_PROFILING_BUILD_VERSION
#define OPENPFC_PROFILING_BUILD_VERSION "unknown"
#endif

namespace pfc::profiling {
namespace {

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
  const auto segs = detail::split_path_segments(path);
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

void write_h5_openpfc_version_string_attr(hid_t loc) {
  hid_t scalar = H5Screate(H5S_SCALAR);
  hid_t vtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(vtype, H5T_VARIABLE);
  hid_t aspace = H5Screate(H5S_SCALAR);
  hid_t attr =
      H5Acreate2(loc, "openpfc_version", vtype, aspace, H5P_DEFAULT, H5P_DEFAULT);
  const char *vptr = OPENPFC_PROFILING_BUILD_VERSION;
  H5Awrite(attr, vtype, &vptr);
  H5Aclose(attr);
  H5Sclose(aspace);
  H5Tclose(vtype);
  H5Sclose(scalar);
}

std::string sanitize_hdf5_attribute_name(const std::string &key) {
  std::string out;
  out.reserve(key.size());
  for (unsigned char c : key) {
    if (std::isalnum(c) != 0 || c == '_' || c == '.') {
      out += static_cast<char>(c);
    } else {
      out += '_';
    }
  }
  return out.empty() ? std::string{"meta_key"} : out;
}

void write_hdf5_export_metadata_attrs(hid_t loc, const nlohmann::json &meta) {
  if (!meta.is_object()) {
    return;
  }
  hid_t vtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(vtype, H5T_VARIABLE);
  for (const auto &el : meta.items()) {
    std::string key = sanitize_hdf5_attribute_name(el.key());
    if (key == "schema_version" || key == "openpfc_version") {
      continue;
    }
    std::string val;
    if (el.value().is_string()) {
      val = el.value().get<std::string>();
    } else {
      val = el.value().dump();
    }
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, key.c_str(), vtype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (a >= 0) {
      const char *p = val.c_str();
      H5Awrite(a, vtype, &p);
      H5Aclose(a);
    }
    H5Sclose(aspace);
  }
  H5Tclose(vtype);
}

void write_profiling_hdf5_payload(hid_t content_root, int size,
                                  const std::vector<int> &row_counts,
                                  const std::vector<std::size_t> &row_offset,
                                  int stride, int nmeta, int kpaths,
                                  const std::vector<std::string> &frame_metric_names,
                                  const std::vector<std::string> &path_names,
                                  const std::vector<double> &all_flat) {
  hid_t scalar = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(content_root, "schema_version", H5T_NATIVE_INT, scalar,
                          H5P_DEFAULT, H5P_DEFAULT);
  int sv = 2;
  H5Awrite(attr, H5T_NATIVE_INT, &sv);
  H5Aclose(attr);
  H5Sclose(scalar);

  write_h5_openpfc_version_string_attr(content_root);

  hid_t vltype = H5Tcopy(H5T_C_S1);
  H5Tset_size(vltype, H5T_VARIABLE);

  if (nmeta > 0) {
    hsize_t nm = static_cast<hsize_t>(nmeta);
    hid_t space_nm = H5Screate_simple(1, &nm, nullptr);
    hid_t ds_names = H5Dcreate2(content_root, "frame_metric_names", vltype, space_nm,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (ds_names < 0) {
      H5Sclose(space_nm);
      H5Tclose(vltype);
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
    hid_t ds_rp = H5Dcreate2(content_root, "region_paths", vltype, space_kp,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (ds_rp < 0) {
      H5Sclose(space_kp);
      H5Tclose(vltype);
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
      H5Gcreate2(content_root, "ranks", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (ranks_root < 0) {
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
}

} // namespace

namespace detail {

void write_profiling_hdf5_file(
    const std::string &path, int size, const std::vector<int> &row_counts,
    const std::vector<std::size_t> &row_offset, int stride, int nmeta, int kpaths,
    const std::vector<std::string> &frame_metric_names,
    const std::vector<std::string> &path_names, const std::vector<double> &all_flat,
    const std::string &run_id, const nlohmann::json &export_metadata) {
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

  try {
    if (run_id.empty()) {
      write_profiling_hdf5_payload(prof, size, row_counts, row_offset, stride, nmeta,
                                   kpaths, frame_metric_names, path_names, all_flat);
    } else {
      write_h5_int_attr(prof, "schema_version", 3);
      write_h5_openpfc_version_string_attr(prof);
      const std::string sanitized = sanitize_profiling_run_id_for_hdf5(run_id);
      hid_t runs_g = H5Gcreate2(prof, "runs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (runs_g < 0) {
        throw std::runtime_error("ProfilingSession: H5Gcreate2 runs failed");
      }
      hid_t run_gr = H5Gcreate2(runs_g, sanitized.c_str(), H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT);
      if (run_gr < 0) {
        H5Gclose(runs_g);
        throw std::runtime_error("ProfilingSession: H5Gcreate2 run group failed");
      }
      write_hdf5_export_metadata_attrs(run_gr, export_metadata);
      write_profiling_hdf5_payload(run_gr, size, row_counts, row_offset, stride,
                                   nmeta, kpaths, frame_metric_names, path_names,
                                   all_flat);
      H5Gclose(run_gr);
      H5Gclose(runs_g);
    }
  } catch (const std::exception &) {
    (void)H5Gclose(prof);
    (void)H5Gclose(root);
    (void)H5Fclose(file);
    throw;
  }

  H5Gclose(prof);
  H5Gclose(root);
  H5Fclose(file);
}

} // namespace detail
} // namespace pfc::profiling
