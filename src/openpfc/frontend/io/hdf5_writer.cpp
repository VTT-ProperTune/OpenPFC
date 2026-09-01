// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/io/hdf5_writer.hpp>

#ifdef OPENPFC_HAS_HDF5

#include <hdf5.h>

#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/mpi/domain_geometry.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

namespace pfc {
namespace {

[[noreturn]] void hdf5_fail(const char *what) {
  throw std::runtime_error(std::string("HDF5Writer: ") + what);
}

void hdf5_require(herr_t status, const char *what) {
  if (status < 0) {
    hdf5_fail(what);
  }
}

void hdf5_require_id(hid_t id, const char *what) {
  if (id < 0) {
    hdf5_fail(what);
  }
}

struct HidGuard {
  hid_t id = -1;
  herr_t (*close)(hid_t) = nullptr;
  HidGuard(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
  ~HidGuard() {
    if (id >= 0 && close != nullptr) {
      (void)close(id);
    }
  }
  HidGuard(const HidGuard &) = delete;
  HidGuard &operator=(const HidGuard &) = delete;
};

std::string xdmf_path_for(const std::string &h5_path) {
  const auto dot = h5_path.find_last_of('.');
  if (dot == std::string::npos || dot == 0) {
    return h5_path + ".xdmf";
  }
  return h5_path.substr(0, dot) + ".xdmf";
}

void write_xdmf_sidecar(const std::string &h5_path, int nx, int ny, int nz) {
  const std::string xdmf = xdmf_path_for(h5_path);
  const std::string h5_name = std::filesystem::path(h5_path).filename().string();
  std::FILE *fp = std::fopen(xdmf.c_str(), "w");
  if (fp == nullptr) {
    hdf5_fail("could not write XDMF sidecar");
  }
  std::fprintf(fp,
               "<?xml version=\"1.0\" ?>\n"
               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
               "<Xdmf Version=\"3.0\">\n"
               "  <Domain>\n"
               "    <Grid Name=\"field\" GridType=\"Uniform\">\n"
               "      <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d "
               "%d\"/>\n"
               "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
               "        <DataItem Dimensions=\"3\" NumberType=\"Float\" "
               "Precision=\"8\" Format=\"XML\">0 0 0</DataItem>\n"
               "        <DataItem Dimensions=\"3\" NumberType=\"Float\" "
               "Precision=\"8\" Format=\"XML\">1 1 1</DataItem>\n"
               "      </Geometry>\n"
               "      <Attribute Name=\"field\" AttributeType=\"Scalar\" "
               "Center=\"Node\">\n"
               "        <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               "Precision=\"8\" Format=\"HDF\">%s:/field</DataItem>\n"
               "      </Attribute>\n"
               "    </Grid>\n"
               "  </Domain>\n"
               "</Xdmf>\n",
               nz, ny, nx, nz, ny, nx, h5_name.c_str());
  std::fclose(fp);
}

void write_hyperslab(hid_t file, const double *data, const hsize_t file_dims[3],
                     const hsize_t mem_dims[3], const hsize_t start[3], hid_t dxpl) {
  HidGuard filespace{H5Screate_simple(3, file_dims, nullptr), H5Sclose};
  hdf5_require_id(filespace.id, "H5Screate_simple filespace");
  hdf5_require(H5Sselect_hyperslab(filespace.id, H5S_SELECT_SET, start, nullptr,
                                   mem_dims, nullptr),
               "H5Sselect_hyperslab");

  HidGuard memspace{H5Screate_simple(3, mem_dims, nullptr), H5Sclose};
  hdf5_require_id(memspace.id, "H5Screate_simple memspace");

  HidGuard dset{H5Dcreate2(file, "field", H5T_NATIVE_DOUBLE, filespace.id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Dclose};
  hdf5_require_id(dset.id, "H5Dcreate2");
  hdf5_require(
      H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, memspace.id, filespace.id, dxpl, data),
      "H5Dwrite");
}

void scatter_brick_into_global(const double *local, const int lx, const int ly,
                               const int lz, const int ox, const int oy,
                               const int oz, const int nx, const int ny,
                               double *global) {
  for (int k = 0; k < lz; ++k) {
    for (int j = 0; j < ly; ++j) {
      for (int i = 0; i < lx; ++i) {
        const std::size_t li =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(lx) *
                static_cast<std::size_t>(ly);
        const std::size_t gi =
            static_cast<std::size_t>(ox + i) +
            static_cast<std::size_t>(oy + j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(oz + k) * static_cast<std::size_t>(nx) *
                static_cast<std::size_t>(ny);
        global[gi] = local[li];
      }
    }
  }
}

} // namespace

void HDF5Writer::set_domain(const std::array<int, 3> &arr_global,
                            const std::array<int, 3> &arr_local,
                            const std::array<int, 3> &arr_offset) {
  pfc::mpi::validate_subarray_domain(arr_global, arr_local, arr_offset,
                                     "HDF5Writer::set_domain");
  m_global = arr_global;
  m_local = arr_local;
  m_offset = arr_offset;
  m_domain_valid = true;
}

MPI_Status HDF5Writer::write(int increment, const RealField &data) {
  int local_ok = 1;
  std::string error_msg;
  std::size_t expected = 0;
  if (!m_domain_valid) {
    local_ok = 0;
    error_msg = "HDF5Writer::write: set_domain() was not called";
  } else {
    expected = pfc::mpi::checked_local_extent_product(m_local, "HDF5Writer::write");
    if (data.size() != expected) {
      std::ostringstream oss;
      oss << "HDF5Writer::write: buffer size mismatch (expected " << expected
          << " elements from set_domain, got " << data.size() << ")";
      error_msg = oss.str();
      local_ok = 0;
    }
  }

  int global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
      "MPI_Allreduce on HDF5Writer buffer size check");
  if (global_ok == 0) {
    if (!error_msg.empty()) {
      throw std::runtime_error(error_msg);
    }
    throw std::runtime_error(
        "HDF5Writer::write: collective buffer size mismatch on peer rank");
  }

  int rank = 0;
  int nproc = 1;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(m_comm, &rank), "MPI_Comm_rank");
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(m_comm, &nproc), "MPI_Comm_size");

  const std::string path = formatted_path(increment);
  const hsize_t file_dims[3] = {static_cast<hsize_t>(m_global[2]),
                                static_cast<hsize_t>(m_global[1]),
                                static_cast<hsize_t>(m_global[0])};
  const hsize_t mem_dims[3] = {static_cast<hsize_t>(m_local[2]),
                               static_cast<hsize_t>(m_local[1]),
                               static_cast<hsize_t>(m_local[0])};
  const hsize_t start[3] = {static_cast<hsize_t>(m_offset[2]),
                            static_cast<hsize_t>(m_offset[1]),
                            static_cast<hsize_t>(m_offset[0])};

#if defined(H5_HAVE_PARALLEL)
  {
    HidGuard fapl{H5Pcreate(H5P_FILE_ACCESS), H5Pclose};
    hdf5_require_id(fapl.id, "H5Pcreate FILE_ACCESS");
    hdf5_require(H5Pset_fapl_mpio(fapl.id, m_comm, MPI_INFO_NULL),
                 "H5Pset_fapl_mpio");
    HidGuard file{H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl.id),
                  H5Fclose};
    hdf5_require_id(file.id, "H5Fcreate (parallel)");
    HidGuard dxpl{H5Pcreate(H5P_DATASET_XFER), H5Pclose};
    hdf5_require_id(dxpl.id, "H5Pcreate DATASET_XFER");
    hdf5_require(H5Pset_dxpl_mpio(dxpl.id, H5FD_MPIO_COLLECTIVE),
                 "H5Pset_dxpl_mpio");
    write_hyperslab(file.id, data.data(), file_dims, mem_dims, start, dxpl.id);
  }
#else
  if (nproc == 1) {
    HidGuard file{H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                  H5Fclose};
    hdf5_require_id(file.id, "H5Fcreate");
    write_hyperslab(file.id, data.data(), file_dims, mem_dims, start, H5P_DEFAULT);
  } else {
    const int count = pfc::mpi::expect_mpi_io_count(expected, "HDF5Writer::write");
    std::vector<int> counts(static_cast<std::size_t>(nproc));
    pfc::mpi::throw_on_mpi_error(
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, m_comm),
        "MPI_Allgather HDF5Writer counts");
    std::vector<int> displs(static_cast<std::size_t>(nproc));
    int total = 0;
    for (int r = 0; r < nproc; ++r) {
      displs[static_cast<std::size_t>(r)] = total;
      total += counts[static_cast<std::size_t>(r)];
    }
    std::vector<double> gathered;
    if (rank == 0) {
      gathered.resize(static_cast<std::size_t>(total));
    }
    pfc::mpi::throw_on_mpi_error(MPI_Gatherv(data.data(), count, MPI_DOUBLE,
                                             rank == 0 ? gathered.data() : nullptr,
                                             counts.data(), displs.data(),
                                             MPI_DOUBLE, 0, m_comm),
                                 "MPI_Gatherv HDF5Writer");

    int meta[6] = {m_local[0],  m_local[1],  m_local[2],
                   m_offset[0], m_offset[1], m_offset[2]};
    std::vector<int> all_meta(static_cast<std::size_t>(nproc) * 6);
    pfc::mpi::throw_on_mpi_error(
        MPI_Allgather(meta, 6, MPI_INT, all_meta.data(), 6, MPI_INT, m_comm),
        "MPI_Allgather HDF5Writer bricks");

    if (rank == 0) {
      const std::size_t nglob = pfc::mpi::checked_local_extent_product(
          m_global, "HDF5Writer::write global");
      std::vector<double> global(nglob, 0.0);
      for (int r = 0; r < nproc; ++r) {
        const int *m = all_meta.data() + static_cast<std::size_t>(r) * 6;
        const int lx = m[0], ly = m[1], lz = m[2];
        const int ox = m[3], oy = m[4], oz = m[5];
        scatter_brick_into_global(
            gathered.data() +
                static_cast<std::size_t>(displs[static_cast<std::size_t>(r)]),
            lx, ly, lz, ox, oy, oz, m_global[0], m_global[1], global.data());
      }
      HidGuard file{H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                    H5Fclose};
      hdf5_require_id(file.id, "H5Fcreate (gather)");
      const hsize_t full_mem[3] = {file_dims[0], file_dims[1], file_dims[2]};
      const hsize_t origin[3] = {0, 0, 0};
      write_hyperslab(file.id, global.data(), file_dims, full_mem, origin,
                      H5P_DEFAULT);
    }
  }
#endif

  pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier HDF5Writer");
  if (rank == 0) {
    write_xdmf_sidecar(path, m_global[0], m_global[1], m_global[2]);
  }
  pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier HDF5Writer xdmf");
  return MPI_Status{};
}

MPI_Status HDF5Writer::write(int, const ComplexField &) {
  throw std::invalid_argument("HDF5Writer does not support complex fields");
}

} // namespace pfc

#endif // OPENPFC_HAS_HDF5
