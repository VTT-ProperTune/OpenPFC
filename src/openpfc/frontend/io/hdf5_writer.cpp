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
  if (!m_domain_valid) {
    throw std::runtime_error("HDF5Writer::write: set_domain() was not called");
  }
  int nproc = 1;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(m_comm, &nproc), "MPI_Comm_size");
  if (nproc != 1) {
    throw std::invalid_argument(
        "HDF5Writer supports nproc=1 only (parallel HDF5 is not wired)");
  }
  const std::size_t expected =
      pfc::mpi::checked_local_extent_product(m_local, "HDF5Writer::write");
  if (data.size() != expected) {
    std::ostringstream oss;
    oss << "HDF5Writer::write: buffer size mismatch (expected " << expected
        << " elements from set_domain, got " << data.size() << ")";
    throw std::runtime_error(oss.str());
  }

  const std::string path = formatted_path(increment);
  HidGuard file{H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                H5Fclose};
  hdf5_require_id(file.id, "H5Fcreate");

  const hsize_t file_dims[3] = {static_cast<hsize_t>(m_global[2]),
                                static_cast<hsize_t>(m_global[1]),
                                static_cast<hsize_t>(m_global[0])};
  const hsize_t mem_dims[3] = {static_cast<hsize_t>(m_local[2]),
                               static_cast<hsize_t>(m_local[1]),
                               static_cast<hsize_t>(m_local[0])};
  const hsize_t start[3] = {static_cast<hsize_t>(m_offset[2]),
                            static_cast<hsize_t>(m_offset[1]),
                            static_cast<hsize_t>(m_offset[0])};

  HidGuard filespace{H5Screate_simple(3, file_dims, nullptr), H5Sclose};
  hdf5_require_id(filespace.id, "H5Screate_simple filespace");
  hdf5_require(H5Sselect_hyperslab(filespace.id, H5S_SELECT_SET, start, nullptr,
                                   mem_dims, nullptr),
               "H5Sselect_hyperslab");

  HidGuard memspace{H5Screate_simple(3, mem_dims, nullptr), H5Sclose};
  hdf5_require_id(memspace.id, "H5Screate_simple memspace");

  HidGuard dset{H5Dcreate2(file.id, "field", H5T_NATIVE_DOUBLE, filespace.id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Dclose};
  hdf5_require_id(dset.id, "H5Dcreate2");
  hdf5_require(H5Dwrite(dset.id, H5T_NATIVE_DOUBLE, memspace.id, filespace.id,
                        H5P_DEFAULT, data.data()),
               "H5Dwrite");

  write_xdmf_sidecar(path, m_global[0], m_global[1], m_global[2]);
  return MPI_Status{};
}

MPI_Status HDF5Writer::write(int, const ComplexField &) {
  throw std::invalid_argument("HDF5Writer does not support complex fields");
}

} // namespace pfc

#endif // OPENPFC_HAS_HDF5
