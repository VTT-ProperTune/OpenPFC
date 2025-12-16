// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/results_writers/vtk_writer.hpp"
#include "openpfc/utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace pfc {

void VTKWriter::set_domain(const std::array<int, 3> &arr_global,
                           const std::array<int, 3> &arr_local,
                           const std::array<int, 3> &arr_offset) {
  m_global_size = arr_global;
  m_local_size = arr_local;
  m_offset = arr_offset;
}

std::string VTKWriter::generate_filename(int increment, int rank) const {
  // Format filename with increment number
  std::string base = utils::format_with_number(m_filename, increment);

  if (m_num_ranks > 1 && rank >= 0) {
    // For parallel: remove .vti extension, add rank, restore extension
    size_t ext_pos = base.find_last_of('.');
    if (ext_pos != std::string::npos) {
      std::string name = base.substr(0, ext_pos);
      std::string ext = base.substr(ext_pos);
      return name + "_" + std::to_string(rank) + ext;
    }
  }
  return base;
}

void VTKWriter::write_vti_header(std::ofstream &file,
                                 [[maybe_unused]] int increment) const {
  file << R"(<?xml version="1.0" encoding="utf-8"?>)" << std::endl;
  file
      << R"(<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">)"
      << std::endl;

  // Whole extent (global domain)
  file << R"(  <ImageData WholeExtent="0 )" << m_global_size[0] - 1 << " 0 "
       << m_global_size[1] - 1 << " 0 " << m_global_size[2] - 1 << R"(" Origin=")"
       << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << R"(" Spacing=")"
       << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << R"(">)"
       << std::endl;

  // Piece extent (local domain)
  file << R"(    <Piece Extent=")" << m_offset[0] << " "
       << m_offset[0] + m_local_size[0] - 1 << " " << m_offset[1] << " "
       << m_offset[1] + m_local_size[1] - 1 << " " << m_offset[2] << " "
       << m_offset[2] + m_local_size[2] - 1 << R"(">)" << std::endl;

  file << R"(      <PointData>)" << std::endl;
  file << R"(        <DataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1" format="appended" offset="0"/>)" << std::endl;
  file << R"(      </PointData>)" << std::endl;
  file << R"(    </Piece>)" << std::endl;
  file << R"(  </ImageData>)" << std::endl;
  file << R"(  <AppendedData encoding="raw">)" << std::endl;
  file << "_"; // Marker for data start
}

void VTKWriter::write_vti_data(std::ofstream &file, const RealField &data) const {
  // Write data size (UInt64)
  size_t data_size = data.size() * sizeof(double);
  file.write(reinterpret_cast<const char *>(&data_size), sizeof(size_t));

  // Write actual data
  file.write(reinterpret_cast<const char *>(data.data()), data_size);
}

void VTKWriter::write_pvti_file(int increment) const {
  if (m_rank != 0) return; // Only rank 0 writes master file

  // Format filename with increment number
  std::string pvti_filename = utils::format_with_number(m_filename, increment);

  // Replace .vti with .pvti
  size_t ext_pos = pvti_filename.find_last_of('.');
  if (ext_pos != std::string::npos) {
    pvti_filename = pvti_filename.substr(0, ext_pos) + ".pvti";
  } else {
    pvti_filename += ".pvti";
  }

  std::ofstream file(pvti_filename);
  if (!file) {
    std::cerr << "Failed to open PVTI file: " << pvti_filename << std::endl;
    return;
  }

  file << R"(<?xml version="1.0" encoding="utf-8"?>)" << std::endl;
  file << R"(<VTKFile type="PImageData" version="1.0" byte_order="LittleEndian">)"
       << std::endl;
  file << R"(  <PImageData WholeExtent="0 )" << m_global_size[0] - 1 << " 0 "
       << m_global_size[1] - 1 << " 0 " << m_global_size[2] - 1 << R"(" Origin=")"
       << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << R"(" Spacing=")"
       << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << R"(">)"
       << std::endl;
  file << R"(    <PPointData>)" << std::endl;
  file << R"(      <PDataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1"/>)" << std::endl;
  file << R"(    </PPointData>)" << std::endl;

  // List all piece files
  for (int r = 0; r < m_num_ranks; ++r) {
    std::string piece_filename = generate_filename(increment, r);
    file << R"(    <Piece Source=")" << piece_filename << R"("/>)" << std::endl;
  }

  file << R"(  </PImageData>)" << std::endl;
  file << R"(</VTKFile>)" << std::endl;
}

MPI_Status VTKWriter::write(int increment, const RealField &data) {
  MPI_Status status;
  MPI_Status_set_cancelled(&status, 0);
  MPI_Status_set_elements(&status, MPI_DOUBLE, data.size());

  std::string filename = generate_filename(increment, m_rank);
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Rank " << m_rank << ": Failed to open VTK file: " << filename
              << std::endl;
    MPI_Abort(m_comm, 1);
  }

  // Write header
  write_vti_header(file, increment);

  // Write data
  write_vti_data(file, data);

  // Write footer
  file << std::endl;
  file << R"(  </AppendedData>)" << std::endl;
  file << R"(</VTKFile>)" << std::endl;
  file.close();

  // Synchronize all ranks before writing master file
  MPI_Barrier(m_comm);

  // Write parallel master file (rank 0 only)
  if (m_num_ranks > 1) {
    write_pvti_file(increment);
  }

  return status;
}

MPI_Status VTKWriter::write(int increment, const ComplexField &data) {
  // Convert complex to real (magnitude)
  RealField magnitude(data.size());
  std::transform(data.begin(), data.end(), magnitude.begin(),
                 [](const std::complex<double> &c) { return std::abs(c); });

  return write(increment, magnitude);
}

} // namespace pfc
