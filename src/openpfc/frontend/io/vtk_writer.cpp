// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/io/vtk_writer_validate.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/utils/logging.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pfc {

namespace {

/**
 * Parallel VTK ImageData pieces use `basename_rank.ext` (same convention as
 * `VTKWriter::generate_filename` for multi-rank runs).
 */
[[nodiscard]] std::string vtk_piece_filename_for_rank(std::string base, int rank) {
  const size_t ext_pos = base.find_last_of('.');
  if (ext_pos == std::string::npos) {
    return base + '_' + std::to_string(rank);
  }
  std::string name = base.substr(0, ext_pos);
  std::string ext = base.substr(ext_pos);
  return name + '_' + std::to_string(rank) + ext;
}

} // namespace

void VTKWriter::set_domain(const std::array<int, 3> &arr_global,
                           const std::array<int, 3> &arr_local,
                           const std::array<int, 3> &arr_offset) {
  io::vtk_validate::validate_writer_domain(arr_global, arr_local, arr_offset,
                                           m_origin, m_spacing);
  m_global_size = arr_global;
  m_local_size = arr_local;
  m_offset = arr_offset;
}

void VTKWriter::set_origin(const std::array<double, 3> &origin) {
  io::vtk_validate::validate_origin_array(origin, "VTKWriter::set_origin");
  m_origin = origin;
}

void VTKWriter::set_spacing(const std::array<double, 3> &spacing) {
  io::vtk_validate::validate_spacing_array(spacing, "VTKWriter::set_spacing");
  m_spacing = spacing;
}

std::string VTKWriter::generate_filename(int increment, int rank) const {
  std::string base = utils::format_with_number(m_filename, increment);

  if (m_num_ranks > 1 && rank >= 0) {
    // Match legacy behavior: only insert `_rank` before the extension when a dot
    // exists.
    if (base.find_last_of('.') != std::string::npos) {
      return vtk_piece_filename_for_rank(std::move(base), rank);
    }
  }
  return base;
}

void VTKWriter::write_vti_header(std::ofstream &file,
                                 [[maybe_unused]] int increment) const {
  file << R"(<?xml version="1.0" encoding="utf-8"?>)" << '\n';
  file
      << R"(<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">)"
      << '\n';

  // Whole extent (global domain)
  file << R"(  <ImageData WholeExtent="0 )" << m_global_size[0] - 1 << " 0 "
       << m_global_size[1] - 1 << " 0 " << m_global_size[2] - 1 << R"(" Origin=")"
       << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << R"(" Spacing=")"
       << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << R"(">)"
       << '\n';

  // Piece extent (local domain)
  file << R"(    <Piece Extent=")" << m_offset[0] << " "
       << m_offset[0] + m_local_size[0] - 1 << " " << m_offset[1] << " "
       << m_offset[1] + m_local_size[1] - 1 << " " << m_offset[2] << " "
       << m_offset[2] + m_local_size[2] - 1 << R"(">)" << '\n';

  file << R"(      <PointData>)" << '\n';
  file << R"(        <DataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1" format="appended" offset="0"/>)" << '\n';
  file << R"(      </PointData>)" << '\n';
  file << R"(    </Piece>)" << '\n';
  file << R"(  </ImageData>)" << '\n';
  file << R"(  <AppendedData encoding="raw">)" << '\n';
  file << "_"; // Marker for data start
}

void VTKWriter::write_vti_data(std::ofstream &file, const RealField &data) {
  // VTK header_type="UInt64": length prefix is always 8 bytes (see
  // write_vti_header).
  const std::uint64_t appended_bytes =
      static_cast<std::uint64_t>(data.size()) * sizeof(double);
  file.write(reinterpret_cast<const char *>(&appended_bytes),
             sizeof(appended_bytes));

  const size_t payload_bytes = data.size() * sizeof(double);
  file.write(reinterpret_cast<const char *>(data.data()),
             static_cast<std::streamsize>(payload_bytes));
}

void VTKWriter::write_pvti_file(int increment) const {
  // Only true rank 0 writes master file
  int current_rank = 0;
  int current_size = 1;
  MPI_Comm_rank(m_comm, &current_rank);
  MPI_Comm_size(m_comm, &current_size);
  if (current_rank != 0) {
    return;
  }

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
    const Logger lg{LogLevel::Warning, /*rank*/ 0};
    log_error(lg, std::string("Failed to open PVTI file: ") + pvti_filename);
    return;
  }

  file << R"(<?xml version="1.0" encoding="utf-8"?>)" << '\n';
  file << R"(<VTKFile type="PImageData" version="1.0" byte_order="LittleEndian">)"
       << '\n';
  file << R"(  <PImageData WholeExtent="0 )" << m_global_size[0] - 1 << " 0 "
       << m_global_size[1] - 1 << " 0 " << m_global_size[2] - 1 << R"(" Origin=")"
       << m_origin[0] << " " << m_origin[1] << " " << m_origin[2] << R"(" Spacing=")"
       << m_spacing[0] << " " << m_spacing[1] << " " << m_spacing[2] << R"(">)"
       << '\n';
  file << R"(    <PPointData>)" << '\n';
  file << R"(      <PDataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1"/>)" << '\n';
  file << R"(    </PPointData>)" << '\n';

  const std::string base = utils::format_with_number(m_filename, increment);
  for (int r = 0; r < current_size; ++r) {
    const std::string piece_filename = vtk_piece_filename_for_rank(base, r);
    file << R"(    <Piece Source=")" << piece_filename << R"("/>)" << '\n';
  }

  file << R"(  </PImageData>)" << '\n';
  file << R"(</VTKFile>)" << '\n';
  file.close();
  if (!file.good()) {
    const Logger lg{LogLevel::Error, /*rank*/ 0};
    const std::string msg =
        std::string("Failed writing PVTI file: ") + pvti_filename;
    log_error(lg, msg);
    throw std::runtime_error(msg);
  }
}

MPI_Status VTKWriter::write(int increment, const RealField &data) {
  const std::size_t expected_pts =
      io::vtk_validate::expect_local_point_count(m_local_size);
  if (data.size() != expected_pts) {
    std::ostringstream oss;
    oss << "VTKWriter::write: field size mismatch for VTK Piece (expected "
        << expected_pts << " points, got " << data.size() << ")";
    throw std::runtime_error(oss.str());
  }

  MPI_Status status;
  MPI_Status_set_cancelled(&status, 0);
  MPI_Status_set_elements(&status, MPI_DOUBLE, static_cast<int>(data.size()));

  std::string filename = generate_filename(increment, m_rank);
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    const Logger lg{LogLevel::Error, m_rank};
    const std::string msg = std::string("Failed to open VTK file: ") + filename;
    log_error(lg, msg);
    throw std::runtime_error(msg);
  }

  // Write header
  write_vti_header(file, increment);

  // Write data
  write_vti_data(file, data);

  // Write footer
  file << '\n';
  file << R"(  </AppendedData>)" << '\n';
  file << R"(</VTKFile>)" << '\n';
  file.close();
  if (!file.good()) {
    const Logger lg{LogLevel::Error, m_rank};
    const std::string msg = std::string("Failed writing VTK file: ") + filename;
    log_error(lg, msg);
    throw std::runtime_error(msg);
  }

  // Collective on m_comm: all ranks must participate before rank 0 writes PVTI.
  pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier");

  // Write parallel master file (rank 0 only)
  int current_size = 1;
  MPI_Comm_size(m_comm, &current_size);
  if (current_size > 1) {
    write_pvti_file(increment);
  }

  return status;
}

MPI_Status VTKWriter::write(int increment, const ComplexField &data) {
  const std::size_t expected_pts =
      io::vtk_validate::expect_local_point_count(m_local_size);
  if (data.size() != expected_pts) {
    std::ostringstream oss;
    oss << "VTKWriter::write: field size mismatch for VTK Piece (expected "
        << expected_pts << " points, got " << data.size() << ")";
    throw std::runtime_error(oss.str());
  }

  // Convert complex to real (magnitude)
  RealField magnitude(data.size());
  std::transform(data.begin(), data.end(), magnitude.begin(),
                 [](const std::complex<double> &c) { return std::abs(c); });

  return write(increment, magnitude);
}

} // namespace pfc
