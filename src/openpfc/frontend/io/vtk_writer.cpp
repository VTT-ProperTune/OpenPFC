// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <limits>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <sstream>
#include <stdexcept>

namespace pfc {

namespace {

[[nodiscard]] constexpr bool fits_half_open_extent(long long offset,
                                                   long long length,
                                                   long long upper_exclusive) {
  return offset >= 0 && length >= 0 && upper_exclusive >= 0 &&
         offset + length <= upper_exclusive;
}

[[nodiscard]] bool spacing_ok(double s) { return std::isfinite(s) && s > 0.0; }

[[nodiscard]] bool origin_ok(double x) { return std::isfinite(x); }

[[nodiscard]] bool vtk_extent_endpoint_safe(int begin_index,
                                            unsigned long long extent_len_ul) {
  if (extent_len_ul > static_cast<unsigned long long>(INT_MAX) + 1ULL) {
    return false;
  }
  const auto extent_len = static_cast<long long>(extent_len_ul);
  const auto endpoint = static_cast<long long>(begin_index) + extent_len - 1LL;
  return endpoint <= static_cast<long long>(INT_MAX);
}

void validate_vtk_domain_for_writer(const std::array<int, 3> &global_size,
                                    const std::array<int, 3> &local_size,
                                    const std::array<int, 3> &offset,
                                    const std::array<double, 3> &origin,
                                    const std::array<double, 3> &spacing) {
  for (int i = 0; i < 3; ++i) {
    const int g = global_size[i];
    const int l = local_size[i];
    const int o = offset[i];
    if (g <= 0 || l <= 0) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: global/local dimensions must be positive");
    }
    if (o < 0) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: offsets must be non-negative");
    }

    const auto gu = static_cast<long long>(g);
    if (!fits_half_open_extent(static_cast<long long>(o), static_cast<long long>(l),
                               gu)) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: piece Extent does not lie inside WholeExtent "
          "(check offset + local_size vs global)");
    }

    const unsigned long long extent_len_ul = static_cast<unsigned long long>(l);
    if (!vtk_extent_endpoint_safe(o, extent_len_ul)) {
      throw std::overflow_error(
          "VTKWriter::set_domain: VTK extents overflow int range");
    }

    if (!origin_ok(origin[i])) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: origin components must be finite");
    }
    if (!spacing_ok(spacing[i])) {
      throw std::invalid_argument(
          "VTKWriter::set_domain: spacing components must be finite and "
          "positive");
    }
  }

  // WholeExtent endpoints are global_size[d]-1; guard overflow when nx==INT_MAX.
  for (int i = 0; i < 3; ++i) {
    const long long whole_hi = static_cast<long long>(global_size[i]) - 1LL;
    if (whole_hi > static_cast<long long>(INT_MAX)) {
      throw std::overflow_error(
          "VTKWriter::set_domain: VTK WholeExtent overflows int range");
    }
  }
}

[[nodiscard]] std::size_t
vtk_local_point_count_or_throw(const std::array<int, 3> &local_size) {
  for (int i = 0; i < 3; ++i) {
    if (local_size[i] <= 0) {
      throw std::invalid_argument(
          "VTKWriter::write: domain not configured (call set_domain with "
          "positive local sizes)");
    }
  }

  unsigned long long n = 1;
  for (int i = 0; i < 3; ++i) {
    const auto li = static_cast<unsigned long long>(local_size[i]);
    const auto max_sz =
        static_cast<unsigned long long>((std::numeric_limits<std::size_t>::max)());
    if (n > max_sz / li) {
      throw std::overflow_error(
          "VTKWriter::write: local field size product overflows size_t");
    }
    n *= li;
  }
  if (n > static_cast<unsigned long long>(INT_MAX)) {
    throw std::overflow_error(
        "VTKWriter::write: local field element count exceeds INT_MAX");
  }
  return static_cast<std::size_t>(n);
}

} // namespace

void VTKWriter::set_domain(const std::array<int, 3> &arr_global,
                           const std::array<int, 3> &arr_local,
                           const std::array<int, 3> &arr_offset) {
  validate_vtk_domain_for_writer(arr_global, arr_local, arr_offset, m_origin,
                                 m_spacing);
  m_global_size = arr_global;
  m_local_size = arr_local;
  m_offset = arr_offset;
}

void VTKWriter::set_origin(const std::array<double, 3> &origin) {
  for (int i = 0; i < 3; ++i) {
    if (!origin_ok(origin[i])) {
      throw std::invalid_argument(
          "VTKWriter::set_origin: origin components must be finite");
    }
  }
  m_origin = origin;
}

void VTKWriter::set_spacing(const std::array<double, 3> &spacing) {
  for (int i = 0; i < 3; ++i) {
    if (!spacing_ok(spacing[i])) {
      throw std::invalid_argument(
          "VTKWriter::set_spacing: spacing components must be finite and "
          "positive");
    }
  }
  m_spacing = spacing;
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
      std::string out = name;
      out += '_';
      out += std::to_string(rank);
      out += ext;
      return out;
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

  // List all piece files
  for (int r = 0; r < current_size; ++r) {
    // Build piece filename deterministically to avoid relying on cached m_num_ranks
    std::string base = utils::format_with_number(m_filename, increment);
    size_t base_ext_pos = base.find_last_of('.');
    std::string name =
        (base_ext_pos != std::string::npos) ? base.substr(0, base_ext_pos) : base;
    std::string ext = (base_ext_pos != std::string::npos) ? base.substr(base_ext_pos)
                                                          : std::string();
    std::string piece_filename = name;
    piece_filename += '_';
    piece_filename += std::to_string(r);
    piece_filename += ext;
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
  const std::size_t expected_pts = vtk_local_point_count_or_throw(m_local_size);
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
  const std::size_t expected_pts = vtk_local_point_count_or_throw(m_local_size);
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
