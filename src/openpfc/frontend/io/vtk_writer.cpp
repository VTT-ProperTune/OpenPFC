// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <limits>
#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/frontend/io/vtk_writer_validate.hpp>
#include <openpfc/kernel/mpi/collective_check.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/utils/logging.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

using Extent = std::array<int, 6>;

Extent intersection(const Extent &a, const Extent &b) {
  Extent result{};
  for (int d = 0; d < 3; ++d) {
    result[2 * d] = std::max(a[2 * d], b[2 * d]);
    result[2 * d + 1] = std::min(a[2 * d + 1], b[2 * d + 1]);
  }
  return result;
}

std::size_t volume(const Extent &extent) {
  std::size_t result = 1;
  for (int d = 0; d < 3; ++d) {
    if (extent[2 * d + 1] < extent[2 * d]) return 0;
    const auto n = static_cast<std::size_t>(extent[2 * d + 1] - extent[2 * d]) + 1;
    if (result > std::numeric_limits<std::size_t>::max() / n)
      throw std::length_error("VTK point count overflows size_t");
    result *= n;
  }
  return result;
}

Extent shared_point_extent(Extent owned, const std::array<int, 3> &global) {
  for (int d = 0; d < 3; ++d)
    if (owned[2 * d + 1] < global[d] - 1) ++owned[2 * d + 1];
  return owned;
}

template <class Function> void visit(const Extent &extent, Function &&function) {
  for (int z = extent[4]; z <= extent[5]; ++z)
    for (int y = extent[2]; y <= extent[3]; ++y)
      for (int x = extent[0]; x <= extent[1]; ++x) function(x, y, z);
}

std::size_t point_index(int x, int y, int z, const Extent &extent) {
  const auto nx = static_cast<std::size_t>(extent[1] - extent[0]) + 1;
  const auto ny = static_cast<std::size_t>(extent[3] - extent[2]) + 1;
  return static_cast<std::size_t>(x - extent[0]) +
         nx * (static_cast<std::size_t>(y - extent[2]) +
               ny * static_cast<std::size_t>(z - extent[4]));
}

struct ParallelPiece {
  std::array<int, 3> size;
  std::vector<double> values;
};

// Only shared faces/edges/corners cross ranks. Owned samples are copied
// locally; there is no global field gather and no assumption about the split
// axes.
ParallelPiece prepare_parallel_piece(pfc::field::FieldView<double> data,
                                     const Extent &local,
                                     const std::array<int, 3> &global, int rank,
                                     int ranks, MPI_Comm comm) {
  static_assert(sizeof(Extent) == 6 * sizeof(int));
  std::vector<Extent> owned;
  bool ok = true;
  try {
    owned.resize(static_cast<std::size_t>(ranks));
  } catch (const std::exception &) {
    ok = false;
  }
  pfc::mpi::require_all_ranks(ok, comm, "VTKWriter: cannot allocate piece metadata");
  pfc::mpi::throw_on_mpi_error(MPI_Allgather(local.data(), 6, MPI_INT,
                                             owned.front().data(), 6, MPI_INT, comm),
                               "MPI_Allgather of VTK owned extents");

  const Extent piece = shared_point_extent(local, global);
  ParallelPiece result{};
  std::vector<int> sends, receives, send_offsets, receive_offsets;
  std::vector<double> outgoing, incoming;
  std::vector<unsigned char> seen;
  try {
    std::size_t total_owned = 0;
    const Extent whole{0, global[0] - 1, 0, global[1] - 1, 0, global[2] - 1};
    const auto whole_count = volume(whole);
    for (const auto &extent : owned) {
      const auto count = volume(extent);
      if (intersection(extent, whole) != extent || count > whole_count - total_owned)
        throw std::invalid_argument("Invalid VTK owned partition");
      total_owned += count;
    }
    if (total_owned != whole_count)
      throw std::invalid_argument("VTK owned partition does not cover the domain");
    for (int d = 0; d < 3; ++d) result.size[d] = piece[2 * d + 1] - piece[2 * d] + 1;
    result.values.resize(volume(piece));
    seen.resize(result.values.size(), 0);
    sends.resize(ranks);
    receives.resize(ranks);
    send_offsets.resize(ranks);
    receive_offsets.resize(ranks);
    std::size_t send_count = 0, receive_count = 0;
    constexpr auto max_count =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (int peer = 0; peer < ranks; ++peer) {
      const auto send = peer == rank
                            ? 0
                            : volume(intersection(
                                  local, shared_point_extent(owned[peer], global)));
      const auto receive =
          peer == rank ? 0 : volume(intersection(owned[peer], piece));
      if (send > max_count - send_count || receive > max_count - receive_count)
        throw std::length_error("VTK boundary exchange exceeds MPI int counts");
      sends[peer] = static_cast<int>(send);
      receives[peer] = static_cast<int>(receive);
      send_offsets[peer] = static_cast<int>(send_count);
      receive_offsets[peer] = static_cast<int>(receive_count);
      send_count += send;
      receive_count += receive;
    }
    outgoing.resize(send_count);
    incoming.resize(receive_count);
    for (int peer = 0; peer < ranks; ++peer) {
      if (peer == rank) continue;
      std::size_t position = static_cast<std::size_t>(send_offsets[peer]);
      visit(intersection(local, shared_point_extent(owned[peer], global)),
            [&](int x, int y, int z) {
              outgoing[position++] = data.data()[point_index(x, y, z, local)];
            });
    }
  } catch (const std::exception &) {
    ok = false;
  }
  pfc::mpi::require_all_ranks(ok, comm,
                              "VTKWriter: invalid partition, excessive "
                              "boundary counts or piece allocation failure");
  double unused = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Alltoallv(outgoing.empty() ? &unused : outgoing.data(), sends.data(),
                    send_offsets.data(), MPI_DOUBLE,
                    incoming.empty() ? &unused : incoming.data(), receives.data(),
                    receive_offsets.data(), MPI_DOUBLE, comm),
      "MPI_Alltoallv of VTK shared boundary points");
  for (int peer = 0; peer < ranks; ++peer) {
    std::size_t position = static_cast<std::size_t>(receive_offsets[peer]);
    visit(intersection(owned[peer], piece), [&](int x, int y, int z) {
      const auto index = point_index(x, y, z, piece);
      if (seen[index]) ok = false;
      seen[index] = 1;
      result.values[index] = peer == rank ? data.data()[point_index(x, y, z, local)]
                                          : incoming[position++];
    });
  }
  ok = ok && std::all_of(seen.begin(), seen.end(),
                         [](unsigned char value) { return value == 1; });
  pfc::mpi::require_all_ranks(
      ok, comm, "VTKWriter: shared points have missing or overlapping owners");
  return result;
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
  std::string base = formatted_path(increment);

  if (m_num_ranks > 1 && rank >= 0) {
    // Match legacy behavior: only insert `_rank` before the extension when a
    // dot exists.
    if (base.find_last_of('.') != std::string::npos) {
      return vtk_piece_filename_for_rank(std::move(base), rank);
    }
  }
  return base;
}

void VTKWriter::write_vti_header(std::ofstream &file, [[maybe_unused]] int increment,
                                 const std::array<int, 3> &piece_size) const {
  file << std::setprecision(std::numeric_limits<double>::max_digits10);
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
       << m_offset[0] + piece_size[0] - 1 << " " << m_offset[1] << " "
       << m_offset[1] + piece_size[1] - 1 << " " << m_offset[2] << " "
       << m_offset[2] + piece_size[2] - 1 << R"(">)" << '\n';

  file << R"(      <PointData>)" << '\n';
  file << R"(        <DataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1" format="appended" offset="0"/>)" << '\n';
  file << R"(      </PointData>)" << '\n';
  file << R"(    </Piece>)" << '\n';
  file << R"(  </ImageData>)" << '\n';
  file << R"(  <AppendedData encoding="raw">)" << '\n';
  file << "_"; // Marker for data start
}

void VTKWriter::write_vti_data(std::ofstream &file,
                               pfc::field::FieldView<double> data) {
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
  // Shared point extents partition cells without a gap at process seams.
  int current_rank = 0;
  int current_size = 1;
  MPI_Comm_rank(m_comm, &current_rank);
  MPI_Comm_size(m_comm, &current_size);
  const auto local_extent = shared_point_extent(
      Extent{
          m_offset[0],
          m_offset[0] + m_local_size[0] - 1,
          m_offset[1],
          m_offset[1] + m_local_size[1] - 1,
          m_offset[2],
          m_offset[2] + m_local_size[2] - 1,
      },
      m_global_size);
  std::vector<int> extents(static_cast<std::size_t>(current_size) * 6);
  pfc::mpi::throw_on_mpi_error(
      MPI_Gather(local_extent.data(), 6, MPI_INT,
                 current_rank == 0 ? extents.data() : nullptr, 6, MPI_INT, 0,
                 m_comm),
      "MPI_Gather of VTK piece extents");
  if (current_rank != 0) {
    return;
  }

  // Format filename with increment number
  std::string pvti_filename = formatted_path(increment);

  // Replace .vti with .pvti
  size_t ext_pos = pvti_filename.find_last_of('.');
  if (ext_pos != std::string::npos) {
    pvti_filename = pvti_filename.substr(0, ext_pos) + ".pvti";
  } else {
    pvti_filename += ".pvti";
  }

  std::ofstream file(pvti_filename);
  file << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (!file) {
    const Logger lg{LogLevel::Error, /*rank*/ 0};
    const std::string msg =
        std::string("Failed to open PVTI file: ") + pvti_filename;
    log_error(lg, msg);
    throw std::runtime_error(msg);
  }

  file << R"(<?xml version="1.0" encoding="utf-8"?>)" << '\n';
  file << R"(<VTKFile type="PImageData" version="1.0" byte_order="LittleEndian">)"
       << '\n';
  file << R"(  <PImageData WholeExtent="0 )" << m_global_size[0] - 1 << " 0 "
       << m_global_size[1] - 1 << " 0 " << m_global_size[2] - 1
       << R"(" GhostLevel="0" Origin=")" << m_origin[0] << " " << m_origin[1] << " "
       << m_origin[2] << R"(" Spacing=")" << m_spacing[0] << " " << m_spacing[1]
       << " " << m_spacing[2] << R"(">)" << '\n';
  file << R"(    <PPointData>)" << '\n';
  file << R"(      <PDataArray type="Float64" Name=")" << m_field_name
       << R"(" NumberOfComponents="1"/>)" << '\n';
  file << R"(    </PPointData>)" << '\n';

  const std::string base = formatted_path(increment);
  for (int r = 0; r < current_size; ++r) {
    const std::string piece_filename =
        std::filesystem::path(vtk_piece_filename_for_rank(base, r))
            .filename()
            .string();
    const int *extent = extents.data() + static_cast<std::size_t>(r) * 6;
    file << R"(    <Piece Extent=")" << extent[0] << " " << extent[1] << " "
         << extent[2] << " " << extent[3] << " " << extent[4] << " " << extent[5]
         << R"(" Source=")" << piece_filename << R"("/>)" << '\n';
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

MPI_Status VTKWriter::write(int increment, pfc::field::FieldView<double> data) {
  const std::size_t expected_pts =
      io::vtk_validate::expect_local_point_count(m_local_size);

  // 1. Initialize local success flag and error message storage
  int local_ok = 1;
  std::string error_msg;

  // 2. Field size mismatch check - sets local_ok=0 on error, delays throw
  if (data.size() != expected_pts) {
    std::ostringstream oss;
    oss << "VTKWriter::write: field size mismatch for VTK Piece (expected "
        << expected_pts << " points, got " << data.size() << ")";
    error_msg = oss.str();
    local_ok = 0;
  }

  // 3. File open check - sets local_ok=0 on error, delays throw
  std::string filename = generate_filename(increment, m_rank);
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    const Logger lg{LogLevel::Error, m_rank};
    error_msg = std::string("Failed to open VTK file: ") + filename;
    log_error(lg, error_msg);
    local_ok = 0;
  }

  // 4. FIRST collective agreement: field size AND file open checks combined
  // BEFORE file write
  int global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
      "MPI_Allreduce on error flag");

  // 5. Throw collectively if any rank failed field size or file open
  if (global_ok == 0) {
    if (!error_msg.empty()) {
      throw std::runtime_error(error_msg);
    } else {
      throw std::runtime_error("VTKWriter::write: collective error detected");
    }
  }

  // Complete point-grid cells at rank seams before publishing their pieces.
  std::array<int, 3> piece_size = m_local_size;
  std::vector<double> shared_values;
  if (m_num_ranks > 1) {
    const Extent owned{m_offset[0], m_offset[0] + m_local_size[0] - 1,
                       m_offset[1], m_offset[1] + m_local_size[1] - 1,
                       m_offset[2], m_offset[2] + m_local_size[2] - 1};
    auto piece = prepare_parallel_piece(data, owned, m_global_size, m_rank,
                                        m_num_ranks, m_comm);
    piece_size = piece.size;
    shared_values = std::move(piece.values);
    data = pfc::field::FieldView<double>(shared_values);
  }
  // 6. Proceed with file writes on all ranks (now guaranteed to reach barrier)
  write_vti_header(file, increment, piece_size);
  write_vti_data(file, data);
  file << '\n';
  file << R"(  </AppendedData>)" << '\n';
  file << R"(</VTKFile>)" << '\n';
  file.close();

  // 7. File write failure check - sets local_ok=0 on error
  if (!file.good()) {
    const Logger lg{LogLevel::Error, m_rank};
    const std::string msg = std::string("Failed writing VTK file: ") + filename;
    log_error(lg, msg);
    local_ok = 0;
    error_msg = msg;
  }

  // 8. SECOND collective agreement: file write failure check AFTER file write
  // completion
  global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
      "MPI_Allreduce on file write error flag");

  // 9. Throw collectively if any rank failed file write check
  if (global_ok == 0) {
    throw std::runtime_error(error_msg);
  }

  // 10. Collective barrier before parallel master file write
  pfc::mpi::throw_on_mpi_error(MPI_Barrier(m_comm), "MPI_Barrier");

  // 11. Write parallel master file (rank 0 only)
  int current_size = 1;
  MPI_Comm_size(m_comm, &current_size);
  if (current_size > 1) {
    write_pvti_file(increment);
  }

  // 12. Return MPI status only after all ranks complete
  MPI_Status status;
  MPI_Status_set_cancelled(&status, 0);
  MPI_Status_set_elements(&status, MPI_DOUBLE, static_cast<int>(expected_pts));
  return status;
}

MPI_Status VTKWriter::write(int increment,
                            pfc::field::FieldView<std::complex<double>> data) {
  const std::size_t expected_pts =
      io::vtk_validate::expect_local_point_count(m_local_size);

  // 1. Initialize local success flag and error message storage
  int local_ok = 1;
  std::string error_msg;

  // 2. Field size mismatch check - sets local_ok=0 on error, delays throw
  if (data.size() != expected_pts) {
    std::ostringstream oss;
    oss << "VTKWriter::write: field size mismatch for VTK Piece (expected "
        << expected_pts << " points, got " << data.size() << ")";
    error_msg = oss.str();
    local_ok = 0;
  }

  // 3. Collective error agreement before field conversion
  int global_ok = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, m_comm),
      "MPI_Allreduce on complex field error flag");

  // 4. Throw collectively if any rank failed
  if (global_ok == 0) {
    if (!error_msg.empty()) {
      throw std::runtime_error(error_msg);
    } else {
      throw std::runtime_error(
          "VTKWriter::write: collective complex field error detected");
    }
  }

  // 5. Convert complex to real (magnitude) - only happens if all ranks passed
  std::vector<double> magnitude(data.size());
  std::transform(data.data(), data.data() + data.size(), magnitude.begin(),
                 [](const std::complex<double> &c) { return std::abs(c); });

  // 6. Delegate to the real FieldView overload which has its own collective
  // error agreements
  return write(increment, magnitude);
}

} // namespace pfc
