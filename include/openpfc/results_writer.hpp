// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file results_writer.hpp
 * @brief Base interface for simulation output and I/O operations
 *
 * @details
 * This file defines the ResultsWriter abstract base class, which provides
 * a unified interface for writing simulation results to various output formats.
 * OpenPFC currently supports:
 * - Binary format (BinaryWriter) for checkpointing and restart
 * - Future: VTK format for visualization (User Story #0025)
 *
 * The ResultsWriter interface supports:
 * - Parallel I/O via MPI (distributed data from decomposed domains)
 * - Writing real and complex fields
 * - Timestep-indexed output (increment parameter)
 * - Domain configuration (global/local sizes, offsets)
 *
 * Typical usage:
 * @code
 * // Create writer
 * auto writer = std::make_unique<pfc::BinaryWriter>("output.bin");
 *
 * // Configure domain (once)
 * writer->set_domain(global_size, local_size, local_offset);
 *
 * // Write data at each save step
 * for (int step = 0; step < num_steps; ++step) {
 *     writer->write(step, field_data);
 * }
 * @endcode
 *
 * This file is part of the I/O module, providing output capabilities for
 * simulation results and checkpointing.
 *
 * @see simulator.hpp for I/O orchestration
 * @see types.hpp for RealField and ComplexField definitions
 * @see binary_reader.hpp for input counterpart
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_RESULTS_WRITER_HPP
#define PFC_RESULTS_WRITER_HPP

#include "types.hpp"
#include "utils.hpp"
#include <array>
#include <iostream>
#include <mpi.h>
#include <vector>

namespace pfc {

/**
 * @brief Abstract base class for writing simulation results to various formats
 *
 * ResultsWriter provides a unified interface for outputting simulation data to
 * different file formats (binary, VTK, HDF5, etc.). It handles parallel I/O
 * via MPI, allowing each rank to write its local subdomain data to a collective
 * output file.
 *
 * ## Key Responsibilities
 *
 * - Define output file name and format
 * - Configure domain decomposition (global size, local size, local offset)
 * - Write real fields (temperature, density, etc.)
 * - Write complex fields (Fourier coefficients, k-space data)
 * - Handle time step indexing (increment parameter)
 * - Coordinate parallel I/O across MPI ranks
 *
 * ## Design Philosophy
 *
 * ResultsWriter follows OpenPFC's principles:
 * - **Interface segregation**: Minimal abstract interface (set_domain, write)
 * - **Format flexibility**: Subclasses implement specific formats
 * - **Parallel-aware**: Built for MPI from the ground up
 * - **Composable**: Multiple writers can be used simultaneously
 *
 * ## Usage Pattern
 *
 * ```cpp
 * using namespace pfc;
 *
 * // 1. Create writer (filename pattern)
 * auto writer = std::make_unique<BinaryWriter>(\"output_%04d.bin\");
 *
 * // 2. Set domain configuration (once)
 * writer->set_domain(global_size, local_size, local_offset);
 *
 * // 3. Write at each time step
 * for (int step = 0; step < num_steps; ++step) {
 *     writer->write(step, field);  // Creates output_0000.bin, output_0001.bin, ...
 * }
 * ```
 *
 * ## Subclasses
 *
 * - **BinaryWriter**: Raw binary format, optimal for checkpointing/restart
 * - **VTKWriter** (planned): Visualization Toolkit format for ParaView
 * - **CustomWriter**: Users can subclass for custom formats
 *
 * @example
 * **Basic Binary Output**
 * ```cpp
 * using namespace pfc;
 *
 * // Create decomposed domain
 * auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, mpi_size);
 * auto local_world = decomposition::get_subworld(decomp, mpi_rank);
 *
 * // Create field and writer
 * auto field = create_real_field(local_world);
 * auto writer = std::make_unique<BinaryWriter>(\"results_%04d.bin\");
 *
 * // Configure writer
 * auto global_size = world::get_size(world);
 * auto local_size = world::get_size(local_world);
 * auto local_offset = compute_offset(local_world);  // Relative to global origin
 *
 * writer->set_domain(global_size, local_size, local_offset);
 *
 * // Write field at step 0
 * writer->write(0, field);  // Creates results_0000.bin
 * ```
 *
 * @example
 * **Multiple Writers (Binary + Statistics)**
 * ```cpp
 * using namespace pfc;
 *
 * // Binary writer for full field
 * auto binary_writer = std::make_unique<BinaryWriter>(\"field_%04d.bin\");
 * binary_writer->set_domain(global_size, local_size, local_offset);
 *
 * // Custom stats writer (rank 0 only)
 * auto stats_writer = std::make_unique<StatsWriter>(\"stats.csv\");
 *
 * // Use both in simulation loop
 * for (int step = 0; step < 1000; ++step) {
 *     // ... compute ...
 *
 *     if (step % 10 == 0) {
 *         binary_writer->write(step, field);  // Every 10 steps
 *     }
 *
 *     stats_writer->write_statistics(step, field);  // Every step
 * }
 * ```
 *
 * @example
 * **Complex Field Output (K-space Data)**
 * ```cpp
 * using namespace pfc;
 *
 * // After FFT forward transform
 * auto k_space_field = create_complex_field(local_world);
 * fft.forward(real_field, k_space_field);
 *
 * // Write complex field
 * auto writer = std::make_unique<BinaryWriter>(\"kspace_%04d.bin\");
 * writer->set_domain(global_size_complex, local_size_complex, local_offset_complex);
 * writer->write(increment, k_space_field);  // Stores complex doubles
 * ```
 *
 * @note The increment parameter is used for filename formatting (e.g., %04d → 0000,
 * 0001, ...).
 * @note set_domain() must be called once before any write() operations.
 * @note Parallel I/O is collective - all ranks must participate in write().
 * @note For checkpointing, store both field data and metadata (time, increment).
 *
 * @warning Subclasses must implement set_domain() and both write() overloads.
 * @warning File format must support parallel I/O for MPI efficiency.
 *
 * @see BinaryWriter - binary format implementation
 * @see Simulator::add_results_writer() - integrate with simulation loop
 */
class ResultsWriter {
public:
  /**
   * @brief Construct a ResultsWriter with output filename pattern
   *
   * The filename can include format specifiers (e.g., %04d) that will be
   * replaced with the increment number during write operations.
   *
   * @param[in] filename Output filename or pattern (e.g., "output_%04d.bin")
   *
   * @example
   * ```cpp
   * using namespace pfc;
   *
   * // Fixed filename (overwrites on each write)
   * auto writer1 = std::make_unique<BinaryWriter>("output.bin");
   *
   * // Timestep-indexed (creates output_0000.bin, output_0001.bin, ...)
   * auto writer2 = std::make_unique<BinaryWriter>("output_%04d.bin");
   *
   * // Custom prefix
   * auto writer3 = std::make_unique<BinaryWriter>("results/field_%06d.bin");
   * ```
   *
   * @see write() - uses filename pattern with increment
   */
  ResultsWriter(const std::string &filename) { m_filename = filename; }
  virtual ~ResultsWriter() = default;

  /**
   * @brief Configure the domain decomposition for parallel I/O
   *
   * Sets up the mapping between local (rank-owned) and global (full domain)
   * data layout. This information is used to write each rank's local data
   * to the correct position in the collective output file.
   *
   * Must be called once before any write() operations.
   *
   * @param[in] arr_global Global domain size [nx, ny, nz]
   * @param[in] arr_local Local subdomain size owned by this rank
   * @param[in] arr_offset Starting offset of local subdomain in global coordinates
   *
   * @example
   * ```cpp
   * using namespace pfc;
   *
   * auto global_size = std::array<int, 3>{256, 256, 256};
   * auto local_size = std::array<int, 3>{128, 128, 256};  // Rank's portion
   * auto offset = std::array<int, 3>{0, 0, 0};  // Rank 0 starts at origin
   *
   * writer->set_domain(global_size, local_size, offset);
   * ```
   *
   * @note Call this once per writer, before any write() operations.
   * @note All ranks must call set_domain() with consistent global_size.
   * @note Offset is in grid indices, not physical coordinates.
   *
   * @see write() - writes data using this domain configuration
   */
  virtual void set_domain(const std::array<int, 3> &arr_global,
                          const std::array<int, 3> &arr_local,
                          const std::array<int, 3> &arr_offset) = 0;

  /**
   * @brief Write a real-valued field to file at specified time step
   *
   * Writes the local portion of a RealField (double precision) to the output
   * file. In MPI contexts, this is a collective operation - all ranks write
   * their local data simultaneously to the correct position in the file.
   *
   * @param[in] increment Time step or frame number (used for filename formatting)
   * @param[in] data Local real field data (vector of doubles)
   * @return MPI_Status Information about the write operation
   *
   * @example
   * ```cpp
   * using namespace pfc;
   *
   * for (int step = 0; step < 1000; ++step) {
   *     if (step % 100 == 0) {
   *         writer->write(step, temperature_field);
   *     }
   * }
   * ```
   *
   * @note This is a collective MPI operation - all ranks must call it.
   * @note Field size must match local_size specified in set_domain().
   */
  virtual MPI_Status write(int increment, const RealField &data) = 0;

  /**
   * @brief Write a complex-valued field to file at specified time step
   *
   * Writes the local portion of a ComplexField (complex doubles) to the output
   * file. Useful for storing Fourier coefficients or k-space data.
   *
   * @param[in] increment Time step or frame number
   * @param[in] data Local complex field data
   * @return MPI_Status Information about the write operation
   *
   * @note Complex field size is typically ~50% of real field size (r2c symmetry).
   * @see FFT::forward() - produces ComplexField from RealField
   */
  virtual MPI_Status write(int increment, const ComplexField &data) = 0;

  template <typename T> MPI_Status write(const std::vector<T> &data) {
    return write(0, data);
  }

protected:
  std::string m_filename;
};

/**
 * @brief Binary format writer for raw field data output
 *
 * BinaryWriter implements ResultsWriter for raw binary output format. This
 * format is optimal for:
 * - Checkpointing and restart (exact data preservation)
 * - Large-scale simulations (minimal storage overhead)
 * - Fast I/O performance (no parsing or conversion)
 *
 * The binary format stores double or complex<double> values directly in native
 * byte order (platform-dependent). Files can be read back using BinaryReader
 * for simulation restart.
 *
 * ## File Format
 *
 * - Extension: `.bin` (by convention)
 * - Data type: double (8 bytes) or complex<double> (16 bytes)
 * - Byte order: Native (platform-dependent)
 * - Structure: Raw array, no header/metadata
 *
 * ## Parallel I/O
 *
 * Uses MPI-IO for collective parallel writes. Each rank writes its local
 * subdomain to the correct position in a single shared file.
 *
 * @example
 * **Basic Binary Output**
 * ```cpp
 * using namespace pfc;
 *
 * auto writer = std::make_unique<BinaryWriter>(\"field_%04d.bin\");
 * writer->set_domain(global_size, local_size, offset);
 *
 * writer->write(0, field);  // Creates field_0000.bin
 * writer->write(1, field);  // Creates field_0001.bin
 * ```
 *
 * @example
 * **Checkpoint for Restart**
 * ```cpp
 * using namespace pfc;
 *
 * // Save checkpoint every 1000 steps\n * auto checkpoint =
 * std::make_unique<BinaryWriter>(\"checkpoint_%04d.bin\");
 * checkpoint->set_domain(global_size, local_size, offset);
 *
 * for (int step = 0; step < 10000; ++step) {
 *     // ... simulation ...
 *
 *     if (step % 1000 == 0) {
 *         checkpoint->write(step, field);
 *         // Also save metadata: time, step, parameters
 *     }
 * }
 * ```
 *
 * @note Binary files are not portable across different architectures (endianness).
 * @note No metadata is stored - you must track time, step number, domain size
 * separately.
 * @note For visualization, use VTKWriter instead (User Story #0025).
 *
 * @see ResultsWriter - base class interface
 * @see BinaryReader - read binary files for restart
 */
class BinaryWriter : public ResultsWriter {
  using ResultsWriter::ResultsWriter;

private:
  MPI_Datatype m_filetype;

  static MPI_Datatype get_type(RealField) { return MPI_DOUBLE; }
  static MPI_Datatype get_type(ComplexField) { return MPI_DOUBLE_COMPLEX; }

public:
  void set_domain(const std::array<int, 3> &arr_global,
                  const std::array<int, 3> &arr_local,
                  const std::array<int, 3> &arr_offset) {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                             arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  };

  MPI_Status write(int increment, const RealField &data) {
    return write_(increment, data);
  }

  MPI_Status write(int increment, const ComplexField &data) {
    return write_(increment, data);
  }

  template <typename T>
  MPI_Status write_(int increment, const std::vector<T> &data) {
    MPI_File fh;
    std::string filename2 = utils::format_with_number(m_filename, increment);
    MPI_File_open(MPI_COMM_WORLD, filename2.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset filesize = 0;
    MPI_Status status;
    const unsigned int disp = 0;
    MPI_Datatype type = get_type(data);
    MPI_File_set_size(fh, filesize); // force overwriting existing data
    MPI_File_set_view(fh, disp, type, m_filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, data.data(), data.size(), type, &status);
    MPI_File_close(&fh);
    return status;
  }
};
} // namespace pfc

#endif // PFC_RESULTS_WRITER_HPP
