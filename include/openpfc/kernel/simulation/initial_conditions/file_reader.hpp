// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file file_reader.hpp
 * @brief Read initial conditions from binary file
 *
 * @details
 * This file defines the FileReader class, which reads field values from a
 * binary file to initialize simulation state. Useful for:
 * - Restarting simulations from checkpoints
 * - Loading pre-computed initial conditions
 * - Continuing interrupted simulations
 *
 * The binary file format must match the expected field layout (domain size,
 * decomposition, data type).
 *
 * Usage:
 * @code
 * auto ic = std::make_unique<pfc::FileReader>("checkpoint.bin");
 * ic->set_field_name("density");
 * simulator.add_initial_condition(std::move(ic));
 * @endcode
 *
 * @see binary_reader.hpp for binary I/O operations
 * @see field_modifier.hpp for base class
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_INITIAL_CONDITIONS_FILE_READER_HPP
#define PFC_INITIAL_CONDITIONS_FILE_READER_HPP

#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <openpfc/frontend/utils/logging.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/model.hpp>

namespace pfc {

class FileReader : public FieldModifier {
private:
  std::string m_filename;
  MPI_Comm m_io_comm = MPI_COMM_WORLD;

public:
  FileReader() = default;

  void set_filename(std::string filename) { m_filename = std::move(filename); }
  const std::string &get_filename() const { return m_filename; }

  explicit FileReader(std::string filename) : m_filename(std::move(filename)) {}

  void set_mpi_comm(MPI_Comm comm) noexcept override { m_io_comm = comm; }

  void apply(const SimulationContext &ctx, Model &m, double time) override {
    (void)time;
    const fft::IFFT &fft = get_fft(m);
    const auto &world = get_world(m);
    const auto world_size = get_size(world);
    const auto inbox_size = get_inbox(fft).size;
    const auto inbox_offset = get_inbox(fft).low;

    Field &f = get_real_field(m, get_field_name());
    if (m.is_rank0()) {
      const pfc::Logger lg{pfc::LogLevel::Info, 0};
      pfc::log_info(lg, std::string("Reading initial condition from file: ") +
                            get_filename());
    }
    try {
      BinaryReader reader{ctx.mpi_comm()};
      reader.set_domain(world_size, inbox_size, inbox_offset);
      reader.read(get_filename(), f);
    } catch (const std::exception &ex) {
      std::ostringstream oss;
      oss << "FileReader failed to read \"" << get_filename() << "\" into field \""
          << get_field_name() << "\": " << ex.what();
      throw std::runtime_error(oss.str());
    }
  }

  void apply(Model &m, double time) override {
    apply(SimulationContext{m_io_comm}, m, time);
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_FILE_READER_HPP
