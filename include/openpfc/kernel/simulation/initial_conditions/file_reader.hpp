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

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/utils/logging.hpp>

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

  void apply(pfc::field::FieldOutput<double> field, const Domain &domain, const Box3i &box,
             double time = 0.0) override {
    apply(SimulationContext{m_io_comm}, field, domain, box, time);
  }

  void apply(const SimulationContext &ctx, pfc::field::FieldOutput<double> field, const Domain &domain,
             const Box3i &inbox, double time = 0.0) override {
    (void)time;
    if (ctx.is_rank0()) {
      const pfc::Logger lg{pfc::LogLevel::Info, 0};
      pfc::log_info(lg, std::string("Reading initial condition from file: ") +
                            get_filename());
    }
    try {
      BinaryReader reader{ctx.mpi_comm()};
      reader.set_domain(pfc::domain::get_size(domain), inbox.size, inbox.low);
      reader.read(get_filename(), field);
    } catch (const std::exception &ex) {
      std::ostringstream oss;
      oss << "FileReader failed to read \"" << get_filename() << "\" into field \""
          << get_field_name() << "\": " << ex.what();
      throw std::runtime_error(oss.str());
    }
  }


};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_FILE_READER_HPP
