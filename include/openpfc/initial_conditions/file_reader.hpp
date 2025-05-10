// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_INITIAL_CONDITIONS_FILE_READER_HPP
#define PFC_INITIAL_CONDITIONS_FILE_READER_HPP

#include <iostream>

#include "../binary_reader.hpp"
#include "../field_modifier.hpp"

namespace pfc {

class FileReader : public FieldModifier {
private:
  std::string m_filename;

public:
  FileReader() = default;

  void set_filename(std::string filename) { m_filename = filename; }
  const std::string &get_filename() const { return m_filename; }

  explicit FileReader(const std::string &filename) : m_filename(filename) {}

  void apply(Model &m, double) override {
    const FFT &fft = m.get_fft();
    const auto &world = m.get_world();
    const auto world_size = get_size(world);
    const auto inbox_size = get_inbox(fft).size;
    const auto inbox_offset = get_inbox(fft).low;

    Field &f = m.get_real_field(get_field_name());
    std::cout << "Reading initial condition from file" << get_filename()
              << std::endl;
    BinaryReader reader;
    reader.set_domain(world_size, inbox_size, inbox_offset);
    reader.read(get_filename(), f);
  }
};

} // namespace pfc

#endif // PFC_INITIAL_CONDITIONS_FILE_READER_HPP
