/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#pragma once

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
    const Decomposition &d = m.get_decomposition();
    Field &f = m.get_real_field(get_field_name());
    std::cout << "Reading initial condition from file" << get_filename() << std::endl;
    BinaryReader reader;
    reader.set_domain(d.get_world().get_size(), d.inbox.size, d.inbox.low);
    reader.read(get_filename(), f);
  }
};

} // namespace pfc
