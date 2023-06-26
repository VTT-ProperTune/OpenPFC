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
    Field &f = m.get_field();
    std::cout << "Reading initial condition from file" << get_filename()
              << std::endl;
    BinaryReader reader;
    reader.set_domain(d.get_world().get_size(), d.inbox.size, d.inbox.low);
    reader.read(get_filename(), f);
  }
};

}  // namespace pfc
