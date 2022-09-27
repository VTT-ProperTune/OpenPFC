#pragma once

#include "fft.hpp"
#include "types.hpp"
#include "world.hpp"

namespace pfc {
class Model {

private:
  World &m_world;
  Decomposition &m_decomposition;
  FFT &m_fft;
  RealFieldSet m_real_fields;
  ComplexFieldSet m_complex_fields;

public:
  const bool rank0;

  Model(World &world, Decomposition &decomposition, FFT &fft)
      : m_world(world), m_decomposition(decomposition), m_fft(fft),
        rank0(m_decomposition.get_id() == 0) {}

  World &get_world() { return m_world; }
  Decomposition &get_decomposition() { return m_decomposition; }
  FFT &get_fft() { return m_fft; }

  /* methods that need to override for concrete implementations */
  virtual void step(double t) = 0;
  virtual void initialize(double dt) = 0;

  bool has_real_field(const std::string &field_name) {
    return m_real_fields.count(field_name) > 0;
  }

  void add_real_field(const std::string &name, RealField &field) {
    m_real_fields.insert({name, field});
  }

  bool has_complex_field(const std::string &field_name) {
    return m_complex_fields.count(field_name) > 0;
  }

  void add_complex_field(const std::string &name, ComplexField &field) {
    m_complex_fields.insert({name, field});
  }

  RealField &get_real_field(const std::string &name) {
    return m_real_fields.find(name)->second;
  }

  ComplexField &get_complex_field(const std::string &name) {
    return m_complex_fields.find(name)->second;
  }

  bool has_field(const std::string &field_name) {
    return has_real_field(field_name) || has_complex_field(field_name);
  }

  virtual Field &get_field() { return get_real_field("default"); };
};
} // namespace pfc
