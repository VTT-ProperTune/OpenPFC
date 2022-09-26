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
  virtual Field &get_field() = 0;
};
} // namespace pfc
