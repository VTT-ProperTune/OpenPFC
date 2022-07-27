#pragma once

#include "model.hpp"
#include "time.hpp"
#include "world.hpp"

namespace pfc {

template <class Model> class Simulator {
private:
  Model m_model;
  World m_world;
  Time m_time;

public:
  Simulator(const World &world, const Time &time)
      : m_model(world), m_world(world), m_time(time) {
    m_model.initialize(m_time.get_dt());
  }

  Model &get_model() { return m_model; }
  bool done() const { return m_time.done(); }
  int get_increment() const { return m_time.get_increments(); }
  double get_time() const { return m_time.get_current(); }

  void step() {
    m_time.next();
    m_model.step(m_time.get_dt());
    return;
  }
};

} // namespace pfc
