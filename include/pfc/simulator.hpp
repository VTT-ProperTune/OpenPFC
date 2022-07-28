#pragma once

#include "field_modifier.hpp"
#include "model.hpp"
#include "results_writer.hpp"
#include "time.hpp"
#include "world.hpp"
#include <memory>

namespace pfc {

template <class Model> class Simulator {

private:
  Model m_model;
  World m_world;
  Time m_time;
  std::vector<std::unique_ptr<ResultsWriter>> m_result_writers;
  std::vector<std::unique_ptr<FieldModifier>> m_initial_conditions;
  int m_result_counter = 0;

public:
  Simulator(const World &world, const Time &time)
      : m_model(world), m_world(world), m_time(time) {
    m_model.initialize(m_time.get_dt());
  }

  Model &get_model() { return m_model; }
  bool done() const { return m_time.done(); }
  int get_increment() const { return m_time.get_increments(); }
  double get_time() const { return m_time.get_current(); }
  bool is_first_increment() const { return get_increment() == 0; }

  void add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    writer->set_domain(m_world.get_size(), m_model.get_inbox_size(),
                       m_model.get_inbox_low());
    m_result_writers.push_back(std::move(writer));
  }

  void add_initial_conditions(std::unique_ptr<FieldModifier> modifier) {
    m_initial_conditions.push_back(std::move(modifier));
  }

  void write_results(int filenum) {
    Model &m = get_model();
    std::vector<double> &field = m.get_field();
    for (const auto &writer : m_result_writers) {
      writer->write(filenum, field);
    }
  }

  void prestep_first_increment() {
    for (const auto &modifier : m_initial_conditions) {
      modifier->apply(get_model(), get_model().get_field(), get_time());
    }
    if (m_time.do_save()) {
      write_results(m_result_counter++);
    }
  }

  void step() {
    if (is_first_increment()) {
      prestep_first_increment();
    }
    m_time.next();
    m_model.step(m_time.get_dt());
    if (m_time.do_save()) {
      write_results(m_result_counter++);
    }
    return;
  }
};

} // namespace pfc
