#pragma once

#include "field_modifier.hpp"
#include "model.hpp"
#include "results_writer.hpp"
#include "time.hpp"
#include "world.hpp"
#include <memory>

namespace pfc {

class Simulator {

private:
  World &m_world;
  Decomposition &m_decomposition;
  FFT &m_fft;
  Model &m_model;
  Time &m_time;

  std::vector<std::unique_ptr<ResultsWriter>> m_result_writers;
  std::vector<std::unique_ptr<FieldModifier>> m_initial_conditions;
  std::vector<std::unique_ptr<FieldModifier>> m_boundary_conditions;
  int m_result_counter = 0;

public:
  Simulator(World &world, Decomposition &decomposition, FFT &fft, Model &model,
            Time &time)
      : m_world(world), m_decomposition(decomposition), m_fft(fft),
        m_model(model), m_time(time) {}

  World &get_world() { return m_world; }
  Decomposition &get_decomposition() { return m_decomposition; }
  FFT &get_fft() { return m_fft; }
  Model &get_model() { return m_model; }
  Time &get_time() { return m_time; }
  Field &get_field() { return get_model().get_field(); }

  void add_results_writer(std::unique_ptr<ResultsWriter> writer) {
    Decomposition &d = get_decomposition();
    writer->set_domain(d.world.size, d.inbox.size, d.inbox.low);
    m_result_writers.push_back(std::move(writer));
  }

  void add_initial_conditions(std::unique_ptr<FieldModifier> modifier) {
    m_initial_conditions.push_back(std::move(modifier));
  }

  void add_boundary_conditions(std::unique_ptr<FieldModifier> modifier) {
    m_boundary_conditions.push_back(std::move(modifier));
  }

  void set_result_counter(int result_counter) {
    m_result_counter = result_counter;
  }

  double get_result_counter() const { return m_result_counter; }

  void write_results() {
    int file_num = get_result_counter();
    Field &field = get_field();
    for (const auto &writer : m_result_writers) {
      writer->write(file_num, field);
    }
    set_result_counter(file_num + 1);
  }

  void apply_initial_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    for (const auto &ic : m_initial_conditions) {
      ic->apply(model, time.get_current());
    }
  }

  void apply_boundary_conditions() {
    Model &model = get_model();
    Time &time = get_time();
    for (const auto &bc : m_boundary_conditions) {
      bc->apply(model, time.get_current());
    }
  }

  void step() {
    Time &time = get_time();
    Model &model = get_model();
    if (time.get_increment() == 0) {
      apply_initial_conditions();
      apply_boundary_conditions();
      if (time.do_save()) {
        write_results();
      }
    }
    time.next();
    apply_boundary_conditions();
    model.step(time.get_current());
    if (time.do_save()) {
      write_results();
    }
    return;
  }

  bool done() {
    Time &time = get_time();
    return time.done();
  }
};

} // namespace pfc
