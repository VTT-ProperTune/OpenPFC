// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file mock_model.hpp
 * @brief Mock implementations of Model for unit testing
 *
 * This file provides reusable mock Model implementations to avoid
 * code duplication across test files. All tests that need a Model
 * instance should use these fixtures instead of creating their own.
 */

#pragma once

#include "openpfc/field_modifier.hpp"
#include "openpfc/model.hpp"

namespace pfc {
namespace testing {

/**
 * @brief Basic mock Model implementation for unit testing
 *
 * Minimal Model implementation that does nothing in step() and initialize().
 * Use this when you need a Model instance but don't care about its behavior.
 *
 * @code
 * auto world = world::create({8, 8, 8});
 * pfc::testing::MockModel model(world);
 * // Use model in your test...
 * @endcode
 */
class MockModel : public Model {
public:
  /**
   * @brief Construct a MockModel with the given world
   * @param world The World object defining the simulation domain
   */
  explicit MockModel(const World &world) : Model(world) {}

  /**
   * @brief Mock step() implementation (does nothing)
   * @param t Current simulation time (unused)
   */
  void step(double /*t*/) override {}

  /**
   * @brief Mock initialize() implementation (does nothing)
   * @param dt Time step size (unused)
   */
  void initialize(double /*dt*/) override {}
};

/**
 * @brief Instrumented mock Model that tracks method calls
 *
 * Use this when you need to verify that certain Model methods were called
 * with specific parameters. Tracks call counts and parameters.
 *
 * @code
 * auto world = world::create({8, 8, 8});
 * pfc::testing::InstrumentedMockModel model(world);
 *
 * model.step(1.0);
 * model.step(2.0);
 *
 * REQUIRE(model.step_call_count == 2);
 * REQUIRE(model.last_step_time == 2.0);
 * @endcode
 */
class InstrumentedMockModel : public MockModel {
public:
  using MockModel::MockModel;

  /**
   * @brief Instrumented step() - tracks calls and parameters
   * @param t Current simulation time
   */
  void step(double t) override {
    step_call_count++;
    last_step_time = t;
  }

  /**
   * @brief Instrumented initialize() - tracks calls and parameters
   * @param dt Time step size
   */
  void initialize(double dt) override {
    initialize_call_count++;
    last_init_dt = dt;
  }

  /// Number of times step() was called
  int step_call_count = 0;

  /// Number of times initialize() was called
  int initialize_call_count = 0;

  /// Last time value passed to step()
  double last_step_time = 0.0;

  /// Last dt value passed to initialize()
  double last_init_dt = 0.0;
};

/**
 * @brief Mock Model with modification flag for FieldModifier testing
 *
 * Extended mock that tracks whether a FieldModifier's apply() method
 * was called. Use this to test FieldModifier implementations.
 *
 * @code
 * pfc::testing::MockModelWithModificationFlag model(world);
 * MockFieldModifier modifier;
 * modifier.apply(model, 0.0);
 * REQUIRE(model.is_modified);
 * @endcode
 */
class MockModelWithModificationFlag : public MockModel {
public:
  using MockModel::MockModel;

  /// Flag indicating whether a field modifier was applied
  bool is_modified = false;
};

/**
 * @brief Mock FieldModifier for testing
 *
 * Simple FieldModifier that sets the modification flag on
 * MockModelWithModificationFlag when applied.
 *
 * @code
 * pfc::testing::MockFieldModifier modifier;
 * pfc::testing::MockModelWithModificationFlag model(world);
 * modifier.apply(model, 0.0);
 * REQUIRE(model.is_modified);
 * @endcode
 */
class MockFieldModifier : public FieldModifier {
public:
  /**
   * @brief Apply mock modification (sets is_modified flag)
   * @param m Model to modify (must be MockModelWithModificationFlag)
   * @param time Current simulation time (unused)
   */
  void apply(Model &m, double /*time*/) override {
    auto &mock_model = dynamic_cast<MockModelWithModificationFlag &>(m);
    mock_model.is_modified = true;
  }
};

/**
 * @brief Mock initial condition that fills a field with a constant value
 *
 * Simple FieldModifier for testing initial conditions. Sets all values
 * in the target field to 1.0.
 *
 * @code
 * pfc::testing::MockIC ic;
 * ic.set_field_name("phi");
 * ic.apply(model, 0.0);
 * // model's "phi" field is now filled with 1.0
 * @endcode
 */
class MockIC : public FieldModifier {
public:
  /**
   * @brief Fill target field with 1.0
   * @param m Model containing the field
   * @param time Current simulation time (unused)
   */
  void apply(Model &m, double /*time*/) override {
    std::vector<double> &field = m.get_real_field(get_field_name());
    std::fill(field.begin(), field.end(), 1.0);
  }
};

} // namespace testing
} // namespace pfc
