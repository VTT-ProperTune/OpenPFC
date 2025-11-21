// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/field_modifier.hpp"
#include "openpfc/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;

// Extended mock model for testing field modifications
class MockModelWithModificationFlag : public pfc::testing::MockModel {
public:
  using pfc::testing::MockModel::MockModel;
  bool is_modified = false; // Track whether apply() was called
};

// Mock field modifier class for testing
class MockFieldModifier : public FieldModifier {
public:
  void apply(Model &m,
             double /*time*/) override { // Suppress unused parameter warning
    MockModelWithModificationFlag &mockModel =
        dynamic_cast<MockModelWithModificationFlag &>(m);
    mockModel.is_modified = true;
  };
};

TEST_CASE("FieldModifier - applies field modification to model",
          "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  MockModelWithModificationFlag model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  MockFieldModifier modifier;

  double current_time = 0.0;
  modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier - polymorphic usage", "[field_modifier][unit]") {
  FieldModifier *modifier = new MockFieldModifier();
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  MockModelWithModificationFlag model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  double current_time = 0.0;
  modifier->apply(model, current_time);

  REQUIRE(model.is_modified);

  delete modifier;
}

TEST_CASE("FieldModifier - move semantics", "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  MockModelWithModificationFlag model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  MockFieldModifier modifier;

  double current_time = 0.0;
  MockFieldModifier moved_modifier = std::move(modifier);
  moved_modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier - field name getter and setter", "[field_modifier][unit]") {
  MockFieldModifier modifier;
  // default field name is "default"
  REQUIRE(modifier.get_field_name() == "default");

  modifier.set_field_name("phi");
  REQUIRE(modifier.get_field_name() == "phi");
}

TEST_CASE("FieldModifier - MockModel integration", "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  REQUIRE(get_size(model.get_world()) == Int3{8, 8, 8});
}
