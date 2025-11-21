// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/field_modifier.hpp"
#include "openpfc/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;

TEST_CASE("FieldModifier - applies field modification to model",
          "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModelWithModificationFlag model(world);
  model.set_fft(fft);

  pfc::testing::MockFieldModifier modifier;

  double current_time = 0.0;
  modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier - polymorphic usage", "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModelWithModificationFlag model(world);
  model.set_fft(fft);

  std::unique_ptr<FieldModifier> modifier =
      std::make_unique<pfc::testing::MockFieldModifier>();

  double current_time = 0.0;
  modifier->apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier - move semantics", "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModelWithModificationFlag model(world);
  model.set_fft(fft);

  pfc::testing::MockFieldModifier modifier;

  double current_time = 0.0;
  pfc::testing::MockFieldModifier moved_modifier = std::move(modifier);
  moved_modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier - field name getter and setter", "[field_modifier][unit]") {
  pfc::testing::MockFieldModifier modifier;
  // default field name is "default"
  REQUIRE(modifier.get_field_name() == "default");

  modifier.set_field_name("phi");
  REQUIRE(modifier.get_field_name() == "phi");
}

TEST_CASE("FieldModifier - works with MockModel", "[field_modifier][unit]") {
  auto world = world::create({8, 8, 8});
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(world);
  model.set_fft(fft);

  REQUIRE(get_size(model.get_world()) == Int3{8, 8, 8});
}
