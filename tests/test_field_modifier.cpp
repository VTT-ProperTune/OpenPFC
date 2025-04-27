// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/field_modifier.hpp>

using namespace pfc;

// Mock model class for testing
class MockModel : public Model {
public:
  bool is_modified = false;

  void step(double) override {}
  void initialize(double) override {}
};

// Mock field modifier class for testing
class MockFieldModifier : public FieldModifier {
public:
  void apply(Model &m, double) override {
    MockModel &mockModel = dynamic_cast<MockModel &>(m);
    mockModel.is_modified = true;
  }
};

TEST_CASE("FieldModifier applies field modification to the model", "[FieldModifier]") {
  MockModel model;
  MockFieldModifier modifier;

  double current_time = 0.0;
  modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier can be used polymorphically", "[FieldModifier]") {
  FieldModifier *modifier = new MockFieldModifier();
  MockModel model;

  double current_time = 0.0;
  modifier->apply(model, current_time);

  REQUIRE(model.is_modified);

  delete modifier;
}

TEST_CASE("FieldModifier can be moved", "[FieldModifier]") {
  MockModel model;
  MockFieldModifier modifier;

  double current_time = 0.0;
  MockFieldModifier moved_modifier = std::move(modifier);
  moved_modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("Field name can be set and retrieved", "[FieldModifier]") {
  MockFieldModifier modifier;
  // default field name is "default"
  REQUIRE(modifier.get_field_name() == "default");
  modifier.set_field_name("phi");
  REQUIRE(modifier.get_field_name() == "phi");
}
