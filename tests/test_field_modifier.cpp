/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

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
