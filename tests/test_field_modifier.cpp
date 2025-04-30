// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/field_modifier.hpp"
#include "openpfc/model.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

using namespace pfc;

// Mock model class for testing
class MockModel : public Model {
public:
  bool is_modified = false; // Add this member to track modifications

  MockModel(const pfc::World &world) : pfc::Model(world) {}
  void step(double /*t*/) override {} // Suppress unused parameter warning
  void initialize(double /*dt*/) override {
  } // Suppress unused parameter warning
};

// Mock field modifier class for testing
class MockFieldModifier : public FieldModifier {
public:
  void apply(Model &m,
             double /*time*/) override { // Suppress unused parameter warning
    MockModel &mockModel = dynamic_cast<MockModel &>(m);
    mockModel.is_modified = true;
  };
};

TEST_CASE("FieldModifier applies field modification to the model",
          "[FieldModifier]") {
  World world = create_world({8, 8, 8});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD,
          heffte::default_options<heffte::backend::fftw>(), world);
  MockModel model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  MockFieldModifier modifier;

  double current_time = 0.0;
  modifier.apply(model, current_time);

  REQUIRE(model.is_modified);
}

TEST_CASE("FieldModifier can be used polymorphically", "[FieldModifier]") {
  FieldModifier *modifier = new MockFieldModifier();
  World world = create_world({8, 8, 8});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD,
          heffte::default_options<heffte::backend::fftw>(), world);
  MockModel model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  double current_time = 0.0;
  modifier->apply(model, current_time);

  REQUIRE(model.is_modified);

  delete modifier;
}

TEST_CASE("FieldModifier can be moved", "[FieldModifier]") {
  World world = create_world({8, 8, 8});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD,
          heffte::default_options<heffte::backend::fftw>(), world);
  MockModel model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

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

TEST_CASE("Field Modifier - MockModel Integration", "[field_modifier]") {
  World world = create_world({8, 8, 8});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD,
          heffte::default_options<heffte::backend::fftw>(), world);
  MockModel model(world);
  model.set_fft(fft); // Ensure FFT object is set

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  REQUIRE(model.get_world().get_size() == World::Int3{8, 8, 8});
}
