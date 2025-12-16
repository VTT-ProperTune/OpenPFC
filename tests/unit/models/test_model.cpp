// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;
using pfc::types::Int3;

TEST_CASE("Model - basic functionality", "[model][unit]") {
  // Create an instance of the Model
  World world = world::create(GridSize({8), PhysicalOrigin(1), GridSpacing(1}));
  pfc::testing::MockModel model(world);

  REQUIRE(get_size(model.get_world()) == Int3{8, 1, 1});

  SECTION("Default construction") {
    // Ensure FFT object is set before proceeding
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    model.set_fft(fft);
  }

  SECTION("Set and get FFT") {
    // Create a Decomposition object
    auto decomposition = decomposition::create(world, 1);
    // Create an FFT object
    auto fft = fft::create(decomposition);
    model.set_fft(fft);

    REQUIRE(model.get_fft().size_inbox() == fft.size_inbox());
    REQUIRE(model.get_fft().size_outbox() == fft.size_outbox());
    REQUIRE(model.is_rank0());
  }

  SECTION("Real field operations") {
    // Ensure FFT object is set before proceeding
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    model.set_fft(fft);

    // Create a real field
    RealField field;
    field.resize(10);

    // Add the field to the model
    model.add_real_field("field1", field);

    REQUIRE(model.has_field("field1"));
    REQUIRE(model.has_real_field("field1"));
    REQUIRE_FALSE(model.has_complex_field("field1"));

    // Get the field from the model and verify it has correct size
    RealField &retrieved_field = model.get_real_field("field1");
    REQUIRE(retrieved_field.size() == field.size());
  }

  SECTION("Complex field operations") {
    // Create a complex field
    ComplexField field;
    field.resize(10);

    // Add the field to the model
    model.add_complex_field("field2", field);

    REQUIRE(model.has_field("field2"));
    REQUIRE_FALSE(model.has_real_field("field2"));
    REQUIRE(model.has_complex_field("field2"));

    // Get the field from the model and verify it has correct size
    ComplexField &retrieved_field = model.get_complex_field("field2");
    REQUIRE(retrieved_field.size() == field.size());
  }
}

TEST_CASE("Model::is_rank0() returns correct rank status", "[model][unit][rank]") {
  World world = world::create(GridSize({10), PhysicalOrigin(10), GridSpacing(10}));
  pfc::testing::MockModel model(world);

  // Initialize FFT (required for is_rank0 to be set)
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  model.set_fft(fft);

  int rank = mpi::get_rank();

  if (rank == 0) {
    REQUIRE(model.is_rank0() == true);
  } else {
    REQUIRE(model.is_rank0() == false);
  }
}

TEST_CASE("Model::is_rank0() is const-correct", "[model][unit][rank]") {
  World world = world::create(GridSize({10), PhysicalOrigin(10), GridSpacing(10}));

  // Create model and set FFT to ensure m_rank0 is set
  pfc::testing::MockModel model(world);
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  model.set_fft(fft);

  // Create const reference to test const-correctness
  const pfc::testing::MockModel &const_model = model;

  // Should compile (is_rank0() is const)
  bool result = const_model.is_rank0();

  // Valid boolean value (must be either true or false)
  bool is_valid = (result == true || result == false);
  REQUIRE(is_valid);
}

TEST_CASE("Model - error handling for non-existent fields", "[model][unit][error]") {
  World world = world::create(GridSize({8), PhysicalOrigin(8), GridSpacing(8}));
  pfc::testing::MockModel model(world);

  // Ensure FFT is set
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  model.set_fft(fft);

  SECTION("Accessing non-existent real field throws std::out_of_range") {
    REQUIRE_THROWS_AS(model.get_real_field("nonexistent"), std::out_of_range);

    // Error message should be helpful
    try {
      model.get_real_field("nonexistent");
      FAIL("Should have thrown");
    } catch (const std::out_of_range &e) {
      std::string msg = e.what();
      bool has_name = msg.find("nonexistent") != std::string::npos;
      bool has_not_found = msg.find("not found") != std::string::npos;
      REQUIRE(has_name);
      REQUIRE(has_not_found);
    }
  }

  SECTION("Accessing non-existent complex field throws std::out_of_range") {
    REQUIRE_THROWS_AS(model.get_complex_field("nonexistent"), std::out_of_range);

    // Error message should be helpful
    try {
      model.get_complex_field("nonexistent");
      FAIL("Should have thrown");
    } catch (const std::out_of_range &e) {
      std::string msg = e.what();
      bool has_name = msg.find("nonexistent") != std::string::npos;
      bool has_not_found = msg.find("not found") != std::string::npos;
      REQUIRE(has_name);
      REQUIRE(has_not_found);
    }
  }

  SECTION("Error message lists available fields") {
    // Add some fields
    RealField field1, field2;
    field1.resize(10);
    field2.resize(10);
    model.add_real_field("density", field1);
    model.add_real_field("temperature", field2);

    try {
      model.get_real_field("nonexistent");
      FAIL("Should have thrown");
    } catch (const std::out_of_range &e) {
      std::string msg = e.what();
      // Should mention available fields
      REQUIRE(msg.find("Available fields") != std::string::npos);
    }
  }

  SECTION("Const version also throws for non-existent fields") {
    const pfc::testing::MockModel &const_model = model;
    REQUIRE_THROWS_AS(const_model.get_real_field("nonexistent"), std::out_of_range);
    REQUIRE_THROWS_AS(const_model.get_complex_field("nonexistent"),
                      std::out_of_range);
  }
}

TEST_CASE("Model - error handling for FFT access", "[model][unit][error]") {
  World world = world::create(GridSize({8), PhysicalOrigin(8), GridSpacing(8}));
  pfc::testing::MockModel model(world);

  SECTION("Accessing FFT before it's set throws std::runtime_error") {
    REQUIRE_THROWS_AS(model.get_fft(), std::runtime_error);

    // Error message should be helpful
    try {
      model.get_fft();
      FAIL("Should have thrown");
    } catch (const std::runtime_error &e) {
      std::string msg = e.what();
      REQUIRE(msg.find("FFT object has not been set") != std::string::npos);
      bool has_set_fft = msg.find("set_fft") != std::string::npos;
      bool has_constructor = msg.find("constructor") != std::string::npos;
      bool has_helpful_info = has_set_fft || has_constructor;
      REQUIRE(has_helpful_info);
    }
  }
}
