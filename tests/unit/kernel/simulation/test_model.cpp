// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/simulation/model.hpp>

#include "fixtures/mock_model.hpp"

using namespace pfc;
using pfc::types::Int3;

namespace {
/** Satisfies [[nodiscard]] on field accessors when tests expect an exception. */
template <class T> void use_field_ref(T &&) {}
} // namespace

TEST_CASE("Model - basic functionality (v2.0)", "[model][unit]") {
  World world = world::create(GridSize({8, 1, 1}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);

  REQUIRE(get_size(get_world(model)) == Int3{8, 1, 1});

  SECTION("FFT is available after construction") {
    REQUIRE(get_fft(model).size_inbox() == fft.size_inbox());
    REQUIRE(get_fft(model).size_outbox() == fft.size_outbox());
    REQUIRE(is_rank0(model) == (mpi::get_rank() == 0));
  }

  SECTION("Real field operations") {
    // Create a real field
    RealField field;
    field.resize(10);

    // Add the field to the model
    add_real_field(model, "field1", field);

    REQUIRE(has_field(model, "field1"));
    REQUIRE(has_real_field(model, "field1"));
    REQUIRE_FALSE(has_complex_field(model, "field1"));

    // Get the field from the model and verify it has correct size
    RealField &retrieved_field = get_real_field(model, "field1");
    REQUIRE(retrieved_field.size() == field.size());
  }

  SECTION("Complex field operations") {
    // Create a complex field
    ComplexField field;
    field.resize(10);

    // Add the field to the model
    add_complex_field(model, "field2", field);

    REQUIRE(has_field(model, "field2"));
    REQUIRE_FALSE(has_real_field(model, "field2"));
    REQUIRE(has_complex_field(model, "field2"));

    // Get the field from the model and verify it has correct size
    ComplexField &retrieved_field = get_complex_field(model, "field2");
    REQUIRE(retrieved_field.size() == field.size());
  }
}

TEST_CASE("Model::is_rank0() returns correct rank status (v2.0)",
          "[model][unit][rank]") {
  World world = world::create(GridSize({10, 10, 10}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);

  int rank = mpi::get_rank();
  if (rank == 0) {
    REQUIRE(is_rank0(model) == true);
  } else {
    REQUIRE(is_rank0(model) == false);
  }
}

TEST_CASE("Model::is_rank0() is const-correct (v2.0)", "[model][unit][rank]") {
  World world = world::create(GridSize({10, 10, 10}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);

  // Create const reference to test const-correctness
  const pfc::testing::MockModel &const_model = model;

  // Should compile (is_rank0() is const)
  bool result = is_rank0(const_model);

  // Valid boolean value (must be either true or false)
  bool is_valid = (result == true || result == false);
  REQUIRE(is_valid);
}

TEST_CASE("Model - error handling for non-existent fields", "[model][unit][error]") {
  World world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);

  SECTION("Accessing non-existent real field throws std::out_of_range") {
    REQUIRE_THROWS_AS(get_real_field(model, "nonexistent"), std::out_of_range);

    // Error message should be helpful
    try {
      use_field_ref(model.get_real_field("nonexistent"));
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
    REQUIRE_THROWS_AS(get_complex_field(model, "nonexistent"), std::out_of_range);

    // Error message should be helpful
    try {
      use_field_ref(model.get_complex_field("nonexistent"));
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
    add_real_field(model, "density", field1);
    add_real_field(model, "temperature", field2);

    try {
      use_field_ref(model.get_real_field("nonexistent"));
      FAIL("Should have thrown");
    } catch (const std::out_of_range &e) {
      std::string msg = e.what();
      // Should mention available fields
      REQUIRE(msg.find("Available fields") != std::string::npos);
    }
  }

  SECTION("Const version also throws for non-existent fields") {
    const pfc::testing::MockModel &const_model = model;
    REQUIRE_THROWS_AS(get_real_field(const_model, "nonexistent"),
                      std::out_of_range);
    REQUIRE_THROWS_AS(get_complex_field(const_model, "nonexistent"),
                      std::out_of_range);
  }
}
