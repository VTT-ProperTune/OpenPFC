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
  World world = world::create({8, 1, 1});
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
  World world = world::create({10, 10, 10});
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
  World world = world::create({10, 10, 10});

  // Create model and set FFT to ensure m_rank0 is set
  pfc::testing::MockModel model(world);
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  model.set_fft(fft);

  // Create const reference to test const-correctness
  const pfc::testing::MockModel &const_model = model;

  // Should compile (is_rank0() is const)
  bool result = const_model.is_rank0();

  // Valid boolean value
  REQUIRE((result == true || result == false));
}
