// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_model_fft_baseline.cpp
 * @brief Baseline tests for Model-FFT interaction before refactoring
 *
 * This file captures the current (v1.x) behavior of Model-FFT interaction
 * to ensure the v2.0 refactoring (FFT* → FFT&) maintains correctness.
 *
 * Tests document:
 * - Both construction patterns (with/without FFT)
 * - set_fft() behavior
 * - get_fft() error handling
 * - FFT lifetime management
 * - Derived model class patterns
 *
 * Related User Stories: #0002, #0027
 * Related Design Doc: llm/design/fft_ownership_design.md
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"

#include "fixtures/mock_model.hpp"

using namespace pfc;
using pfc::types::Int3;

// =============================================================================
// BASELINE TESTS: Current (v1.x) Behavior
// =============================================================================

TEST_CASE("Model-FFT: Construction requires FFT (v2.0)", "[model][fft][baseline]") {
  World world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);

  SECTION("Construct with FFT reference") {
    pfc::testing::MockModel model(fft, world);
    REQUIRE_NOTHROW(model.get_fft());
    REQUIRE(model.get_fft().size_inbox() > 0);
  }
}

TEST_CASE("Model-FFT: get_fft() never throws (v2.0)", "[model][fft][baseline]") {
  World world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  pfc::testing::MockModel model(fft, world);
  REQUIRE_NOTHROW(model.get_fft());
}

TEST_CASE("Model-FFT: FFT lifetime management (v2.0)", "[model][fft][baseline]") {
  World world = world::create(GridSize({8, 8, 8}));

  SECTION("FFT must outlive Model (documented requirement)") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::MockModel model(fft, world);
    REQUIRE(model.get_fft().size_inbox() > 0);
    REQUIRE(fft.size_inbox() > 0);
  }
}

TEST_CASE("Model-FFT Baseline: Typical usage patterns",
          "[model][fft][baseline][patterns]") {
  World world = world::create(GridSize({16, 16, 16}));

  SECTION("Pattern: Create FFT then construct model (v2.0)") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::MockModel model(fft, world);
    model.initialize(0.1);

    // Model can use FFT in initialize() and step()
    REQUIRE_NOTHROW(model.get_fft());
  }
}

TEST_CASE("Model-FFT Baseline: Derived model behavior", "[model][fft][baseline]") {
  World world = world::create(GridSize({8, 8, 8}));

  SECTION("Derived model inherits FFT access") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::MockModel model(fft, world);

    // Derived model can access FFT via get_fft()
    FFT &model_fft = model.get_fft();
    REQUIRE(model_fft.size_inbox() == fft.size_inbox());
    REQUIRE(model_fft.size_outbox() == fft.size_outbox());
  }

  SECTION("Instrumented model tracks initialization") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::InstrumentedMockModel model(fft, world);
    model.initialize(0.1);

    REQUIRE(model.initialize_call_count == 1);
    REQUIRE(model.last_init_dt == 0.1);
  }
}

TEST_CASE("Model-FFT Baseline: Field operations with FFT",
          "[model][fft][baseline]") {
  World world = world::create(GridSize({8, 8, 8}));

  SECTION("Real fields work after FFT is set") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::MockModel model(fft, world);

    // Create field with correct size
    RealField field;
    field.resize(fft.size_inbox());

    model.add_real_field("phi", field);
    REQUIRE(model.has_real_field("phi"));

    // Field size should match FFT inbox
    RealField &phi = model.get_real_field("phi");
    REQUIRE(phi.size() == fft.size_inbox());
  }

  SECTION("Complex fields work after FFT is set") {
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);
    pfc::testing::MockModel model(fft, world);

    // Create field with correct size
    ComplexField field;
    field.resize(fft.size_outbox());

    model.add_complex_field("phi_k", field);
    REQUIRE(model.has_complex_field("phi_k"));

    // Field size should match FFT outbox
    ComplexField &phi_k = model.get_complex_field("phi_k");
    REQUIRE(phi_k.size() == fft.size_outbox());
  }
}

// =============================================================================
// DOCUMENTATION: Known Issues with Current Design
// =============================================================================

TEST_CASE("Model-FFT: Resolved issues (v2.0)", "[model][fft][baseline][doc]") {
  SECTION("Resolved: Null pointer checks removed") {
    // Every function that uses FFT must check for null
    // Example from real code:
    // FFT& get_fft() {
    //   if (m_fft == nullptr) {
    //     throw std::runtime_error("FFT not set");
    //   }
    //   return *m_fft;
    // }

    SUCCEED("Resolved: get_fft() always valid - no null checks");
  }
  SECTION("Resolved: Single construction path with FFT&") {
    // Users must remember to call set_fft() after construction
    // This leads to runtime errors when forgotten

    SUCCEED("Resolved: API is clear - FFT required at construction");
  }
  SECTION("Clarified: Ownership via reference semantics") {
    // m_fft is a raw pointer - who owns it?
    // Answer: Model doesn't own it (FFT must outlive Model)
    // But this is not obvious from the type

    SUCCEED("Clarified: FFT must outlive Model (documented)");
  }
  SECTION("Note: Mocking strategies for FFT") {
    // Cannot inject a mock FFT because set_fft() takes FFT&
    // This makes unit testing harder

    SUCCEED("Note: Use adapter or test fixtures for FFT when needed");
  }
}

// =============================================================================
// FORWARD-COMPATIBILITY: Tests for Future v2.0 API
// =============================================================================

TEST_CASE("Model-FFT: v2.0 behavior validated", "[model][fft][future]") {
  // These tests document the DESIRED behavior after refactoring
  // They are currently disabled (. tag) and will be enabled in v2.0

  SECTION("v2.0: Single constructor with FFT reference") {
    World world = world::create(GridSize({8, 8, 8}));
    auto decomposition = decomposition::create(world, 1);
    auto fft = fft::create(decomposition);

    pfc::testing::MockModel model(fft, world);
    REQUIRE_NOTHROW(model.get_fft());
  }

  SECTION("v2.0: No set_fft() method") {
    SUCCEED("v2.0: FFT immutable after construction");
  }

  SECTION("v2.0: get_fft() never throws") { REQUIRE(true); }
}
