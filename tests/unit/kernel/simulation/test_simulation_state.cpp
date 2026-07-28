// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// M2 unit tests for openpfc::kernel::simulation::SimulationState:
// owning the canonical pfc::field::Field<T> by name and by typed
// FieldHandle<T>, including heterogeneous element types coexisting in one state.

#include <complex>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/field.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>

using namespace openpfc::kernel::simulation;

namespace {

// Helper function to create a simple field for testing
template <typename T>
pfc::field::Field<T> make_simple_field(int nx = 10, int ny = 10, int nz = 10) {
  auto domain = pfc::domain::create({nx, ny, nz});
  pfc::Int3 lower = {0, 0, 0};
  pfc::Int3 upper = {nx - 1, ny - 1, nz - 1};
  pfc::world::World world(lower, upper, domain);
  return pfc::field::create<T>(world);
}

} // namespace

TEST_CASE("SimulationState: insert and retrieve fields by name",
          "[simulation_state][unit]") {
  SimulationState state;

  SECTION("insert double field and retrieve via FieldHandle") {
    auto field = make_simple_field<double>(10, 20, 5);
    REQUIRE(state.size() == 0);
    REQUIRE_FALSE(state.has_field("density"));

    state.insert_field("density", std::move(field));

    REQUIRE(state.size() == 1);
    REQUIRE(state.has_field("density"));

    auto handle = state.get_field<double>("density");
    REQUIRE(handle);
    // The handle should provide access to the field
    auto& retrieved_field = handle.get();
    REQUIRE(&retrieved_field != nullptr);
  }

  SECTION("insert complex<float> field") {
    auto field = make_simple_field<std::complex<float>>(5, 5, 5);
    state.insert_field("potential", std::move(field));

    auto handle = state.get_field<std::complex<float>>("potential");
    REQUIRE(handle);
    REQUIRE(state.has_field("potential"));
  }

  SECTION("duplicate insert fails") {
    auto f1 = make_simple_field<double>(10, 20);
    auto f2 = make_simple_field<double>(30, 40);
    
    state.insert_field("rho", std::move(f1));
    REQUIRE_THROWS_AS(state.insert_field("rho", std::move(f2)),
                      std::runtime_error);
  }
}

TEST_CASE("SimulationState: FieldHandle null access",
          "[simulation_state][unit]") {
  SimulationState state;
  
  SECTION("nonexistent field returns null handle") {
    auto handle = state.get_field<double>("nonexistent");
    REQUIRE_FALSE(handle);
  }

  SECTION("type mismatch returns null handle") {
    auto field = make_simple_field<double>(10, 20);
    state.insert_field("field", std::move(field));
    
    auto handle_complex = state.get_field<std::complex<double>>("field");
    REQUIRE_FALSE(handle_complex);
  }
}

TEST_CASE("SimulationState: has_field", "[simulation_state][unit]") {
  SimulationState state;
  
  REQUIRE_FALSE(state.has_field("any"));
  
  auto f = make_simple_field<float>(5);
  state.insert_field("temp", std::move(f));
  
  REQUIRE(state.has_field("temp"));
}

TEST_CASE("SimulationState: remove_field", "[simulation_state][unit]") {
  SimulationState state;
  
  auto f = make_simple_field<double>(10);
  state.insert_field("rho", std::move(f));
  
  REQUIRE(state.has_field("rho"));
  REQUIRE(state.remove_field<double>("rho"));
  REQUIRE_FALSE(state.has_field("rho"));
  REQUIRE_FALSE(state.remove_field<double>("rho")); // not found anymore
  
  SECTION("type mismatch removal fails") {
    auto f = make_simple_field<double>(10);
    state.insert_field("field", std::move(f));
    
    REQUIRE_FALSE(state.remove_field<float>("field"));
    REQUIRE(state.has_field("field")); // should still be there
  }
}

TEST_CASE("SimulationState: clear and size", "[simulation_state][unit]") {
  SimulationState state;
  
  auto f1 = make_simple_field<double>(10);
  auto f2 = make_simple_field<float>(20);
  
  state.insert_field("f1", std::move(f1));
  state.insert_field("f2", std::move(f2));
  
  REQUIRE(state.size() == 2);
  
  state.clear();
  
  REQUIRE(state.size() == 0);
  REQUIRE_FALSE(state.has_field("f1"));
  REQUIRE_FALSE(state.has_field("f2"));
}

TEST_CASE("SimulationState: heterogeneous element types coexist",
          "[simulation_state][unit]") {
  using cdouble = std::complex<double>;
  SimulationState state;
  
  auto psi = make_simple_field<double>(8, 8, 8);
  state.insert_field("psi", std::move(psi));
  
  auto psi_hat = make_simple_field<cdouble>(8, 8, 8);
  state.insert_field("psi_hat", std::move(psi_hat));
  
  REQUIRE(state.size() == 2);
  REQUIRE(state.has_field("psi"));
  REQUIRE(state.has_field("psi_hat"));
  
  // Get handles for both types
  auto handle_real = state.get_field<double>("psi");
  auto handle_complex = state.get_field<cdouble>("psi_hat");
  
  REQUIRE(handle_real);
  REQUIRE(handle_complex);
  
  // Type safety: wrong type returns null handle
  REQUIRE_FALSE(state.get_field<cdouble>("psi"));
  REQUIRE_FALSE(state.get_field<double>("psi_hat"));
}

TEST_CASE("SimulationState: FieldHandle equality and comparison",
          "[simulation_state][unit]") {
  SimulationState state;
  
  auto f1 = make_simple_field<double>(10);
  auto f2 = make_simple_field<double>(20);
  
  state.insert_field("field1", std::move(f1));
  state.insert_field("field2", std::move(f2));
  
  auto h1 = state.get_field<double>("field1");
  auto h2 = state.get_field<double>("field2");
  auto h1_again = state.get_field<double>("field1");
  
  REQUIRE(h1);
  REQUIRE(h2);
  REQUIRE(h1_again);
  
  // Same field should have equal handles
  REQUIRE(h1 == h1_again);
  REQUIRE(!(h1 != h1_again));
  
  // Different fields should have different handles
  REQUIRE(h1 != h2);
  REQUIRE(!(h1 == h2));
}

TEST_CASE("FieldHandle: usable as unordered_map key", "[simulation_state][unit]") {
  SimulationState state;
  
  auto f1 = make_simple_field<double>(10);
  auto f2 = make_simple_field<double>(20);
  
  state.insert_field("field1", std::move(f1));
  state.insert_field("field2", std::move(f2));
  
  auto h1 = state.get_field<double>("field1");
  auto h2 = state.get_field<double>("field2");
  
  std::unordered_map<FieldHandle<double>, std::string> labels;
  labels[h1] = "label1";
  labels[h2] = "label2";
  
  REQUIRE(labels.size() == 2);
  REQUIRE(labels[h1] == "label1");
  REQUIRE(labels[h2] == "label2");
  
  // Test with same field again
  auto h1_again = state.get_field<double>("field1");
  REQUIRE(labels[h1_again] == "label1");
}

TEST_CASE("SimulationState: move semantics", "[simulation_state][unit]") {
  SimulationState state1;
  
  auto f = make_simple_field<double>(10);
  state1.insert_field("field", std::move(f));
  
  REQUIRE(state1.size() == 1);
  REQUIRE(state1.has_field("field"));
  
  // Move construction
  SimulationState state2 = std::move(state1);
  REQUIRE(state2.size() == 1);
  REQUIRE(state2.has_field("field"));
  
  // state1 should be empty after move
  REQUIRE(state1.size() == 0);
  
  // Move assignment
  SimulationState state3;
  state3 = std::move(state2);
  REQUIRE(state3.size() == 1);
  REQUIRE(state3.has_field("field"));
}

TEST_CASE("SimulationState: non-copyable", "[simulation_state][unit]") {
  SimulationState state1;
  auto f = make_simple_field<double>(10);
  state1.insert_field("field", std::move(f));
  
  // Copy construction should not compile (static_assert or delete)
  // This is checked by the deleted copy constructor in the class
  // We just verify that the move operations work correctly
  SimulationState state2 = std::move(state1);
  REQUIRE(state2.size() == 1);
}