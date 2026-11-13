// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/apply_field_modifier.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>

#include "fixtures/mock_field_modifier.hpp"

using namespace pfc;

TEST_CASE("FieldModifier - applies field modification", "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{8, 8, 8});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);
  pfc::testing::MockFieldModifier modifier;
  modifier.apply(psi, domain, box, 0.0);
  REQUIRE(modifier.applied);
}

TEST_CASE("FieldModifier - polymorphic usage", "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{8, 8, 8});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);
  std::unique_ptr<FieldModifier> modifier =
      std::make_unique<pfc::testing::MockFieldModifier>();
  modifier->apply(psi, domain, box, 0.0);
  REQUIRE(static_cast<pfc::testing::MockFieldModifier *>(modifier.get())->applied);
}

TEST_CASE("FieldModifier - move semantics", "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{8, 8, 8});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);
  pfc::testing::MockFieldModifier modifier;
  pfc::testing::MockFieldModifier moved_modifier = std::move(modifier);
  moved_modifier.apply(psi, domain, box, 0.0);
  REQUIRE(moved_modifier.applied);
}

TEST_CASE("FieldModifier - field name getter and setter", "[field_modifier][unit]") {
  pfc::testing::MockFieldModifier modifier;
  REQUIRE(modifier.get_field_name() == "default");
  modifier.set_field_name("phi");
  REQUIRE(modifier.get_field_name() == "phi");
}

TEST_CASE("FieldModifier - input validation", "[field_modifier][unit][error]") {
  pfc::testing::MockFieldModifier modifier;

  SECTION("Empty field name throws std::invalid_argument") {
    REQUIRE_THROWS_AS(modifier.set_field_name(""), std::invalid_argument);
  }

  SECTION("Valid field name is accepted") {
    REQUIRE_NOTHROW(modifier.set_field_name("density"));
    REQUIRE(modifier.get_field_name() == "density");
  }

  SECTION("Empty field name list throws") {
    REQUIRE_THROWS_AS(modifier.set_field_names({}), std::invalid_argument);
  }

  SECTION("Multi-field names") {
    modifier.set_field_names({"a", "b"});
    REQUIRE(modifier.get_field_names().size() == 2);
    REQUIRE(modifier.get_field_name() == "a");
  }
}

TEST_CASE("apply_field_modifier wraps a host Field and records the host write",
          "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{4, 3, 2});
  auto box = pfc::domain::index_box(domain);
  pfc::Field<double> psi(domain, box, 0);
  pfc::testing::MockIC ic;
  pfc::apply_field_modifier(ic, psi, 0.0);
  for (std::size_t i = 0; i < psi.size(); ++i) {
    REQUIRE(psi.data()[i] == 1.0);
  }
  REQUIRE(psi.residency().host_valid());
}

TEST_CASE("apply_field_modifier rejects halo-padded storage",
          "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{4, 4, 4});
  auto box = pfc::domain::index_box(domain);
  pfc::Field<double> padded(domain, box, 1);
  pfc::testing::MockIC ic;
  REQUIRE_THROWS_AS(pfc::apply_field_modifier(ic, padded, 0.0),
                    std::invalid_argument);
}

TEST_CASE("Field::output() and Field::view() alias the same storage",
          "[field_modifier][unit]") {
  auto domain = pfc::domain::create(pfc::Int3{3, 2, 2});
  auto box = pfc::domain::index_box(domain);
  pfc::Field<double> psi(domain, box, 0);
  auto out = psi.output();
  REQUIRE(out.size() == psi.size());
  out[5] = 2.5;
  auto view = psi.view();
  REQUIRE(view.size() == psi.size());
  REQUIRE(view.data()[5] == 2.5);
  REQUIRE(view.extents() == pfc::Int3{3, 2, 2});
  REQUIRE(view.spacing() == psi.spacing());
}

// ADR 0006: modifiers are written for double; a float field is widened into a
// double scratch, modified, and narrowed back by apply_field_modifier.
TEST_CASE("apply_field_modifier serves a float field through a double scratch",
          "[field_modifier][unit][float]") {
  auto domain = pfc::domain::create(pfc::Int3{4, 4, 4});
  auto box = pfc::domain::index_box(domain);
  pfc::data::Field<float> psi(domain, box, 0);
  for (float &v : psi.vec()) {
    v = -1.0f;
  }
  pfc::Constant ic(0.25);
  pfc::apply_field_modifier(ic, psi, 0.0);
  for (float v : psi.vec()) {
    REQUIRE(v == 0.25f);
  }

  pfc::data::Field<float> halo(domain, box, 1);
  REQUIRE_THROWS_AS(pfc::apply_field_modifier(ic, halo, 0.0), std::invalid_argument);
}
