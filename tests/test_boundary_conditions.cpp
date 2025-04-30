// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/boundary_conditions/fixed_bc.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/model.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace pfc;

/*

class MockModel : public Model {
public:
  MockModel(const World &world) : Model(world) {}

  // Implement abstract methods with mock behavior
  Field &get_field() override {
    static RealField dummyField;
    return dummyField;
  }

  const Decomposition &get_decomposition() const {
    static Decomposition dummyDecomp;
    return dummyDecomp;
  }

  const World &get_world() const { return Model::get_world(); }
};

TEST_CASE("FixedBC apply method triggers error", "[BoundaryConditions]") {
  // Create a dummy World object
  const World world = create_world({128, 128, 128});

  // Create a MockModel object
  MockModel model(world);

  // Create a FixedBC object
  FixedBC fixedBC;

  // Call the apply method to trigger the error
  double time = 0.0;
  fixedBC.apply(model, time);
}
*/
