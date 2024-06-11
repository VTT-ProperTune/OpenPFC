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
#include <openpfc/initial_conditions/constant.hpp>
#include <openpfc/model.hpp>
#include <vector>

using namespace pfc;

// Mock model class for testing
class ModelWithConstantIC : public Model {
public:
  void step(double) override {}
  void initialize(double) override {}
};

TEST_CASE("Constant Field Modifier") {

  SECTION("Density value") {
    Constant c(1.0);
    REQUIRE(c.get_density() == 1.0);
    c.set_density(2.5);
    REQUIRE(c.get_density() == 2.5);
  }

  SECTION("Apply field modifier") {
    MPI_Init(0, nullptr);
    World world({8, 1, 1});
    Decomposition decomp(world);
    std::vector<double> psi(8);
    FFT fft(decomp);
    ModelWithConstantIC m;
    m.add_real_field("default", psi);
    // TODO: This should be possible without defining fft
    m.set_fft(fft);
    Constant c(1.0);
    c.apply(m, 0.0);
    const Field &field = m.get_real_field("default");
    for (const auto &value : field) {
      REQUIRE(value == 1.0);
    }
    MPI_Finalize();
  }
}
