// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/field_modifier.hpp>

using namespace pfc;
using namespace pfc::test;

// Minimal FieldModifier that sets a constant value
class ConstantIC : public FieldModifier {
public:
  explicit ConstantIC(const std::string &field_name, double value) : value_(value) {
    set_field_name(field_name);
  }

  void apply(Model &m, double /*t*/) override {
    auto &field = m.get_real_field(get_field_name());
    for (double &elem : field) {
      elem = value_;
    }
  }

private:
  double value_;
};

TEST_CASE("FieldModifier integration: constant IC",
          "[integration][field][modifier]") {
  auto world = world::uniform(16, 1.0);
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  DiffusionModel model(fft, world);
  model.initialize(1.0);

  // Apply constant IC to density
  ConstantIC ic("density", 0.25);
  ic.apply(model, 0.0);

  // Check the field was set
  auto &psi = model.get_real_field("density");
  REQUIRE_FALSE(psi.empty());
  for (const auto &v : psi) {
    REQUIRE(v == Catch::Approx(0.25).margin(1e-12));
  }
}
