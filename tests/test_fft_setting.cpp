// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/model.hpp"
// #include "openpfc/openpfc.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream> // For debugging output

using namespace pfc;

class MockModel : public Model {
public:
  MockModel(const World &world) : Model(world) {}
  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

TEST_CASE("Model - FFT Setting and Retrieval", "[fft_setting]") {
  World world = create_world({8, 8, 8});
  Decomposition decomp = make_decomposition(world, 0, 1);
  FFT fft(decomp, MPI_COMM_WORLD, heffte::default_options<heffte::backend::fftw>(), world);

  MockModel model(world);

  model.set_fft(fft);

  // Ensure FFT object is set before proceeding
  REQUIRE_NOTHROW(model.get_fft());

  SECTION("Retrieve FFT object") {
    FFT &retrieved_fft = model.get_fft();

    REQUIRE(&retrieved_fft == &fft); // Ensure the retrieved FFT object matches the original
  }

  SECTION("Ensure FFT object is not null") {
    REQUIRE_NOTHROW(model.get_fft()); // Ensure no exception is thrown
  }
}
