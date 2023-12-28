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
