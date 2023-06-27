#include <catch2/catch_test_macros.hpp>
#include <openpfc/initial_conditions/constant.hpp>
#include <openpfc/model.hpp>
#include <vector>

using namespace pfc;

// Mock model class for testing
class ModelWithConstantIC : public Model {
private:
  std::vector<double> psi;

public:
  ModelWithConstantIC() {
    // TODO: FieldModifier can only modify default field.
    add_real_field("default", psi);
  }
  void step(double) override {}
  void initialize(double) override { psi.resize(8); }
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
    FFT fft(decomp);
    ModelWithConstantIC m;
    // TODO: This should be possible without defining fft
    m.set_fft(fft);
    m.initialize(1.0);
    Constant c(1.0);
    c.apply(m, 0.0);
    const Field &field = m.get_field();
    for (const auto &value : field) {
      REQUIRE(value == 1.0);
    }
    MPI_Finalize();
  }
}
