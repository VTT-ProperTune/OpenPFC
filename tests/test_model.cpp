#include <catch2/catch_test_macros.hpp>
#include <openpfc/model.hpp>

using namespace pfc;

// Define a mock implementation of the Model class for testing
class MockModel : public Model {
public:
  void step(double) override {}
  void initialize(double) override {}
};

TEST_CASE("Model functionality", "[Model]") {
  // Create an instance of the Model
  MockModel model;

  SECTION("Default construction") {
    REQUIRE_FALSE(model.is_rank0());
  }

  SECTION("Set and get FFT") {
    MPI_Init(0, nullptr);
    // Create a Decomposition object
    World world({8, 1, 1}); // Assuming the World class is defined
    Decomposition decomposition(world);
    // Create an FFT object
    FFT fft(decomposition);
    model.set_fft(fft);
    REQUIRE(&model.get_fft() == &fft);
    REQUIRE(model.is_rank0());
    MPI_Finalize();
  }

  SECTION("Real field operations") {
    // Create a real field
    RealField field;
    field.resize(10);

    // Add the field to the model
    model.add_real_field("field1", field);

    REQUIRE(model.has_field("field1"));
    REQUIRE(model.has_real_field("field1"));
    REQUIRE_FALSE(model.has_complex_field("field1"));

    // Get the field from the model
    RealField &retrieved_field = model.get_real_field("field1");
    REQUIRE(&retrieved_field == &field);
  }

  SECTION("Try to access 'default' field without being defined") {
    REQUIRE_THROWS_AS(model.get_field(), std::runtime_error);
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

    // Get the field from the model
    ComplexField &retrieved_field = model.get_complex_field("field2");
    REQUIRE(&retrieved_field == &field);
  }
}
