#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/fft.hpp>
#include <vector>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("FFT forward transformation", "[FFT]") {
  MPI_Init(0, nullptr);
  // Create a Decomposition object
  World world({8, 1, 1}); // Assuming the World class is defined
  Decomposition decomposition(world);

  // Create an FFT object
  FFT fft(decomposition);

  // Generate input data
  std::vector<double> input = {0.000, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498};
  REQUIRE(input.size() == fft.size_inbox());

  // Perform the forward transformation
  std::vector<std::complex<double>> output(fft.size_outbox());
  fft.forward(input, output);

  REQUIRE_THAT(std::real(output[0]), WithinAbs(21.991, 0.01));
  MPI_Finalize();
}

TEST_CASE("FFT backward transformation", "[FFT]") {
  MPI_Init(0, nullptr);
  // Create a Decomposition object
  World world({2, 1, 1}); // Assuming the World class is defined
  Decomposition decomposition(world);

  // Create an FFT object
  FFT fft(decomposition);

  // Generate input data
  std::vector<std::complex<double>> input = {std::complex<double>(1.0, 0.0), std::complex<double>(2.0, 0.0)};

  // Perform the backward transformation
  std::vector<double> output(fft.size_inbox());
  fft.backward(input, output);

  // Perform assertions on the output
  REQUIRE(output.size() == fft.size_inbox());
  REQUIRE_THAT(output[0], WithinAbs(1.5, 0.01));
  // Add more assertions as needed
  MPI_Finalize();
}
