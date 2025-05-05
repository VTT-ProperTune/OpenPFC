// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/openpfc.hpp>
#include <random>

using namespace pfc;
using namespace std;

int main(int argc, char *argv[]) {

  // Create MPI session, World and Decomposition
  MPI_Worker worker(argc, argv);
  World world = world::create({4, 3, 2});
  Decomposition decomp = make_decomposition(world);

  // Create input field
  // DiscreteField<double, 3> input(decomp);

  auto dimensions = decomp.m_inbox.size;
  auto offsets = decomp.m_inbox.low;
  auto origin = get_origin(world);
  auto discretization = get_spacing(world);
  DiscreteField<double, 3> input(dimensions, offsets, origin, discretization);

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // Fill input field with random numbers
  apply(input, [&](auto, auto, auto) { return dis(gen); });

  auto plan_options = heffte::default_options<heffte::backend::fftw>();
  FFT fft(decomp, MPI_COMM_WORLD, plan_options, world);

  // Create output array to store FFT results. If requested array is of type T =
  // complex<double>, then array will be constructed using complex indices so
  // that it matches the Fourier-space, i.e. first dimension is floor(Lx/2) + 1.
  Array<complex<double>, 3> output(get_outbox_size(fft));

  std::cout << "input: " << input << std::endl;   // this is {4, 3, 2}
  std::cout << "output: " << output << std::endl; // this is {3, 3, 2}

  // This would construct an array of type T = <double> with different indices
  // Array<double, 3> output2(fft);
  // std::cout << output2 << std::endl; // this is {4, 3, 2}

  fft.forward(input, output);

  // Display results
  show(input);
  show(output);

  return 0;
}
