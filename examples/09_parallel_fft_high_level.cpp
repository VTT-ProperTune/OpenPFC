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

#include <openpfc/openpfc.hpp>
#include <random>

using namespace pfc;
using namespace std;

int main(int argc, char *argv[]) {

  // Create MPI session, World and Decomposition
  MPI_Worker worker(argc, argv);
  World world({4, 3, 2});
  Decomposition decomp(world);

  // Create input field
  DiscreteField<double, 3> input(decomp);
  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // Fill input field with random numbers
  apply(input, [&](auto, auto, auto) { return dis(gen); });

  // Create output array to store FFT results. If requested array is of type T =
  // complex<double>, then array will be constructed using complex indices so
  // that it matches the Fourier-space, i.e. first dimension is floor(Lx/2) + 1.
  Array<complex<double>, 3> output(decomp);

  std::cout << "input: " << input << std::endl;   // this is {4, 3, 2}
  std::cout << "output: " << output << std::endl; // this is {3, 3, 2}

  // This would construct an array of type T = <double> with different indices
  // Array<double, 3> output2(decomp);
  // std::cout << output2 << std::endl; // this is {4, 3, 2}

  // Create FFT object and perform parallel FFT
  FFT fft(decomp);
  fft.forward(input, output);

  // Display results
  show(input);
  show(output);

  return 0;
}
