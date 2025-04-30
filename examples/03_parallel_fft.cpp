// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>

#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/factory/decomposition_factory.hpp>
#include <openpfc/fft.hpp>

using namespace std;
using namespace pfc;

/* Print vector contents */
template <typename T> void print_vec(const vector<T> &v) {
  std::cout << "[";
  for (auto &e : v) {
    std::cout << e;
    if (&e != &v.back()) std::cout << ", ";
  }
  std::cout << "]\n";
}

/* A naive DFT implementation */
void dft_forward(const vector<double> x, vector<complex<double>> &X) {
  const double pi = 4.0 * atan(1.0);
  fill(X.begin(), X.end(), 0.0);
  for (int n = 0, N = x.size(); n < N; n++) {
    for (int k = 0; k < N / 2 + 1; k++) {
      X[k] += x[n] * complex<double>(cos(2 * pi / N * k * n),
                                     -sin(2 * pi / N * k * n));
    }
  }
}

/** \example 03_parallel_fft.cpp
 *
 * When the World and Decomposition are defined, the next step is to specify the
 * FFT, which serves as the true workhorse of OpenPFC. It is responsible for
 * performing the parallel FFT computations. OpenPFC leverages the remarkable
 * algorithm provided by the HeFFTe library, which is widely recognized and
 * proven to be an efficient implementation. With these three fundamental
 * components in place, one can already embark on building programs that involve
 * distributed FFT transformations. The following example demonstrates the power
 * of these components by showcasing a distributed FFT computation.
 *
 * The provided example demonstrates the usage of the parallel Fast Fourier
 * Transform (FFT) implementation in OpenPFC. Here's a description of what the
 * example does:
 *
 * The program begins by including the necessary headers and namespaces for the
 * OpenPFC library and standard C++ functionality. It also defines a utility
 * function `print_vec` to print the contents of a vector.
 *
 * The next part of the code defines a naive Discrete Fourier Transform (DFT)
 * implementation called `dft_forward`, which performs a forward DFT
 * transformation using complex arithmetic.
 *
 * The main function initializes the MPI process, retrieves the rank and number
 * of processes, and sets up the necessary components for parallel computing
 * using OpenPFC. It creates a `World` object with a size of 8 in the
 * x-direction and 1 in the y- and z-directions. It then constructs a
 * `Decomposition` object based on the created `World` and the MPI communicator.
 * Finally, it creates an `FFT` object using the `Decomposition` and MPI
 * communicator.
 *
 * Two vectors, `in` and `out`, are created to store the input data and the
 * results of the FFT computation, respectively. The input data is initialized
 * with values based on the index. The `forward` method of the `FFT` object is
 * called to perform the parallel FFT transformation on the input data.
 *
 * To validate the results, the program also applies the `dft_forward` function
 * to the input data and stores the results in the `out2` vector.
 *
 * The program then prints the input and output data of the parallel FFT
 * implementation and the naive DFT implementation (only done by rank 0). It
 * calculates and prints the norms of the output data from both implementations
 * and computes the error by comparing the norms.
 *
 * Finally, the MPI process is finalized, and the program terminates.
 *
 * The example essentially compares the results of the parallel FFT
 * implementation provided by OpenPFC with a naive DFT implementation to verify
 * the correctness of the parallel FFT.
 *
 * Expected output is:
 *
 *      HeFFTe parallel FFT implementation
 *
 *      Input data:
 *      [0.000, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498]
 *      Output data:
 *      [(21.991,0.000), (-3.142,7.584), (-3.142,3.142), (-3.142,1.301),
 * (-3.142,0.000)]
 *
 *      Naive DFT implementation
 *
 *      Input data:
 *      [0.000, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498]
 *      Output data:
 *      [(21.991,0.000), (-3.142,7.584), (-3.142,3.142), (-3.142,1.301),
 * (-3.142,-0.000)]
 *
 *      Norms:
 *
 *      0: 483.611, 483.611
 *      1: 67.394, 67.394
 *      2: 19.739, 19.739
 *      3: 11.563, 11.563
 *      4: 9.870, 9.870
 *
 *      Error: -0.000000000000
 *
 */
int main(int argc, char *argv[]) {

  cout << fixed;
  cout.precision(3);

  // Initialize MPI process, get number of ranks and current rank number
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, num_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_procs);

  // Construct world, decomposition and fft
  World world = create_world({8, 1, 1});
  Decomposition decomposition = make_decomposition(world, comm);
  auto plan_options = heffte::default_options<heffte::backend::fftw>();
  FFT fft(decomposition, comm, plan_options, world);

  // Create two vectors; in contains input data and results are stored to out
  vector<double> in(fft.size_inbox());
  vector<complex<double>> out(fft.size_outbox());
  for (int i = 0, N = in.size(); i < N; i++) {
    in[i] = i * atan(1.0);
  }

  // Perform FFT for input data using parallel FFT
  fft.forward(in, out);

  // Let's use our own implementation to check the results
  vector<complex<double>> out2(fft.size_outbox());
  dft_forward(in, out2);

  cout << "HeFFTe parallel FFT implementation\n\n";
  if (rank == 0) {
    cout << "Input data:\n";
    print_vec(in);
    cout << "Output data:\n";
    print_vec(out);
  }

  if (rank == 0) {
    cout << "\nNaive DFT implementation\n\n";
    cout << "Input data:\n";
    print_vec(in);
    cout << "Output data:\n";
    print_vec(out2);
  }

  double err = 0.0;
  cout << "\nNorms:\n\n";
  for (int i = 0, N = out.size(); i < N; i++) {
    cout << i << ": " << norm(out[i]) << ", " << norm(out2[i]) << "\n";
    err += norm(out[i]);
    err -= norm(out2[i]);
  }

  cout.precision(12);
  cout << "\nError: " << err << "\n";

  MPI_Finalize();
  return 0;
}
