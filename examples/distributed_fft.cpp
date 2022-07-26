#include <iostream>
#include <pfc/fft.hpp>
#include <vector>

using namespace std;
using namespace PFC;

void print_vec(const vector<double> &v) {
  std::cout << "[";
  for (auto &e : v) {
    std::cout << e;
    if (&e != &v.back()) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

void print_vec(const vector<complex<double>> &v) {
  std::cout << "[";
  for (auto &e : v) {
    std::cout << e;
    if (&e != &v.back()) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

/* A naive DFT implementation */
void r2c(const vector<double> x, vector<complex<double>> &X) {
  const double pi = 4.0 * atan(1.0);
  fill(X.begin(), X.end(), 0.0);
  for (int n = 0, N = x.size(); n < N; n++) {
    for (int k = 0; k < N / 2 + 1; k++) {
      X[k] += x[n] * complex<double>(cos(2 * pi / N * k * n),
                                     -sin(2 * pi / N * k * n));
    }
  }
}

int main(int argc, char *argv[]) {

  cout << fixed;
  cout.precision(3);

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  FFT fft({8, 1, 1});
  vector<double> in(fft.size_inbox());
  vector<complex<double>> out(fft.size_outbox());
  for (int i = 0, N = in.size(); i < N; i++) {
    in[i] = i * atan(1.0);
  }
  fft.forward(in, out);

  cout << "HeFFTe implementation\n\n";
  if (rank == 0) {
    cout << "Input data:\n";
    print_vec(in);
    cout << "Output data:\n";
    print_vec(out);
  }

  // let's use our own implementation to check the results
  vector<complex<double>> out2(fft.size_outbox());
  r2c(in, out2);
  if (rank == 0) {
    cout << "\nOwn implementation\n\n";
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
