#include <cmath>
#include <iostream>
#include <openpfc/openpfc.hpp>
#include <vector>

using namespace std;
using namespace pfc;

int main(int argc, char *argv[]) {

  // initialize mpi
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // define world
  int Lx = 16;
  double pi = 3.14159265358979323846;
  double dx = 2 * pi / Lx;
  double x0 = -0.5 * Lx * dx;
  World world({Lx, 1, 1}, {x0, 0, 0}, {dx, 1, 1});

  // define domain decomposition
  Decomposition decomp(world, 0, 1);

  // construct FFT object
  FFT fft(decomp, comm);
  size_t size_inbox = fft.size_inbox();
  size_t size_outbox = fft.size_outbox();

  // allocate space for this particular mpi process:
  // - size_inbox does not equal to Lx (in general)
  // - size_outbox = floor(size_inbox/2) + 1 due to the symmetry
  vector<double> y(size_inbox), dy(size_inbox);
  vector<double> op(size_outbox);
  vector<complex<double>> Y(size_outbox);

  // get the lower and upper limits of outbox for this particular mpi process
  array<int, 3> olow = decomp.outbox.low;
  array<int, 3> ohigh = decomp.outbox.high;

  // construct operator
  const double fx = 2.0 * pi / (dx * Lx);
  int idx = 0;
  for (int i = olow[0]; i <= ohigh[0]; i++) {
    op[idx] = (i < Lx / 2) ? i * fx : (i - Lx) * fx;
    idx += 1;
  }

  // get the lower and upper limits of outbox for this particular mpi process
  array<int, 3> ilow = decomp.inbox.low;
  array<int, 3> ihigh = decomp.inbox.high;

  // generate data y = f(x)
  idx = 0;
  for (int i = ilow[0]; i <= ihigh[0]; i++) {
    double x = x0 + i * dx;
    y[idx] = sin(x);
    idx += 1;
  }

  // calculate FFT
  fft.forward(y, Y);
  // apply operator to Y
  complex<double> im(0, 1);
  for (int i = 0; i < size_outbox; i++) {
    Y[i] = im * op[i] * Y[i];
  }
  // calculate inverse-FFT
  fft.backward(Y, dy);

  // check the results
  idx = 0;
  for (int i = ilow[0]; i <= ihigh[0]; i++) {
    double x = x0 + i * dx;
    double diff = abs(dy[i] - cos(x));
    printf("i=%2d, x(i)=%5.2f, dy(x)=%5.2f, |dy(x)-dy_true(x)|=%0.5f\n", i, x,
           dy[i], diff);
    idx += 1;
  }

  MPI_Finalize();
  return 0;
}
