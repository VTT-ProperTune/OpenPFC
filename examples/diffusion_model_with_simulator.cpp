#include "diffusion_model.hpp"
#include <iostream>
#include <memory>
#include <openpfc/results_writer.hpp>
#include <openpfc/simulator.hpp>
#include <openpfc/time.hpp>

using namespace std;

void print_stats(Simulator &simulator) {
  // we can still access the model:
  auto &model = dynamic_cast<Diffusion &>(simulator.get_model());
  auto &field = simulator.get_field();
  auto &time = simulator.get_time();
  int idx = model.get_midpoint_idx();
  if (idx == -1) return;
  cout << "n = " << time.get_increment() << ", t = " << time.get_current()
       << ", psi[" << idx << "] = " << field[idx] << endl;
}

void run_test(Simulator &simulator) {
  auto &model = dynamic_cast<Diffusion &>(simulator.get_model());
  auto idx = model.get_midpoint_idx();
  if (idx != -1) {
    auto &field = model.get_field();
    if (abs(field[idx] - 0.5) < 0.01) {
      cout << "Test pass!" << endl;
    } else {
      cerr << "Test failed!" << endl;
    }
  }
}

void run() {

  int Lx = 64, Ly = Lx, Lz = Lx;
  double dx = 2.0 * constants::pi / 8.0, dy = dx, dz = dx;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;
  Vec3<int> dimensions{Lx, Ly, Lz};
  Vec3<double> origo{x0, y0, z0};
  Vec3<double> discretization{dx, dy, dz};
  World world(dimensions, origo, discretization);

  double t0 = 0.0;
  double t1 = 0.5874010519681994;
  double dt = (t1 - t0) / 42;
  double saveat = 1.0;
  Vec3<double> tspan{t0, t1, dt};
  Time time(tspan, saveat);

  MPI_Comm comm = MPI_COMM_WORLD;
  Decomposition decomposition(world, comm);
  FFT fft(decomposition, comm);
  Diffusion model(fft);
  Simulator simulator(model, time);

  // write results to binary file
  simulator.add_results_writer(make_unique<BinaryWriter>("diffusion_%04d.bin"));

  print_stats(simulator);
  while (!simulator.done()) {
    simulator.step();
    print_stats(simulator);
  }

  run_test(simulator);
}

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
  return 0;
}
