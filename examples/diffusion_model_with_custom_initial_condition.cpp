#include "diffusion_model.hpp"
#include <iostream>
#include <memory>
#include <pfc/field_modifier.hpp>
#include <pfc/results_writer.hpp>
#include <pfc/simulator.hpp>
#include <pfc/time.hpp>

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

class Gaussian : public FieldModifier {

private:
  double m_D;

public:
  Gaussian(double D) : m_D(D) {}

  void apply(Model &m, double t) override {
    if (m.rank0) {
      cout << "Applying custom initial condition at time " << t << endl;
    }
    World &w = m.get_world();
    Decomposition &d = m.get_decomposition();
    Field &f = m.get_field();
    auto low = d.inbox.low;
    auto high = d.inbox.high;
    long int idx = 0;

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          double y = w.y0 + j * w.dy;
          double z = w.z0 + k * w.dz;
          f[idx] = exp(-(x * x + y * y + z * z) / (4.0 * m_D));
          idx += 1;
        }
      }
    }
  }
};

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
  Diffusion model(world, decomposition, fft);
  Simulator simulator(world, decomposition, fft, model, time);

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