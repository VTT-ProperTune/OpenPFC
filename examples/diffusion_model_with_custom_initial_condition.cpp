#include "diffusion_model.hpp"
#include <iostream>
#include <memory>
#include <pfc/field_modifier.hpp>
#include <pfc/results_writer.hpp>
#include <pfc/simulator.hpp>
#include <pfc/time.hpp>

using namespace std;

void print_stats(Simulator<Diffusion> &S) {
  // we can still access the model:
  auto &M = S.get_model();
  auto &psi = M.get_field();
  int idx = M.get_midpoint_idx();
  if (idx == -1) return;
  cout << "n = " << S.get_increment() << ", t = " << S.get_time() << ", psi["
       << idx << "] = " << psi[idx] << endl;
}

World make_world() {
  int Lx = 64;
  int Ly = Lx;
  int Lz = Lx;
  double dx = 2.0 * constants::pi / 8.0;
  double dy = dx;
  double dz = dx;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;
  return World({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
}

Time make_time() {
  double t0 = 0.0;
  double t1 = 0.5874010519681994;
  double dt = (t1 - t0) / 42;
  return Time({t0, t1, dt});
}

void run_test(Simulator<Diffusion> &S) {
  auto &model = S.get_model();
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

  void apply(Model &m, Field &f, double t) override {
    if (m.id == 0) {
      cout << "Applying custom initial condition at time " << t << endl;
    }
    auto low = m.get_inbox_low();
    auto high = m.get_inbox_high();
    auto w = m.get_world();
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

  World world = make_world();
  Time time = make_time();
  Simulator<Diffusion> S(world, time);
  S.add_initial_conditions(make_unique<Gaussian>(1.0));

  print_stats(S);
  while (!(S.done())) {
    S.step();
    print_stats(S);
  }

  run_test(S);
}

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
  return 0;
}
