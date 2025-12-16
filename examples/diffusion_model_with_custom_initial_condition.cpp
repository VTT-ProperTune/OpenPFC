// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "diffusion_model.hpp"
#include <iostream>
#include <memory>
#include <openpfc/constants.hpp>
#include <openpfc/factory/decomposition_factory.hpp>
#include <openpfc/field_modifier.hpp>
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

class Gaussian : public FieldModifier {

private:
  double m_D;

public:
  Gaussian(double D) : m_D(D) {}

  void apply(Model &m, double t) override {
    if (m.is_rank0()) {
      cout << "Applying custom initial condition at time " << t << endl;
    }
    auto &world = m.get_world();
    auto &field = m.get_field();
    auto &fft = m.get_fft();
    auto origin = get_origin(world);
    auto spacing = get_spacing(world);

    auto low = get_inbox(fft).low;
    auto high = get_inbox(fft).high;
    long int idx = 0;

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = origin[0] + i * spacing[0];
          double y = origin[1] + j * spacing[1];
          double z = origin[2] + k * spacing[2];
          field[idx] = exp(-(x * x + y * y + z * z) / (4.0 * m_D));
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
  World world = world::create(GridSize(dimensions), PhysicalOrigin(origo),
                              GridSpacing(discretization));

  double t0 = 0.0;
  double t1 = 0.5874010519681994;
  double dt = (t1 - t0) / 42;
  double saveat = 1.0;
  Vec3<double> tspan{t0, t1, dt};
  Time time(tspan, saveat);

  MPI_Comm comm = MPI_COMM_WORLD;
  auto decomposition = make_decomposition(world, comm);
  auto fft = fft::create(decomposition);
  Diffusion model(world);
  Simulator simulator(model, time);

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
