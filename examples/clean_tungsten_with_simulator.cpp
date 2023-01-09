#include "clean_tungsten.hpp"

#include <openpfc/results_writer.hpp>
#include <openpfc/simulator.hpp>
#include <openpfc/time.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace pfc;
using namespace std;

struct ui {
  int Lx, Ly, Lz;
  filesystem::path results_dir;

  ui(int argc, char *argv[]) {
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    if (argc < 5) {
      if (me == 0)
        cerr << "usage: " << argv[0] << " <Lx> <Ly> <Lz> <results-dir>\n";
      throw runtime_error("invalid command line arguments");
    }

    Lx = std::stoi(argv[1]);
    Ly = std::stoi(argv[2]);
    Lz = std::stoi(argv[3]);
    results_dir = argv[4];
    if (me == 0) {
      if (!filesystem::exists(results_dir)) {
        cout << "Results dir " << results_dir << " does not exist, creating\n";
        filesystem::create_directories(results_dir);
      }
    }
  }
};

void run(int Lx, int Ly, int Lz, filesystem::path results_dir) {

  // make world
  double pi = std::atan(1.0) * 4.0;
  // 'lattice' constants for the three ordered phases that exist in 2D/3D PFC
  // const double a1D = 2 * pi;               // stripes
  // const double a2D = 2 * pi * 2 / sqrt(3); // triangular
  double a3D = 2 * pi * sqrt(2); // BCC
  double dx = a3D / 8.0;
  double dy = a3D / 8.0;
  double dz = a3D / 8.0;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;

  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
  Decomposition decomposition(world, MPI_COMM_WORLD);
  FFT fft(decomposition, MPI_COMM_WORLD);
  Tungsten model(fft);

  // make time
  double t0 = 0.0;
  double t1 = 1.0;
  double dt = 1.0;
  double saveat = 1.0;
  Time time({t0, t1, dt}, saveat);

  // make simulator
  Simulator simulator(world, decomposition, fft, model, time);
  // add results writers, one for field psi and another for mean field psi
  simulator.add_results_writer(
      "psi", make_unique<BinaryWriter>(results_dir / "tungsten_psi_%04d.bin"));
  simulator.add_results_writer(
      "psiMF",
      make_unique<BinaryWriter>(results_dir / "tungsten_psiMF_%04d.bin"));

  // initialize model
  model.initialize(dt);

  // run loops
  cout << "Running from t0 = " << t0 << " to " << t1
       << " with step size dt = " << dt << endl;
  while (!simulator.done()) {
    simulator.step();
  }
}

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  ui opts(argc, argv);
  run(opts.Lx, opts.Ly, opts.Lz, opts.results_dir);
  MPI_Finalize();
  return 0;
}
