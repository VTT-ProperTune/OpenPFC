/**
 * \example 05_simulator.cpp
 *
 * In the previous example, a simple diffusion model was implemented. The
 * implementation was done at a rather low level for demonstration reasons and
 * it had a few flaws. First, time stepping should be performed at a higher
 * level. Hard-coding the initial condition inside the model also does not
 * follow the good implementation practices of modular code. Later, we also
 * want, for example, boundary conditions or to write the results of the
 * simulation to the hard disk. These are all examples of things that basically
 * should not be implemented in the model, because the model mainly includes
 * physics. We want to use initial conditions and boundary conditions more
 * widely in more different models.
 *
 * This example introduces a few new classes: Time, Simulator, and
 * FieldModifier. The responsibility of the Time object is to take care of the
 * time step and to tell when the results of the calculation should be saved.
 * The Simulator class combines model, time, initial conditions, and boundary
 * conditions, being a higher-level abstraction of computation. The initial
 * condition is implemented by inheriting the class FieldModifier. Initial
 * conditions and boundary conditions can be implemented in the same class,
 * therefore a slightly more generic name for the initial condition class.
 */

#include <array>
#include <iostream>
#include <memory>
#include <openpfc/decomposition.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/field_modifier.hpp>
#include <openpfc/model.hpp>
#include <openpfc/simulator.hpp>
#include <openpfc/world.hpp>
#include <vector>

using namespace pfc;

const double PI = 3.141592653589793238463;

class GaussianIC : public FieldModifier {
 private:
  double D = 1.0;

 public:
  void apply(Model &m, double t) override {
    (void)t; // suppress compiler warning about unused parameter
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    std::vector<double> &field = m.get_real_field("psi");
    std::array<int, 3> low = decomp.inbox.low;
    std::array<int, 3> high = decomp.inbox.high;

    if (m.is_rank0()) std::cout << "Create initial condition" << std::endl;
    size_t idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          double y = w.y0 + j * w.dy;
          double z = w.z0 + k * w.dz;
          field[idx] = exp(-(x * x + y * y + z * z) / (4.0 * D));
          idx += 1;
        }
      }
    }
  }
};

class Diffusion : public Model {
  using Model::Model;

 private:
  std::vector<double> opL, psi;
  std::vector<std::complex<double>> psi_F;
  double psi_min = 0.0, psi_max = 1.0;

 public:
  double get_psi_min() const { return psi_min; }
  double get_psi_max() const { return psi_max; }

  void allocate() {
    if (is_rank0()) std::cout << "Allocate space" << std::endl;
    FFT &fft = get_fft();
    psi.resize(fft.size_inbox());
    psi_F.resize(fft.size_outbox());
    opL.resize(fft.size_outbox());

    // "Register" real field psi with a name "psi" so that we can access it from
    // initial condition.
    add_real_field("psi", psi);
  }

  void prepare_operators(double dt) {
    const World &w = get_world();
    const Decomposition &decomp = get_decomposition();
    std::array<int, 3> low = decomp.outbox.low;
    std::array<int, 3> high = decomp.outbox.high;

    if (is_rank0()) std::cout << "Prepare operators" << std::endl;
    size_t idx = 0;
    double fx = 2.0 * PI / (w.dx * w.Lx);
    double fy = 2.0 * PI / (w.dy * w.Ly);
    double fz = 2.0 * PI / (w.dz * w.Lz);
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          // Laplacian operator -k^2
          double ki = (i <= w.Lx / 2) ? i * fx : (i - w.Lx) * fx;
          double kj = (j <= w.Ly / 2) ? j * fy : (j - w.Ly) * fy;
          double kk = (k <= w.Lz / 2) ? k * fz : (k - w.Lz) * fz;
          double kLap = -(ki * ki + kj * kj + kk * kk);
          opL[idx++] = 1.0 / (1.0 - dt * kLap);
        }
      }
    }
  }

  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  void step(double) override {
    FFT &fft = get_fft();
    fft.forward(psi, psi_F);
    for (int k = 0, N = psi_F.size(); k < N; k++) psi_F[k] = opL[k] * psi_F[k];
    fft.backward(psi_F, psi);
    find_minmax();
  }

  void find_minmax() {
    double local_min = std::numeric_limits<double>::max();
    double local_max = std::numeric_limits<double>::min();
    auto min_max_finder = [&local_min, &local_max](const double &value) {
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
    };
    std::for_each(psi.begin(), psi.end(), min_max_finder);
    MPI_Reduce(&local_min, &psi_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &psi_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }
};

void print_statline(Simulator &s) {
  if (!s.is_rank0()) return;
  int n = s.get_increment();
  double t = s.get_time();
  Model &model = s.get_model();
  Diffusion &diffusion_model = dynamic_cast<Diffusion &>(model);
  double min = diffusion_model.get_psi_min();
  double max = diffusion_model.get_psi_max();
  std::cout << "n = " << n << ", t = " << t << ", min = " << min
            << ", max = " << max << std::endl;
}

void run_simulator(Simulator &s) {
  // Initialize the simulator before starting time stepping.
  // This also initializes model.
  s.initialize();

  // Run the simulator until we are done
  print_statline(s);
  while (!s.done()) {
    s.step();
    print_statline(s);
  }
}

void run() {
  // Construct world, decomposition, fft and model
  int L = 64;
  double h = 2.0 * PI / 8.0;
  double o = -0.5 * L * h;
  std::array<int, 3> dimensions = {L, L, L};
  std::array<double, 3> discretization = {h, h, h};
  std::array<double, 3> origin = {o, o, o};
  World world(dimensions, origin, discretization);

  Decomposition decomp(world);
  FFT fft(decomp);
  Diffusion model(fft);

  // Define time
  double t0 = 0.0;
  double t1 = 0.5874010519681994;
  double dt = (t1 - t0) / 42;
  double saveat = dt;  // when to save results
  std::array<double, 3> tspan{t0, t1, dt};
  Time time(tspan, saveat);

  // Define simulator
  Simulator simulator(model, time);

  // Define initial condition and add it to simulator
  simulator.add_initial_conditions(std::make_unique<GaussianIC>());

  run_simulator(simulator);

  // Check the result, we should be very close to 0.5
  if (model.rank0) {
    if (abs(model.get_psi_max() - 0.5) < 0.01) {
      std::cout << "Test pass!" << std::endl;
    } else {
      std::cerr << "Test failed!" << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  std::cout << std::fixed;
  std::cout.precision(12);
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
}
