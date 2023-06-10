#include <algorithm>
#include <iostream>
#include <limits>

#include <openpfc/decomposition.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/model.hpp>
#include <openpfc/world.hpp>

using namespace std;
using namespace pfc;

/** \example 04_diffusion_model.cpp
 *
 * Let's embark on the journey of physics modeling. Together, we will construct
 * a captivating diffusion model and unleash its mysteries using OpenPFC.
 * Picture this: we inherit the noble class of 'Model' and breathe life into it
 * by crafting two enchanting functions: 'initialize' and 'step'. Brace yourself
 * for a wondrous exploration that intertwines the realms of science and
 * imagination.
 *
 * This example implements a simple diffusion model using the OpenPFC library.
 * The diffusion model is solved using a finite difference method with a Fourier
 * spectral method for solving the diffusion equation in three dimensions. The
 * code demonstrates a low-level implementation of the model, where the
 * simulation is manually stepped and the initial conditions are defined inside
 * the model.
 *
 * The Diffusion class is derived from the base class Model provided by the
 * OpenPFC library. It overrides two class methods:
 *
 * 1. `initialize(double dt)`: This method is called once to initialize the
 *    necessary parameters and data structures for the simulation. It allocates
 *    memory for the main variable, psi, and its Fourier transform, psi_F. It
 *    also constructs the linear operator L, which is used to solve the
 *    diffusion equation.
 *
 * 2. `step(double dt)`: This method is called to step the model forward in time
 *    by one time increment, dt. It applies the linear operator L to the Fourier
 *    transform of psi, psi_F, and then performs an inverse Fourier transform to
 *    obtain the updated psi values. It also calculates the minimum and maximum
 *    values of psi locally for each MPI rank and performs reduction operations
 *    to obtain the global minimum and maximum values.
 *
 * The `run()` function defines the world dimensions and discretization
 * parameters, constructs the World, Decomposition, FFT, and Diffusion objects,
 * and initializes the simulation. It then enters a loop where the model is
 * stepped forward in time until a specified stopping time is reached. During
 * each iteration, the current time, the iteration number, and the minimum and
 * maximum values of psi are printed.
 *
 * Finally, the main function initializes the MPI environment, calls the run()
 * function, and cleans up the MPI environment before exiting.
 *
 * The code demonstrates how to implement a diffusion model using the OpenPFC
 * library and manually step the simulation. It also shows how to perform local
 * and global reductions using MPI to calculate global properties of the field
 * variable.
 *
 * Expected output is:
 *
 *      ( initialization messages ... )
 *      n = 0, t = 0.000000000000, psi[133152] = 1.000000000000
 *      n = 1, t = 0.013985739333, psi[133152] = 0.979721090279
 *      n = 2, t = 0.027971478665, psi[133152] = 0.960110027682
 *      n = 3, t = 0.041957217998, psi[133152] = 0.941136780128
 *      n = 4, t = 0.055942957330, psi[133152] = 0.922773010503
 *      ( time stepping continues ...)
 *      n = 40, t = 0.559429573303, psi[133152] = 0.516585236400
 *      n = 41, t = 0.573415312636, psi[133152] = 0.509734461852
 *      n = 42, t = 0.587401051968, psi[133152] = 0.503032957135
 */
class Diffusion : public Model {
  using Model::Model; // "Inherit" the default constructor of base class

private:
  vector<double> opL, psi;       // Define linear operator opL and unknown (real) psi
  vector<complex<double>> psi_F; // Define (complex) psi

public:
  double psi_min, psi_max; // minimum and maximum values of psi for this rank

  /**
   * @brief Initialize the diffusion model
   *
   * This function is called before the actual time stepping starts. This is the
   * right place to allocate memory for simulation as well as pre-calculate
   * operators and other things needed in order to start the simulation.
   *
   * @param dt Time step interval
   */
  void initialize(double dt) override {
    if (rank0) cout << "Allocate space" << endl;

    // Get references to world, fft and domain decomposition
    const World &w = get_world();
    FFT &fft = get_fft();
    const Decomposition &decomp = get_decomposition();

    // Allocate space for the main variable and it's fourier transform
    psi.resize(fft.size_inbox());
    psi_F.resize(fft.size_outbox());

    /*
    Construct linear operator L

    Because we are doing FFT between real and complex using symmetry,
    it's enough to define only a half of the operator. Thus, the operator size
    matches with the outbox.
    */
    opL.resize(fft.size_outbox());

    /*
    World is defining the global dimensions of the problem as well as origin and
    chosen discretization parameters.
    */
    if (rank0) cout << "World: " << w << endl;

    /*
    Upper and lower limits for this particular MPI rank, in both inbox and
    outbox, are given by domain decomposition object
    */
    Vec3<int> i_low = decomp.inbox.low;
    Vec3<int> i_high = decomp.inbox.high;
    Vec3<int> o_low = decomp.outbox.low;
    Vec3<int> o_high = decomp.outbox.high;

    /*
    Typically initial conditions are constructed elsewhere. However, to keep
    things as simple as possible, the initial condition can be also constructed
    here.
    */
    if (rank0) cout << "Create initial condition" << endl;
    int idx = 0;
    double D = 1.0;
    for (int k = i_low[2]; k <= i_high[2]; k++) {
      for (int j = i_low[1]; j <= i_high[1]; j++) {
        for (int i = i_low[0]; i <= i_high[0]; i++) {
          double x = w.x0 + i * w.dx;
          double y = w.y0 + j * w.dy;
          double z = w.z0 + k * w.dz;
          psi[idx] = exp(-(x * x + y * y + z * z) / (4.0 * D));
          idx += 1;
        }
      }
    }

    /*
    The main thing along with allocating workspace for simulation is to prepare
    operators, thus making the actual time stepping as fast as possible.
    */
    if (rank0) cout << "Prepare operators" << endl;
    idx = 0;
    double pi = std::atan(1.0) * 4.0;
    double fx = 2.0 * pi / (w.dx * w.Lx);
    double fy = 2.0 * pi / (w.dy * w.Ly);
    double fz = 2.0 * pi / (w.dz * w.Lz);
    for (int k = o_low[2]; k <= o_high[2]; k++) {
      for (int j = o_low[1]; j <= o_high[1]; j++) {
        for (int i = o_low[0]; i <= o_high[0]; i++) {
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

  /**
   * @brief The actual time stepping function.
   *
   * In this particular case, with a simple linear model, is basically
   *
   *    u(t+Δt) = ifft(opL*fft(u(t)))
   *
   */
  void step(double) override {
    FFT &fft = get_fft();    // Get reference to FFT object
    fft.forward(psi, psi_F); // Perform forward FFT, psi_F = fft(psi)
    for (int k = 0, N = psi_F.size(); k < N; k++) {
      psi_F[k] = opL[k] * psi_F[k]; // Calculate result psi_F = opL*psi_F
    }
    fft.backward(psi_F, psi); // Perform backward FFT, psi = ifft(psi_F)
    find_minmax();            // find minimum and maximum values of psi
  }

  /**
   * @brief A simple MPI communication example.
   *
   * We find minimum and maximum of psi locally, and then communicate them
   * between different MPI ranks.
   */
  void find_minmax() {
    // Count local minimum and maximum for this particular rank
    double local_min = std::numeric_limits<double>::max();
    double local_max = std::numeric_limits<double>::min();
    auto min_max_finder = [&local_min, &local_max](const double &value) {
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
    };
    std::for_each(psi.begin(), psi.end(), min_max_finder);

    // This would print maximum only for this particular part of domain attached
    // to this MPI process, but usually that is NOT what we are trying to do:
    // cout << "max = " << local_max << endl;

    // Thus, we need some MPI communication. We use MPI_Reduce to make MIN and
    // MAX reductions and send results to rank 0.
    MPI_Reduce(&local_min, &psi_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &psi_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //                               ^ size                  ^ rank where to send

    // If the result is needed in all other ranks also, we can use MPI_Allreduce
    // to do that:
    /*
    MPI_Allreduce(&local_min, &psi_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &psi_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    */
  }
};

void run() {
  // Define world
  int Lx = 64;
  int Ly = Lx;
  int Lz = Lx;
  const double pi = 3;
  double dx = 2.0 * pi / 8.0;
  double dy = dx;
  double dz = dx;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;

  // Construct world, decomposition, fft and model
  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
  Decomposition decomp(world);
  FFT fft(decomp);
  Diffusion model(fft);

  // Define time
  double t = 0.0;
  double t_stop = 0.5874010519681994;
  double dt = (t_stop - t) / 42;
  int n = 0; // increment counter

  // Initialize the model before starting time stepping
  model.initialize(dt);

  // Loop until we are in t_stop
  if (model.rank0) cout << "n = 0, t = 0, min = 0.0, max = 1.0" << endl;
  while (t <= t_stop) {
    t += dt;
    n += 1;
    model.step(dt);
    if (model.rank0)
      cout << "n = " << n << ", t = " << t << ", min = " << model.psi_min << ", max = " << model.psi_max << endl;
  }

  // Check the result, we should be very close to 0.5
  if (model.rank0) {
    if (abs(model.psi_max - 0.5) < 0.01) {
      cout << "Test pass!" << endl;
    } else {
      cerr << "Test failed!" << endl;
    }
  }
}

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
}
