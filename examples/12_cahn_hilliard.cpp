#include "11_write_results.hpp"
#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>
#include <random>

using namespace pfc;

const double PI = 3.141592653589793238463;

class CahnHilliard : public Model {
private:
  std::vector<double> opL, opN, c;             // Define linear operator opL and unknown (real) psi
  std::vector<std::complex<double>> c_F, c_NF; // Define (complex) psi
  double gamma = 1.0e-2;                       // Surface tension
  double D = 1.0;                              // Diffusion coefficient

public:
  void initialize(double dt) override {
    FFT &fft = get_fft();
    const Decomposition &decomp = get_decomposition();

    // Allocate space for the main variable and it's fourier transform
    c.resize(fft.size_inbox());
    c_F.resize(fft.size_outbox());
    c_NF.resize(fft.size_outbox());
    opL.resize(fft.size_outbox());
    opN.resize(fft.size_outbox());
    add_real_field("concentration", c);

    // prepare operators
    World w = get_world();
    std::array<int, 3> o_low = decomp.outbox.low;
    std::array<int, 3> o_high = decomp.outbox.high;
    size_t idx = 0;
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
          double L = kLap * (-D - D * gamma * kLap);
          opL[idx] = std::exp(L * dt);
          opN[idx] = (L != 0.0) ? (opL[idx] - 1.0) / L * kLap : 0.0;
          idx++;
        }
      }
    }
  }

  void step(double) override {
    FFT &fft = get_fft();
    fft.forward(c, c_F);
    for (auto &elem : c) elem = D * elem * elem * elem;
    fft.forward(c, c_NF);
    for (size_t i = 0; i < c_F.size(); i++) c_F[i] = opL[i] * c_F[i] + opN[i] * c_NF[i];
    fft.backward(c_F, c);
  }
};

/**
 * @brief sprintf function for std::string
 *
 * @param fmt
 * @param ...
 * @return std::string
 */
std::string sprintf(const char *fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

/**
 * @brief Main function
 *
 * @return int
 */
int main(int argc, char **argv) {
  MPI_Worker worker(argc, argv);

  int Lx = 512;
  int Ly = 512;
  int Lz = 1;
  double dx = 20.0 / Lx;
  double dy = 20.0 / Ly;
  double dz = 1.0;
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;

  // Construct world, decomposition, fft and model
  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
  Decomposition decomp(world);
  FFT fft(decomp);
  CahnHilliard model;
  model.set_fft(fft);

  // Define time
  double t = 0.0;
  double t_stop = 1.0;
  double dt = 1.0e-3;
  int n = 0; // increment counter

  // Initialize the model before starting time stepping
  model.initialize(dt);

  // get the concentration field and fill it with random numbers
  std::vector<double> &field = model.get_real_field("concentration");
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &elem : field) elem = dist(rng);

  // initialize VtkWriter
  VtkWriter<double> writer;
  int file_count = 0;
  // set uri as format cahn_hilliard_%04i.vti, where %04i is replaced by file_count
  writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
  writer.set_field_name("concentration");
  writer.set_domain(world.get_size(), decomp.get_inbox_size(), decomp.get_inbox_offset());
  writer.set_origin(world.get_origin());
  writer.set_spacing(world.get_discretization());
  writer.initialize();
  writer.write(field);

  // Initialize high-precision clock
  auto t_start = std::chrono::high_resolution_clock::now();
  // Loop until we are in t_stop
  while (t <= t_stop) {
    model.step(dt);
    if (n % 10 == 0) {
      if (worker.get_rank() == 0) std::cout << "t = " << t << std::endl;
      writer.set_uri(sprintf("cahn_hilliard_%04i.vti", file_count));
      writer.write(field);
      file_count++;
    }
    t += dt;
    n += 1;
  }

  // Stop the clock
  auto t_end = std::chrono::high_resolution_clock::now();
  // Compute the time difference
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  // Print the time difference
  if (worker.get_rank() == 0) std::cout << "Solution time: " << duration << " ms" << std::endl;

  return 0;
}
