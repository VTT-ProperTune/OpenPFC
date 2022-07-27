#include <iostream>
#include <pfc/model.hpp>
#include <pfc/world.hpp>

using namespace std;
using namespace pfc;

class Diffusion : public Model {
  using Model::Model;

private:
  vector<double> opL, psi;
  vector<complex<double>> psi_F;
  const bool verbose = false;

public:
  void allocate() {
    if (master) cout << "Allocate space" << endl;
    opL.resize(size_outbox());
    psi.resize(size_inbox());
    psi_F.resize(size_outbox());
  }

  void create_initial_condition() {
    if (master) cout << "Create initial condition" << endl;
    auto w = get_world();
    auto low = get_inbox_low();
    auto high = get_inbox_high();
    double D = w.Lx / 4.0;
    int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          psi[idx++] = exp(-(x * x) / (4.0 * D));
        }
      }
    }
    if (master && verbose)
      for (int i = 0, N = psi.size(); i < N; i++) {
        cout << "psi[" << i << "] = " << psi[i] << endl;
      }
  }

  void prepare_operators(double dt) {
    if (master) cout << "Prepare operators" << endl;
    auto w = get_world();
    auto low = get_outbox_low();
    auto high = get_outbox_high();
    int idx = 0;
    const double pi = std::atan(1.0) * 4.0;
    const double fx = 2.0 * pi / (w.dx * w.Lx);
    const double fy = 2.0 * pi / (w.dy * w.Ly);
    const double fz = 2.0 * pi / (w.dz * w.Lz);
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          // laplacian operator -k^2
          const double ki = (i <= w.Lx / 2) ? i * fx : (i - w.Lx) * fx;
          const double kj = (j <= w.Ly / 2) ? j * fy : (j - w.Ly) * fy;
          const double kk = (k <= w.Lz / 2) ? k * fz : (k - w.Lz) * fz;
          const double kLap = -(ki * ki + kj * kj + kk * kk);
          if (master && verbose)
            cout << "idx = " << idx << ", ki = " << ki << ", kj " << kj
                 << ", kk = " << kk << ", kLap = " << kLap << endl;
          opL[idx++] = 1.0 / (1.0 - dt * kLap);
        }
      }
    }
    if (master && verbose)
      for (int i = 0, N = opL.size(); i < N; i++) {
        cout << "opL[" << i << "] = " << opL[i] << endl;
      }
  }

  void initialize(double dt) {
    allocate();
    create_initial_condition();
    prepare_operators(dt);
  }

  vector<double> &get_field() { return psi; }

  void step() {
    fft_r2c(psi, psi_F);
    for (int k = 0, N = psi_F.size(); k < N; k++) {
      psi_F[k] = opL[k] * psi_F[k];
    }
    fft_c2r(psi_F, psi);
  }
};

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(12);
  MPI_Init(&argc, &argv);
  {
    int Lx = 64;
    double dx = 2.0 * constants::pi / 8.0;
    double x0 = -0.5 * Lx * dx;
    double dt = 1.0;
    Diffusion D({Lx, 1, 1}, {x0, 0.0, 0.0}, {dx, 1.0, 1.0});
    D.initialize(dt);
    vector<double> &field = D.get_field();
    double t = 0.0;
    cout << "field[32] = " << field[32] << endl;
    while (t <= 3.0 / 4.0 * Lx) {
      D.step();
      cout << "field[32] = " << field[32] << endl;
      t += dt;
    }
  }
  MPI_Finalize();
  return 0;
}
