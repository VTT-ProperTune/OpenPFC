#include <pfc/pfc.hpp>

struct diffusion : pfc::simulation {

  const double a = 1.0; // diffusion constant

  double L(double x, double y, double z) {
    return -a * k2(x, y, z);
  }

  void write_results(unsigned long int n, double t, MPI_Datatype) {
    /*
    int i = rint((x - x0) / dx);
    int j = rint((y - y0) / dy);
    int k = rint((z - z0) / dz);
    bool inside = low[0] <= i && i < high[0] && low[1] <= j && j < high[1] &&
                  low[2] <= k && k < high[2];
    if (!inside) {
      return;
    }
    int idx = (k - low[2]) * size[0] * size[1] + (j - low[1]) * size[0] +
              (i - low[0]);
    std::cout << "n = " << n << ", t = " << t << ", u = " << idx << u[idx]
              << std::endl;
    */
  }
};

int main(int argc, char *argv[]) {

  pfc::simulation *s = new diffusion();
  // define simulation domain
  // X(i, j, k) = { x0 + i*dx, y0 + j*dy, z0 + k*dz }
  std::array<double, 3> O = {-64.0, -64.0, -64.0};
  std::array<double, 3> d = {1.0, 1.0, 1.0};
  std::array<int, 3> L = {128, 128, 128};
  s->set_domain(O, d, L);
  // define simulation time
  // t(n) = t0 + n*dt
  s->set_time(0.0, 10.0, 1.0);

  MPI_Init(&argc, &argv);
  MPI_Solve(s);
  MPI_Finalize();

  return 0;
}
