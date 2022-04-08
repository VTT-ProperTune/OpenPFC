// heFFTe implementation of pfc code

#include <argparse/argparse.hpp>
#include <pfc/pfc.hpp>

struct basicpfc : pfc::simulation {

  const std::string description = "A basic phase field crystal model";
  const double Bx = 1.3;
  const double Bl = 1.0;
  const double p2 = -1.0 / 2.0;
  const double p3 = 1.0 / 3.0;

  double L(double x, double y, double z) {
    auto k2i = k2(x, y, z);
    auto k4i = pow(k2i, 2);
    auto C = -Bx * (-2.0 * k2i + k4i);
    return -k2i * (Bl - C);
  }

  double f(double u) {
    return p2 * u * u + p3 * u * u * u;
  }

  double u0(double x, double y, double z) {
    const double A = 1.0;
    const double n_os = -0.04;
    const double n_ol = -0.05;
    auto R = 20.0;
    if (x * x + y * y + z * z > R * R) {
      return n_ol;
    }
    double cx = cos(x) * dx;
    double cy = cos(y) * dy;
    double cz = cos(z) * dz;
    return n_os + A * (cx * cy + cy * cz + cz * cx);
  }

  void tune_dt(unsigned long int n, double t) {
    return;
    int nmax = max_iters;
    double tmax = t1;
    double dt0 = 1.0;
    double tau = 3.0;
    double tnext = dt0 * (n + 1.0) +
                   pow(1.0 * (n + 1.0) / nmax, tau) * (t1 - dt0 * (n + 1.0));
    set_dt(std::max(dt0, tnext - t));
  }

  bool writeat(unsigned long int n, double t) {
    return true;
  }
};

int main(int argc, char *argv[]) {

  argparse::ArgumentParser program("diffusion");

  program.add_argument("--verbose")
      .help("increase output verbosity")
      .default_value(true)
      .implicit_value(true);

  program.add_argument("--Lx")
      .help("Number of grid points in x direction")
      .scan<'i', int>()
      .default_value(128);

  program.add_argument("--Ly")
      .help("Number of grid points in y direction")
      .scan<'i', int>()
      .default_value(128);

  program.add_argument("--Lz")
      .help("Number of grid points in z direction")
      .scan<'i', int>()
      .default_value(128);

  program.add_argument("--results-dir")
      .help("Where to write results")
      .default_value(".");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  pfc::simulation *s = new diffusion();
  auto Lx = program.get<int>("--Lx");
  auto Ly = program.get<int>("--Ly");
  auto Lz = program.get<int>("--Lz");
  double pi = std::atan(1.0) * 4.0;
  double dx = 2.0 * pi / 8.0;
  double dy = dx;
  double dz = dx;
  double x0 = -0.5 * Lx * dx;
  double y0 = -0.5 * Ly * dy;
  double z0 = -0.5 * Lz * dz;

  s->set_domain({x0, y0, z0}, {dx, dx, dx}, {Lx, Ly, Lz});
  s->set_time(0.0, 10.0, 1.0);
  s->set_max_iters(10);
  // auto results_dir = program.get<std::string>("--results-dir");
  s->set_results_dir(".");

  MPI_Init(&argc, &argv);
  MPI_Solve(s);
  MPI_Finalize();

  return 0;
}
