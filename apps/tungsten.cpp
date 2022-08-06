#include <nlohmann/json.hpp>
#include <pfc/model.hpp>
#include <pfc/results_writer.hpp>
#include <pfc/simulator.hpp>
#include <pfc/time.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

using namespace pfc;
using namespace std;

/*
Model parameters
*/
struct params {
  // average density of the metastable fluid
  double n0 = -0.4;

  // Bulk densities at coexistence, obtained from phase diagram for chosen
  // temperature
  double n_sol = -0.047;
  double n_vap = -0.464;

  // Effective temperature parameters. Temperature in K. Remember to change
  // n_sol and n_vap according to phase diagram when T is changed.
  double T = 3300.0;
  double T0 = 156000.0;
  double Bx = 0.8582;

  // parameters that affect elastic and interface energies

  // width of C2's peak
  double alpha = 0.50;

  // how much we allow the k=1 peak to affect the k=0 value of the
  // correlation, by changing the higher order components of the Gaussian
  // function
  double alpha_farTol = 1.0 / 1000.0;

  // power of the higher order component of the gaussian function. Should be a
  // multiple of 2. Setting this to zero also disables the tolerance setting.
  int alpha_highOrd = 4;

  // derived dimensionless values used in calculating vapor model parameters
  double tau = T / T0;

  // Strength of the meanfield filter. Avoid values higher than ~0.28, to
  // avoid lattice-wavelength variations in the mean field
  double lambda = 0.22;

  // numerical stability parameter for the exponential integrator method
  double stabP = 0.2;

  // Vapor-model parameters
  double shift_u = 0.3341;
  double shift_s = 0.1898;

  double p2 = 1.0;
  double p3 = -1.0 / 2.0;
  double p4 = 1.0 / 3.0;
  double p2_bar = p2 + 2 * shift_s * p3 + 3 * pow(shift_s, 2) * p4;
  double p3_bar = shift_u * (p3 + 3 * shift_s * p4);
  double p4_bar = pow(shift_u, 2) * p4;

  double q20 = -0.0037;
  double q21 = 1.0;
  double q30 = -12.4567;
  double q31 = 20.0;
  double q40 = 45.0;

  double q20_bar = q20 + 2.0 * shift_s * q30 + 3.0 * pow(shift_s, 2) * q40;
  double q21_bar = q21 + 2.0 * shift_s * q31;
  double q30_bar = shift_u * (q30 + 3.0 * shift_s * q40);
  double q31_bar = shift_u * q31;
  double q40_bar = pow(shift_u, 2) * q40;

  double q2_bar = q21_bar * tau + q20_bar;
  double q3_bar = q31_bar * tau + q30_bar;
  double q4_bar = q40_bar;

  // calculating approx amplitude. This is related to the phase diagram
  // calculations.
  double rho_seed = n_sol;
  double A_phi = 135.0 * p4_bar;
  double B_phi = 16.0 * p3_bar + 48.0 * p4_bar * rho_seed;
  double C_phi = -6.0 * (Bx * exp(-T / T0)) + 6.0 * p2_bar +
                 12.0 * p3_bar * rho_seed + 18.0 * p4_bar * pow(rho_seed, 2);
  double d = abs(9.0 * pow(B_phi, 2) - 32.0 * A_phi * C_phi);
  double amp_eq = (-3.0 * B_phi + sqrt(d)) / (8.0 * A_phi);

  // for boundary condition
  double rho_low = n_vap;
  double rho_high = n0;
};

class Tungsten : public Model {
  using Model::Model;

private:
  std::vector<double> filterMF, opL, opN;
#ifdef MAHTI_HACK
  // in principle, we can reuse some of the arrays ...
  std::vector<double> psiMF, psi, &psiN = psiMF;
  std::vector<std::complex<double>> psiMF_F, psi_F, &psiN_F = psiMF_F;
#else
  std::vector<double> psiMF, psi, psiN;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F;
#endif
  std::array<double, 10> timing = {0};
  size_t mem_allocated = 0;
  bool m_first = true;
  params p;

public:
  void allocate() {
    FFT &fft = get_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // operators are only half size due to the symmetry of fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);

    // psi, psiMF, psiN
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);

    // psi_F, psiMF_F, psiN_F, where suffix F means in fourier space
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);

    mem_allocated = 0;
    mem_allocated += utils::sizeof_vec(filterMF);
    mem_allocated += utils::sizeof_vec(opL);
    mem_allocated += utils::sizeof_vec(opN);
    mem_allocated += utils::sizeof_vec(psi);
    mem_allocated += utils::sizeof_vec(psiMF);
    mem_allocated += utils::sizeof_vec(psiN);
    mem_allocated += utils::sizeof_vec(psi_F);
    mem_allocated += utils::sizeof_vec(psiMF_F);
    mem_allocated += utils::sizeof_vec(psiN_F);
  }

  void prepare_operators(double dt) {
    World w = get_world();
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;

    Decomposition &decomp = get_decomposition();
    std::array<int, 3> low = decomp.outbox.low;
    std::array<int, 3> high = decomp.outbox.high;

    int idx = 0;
    const double pi = std::atan(1.0) * 4.0;
    const double fx = 2.0 * pi / (dx * Lx);
    const double fy = 2.0 * pi / (dy * Ly);
    const double fz = 2.0 * pi / (dz * Lz);

    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {

          // laplacian operator -k^2
          double ki = (i <= Lx / 2) ? i * fx : (i - Lx) * fx;
          double kj = (j <= Ly / 2) ? j * fy : (j - Ly) * fy;
          double kk = (k <= Lz / 2) ? k * fz : (k - Lz) * fz;
          double kLap = -(ki * ki + kj * kj + kk * kk);

          // mean-field filtering operator (chi) make a C2 that's quasi-gaussian
          // on the left, and ken-style on the right
          double alpha2 = 2.0 * p.alpha * p.alpha;
          double lambda2 = 2.0 * p.lambda * p.lambda;
          double fMF = exp(kLap / lambda2);
          double k = sqrt(-kLap) - 1.0;
          double k2 = k * k;

          double rTol = -alpha2 * log(p.alpha_farTol) - 1.0;
          double g1 = 0;
          if (p.alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k, p.alpha_highOrd)) / alpha2);
          }

          // taylor expansion of gaussian peak to order 2
          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
          double gf = (k < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
          double opPeak = -p.Bx * exp(-p.T / p.T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double opCk = p.stabP + p.p2_bar + opPeak + p.q2_bar * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;
          idx += 1;
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

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (long int idx = 0, N = psiMF_F.size(); idx < N; idx++)
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    fft.backward(psiMF_F, psiMF);

    // Calculate the nonlinear part of the evolution equation in a real space
    for (long int idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx];
      double v = psiMF[idx];
      psiN[idx] = p.p3_bar * u * u + p.p4_bar * u * u * u + p.q3_bar * v * v +
                  p.q4_bar * v * v * v;
    }

    // Apply stabilization factor if given in parameters
    if (p.stabP != 0.0)
      for (long int idx = 0, N = psiN.size(); idx < N; idx++)
        psiN[idx] = psiN[idx] - p.stabP * psi[idx];

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (long int idx = 0, N = psi_F.size(); idx < N; idx++)
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

  Field &get_field() {
    return psi;
  }

}; // end of class

class Seed {

private:
  typedef std::array<double, 3> vec3;
  typedef std::array<vec3, 3> mat3;
  typedef std::array<vec3, 6> vec36;
  typedef std::array<vec3, 2> vec32;

  const vec3 location_;
  const vec3 orientation_;
  const vec36 q_;
  const vec32 bbox_;
  const double rho_;
  const double radius_;
  const double amplitude_;

  mat3 yaw(double a) {
    double ca = cos(a);
    double sa = sin(a);
    return {vec3({ca, -sa, 0.0}), vec3({sa, ca, 0.0}), vec3({0.0, 0.0, 1.0})};
  }

  mat3 pitch(double b) {
    double cb = cos(b);
    double sb = sin(b);
    return {vec3({cb, 0.0, sb}), vec3({0.0, 1.0, 0.0}), vec3({-sb, 0.0, cb})};
  }

  mat3 roll(double c) {
    double cc = cos(c);
    double sc = sin(c);
    return {vec3({1.0, 0.0, 0.0}), vec3({0.0, cc, -sc}), vec3({0.0, sc, cc})};
  }

  mat3 mult3(const mat3 &A, const mat3 &B) {
    mat3 C = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  vec3 mult3(const mat3 &A, const vec3 &b) {
    vec3 c = {0};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        c[i] += A[i][j] * b[j];
      }
    }
    return c;
  }

  vec36 rotate(const vec3 &orientation) {
    const double s = 1.0 / sqrt(2.0);
    const vec3 q1 = {s, s, 0};
    const vec3 q2 = {s, 0, s};
    const vec3 q3 = {0, s, s};
    const vec3 q4 = {s, 0, -s};
    const vec3 q5 = {s, -s, 0};
    const vec3 q6 = {0, s, -s};
    mat3 Ra = yaw(orientation[0]);
    mat3 Rb = pitch(orientation[1]);
    mat3 Rc = roll(orientation[2]);
    mat3 R = mult3(Ra, mult3(Rb, Rc));
    const vec36 q = {mult3(R, q1), mult3(R, q2), mult3(R, q3),
                     mult3(R, q4), mult3(R, q5), mult3(R, q6)};
    return q;
  }

  vec32 bounding_box(const vec3 &location, double radius) {
    const vec3 low = {location[0] - radius, location[1] - radius,
                      location[2] - radius};
    const vec3 high = {location[0] + radius, location[1] + radius,
                       location[2] + radius};
    const vec32 bbox = {low, high};
    return bbox;
  }

  inline bool is_inside_bbox(const vec3 &location) const {
    const vec32 bbox = get_bbox();
    return (location[0] > bbox[0][0]) && (location[0] < bbox[1][0]) &&
           (location[1] > bbox[0][1]) && (location[1] < bbox[1][1]) &&
           (location[2] > bbox[0][2]) && (location[2] < bbox[1][2]);
  }

  double get_radius() const { return radius_; }
  double get_rho() const { return rho_; }
  double get_amplitude() const { return amplitude_; }
  vec3 get_location() const { return location_; }
  vec36 get_q() const { return q_; }
  vec32 get_bbox() const { return bbox_; }

public:
  Seed(const vec3 &location, const vec3 &orientation, const double radius,
       const double rho, const double amplitude)
      : location_(location), orientation_(orientation), q_(rotate(orientation)),
        bbox_(bounding_box(location, radius)), rho_(rho), radius_(radius),
        amplitude_(amplitude) {}

  bool is_inside(const vec3 &X) const {
    /*
    if (!is_inside_bbox(X)) {
      return false;
    }
    */
    const vec3 Y = get_location();
    double x = X[0] - Y[0];
    double y = X[1] - Y[1];
    double z = X[2] - Y[2];
    double r = get_radius();

    return x * x + y * y + z * z < r * r;
  }

  double get_value(const vec3 &location) const {
    double x = location[0];
    double y = location[1];
    double z = location[2];
    double u = get_rho();
    double a = get_amplitude();
    vec36 q = get_q();
    for (int i = 0; i < 6; i++) {
      u += 2.0 * a * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
    }
    return u;
  }
};

class SingleSeed : public FieldModifier {

private:
  params p;

public:
  void apply(Model &m, double) override {
    World &w = m.get_world();
    Decomposition &decomp = m.get_decomposition();
    Field &f = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    double s = 1.0 / sqrt(2.0);
    std::array<double, 3> q1 = {s, s, 0};
    std::array<double, 3> q2 = {s, 0, s};
    std::array<double, 3> q3 = {0, s, s};
    std::array<double, 3> q4 = {s, 0, -s};
    std::array<double, 3> q5 = {s, -s, 0};
    std::array<double, 3> q6 = {0, s, -s};
    std::array<std::array<double, 3>, 6> q = {q1, q2, q3, q4, q5, q6};

    long int idx = 0;
    // double r2 = pow(0.2 * (Lx * dx), 2);
    double r2 = pow(64.0, 2);
    double u;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = x0 + i * dx;
          double y = y0 + j * dy;
          double z = z0 + k * dz;
          bool seedmask = x * x + y * y + z * z < r2;
          if (!seedmask) {
            u = p.n0;
          } else {
            u = p.rho_seed;
            for (int i = 0; i < 6; i++) {
              u +=
                  2.0 * p.amp_eq * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
            }
          }
          f[idx] = u;
          idx += 1;
        }
      }
    }
  }
};

class RandomSeeds : public FieldModifier {
public:
  params p;

  void apply(Model &m, double) override {
    World &w = m.get_world();
    Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    const int nseeds = 150;
    const double radius = 20.0;
    const double rho = p.rho_seed;
    const double amplitude = p.amp_eq;
    const double lower_x = -128.0 + radius;
    const double upper_x = -128.0 + 3 * radius;
    const double lower_y = -128.0;
    const double upper_y = 128.0;
    const double lower_z = -128.0;
    const double upper_z = 128.0;
    srand(42);
    std::uniform_real_distribution<double> rx(lower_x, upper_x);
    std::uniform_real_distribution<double> ry(lower_y, upper_y);
    std::uniform_real_distribution<double> rz(lower_z, upper_z);
    std::uniform_real_distribution<double> ro(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;
    typedef std::array<double, 3> vec3;
    auto random_location = [&re, &rx, &ry, &rz]() {
      return vec3({rx(re), ry(re), rz(re)});
    };
    auto random_orientation = [&re, &ro]() {
      return vec3({ro(re), ro(re), ro(re)});
    };

    for (int i = 0; i < nseeds; i++) {
      const std::array<double, 3> location = random_location();
      const std::array<double, 3> orientation = random_orientation();
      const Seed seed(location, orientation, radius, rho, amplitude);
      seeds.push_back(seed);
    }

    std::fill(field.begin(), field.end(), p.n0);
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              field[idx] = seed.get_value(X);
            }
          }
          idx += 1;
        }
      }
    }
  }
};

class SeedGrid : public FieldModifier {
public:
  params p;

  void apply(Model &m, double) override {
    World &w = m.get_world();
    Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<Seed> seeds;

    int Nx = 1;
    int Ny = 6;
    int Nz = 6;

    double radius = 30.0;
    double rho = p.rho_seed;
    double amplitude = p.amp_eq;

    double Dy = dy * Ly / Ny;
    double Dz = dz * Lz / Nz;
    double X0 = 3 * radius;
    double Y0 = Dy / 2.0;
    double Z0 = Dz / 2.0;
    int nseeds = Nx * Ny * Nz;

    cout << "Generating " << nseeds << " regular seeds with radius " << radius
         << "\n";

    srand(42);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
    std::default_random_engine re;

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {
            X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const Seed seed(location, orientation, radius, rho, amplitude);
        seeds.push_back(seed);
      }
    }

    std::fill(field.begin(), field.end(), p.n0);
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          const double x = x0 + i * dx;
          const double y = y0 + j * dy;
          const double z = z0 + k * dz;
          const std::array<double, 3> X = {x, y, z};
          for (const auto &seed : seeds) {
            if (seed.is_inside(X)) {
              field[idx] = seed.get_value(X);
              break;
            }
          }
          idx += 1;
        }
      }
    }
  }
};

class BinaryReader {

private:
  MPI_Datatype m_filetype;

public:
  void set_domain(const Vec3<int> &arr_global, const Vec3<int> &arr_local,
                  const Vec3<int> &arr_offset) {
    MPI_Type_create_subarray(3, arr_global.data(), arr_local.data(),
                             arr_offset.data(), MPI_ORDER_FORTRAN, MPI_DOUBLE,
                             &m_filetype);
    MPI_Type_commit(&m_filetype);
  };

  MPI_Status read(const std::string &filename, Field &data) {
    MPI_File fh;
    MPI_Status status;
    if (MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY,
                      MPI_INFO_NULL, &fh)) {
      std::cout << "Unable to open file!" << std::endl;
    }
    MPI_File_set_view(fh, 0, MPI_DOUBLE, m_filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, data.data(), data.size(), MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    return status;
  }
};

class FileReader : public FieldModifier {

private:
  std::string m_filename;

public:
  explicit FileReader(const std::string &filename) : m_filename(filename) {}

  void apply(Model &m, double) override {
    Decomposition &d = m.get_decomposition();
    Field &f = m.get_field();
    cout << "Reading initial condition from file" << m_filename << endl;
    BinaryReader reader;
    reader.set_domain(d.world.size, d.inbox.size, d.inbox.low);
    reader.read(m_filename, f);
  }
};

class FixedBC : public FieldModifier {

private:
  params p;

public:
  void apply(Model &m, double) override {
    Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    World &w = m.get_world();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    double xwidth = 20.0;
    double alpha = 1.0;
    double xpos = w.Lx * w.dx - xwidth;
    long int idx = 0;
    for (int k = low[2]; k <= high[2]; k++) {
      for (int j = low[1]; j <= high[1]; j++) {
        for (int i = low[0]; i <= high[0]; i++) {
          double x = w.x0 + i * w.dx;
          if (std::abs(x - xpos) < xwidth) {
            double S = 1.0 / (1.0 + exp(-alpha * (x - xpos)));
            field[idx] = p.rho_low * S + p.rho_high * (1.0 - S);
          }
          idx += 1;
        }
      }
    }
  }
};

/*
Helper functions to construct objects from json file
*/

using json = nlohmann::json;

template <class T> T from_json(const json &settings);

template <> World from_json<World>(const json &settings) {
  int Lx = settings["Lx"];
  int Ly = settings["Ly"];
  int Lz = settings["Lz"];
  double dx = settings["dx"];
  double dy = settings["dy"];
  double dz = settings["dz"];
  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;
  string origo = settings["origo"];
  if (origo == "center") {
    x0 = -0.5 * dx * Lx;
    y0 = -0.5 * dy * Ly;
    z0 = -0.5 * dz * Lz;
  }
  World world({Lx, Ly, Lz}, {x0, y0, z0}, {dx, dy, dz});
  return world;
}

template <> Time from_json<Time>(const json &settings) {
  double t0 = settings["t0"];
  double t1 = settings["t1"];
  double dt = settings["dt"];
  double saveat = settings["saveat"];
  Time time({t0, t1, dt}, saveat);
  return time;
}

class MPI_Worker {
  MPI_Comm m_comm;
  int m_rank, m_num_procs;

public:
  MPI_Worker(int argc, char *argv[], MPI_Comm comm) : m_comm(comm) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_num_procs);
    if (m_rank != 0) mute();
    cout << "MPI_Init(): initialized " << m_num_procs << " processes" << endl;
  }

  ~MPI_Worker() { MPI_Finalize(); }
  int get_rank() const { return m_rank; }
  void mute() { cout.setstate(ios::failbit); }
  void unmute() { cout.clear(); }
};

/*
The main application
*/

class App {
private:
  MPI_Worker m_worker;
  bool rank0;
  json m_settings;
  World m_world;
  Decomposition m_decomp;
  FFT m_fft;
  Time m_time;
  Tungsten m_model;
  Simulator m_simulator;
  double m_steptime = 0.0;
  double m_avg_steptime = 0.0;
  double m_alpha = 0.01;

  // read settings from file if or standard input
  json read_settings(int argc, char *argv[]) {
    json settings;
    if (argc > 1) {
      if (rank0) cout << "Reading input from file " << argv[1] << "\n\n";
      filesystem::path file(argv[1]);
      if (!filesystem::exists(file)) {
        if (rank0) cerr << "File " << file << " does not exist!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::ifstream input_file(file);
      input_file >> settings;
    } else {
      if (rank0) std::cout << "Reading simulation settings from stdin\n\n";
      std::cin >> settings;
    }
    return settings;
  }

public:
  App(int argc, char *argv[], MPI_Comm comm = MPI_COMM_WORLD)
      : m_worker(MPI_Worker(argc, argv, comm)), rank0(m_worker.get_rank() == 0),
        m_settings(read_settings(argc, argv)),
        m_world(from_json<World>(m_settings)),
        m_decomp(Decomposition(m_world, comm)), m_fft(FFT(m_decomp, comm)),
        m_time(from_json<Time>(m_settings)),
        m_model(Tungsten(m_world, m_decomp, m_fft)),
        m_simulator(Simulator(m_world, m_decomp, m_fft, m_model, m_time)) {}

  bool create_results_dir() {
    filesystem::path results_dir(m_settings["results"].get<string>());
    if (results_dir.has_filename()) results_dir = results_dir.parent_path();
    if (!std::filesystem::exists(results_dir)) {
      cout << "Results dir " << results_dir << " does not exist, creating\n";
      filesystem::create_directories(results_dir);
      return true;
    } else {
      cout << "Warning: results dir " << results_dir << "already exists\n";
      return false;
    }
  }

  int main() {
    cout << m_settings.dump(4) << "\n\n";
    cout << "World: " << m_world << endl;
    if (rank0) create_results_dir();

    cout << "Adding results writer" << endl;
    m_simulator.add_results_writer(
        make_unique<BinaryWriter>(m_settings["results"]));

    cout << "Adding initial conditions" << endl;
    auto ic = m_settings["initial_condition"];
    if (ic["type"] == "single_seed") {
      cout << "Adding single seed initial condition" << endl;
      m_simulator.add_initial_conditions(make_unique<SingleSeed>());
    } else if (ic["type"] == "random_seeds") {
      cout << "Adding randomized seeds initial condition" << endl;
      m_simulator.add_initial_conditions(make_unique<RandomSeeds>());
    } else if (ic["type"] == "seed_grid") {
      cout << "Adding seed grid initial condition" << endl;
      m_simulator.add_initial_conditions(make_unique<SeedGrid>());
    } else if (ic["type"] == "from_file") {
      cout << "Reading initial condition from file" << endl;
      string filename = ic["filename"];
      cout << "Reading from file: " << filename << endl;
      m_simulator.add_initial_conditions(make_unique<FileReader>(filename));
      int result_counter = ic["result_counter"];
      result_counter += 1;
      m_simulator.set_result_counter(result_counter);
      m_time.set_increment(ic["increment"]);
    } else {
      cout << "Warning: unknown initial condition " << ic["type"] << endl;
    }

    cout << "Adding boundary conditions" << endl;
    auto bc = m_settings["boundary_condition"];
    if (bc["type"] == "fixed") {
      cout << "Adding fixed bc" << endl;
      m_simulator.add_boundary_conditions(make_unique<FixedBC>());
    } else {
      cout << "Warning: unknown boundary condition " << bc["type"] << endl;
    }

    m_simulator.apply_initial_conditions();
    if (m_time.get_increment() == 0) {
      m_simulator.apply_boundary_conditions();
      m_simulator.write_results();
    }

    while (!m_time.done()) {
      m_time.next(); // increase increment counter by 1
      m_simulator.apply_boundary_conditions();
      m_steptime = -MPI_Wtime();
      m_model.step(m_time.get_dt());
      m_steptime += MPI_Wtime();
      m_avg_steptime =
          (m_time.get_increment() <= 5)
              ? m_steptime
              : m_alpha * m_steptime + (1.0 - m_alpha) * m_avg_steptime;
      if (m_time.do_save()) {
        m_simulator.write_results();
      }
      cout << "Step " << m_time.get_increment() << " done in " << m_steptime
           << " seconds. Simulation time: " << m_time.get_current() << " / "
           << m_time.get_t1() << ". ETA: ";
      double eta_i = (m_time.get_t1() - m_time.get_current()) / m_time.get_dt();
      double eta_t = eta_i * m_avg_steptime;
      if (eta_t > 86400.0) {
        cout << eta_t / 86400 << " days" << endl;
      } else if (eta_t > 3600) {
        cout << eta_t / 3600 << " hours " << endl;
      } else if (eta_t > 60) {
        cout << eta_t / 60 << " minutes" << endl;
      } else {
        cout << eta_t << " seconds" << endl;
      }
    }

    return 0;
  }
};

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  return App(argc, argv).main();
}
