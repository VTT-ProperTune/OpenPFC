#include <openpfc/openpfc.hpp>
#include <openpfc/ui.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace pfc;
using namespace pfc::utils;
using namespace pfc::ui;
using namespace std;

class Aluminum : public Model {
  using Model::Model;

private:
  std::vector<double> filterMF, opL, opN, opEps, P_F;
#ifdef MAHTI_HACK
  // in principle, we can reuse some of the arrays ...
  std::vector<double> psiMF, psi, &psiN = psiMF;
  std::vector<std::complex<double>> psiMF_F, psi_F, &psiN_F = psiMF_F;
#else
  std::vector<double> psiMF, psi, psiN, P_star_psi, temperature, stress;
  std::vector<std::complex<double>> psiMF_F, psi_F, psiN_F, P_psi_F, temperature_F, stress_F;
#endif
  size_t mem_allocated = 0;
  bool m_first = true;

public:
  /**
   * @brief Model parameters, which can be overridden from json file
   *
   */
  struct {
    // average density of the metastable fluid
    double n0;
    // Bulk densities at coexistence, obtained from phase diagram for chosen
    // temperature
    double n_sol, n_vap;
    // Effective temperature parameters. Temperature in K. Remember to change
    // n_sol and n_vap according to phase diagram when T is changed.
    double T, T0, Bx;
	double T_const, T_max, T_min;
	double G_grid, V_grid, x_initial;
	double m_xpos;
    // width of C2's peak
    double alpha;
    // how much we allow the k=1 peak to affect the k=0 value of the
    // correlation, by changing the higher order components of the Gaussian
    // function
    double alpha_farTol;
    // power of the higher order component of the gaussian function. Should be a
    // multiple of 2. Setting this to zero also disables the tolerance setting.
    int alpha_highOrd;
    // derived dimensionless values used in calculating vapor model parameters
    double tau_const;
    // Strength of the meanfield filter. Avoid values higher than ~0.28, to
    // avoid lattice-wavelength variations in the mean field
    double lambda;
    // numerical stability parameter for the exponential integrator method
    double stabP;
    // Vapor-model parameters
    double shift_u, shift_s;
    double p2, p3, p4, p2_bar, p3_bar, p4_bar;
    double q20, q21, q30, q31, q40;
    double q20_bar, q21_bar, q30_bar, q31_bar, q40_bar, q2_bar, q3_bar, q4_bar;
	double q2_bar_L;
  } params;

  void allocate() {
    FFT &fft = get_fft();
    auto size_inbox = fft.size_inbox();
    auto size_outbox = fft.size_outbox();

    // operators are only half size due to the symmetry of fourier space
    filterMF.resize(size_outbox);
    opL.resize(size_outbox);
    opN.resize(size_outbox);
	opEps.resize(size_outbox);
	P_F.resize(size_outbox);

    // psi, psiMF, psiN
    psi.resize(size_inbox);
    psiMF.resize(size_inbox);
    psiN.resize(size_inbox);
	P_star_psi.resize(size_inbox);
	temperature.resize(size_inbox);
	stress.resize(size_inbox);

    // psi_F, psiMF_F, psiN_F, where suffix F means in fourier space
    psi_F.resize(size_outbox);
    psiMF_F.resize(size_outbox);
    psiN_F.resize(size_outbox);
	P_psi_F.resize(size_outbox);
	stress_F.resize(size_outbox);

    add_real_field("psi", psi);
    add_real_field("default", psi); // for backward compatibility
    add_real_field("psiMF", psiMF);
	add_real_field("psiN", psiN);
	add_real_field("P_star_psi", P_star_psi);
	add_real_field("temperature", temperature);
	add_real_field("stress", stress);

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
	mem_allocated += utils::sizeof_vec(P_F);
	mem_allocated += utils::sizeof_vec(P_psi_F);
	mem_allocated += utils::sizeof_vec(P_star_psi);
	mem_allocated += utils::sizeof_vec(temperature);
	mem_allocated += utils::sizeof_vec(stress);
	mem_allocated += utils::sizeof_vec(stress_F);
  }

  void prepare_operators(double dt) {
    World w = get_world();
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;

    const Decomposition &decomp = get_decomposition();
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
          double alpha2 = 2.0 * params.alpha * params.alpha;
          double lambda2 = 2.0 * params.lambda * params.lambda;
          double fMF = exp(kLap / lambda2);
          double k = sqrt(-kLap) - 1.0;
          double k2 = k * k;

	double kp = sqrt(-kLap) - 2. / sqrt(3.0);
	double kp2 = kp * kp;

          double rTol = -alpha2 * log(params.alpha_farTol) - 1.0;

/*
          double g1 = 0;
          if (params.alpha_highOrd == 0) { // gaussian peak
            g1 = exp(-k2 / alpha2);
          } else { // quasi-gaussian peak with higher order component to make it
                   // decay faster towards k=0
            g1 = exp(-(k2 + rTol * pow(k, params.alpha_highOrd)) / alpha2);
          }
*/

	double g1 = exp(-k2 / alpha2);
	double gp1 = exp(-kp2 / alpha2);

	double peak = (g1 > gp1) ? g1 : gp1;

	P_F[idx] = -params.Bx * exp(-params.tau_const) * peak;

          // taylor expansion of gaussian peak to order 2
//          double g2 = 1.0 - 1.0 / alpha2 * k2;
          // splice the two sides of the peak
//          double gf = (k < 0.0) ? g1 : g2;
          // we separate this out because it is needed in the nonlinear
          // calculation when T is not constant in space
//          double opPeak = -params.Bx * exp(-params.T / params.T0) * gf;
          // includes the lowest order n_mf term since it is a linear term
          double opCk =
              params.stabP + params.p2_bar + P_F[idx] + params.q2_bar_L * fMF;

          filterMF[idx] = fMF;
          opL[idx] = exp(kLap * opCk * dt);
          opN[idx] = (opCk == 0.0) ? kLap * dt : (opL[idx] - 1.0) / opCk;

	double alpha2new = 2.0 * params.alpha * params.alpha / 10.0;
	double g1new = exp(-k2 / alpha2new);
	double gp1new = exp(-kp2 / alpha2new);

	double peaknew = (g1new > gp1new) ? g1new : gp1new;

	opEps[idx] = peaknew;

          idx += 1;
        }
      }
    }
  }

  void initialize(double dt) override {
    allocate();
    prepare_operators(dt);
  }

  void step(double t) override {

    (void)t; // suppress compiler warning about unused parameter

    FFT &fft = get_fft();

    // Calculate mean-field density n_mf
    fft.forward(psi, psi_F);
    for (size_t idx = 0, N = psiMF_F.size(); idx < N; idx++)
      psiMF_F[idx] = filterMF[idx] * psi_F[idx];
    fft.backward(psiMF_F, psiMF);

	for (size_t idx = 0, N = P_psi_F.size(); idx < N; idx++) {
	  P_psi_F[idx] = P_F[idx] * psi_F[idx];
	}

	fft.backward(P_psi_F, P_star_psi);

/*

    // Calculate the nonlinear part of the evolution equation in a real space
    for (size_t idx = 0, N = psiN.size(); idx < N; idx++) {
      double u = psi[idx], v = psiMF[idx];
      double u2 = u * u, u3 = u * u * u, v2 = v * v, v3 = v * v * v;
      double p3 = params.p3_bar, p4 = params.p4_bar;
      double q3 = params.q3_bar, q4 = params.q4_bar;
      psiN[idx] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
    }
*/

	World w = get_world();
	auto dx = w.dx;
	auto x0 = w.x0;
	auto Lx = w.Lx;

	const Decomposition &decomp = get_decomposition();

	std::array<int, 3> low = decomp.inbox.low;
	std::array<int, 3> high = decomp.inbox.high;

	double l = Lx * dx;
	double xpos = fmod(params.m_xpos, l);
	double local_FE = 0;

	size_t idx = 0;
	for (int k = low[2]; k <= high[2]; k++) {
		for (int j = low[1]; j <= high[1]; j++) {
			for (int i = low[0]; i <= high[0]; i++) {
			  double x = x0 + i * dx;
			  double T_var = params.G_grid * (x - params.x_initial - params.V_grid * t);
			  temperature[idx] = T_var;
			  double q2_bar_N = params.q21_bar * T_var / params.T0;
			  double q3_bar = params.q31_bar * (params.T_const + T_var) / params.T0 + params.q30_bar;
			  double u = psi[idx];
			  double v = psiMF[idx];
			  double kernel_term_N = (1.0 - exp(-T_var / params.T0)) * P_star_psi[idx];
			  psiN[idx] = params.p3_bar * u * u + params.p4_bar * u * u * u +
			    q2_bar_N * v + q3_bar * v * v + params.q4_bar * v * v * v + kernel_term_N;

			  local_FE += params.p3_bar * u * u * u / 3. +
			    params.p4_bar * u * u * u * u / 4. + q2_bar_N * u * v / 2. +
			    q3_bar * u * v * v / 3. + params.q4_bar * u * v * v * v / 4. +
			    -u * kernel_term_N * u / 2. + -u * P_star_psi[idx] / 2. +
			    params.p2_bar * u * u / 2. + params.q2_bar * u * v / 2.;

			  idx++;
			}
		}
	}

    // Apply stabilization factor if given in parameters
    if (params.stabP != 0.0)
      for (size_t idx = 0, N = psiN.size(); idx < N; idx++)
        psiN[idx] = psiN[idx] - params.stabP * psi[idx];

    // Fourier transform of the nonlinear part of the evolution equation
    fft.forward(psiN, psiN_F);

    // Apply one step of the evolution equation
    for (size_t idx = 0, N = psi_F.size(); idx < N; idx++)
      psi_F[idx] = opL[idx] * psi_F[idx] + opN[idx] * psiN_F[idx];

    // Inverse Fourier transform result back to real space
    fft.backward(psi_F, psi);
  }

}; // end of class

/**
 * @brief Read model configuration from json file, under model/params.
 *
 * @param j json file
 * @param m model
 */
void from_json(const json &j, Aluminum &m) {
  auto &p = m.params;
  j.at("n0").get_to(p.n0);
  j.at("n_sol").get_to(p.n_sol);
  j.at("n_vap").get_to(p.n_vap);
  j.at("T0").get_to(p.T0);
  j.at("Bx").get_to(p.Bx);
	j.at("G_grid").get_to(p.G_grid);
	j.at("V_grid").get_to(p.V_grid);
	j.at("x_initial").get_to(p.x_initial);
	  p.m_xpos = p.x_initial;
	j.at("T_const").get_to(p.T_const);
	j.at("T_max").get_to(p.T_max);
	j.at("T_min").get_to(p.T_min);
  j.at("alpha").get_to(p.alpha);
  j.at("alpha_farTol").get_to(p.alpha_farTol);
  j.at("alpha_highOrd").get_to(p.alpha_highOrd);
	p.tau_const = p.T_const / p.T0;
  j.at("lambda").get_to(p.lambda);
  j.at("stabP").get_to(p.stabP);
  j.at("shift_u").get_to(p.shift_u);
  j.at("shift_s").get_to(p.shift_s);
  j.at("p2_bar").get_to(p.p2_bar);
  j.at("p3_bar").get_to(p.p3_bar);
  j.at("p4_bar").get_to(p.p4_bar);
/*
  p.p2_bar = p.p2 + 2 * p.shift_s * p.p3 + 3 * pow(p.shift_s, 2) * p.p4;
  p.p3_bar = p.shift_u * (p.p3 + 3 * p.shift_s * p.p4);
  p.p4_bar = pow(p.shift_u, 2) * p.p4;
*/
  j.at("q20_bar").get_to(p.q20_bar);
  j.at("q21_bar").get_to(p.q21_bar);
  j.at("q30_bar").get_to(p.q30_bar);
  j.at("q31_bar").get_to(p.q31_bar);
  j.at("q40_bar").get_to(p.q40_bar);
/*
  p.q20_bar = p.q20 + 2.0 * p.shift_s * p.q30 + 3.0 * pow(p.shift_s, 2) * p.q40;
  p.q21_bar = p.q21 + 2.0 * p.shift_s * p.q31;
  p.q30_bar = p.shift_u * (p.q30 + 3.0 * p.shift_s * p.q40);
  p.q31_bar = p.shift_u * p.q31;
  p.q40_bar = pow(p.shift_u, 2) * p.q40;
*/
  p.q2_bar = p.q21_bar * p.tau_const + p.q20_bar;
	p.q2_bar_L = p.q2_bar;
  p.q3_bar = p.q31_bar * p.tau_const + p.q30_bar;
  p.q4_bar = p.q40_bar;
}

class SeedFCC {

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
    const double s = 1.0 / sqrt(3.0);
    const vec3 q1 = {-s, s, s};
    const vec3 q2 = {s, -s, s};
    const vec3 q3 = {s, s, -s};
    const vec3 q4 = {s, s, s};
    mat3 Ra = yaw(orientation[0]);
    mat3 Rb = pitch(orientation[1]);
    mat3 Rc = roll(orientation[2]);
    mat3 R = mult3(Ra, mult3(Rb, Rc));
    const vec36 q = {mult3(R, q1), mult3(R, q2), mult3(R, q3),
                     mult3(R, q4)};
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
  SeedFCC(const vec3 &location, const vec3 &orientation, const double radius,
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
    for (int i = 0; i < 4; i++) {
      u += 2.0 * a * cos(q[i][0] * x + q[i][1] * y + q[i][2] * z);
    }
    return u;
  }
}; // SeedFCC

class SeedGridFCC : public FieldModifier {
private:
  int m_Nx, m_Ny, m_Nz;
  double m_X0, m_radius;
  double m_amplitude;
  double m_rho;
  double m_rseed;

public:
  SeedGridFCC(int Ny, int Nz, double X0, double radius, double amplitude, double rho, double rseed)
      : m_Nx(1), m_Ny(Ny), m_Nz(Nz), m_X0(X0), m_radius(radius), m_amplitude(amplitude), m_rho(rho), m_rseed(rseed) {}

  void apply(Model &m, double) override {
    auto &p = dynamic_cast<Aluminum &>(m).params;
    const World &w = m.get_world();
    const Decomposition &decomp = m.get_decomposition();
    Field &field = m.get_field();
    Vec3<int> low = decomp.inbox.low;
    Vec3<int> high = decomp.inbox.high;

    // auto Lx = w.Lx;
    auto Ly = w.Ly;
    auto Lz = w.Lz;
    auto dx = w.dx;
    auto dy = w.dy;
    auto dz = w.dz;
    auto x0 = w.x0;
    auto y0 = w.y0;
    auto z0 = w.z0;

    std::vector<SeedFCC> seeds;

    int Nx = m_Nx;
    int Ny = m_Ny;
    int Nz = m_Nz;
    double radius = m_radius;
    double rseed = m_rseed;

    double rho = p.n_sol;
    // double amplitude = p.amp_eq;
    double amplitude = m_amplitude;

    double Dy = dy * Ly / Ny;
    double Dz = dz * Lz / Nz;
    double X0 = m_X0;
    double Y0 = Dy / 2.0;
    double Z0 = Dz / 2.0;
    int nseeds = Nx * Ny * Nz;

    cout << "Generating " << nseeds << " regular seeds with radius " << radius
         << "\n";

//    srand(42);
    std::mt19937_64 re(rseed);
    std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
    std::uniform_real_distribution<double> rr(0.0, 8.0 * atan(1.0));
//    std::default_random_engine re;

    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        const std::array<double, 3> location = {
            X0 + rt(re), Y0 + Dy * j + rt(re), Z0 + Dz * k + rt(re)};
        const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
        const SeedFCC seed(location, orientation, radius, rho, amplitude);
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
}; // SeedGridFCC

// Parse initial condition from json file
namespace pfc {
namespace ui {
template <>
std::unique_ptr<SeedGridFCC> from_json<std::unique_ptr<SeedGridFCC>>(const json &params) {
  std::cout << "Parsing SeedGridFCC from json" << std::endl;
  if (!params.contains("amplitude") || !params["amplitude"].is_number()) {
    throw std::invalid_argument(
        "Reading SeedGridFCC failed: missing or invalid 'amplitude' field.");
  }
  double Ny = params["Ny"];
  double Nz = params["Nz"];
  double X0 = params["X0"];
  double radius = params["radius"];
  double amplitude = params["amplitude"];
  double rho = params["rho"];
  double rseed;
  if (!params.contains("rseed") || !params["rseed"].is_number()) {
    std::cout << "No valid random seed detected, using default value 0." << std::endl;
    rseed = 0.;
  } else {
    rseed = params["rseed"];
  }
  return std::make_unique<SeedGridFCC>(Ny, Nz, X0, radius, amplitude, rho, rseed);
}
}  // namespace ui
}  // namespace pfc


int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  register_field_modifier<SeedGridFCC>("seed_grid_fcc");
  App<Aluminum> app(argc, argv);
  return app.main();
}
