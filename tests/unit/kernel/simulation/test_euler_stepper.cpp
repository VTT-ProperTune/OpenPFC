// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace pfc;
using namespace pfc::sim::steppers;
using Catch::Matchers::WithinRel;

// Helper RHS callable for constant du/dt = c
struct ConstantRHS {
  double c;

  void operator()(double /*t*/, std::vector<double> & /*u*/,
                  std::vector<double> &du) const {
    for (size_t i = 0; i < du.size(); ++i) {
      du[i] = c;
    }
  }
};

// Helper linear ODE RHS: du/dt = -lambda * u
struct LinearDecayRHS {
  double lambda;

  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    for (size_t i = 0; i < u.size(); ++i) {
      du[i] = -lambda * u[i];
    }
  }
};

// Analytical solution for exponential decay: u(t) = u0 * exp(-lambda * t)
double analytical_solution(double u0, double lambda, double t) {
  return u0 * std::exp(-lambda * t);
}

// Helper for L2 error computation
double compute_l2_error(const std::vector<double> &numerical,
                        const std::vector<double> &analytical) {
  double error_sq = 0.0;
  for (size_t i = 0; i < numerical.size(); ++i) {
    double diff = numerical[i] - analytical[i];
    error_sq += diff * diff;
  }
  return std::sqrt(error_sq / numerical.size());
}

// Helper to initialize vector with given value
void init_vector(std::vector<double> &vec, double value) {
  for (auto &v : vec) v = value;
}

// Local problem size for tests
constexpr size_t LOCAL_SIZE = 1000;

TEST_CASE("test_forward_euler_constant_rhs") {
  double c = 2.5;
  ConstantRHS rhs{c};

  std::vector<double> dt_values = {0.001, 0.01, 0.1};
  for (double dt : dt_values) {
    std::vector<double> u(LOCAL_SIZE);
    init_vector(u, 1.0);
    std::vector<double> u_initial = u;

    EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
    double t = 0.0;
    t = stepper.step(t, u);

    for (size_t i = 0; i < u.size(); ++i) {
      double expected = u_initial[i] + dt * c;
      REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
    }
  }

  SECTION("multiple steps") {
    double dt = 0.01;
    int steps = 10;
    ConstantRHS rhs{c};

    std::vector<double> u(LOCAL_SIZE);
    init_vector(u, 1.0);
    std::vector<double> u_initial = u;

    EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
      t = stepper.step(t, u);
    }

    for (size_t i = 0; i < u.size(); ++i) {
      double expected = u_initial[i] + steps * dt * c;
      REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
    }
  }
}

TEST_CASE("test_multieuler_tuple_protocol") {
  // Composite RHS that assigns different constants to each field using tuple
  // protocol
  struct CompositeRHS {
    double c_field1;
    double c_field2;

    void operator()(
        double /*t*/,
        std::tuple<std::vector<double> &, std::vector<double> &> u_pack,
        std::tuple<std::vector<double> &, std::vector<double> &> du_pack) const {
      auto &u1 = std::get<0>(u_pack);
      auto &u2 = std::get<1>(u_pack);
      auto &du1 = std::get<0>(du_pack);
      auto &du2 = std::get<1>(du_pack);

      for (size_t i = 0; i < u1.size(); ++i) {
        du1[i] = c_field1;
        du2[i] = c_field2;
      }
    }
  };

  constexpr int N_FIELDS = 2;
  double c1 = 2.0;
  double c2 = 3.0;

  std::array<std::size_t, N_FIELDS> local_sizes = {LOCAL_SIZE, LOCAL_SIZE};

  double dt = 0.01;

  std::vector<double> u1(LOCAL_SIZE);
  std::vector<double> u2(LOCAL_SIZE);
  init_vector(u1, 1.0);
  init_vector(u2, 1.0);
  std::vector<double> u1_initial = u1;
  std::vector<double> u2_initial = u2;

  CompositeRHS rhs{c1, c2};

  // Use correct template order: Rhs first, N second
  MultiEulerStepper<CompositeRHS, N_FIELDS> multi_stepper(dt, local_sizes, rhs);

  double t = 0.0;
  t = multi_stepper.step(t, u1, u2);

  // Verify tuple protocol: both fields stepped correctly with their RHS values
  for (size_t i = 0; i < u1.size(); ++i) {
    double expected1 = u1_initial[i] + dt * c1;
    double expected2 = u2_initial[i] + dt * c2;
    REQUIRE_THAT(u1[i], WithinRel(expected1, 1e-10));
    REQUIRE_THAT(u2[i], WithinRel(expected2, 1e-10));
  }
}

TEST_CASE("test_linear_ode_exponential_decay") {
  double lambda = 1.0;
  LinearDecayRHS rhs{lambda};

  double dt_coarse = 0.01;
  double dt_fine = 0.005;
  double dt_finer = 0.001;

  double t_final = 0.1;

  // Coarse grid solution
  std::vector<double> u_coarse(LOCAL_SIZE);
  init_vector(u_coarse, 1.0);
  EulerStepper<LinearDecayRHS> stepper_coarse(dt_coarse, LOCAL_SIZE, rhs);
  double t_coarse = 0.0;
  int steps_coarse = static_cast<int>(t_final / dt_coarse);
  for (int i = 0; i < steps_coarse; ++i) {
    t_coarse = stepper_coarse.step(t_coarse, u_coarse);
  }

  // Fine grid solution
  std::vector<double> u_fine(LOCAL_SIZE);
  init_vector(u_fine, 1.0);
  EulerStepper<LinearDecayRHS> stepper_fine(dt_fine, LOCAL_SIZE, rhs);
  double t_fine = 0.0;
  int steps_fine = static_cast<int>(t_final / dt_fine);
  for (int i = 0; i < steps_fine; ++i) {
    t_fine = stepper_fine.step(t_fine, u_fine);
  }

  // Finer grid solution
  std::vector<double> u_finer(LOCAL_SIZE);
  init_vector(u_finer, 1.0);
  EulerStepper<LinearDecayRHS> stepper_finer(dt_finer, LOCAL_SIZE, rhs);
  double t_finer = 0.0;
  int steps_finer = static_cast<int>(t_final / dt_finer);
  for (int i = 0; i < steps_finer; ++i) {
    t_finer = stepper_finer.step(t_finer, u_finer);
  }

  // Analytical solution (uniform)
  std::vector<double> u_analytical(LOCAL_SIZE);
  double u0 = 1.0;
  double exact = analytical_solution(u0, lambda, t_final);
  for (auto &val : u_analytical) val = exact;

  // Compute L2 errors
  double error_coarse = compute_l2_error(u_coarse, u_analytical);
  double error_fine = compute_l2_error(u_fine, u_analytical);
  double error_finer = compute_l2_error(u_finer, u_analytical);

  REQUIRE(error_fine <= error_coarse * 0.6 + 1e-12);
  REQUIRE(error_finer <= error_fine * 0.6 + 1e-12);
}

TEST_CASE("test_rhs_read_only_contract") {
  double c = 1.5;

  // RHS that strictly reads from u and writes only to du
  struct ReadOnlyRHS {
    double c;

    void operator()(double /*t*/, std::vector<double> & /*u*/,
                    std::vector<double> &du) const {
      for (size_t i = 0; i < du.size(); ++i) {
        du[i] = c;
      }
    }
  };

  ReadOnlyRHS rhs{c};

  double dt = 0.01;
  int steps = 5;

  std::vector<double> u(LOCAL_SIZE);
  init_vector(u, 1.0);
  std::vector<double> u_initial = u;

  EulerStepper<ReadOnlyRHS> stepper(dt, LOCAL_SIZE, rhs);
  double t = 0.0;
  for (int i = 0; i < steps; ++i) {
    t = stepper.step(t, u);
  }

  for (size_t i = 0; i < u.size(); ++i) {
    double expected = u_initial[i] + steps * dt * c;
    REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
  }
}

TEST_CASE("test_multistep_consistency") {
  double c = 2.0;
  ConstantRHS rhs{c};

  double dt = 0.01;
  int N = 10;

  // Multi-step evolution
  std::vector<double> u_multi(LOCAL_SIZE);
  init_vector(u_multi, 1.0);
  EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
  double t_multi = 0.0;
  for (int i = 0; i < N; ++i) {
    t_multi = stepper.step(t_multi, u_multi);
  }

  // Single equivalent step
  std::vector<double> u_single(LOCAL_SIZE);
  init_vector(u_single, 1.0);
  EulerStepper<ConstantRHS> stepper_single(dt * N, LOCAL_SIZE, rhs);
  double t_single = 0.0;
  t_single = stepper_single.step(t_single, u_single);

  REQUIRE_THAT(t_multi, WithinRel(t_single, 1e-10));
  for (size_t i = 0; i < u_multi.size(); ++i) {
    REQUIRE_THAT(u_multi[i], WithinRel(u_single[i], 1e-10));
  }
}

TEST_CASE("test_dt_zero_edge_case") {
  double c = 1.0;
  ConstantRHS rhs{c};

  std::vector<double> u(LOCAL_SIZE);
  init_vector(u, 1.0);
  std::vector<double> u_initial = u;

  double t = 1.0;
  double dt_zero = 0.0;

  EulerStepper<ConstantRHS> stepper(dt_zero, LOCAL_SIZE, rhs);
  double t_out = stepper.step(t, u);

  REQUIRE(t_out == t);
  for (size_t i = 0; i < u.size(); ++i) {
    REQUIRE(u[i] == u_initial[i]);
  }
}

using Complex = std::complex<double>;

struct ConstantComplexRHS {
  Complex c{};
  void operator()(double /*t*/, std::vector<Complex> & /*u*/,
                  std::vector<Complex> &du) const {
    for (auto &d : du) {
      d = c;
    }
  }
};

struct TwoFieldConstantComplexRHS {
  Complex c0{};
  Complex c1{};
  void
  operator()(double /*t*/,
             std::tuple<std::vector<Complex> &, std::vector<Complex> &> /*u*/,
             std::tuple<std::vector<Complex> &, std::vector<Complex> &> du) const {
    std::get<0>(du)[0] = c0;
    std::get<1>(du)[0] = c1;
  }
};

TEST_CASE("euler_complex_constant_rhs", "[stepper][euler][complex]") {
  constexpr Complex c{0.25, -0.5};
  constexpr Complex u0{1.0, 2.0};
  constexpr double dt = 0.1;
  ConstantComplexRHS rhs{c};
  EulerStepper<ConstantComplexRHS, Complex> stepper(dt, 1, rhs);
  std::vector<Complex> u{u0};
  const auto fingerprint = u;
  auto attempt = stepper.attempt(0.0, u);
  REQUIRE(attempt.success);
  REQUIRE(u == fingerprint);
  const Complex expected = u0 + Complex(dt) * c;
  REQUIRE(attempt.candidate[0].real() ==
          Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(attempt.candidate[0].imag() ==
          Catch::Approx(expected.imag()).margin(1e-12));
  commit_step_attempt(u, attempt);
  REQUIRE(u[0].real() == Catch::Approx(expected.real()).margin(1e-12));
  REQUIRE(u[0].imag() == Catch::Approx(expected.imag()).margin(1e-12));
  REQUIRE(stepper.step(dt, u) == Catch::Approx(2.0 * dt));
  REQUIRE(u[0].real() ==
          Catch::Approx((expected + Complex(dt) * c).real()).margin(1e-12));
}

TEST_CASE("multi_euler_complex_two_field", "[stepper][euler][complex]") {
  constexpr double dt = 0.2;
  constexpr Complex u0{0.5, -0.25};
  constexpr Complex u1{-1.0, 0.75};
  constexpr Complex c0{0.1, 0.2};
  constexpr Complex c1{-0.3, 0.05};
  TwoFieldConstantComplexRHS rhs{c0, c1};
  MultiEulerStepper<TwoFieldConstantComplexRHS, 2, Complex> stepper(dt, {1, 1}, rhs);
  std::vector<Complex> a{u0};
  std::vector<Complex> b{u1};
  const auto fa = a;
  const auto fb = b;
  auto attempt = stepper.attempt(0.0, a, b);
  REQUIRE(attempt.success);
  REQUIRE(a == fa);
  REQUIRE(b == fb);
  const Complex e0 = u0 + Complex(dt) * c0;
  const Complex e1 = u1 + Complex(dt) * c1;
  REQUIRE(attempt.candidate(0)[0].real() == Catch::Approx(e0.real()).margin(1e-12));
  REQUIRE(attempt.candidate(1)[0].imag() == Catch::Approx(e1.imag()).margin(1e-12));
  commit_step_attempt(a, b, attempt);
  REQUIRE(a[0].real() == Catch::Approx(e0.real()).margin(1e-12));
  REQUIRE(b[0].imag() == Catch::Approx(e1.imag()).margin(1e-12));
}

namespace {

pfc::data::Field<double> slab(int nx, int ny, int halo) {
  const auto domain = pfc::domain::with_spacing({nx, ny, 1}, {1.0, 1.0, 1.0});
  const pfc::Box3i box = pfc::Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, 0});
  return pfc::data::Field<double>(domain, box, halo);
}

pfc::data::Field<double> box3(int nx, int ny, int nz, int halo) {
  const auto domain = pfc::domain::with_spacing({nx, ny, nz}, {1.0, 1.0, 1.0});
  const pfc::Box3i box =
      pfc::Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
  return pfc::data::Field<double>(domain, box, halo);
}

double pattern_value(int i, int j, int k) {
  return 0.5 + 0.1 * i + 0.01 * j + 0.001 * k;
}

void fill_pattern(pfc::data::Field<double> &f, double halo_value) {
  const int h = f.storage_halo();
  const auto n = f.local_size();
  for (int k = -h; k < n[2] + h; ++k) {
    for (int j = -h; j < n[1] + h; ++j) {
      for (int i = -h; i < n[0] + h; ++i) {
        const bool owned =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        f(i, j, k) = owned ? pattern_value(i, j, k) : halo_value;
      }
    }
  }
}

bool same_storage(const pfc::data::Field<double> &a,
                  const pfc::data::Field<double> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t q = 0; q < a.size(); ++q) {
    if (a.vec()[q] != b.vec()[q]) return false;
  }
  return true;
}

void fill_sentinel(pfc::data::Field<double> &f, double halo_value,
                   double owned_value) {
  const int h = f.storage_halo();
  const auto n = f.local_size();
  for (int k = -h; k < n[2] + h; ++k) {
    for (int j = -h; j < n[1] + h; ++j) {
      for (int i = -h; i < n[0] + h; ++i) {
        const bool owned =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        f(i, j, k) = owned ? owned_value : halo_value;
      }
    }
  }
}

bool halo_untouched(const pfc::data::Field<double> &f, double halo_value) {
  const int h = f.storage_halo();
  const auto n = f.local_size();
  for (int k = -h; k < n[2] + h; ++k) {
    for (int j = -h; j < n[1] + h; ++j) {
      for (int i = -h; i < n[0] + h; ++i) {
        const bool owned =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        if (!owned && f(i, j, k) != halo_value) return false;
      }
    }
  }
  return true;
}

struct OwnedEval {
  const pfc::data::Field<double> *field = nullptr;
  double rate = 0.0;
  void prepare() {}
  [[nodiscard]] double operator()(int, int, int) const { return rate; }
  [[nodiscard]] std::size_t idx(int i, int j, int k) const {
    return field->idx(i, j, k);
  }
  [[nodiscard]] int imin() const { return 0; }
  [[nodiscard]] int imax() const { return field->local_size()[0]; }
  [[nodiscard]] int jmin() const { return 0; }
  [[nodiscard]] int jmax() const { return field->local_size()[1]; }
  [[nodiscard]] int kmin() const { return 0; }
  [[nodiscard]] int kmax() const { return field->local_size()[2]; }
};

struct IdentityModel {
  [[nodiscard]] double rhs(double, double g) const { return g; }
};

struct CoupledEval {
  const pfc::data::Field<double> *phi = nullptr;
  void prepare() {}
  struct View {
    double phi;
  };
  [[nodiscard]] View operator()(int i, int j, int k) const {
    return {(*phi)(i, j, k)};
  }
  [[nodiscard]] std::size_t idx(int i, int j, int k) const {
    return phi->idx(i, j, k);
  }
  [[nodiscard]] int imin() const { return 0; }
  [[nodiscard]] int imax() const { return phi->local_size()[0]; }
  [[nodiscard]] int jmin() const { return 0; }
  [[nodiscard]] int jmax() const { return phi->local_size()[1]; }
  [[nodiscard]] int kmin() const { return 0; }
  [[nodiscard]] int kmax() const { return phi->local_size()[2]; }
};

struct CoupledModel {
  struct Inc {
    double phi_rate;
    double u_rate;
    [[nodiscard]] std::tuple<double, double> as_tuple() const {
      return {phi_rate, u_rate};
    }
  };
  /// Temperature rate is the pre-step phase value. An update of phi inside
  /// this call would contaminate u_rate.
  [[nodiscard]] Inc rhs(double, const CoupledEval::View &g) const {
    return {1.0, g.phi};
  }
};

} // namespace

TEST_CASE("owned euler updates a padded 2D slab and leaves the halo",
          "[stepper][euler][owned]") {
  constexpr int nx = 6;
  constexpr int ny = 5;
  constexpr int halo = 2;
  constexpr double dt = 0.25;
  constexpr double halo_value = 9.0;
  auto u = slab(nx, ny, halo);
  fill_sentinel(u, halo_value, 1.0);
  REQUIRE(u.size() == static_cast<std::size_t>((nx + 2 * halo) * (ny + 2 * halo) *
                                               (1 + 2 * halo)));

  OwnedEval eval{&u, 4.0};
  IdentityModel model;
  auto stepper = create_owned(std::tie(u), eval, model, dt);
  REQUIRE_FALSE(stepper.can_rollback());
  const std::size_t bytes = stepper.workspace_bytes();
  REQUIRE(bytes == u.size() * sizeof(double));
  REQUIRE(stepper.step(0.0, u) == Catch::Approx(dt));
  REQUIRE(stepper.workspace_bytes() == bytes);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      REQUIRE(u(i, j, 0) == Catch::Approx(1.0 + dt * 4.0));
    }
  }
  REQUIRE(halo_untouched(u, halo_value));
}

TEST_CASE("owned euler evaluates coupled rates before either update",
          "[stepper][euler][owned]") {
  constexpr double dt = 0.5;
  auto phi = slab(4, 3, 2);
  auto temperature = slab(4, 3, 2);
  fill_sentinel(phi, -1.0, 2.0);
  fill_sentinel(temperature, -1.0, 8.0);
  const double phi0 = phi(1, 1, 0);
  const double u0 = temperature(1, 1, 0);

  CoupledEval eval{&phi};
  CoupledModel model;
  auto stepper = create_owned(std::tie(phi, temperature), eval, model, dt);
  stepper.step(0.0, phi, temperature);

  const auto n = phi.local_size();
  for (int j = 0; j < n[1]; ++j) {
    for (int i = 0; i < n[0]; ++i) {
      REQUIRE(phi(i, j, 0) == Catch::Approx(phi0 + dt * 1.0));
      // Temperature uses the pre-step phase, not the updated phase.
      REQUIRE(temperature(i, j, 0) == Catch::Approx(u0 + dt * phi0));
    }
  }
  REQUIRE(halo_untouched(phi, -1.0));
  REQUIRE(halo_untouched(temperature, -1.0));
}

TEST_CASE("owned euler leaves owned cells unchanged when the rhs throws",
          "[stepper][euler][owned]") {
  auto u = slab(4, 4, 1);
  fill_sentinel(u, 3.0, 1.0);
  // The throw is in the stepper's RHS callable, before the owned update.
  // A throw inside for_each_interior's OpenMP loop is a different contract:
  // that loop does not propagate exceptions (it terminates). This test is
  // the stepper's own guarantee.
  struct Boom {
    void operator()(double, std::tuple<std::vector<double> &> &) const {
      throw std::runtime_error("owned euler rhs failed");
    }
  };
  OwnedEulerStepper<Boom, 1> stepper(0.1, {u.size()}, Boom{});
  REQUIRE_THROWS_AS(stepper.step(0.0, u), std::runtime_error);
  REQUIRE(u(0, 0, 0) == Catch::Approx(1.0));
  REQUIRE(u(3, 3, 0) == Catch::Approx(1.0));
  REQUIRE(halo_untouched(u, 3.0));

  // A halo write that happens before the throw is outside the owned-cell
  // guarantee. The stepper does not roll it back.
  struct HaloThenThrow {
    pfc::data::Field<double> *field = nullptr;
    void operator()(double, std::tuple<std::vector<double> &> &) const {
      (*field)(-1, 0, 0) = 42.0;
      throw std::runtime_error("owned euler halo then throw");
    }
  };
  HaloThenThrow rhs{&u};
  OwnedEulerStepper<HaloThenThrow, 1> later(0.1, {u.size()}, rhs);
  REQUIRE_THROWS_AS(later.step(0.0, u), std::runtime_error);
  REQUIRE(u(0, 0, 0) == Catch::Approx(1.0));
  REQUIRE(u(-1, 0, 0) == Catch::Approx(42.0));
}

TEST_CASE("owned euler rejects a non-finite or non-positive dt",
          "[stepper][euler][owned]") {
  struct Idle {
    void operator()(double, std::tuple<std::vector<double> &> &) const {}
  };
  const std::array<std::size_t, 1> sizes{4};
  REQUIRE_THROWS_AS((OwnedEulerStepper<Idle, 1>(0.0, sizes, Idle{})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((OwnedEulerStepper<Idle, 1>(-0.2, sizes, Idle{})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((OwnedEulerStepper<Idle, 1>(
                        std::numeric_limits<double>::infinity(), sizes, Idle{})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((OwnedEulerStepper<Idle, 1>(
                        std::numeric_limits<double>::quiet_NaN(), sizes, Idle{})),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((OwnedEulerStepper<Idle, 1>(0.1, {0}, Idle{})),
                    std::invalid_argument);
}

TEST_CASE("owned euler matches MultiEuler and a pre-step reference on a 3D box",
          "[stepper][euler][owned]") {
  constexpr int nx = 5;
  constexpr int ny = 4;
  constexpr int nz = 3;
  constexpr int halo = 2;
  constexpr double dt = 0.125;
  constexpr double lambda = 0.25;
  constexpr double halo_value = 7.0;
  constexpr int steps = 3;

  auto owned_field = box3(nx, ny, nz, halo);
  auto multi_field = box3(nx, ny, nz, halo);
  fill_pattern(owned_field, halo_value);
  fill_pattern(multi_field, halo_value);
  REQUIRE(owned_field.local_size()[2] == nz);
  REQUIRE(owned_field.size() >
          static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * nz);

  struct CellEval {
    const pfc::data::Field<double> *field = nullptr;
    void prepare() {}
    [[nodiscard]] double operator()(int i, int j, int k) const {
      return (*field)(i, j, k);
    }
    [[nodiscard]] std::size_t idx(int i, int j, int k) const {
      return field->idx(i, j, k);
    }
    [[nodiscard]] int imin() const { return 0; }
    [[nodiscard]] int imax() const { return field->local_size()[0]; }
    [[nodiscard]] int jmin() const { return 0; }
    [[nodiscard]] int jmax() const { return field->local_size()[1]; }
    [[nodiscard]] int kmin() const { return 0; }
    [[nodiscard]] int kmax() const { return field->local_size()[2]; }
  };
  struct DecayModel {
    double lambda = 0.0;
    [[nodiscard]] double rhs(double, double g) const { return -lambda * g; }
  };

  CellEval owned_eval{&owned_field};
  CellEval multi_eval{&multi_field};
  DecayModel model{lambda};
  auto owned = create_owned(std::tie(owned_field), owned_eval, model, dt);
  auto cloning = create(std::tie(multi_field), multi_eval, model, dt);
  REQUIRE(owned.workspace_bytes() == owned_field.size() * sizeof(double));
  REQUIRE_FALSE(owned.can_rollback());

  const double factor = 1.0 - dt * lambda;
  for (int step = 0; step < steps; ++step) {
    REQUIRE(owned.step(step * dt, owned_field) == Catch::Approx((step + 1) * dt));
    REQUIRE(cloning.step(step * dt, multi_field) == Catch::Approx((step + 1) * dt));
  }

  REQUIRE(same_storage(owned_field, multi_field));
  REQUIRE(halo_untouched(owned_field, halo_value));
  const double scale = std::pow(factor, steps);
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        REQUIRE(owned_field(i, j, k) ==
                Catch::Approx(pattern_value(i, j, k) * scale));
      }
    }
  }
  REQUIRE(owned.workspace_bytes() == owned_field.size() * sizeof(double));
}
