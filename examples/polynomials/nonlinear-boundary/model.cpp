// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../collocation.hpp"
#include <iostream>
#include <openpfc/solvers/newton.hpp>
#include <openpfc/solvers/scalar_root.hpp>

// start snippet model
struct BoundaryProblem {
  polynomial::Matrix d2;
  polynomial::Vector exponential;
  void residual(const auto &u, auto &r) {
    r = polynomial::apply(d2, u);
    for (std::size_t i = 0; i < u.size(); ++i) r[i] -= std::exp(u[i]);
  }
  void linearize(const auto &u) {
    exponential.resize(u.size());
    for (std::size_t i = 0; i < u.size(); ++i) exponential[i] = std::exp(u[i]);
  }
  void jacobian(const auto &v, auto &jv) {
    jv = polynomial::apply(d2, v);
    for (std::size_t i = 0; i < v.size(); ++i) jv[i] -= exponential[i] * v[i];
  }
  void precondition(const auto &r, auto &z, int) {
    z.resize(r.size());
    for (std::size_t i = 0; i < r.size(); ++i)
      z[i] = r[i] / (d2[i][i] - exponential[i]);
  }
  // This model caches only the Jacobian at the accepted iterate.
  void begin_trial(const auto &) {}
  void accept_trial(const auto &) {}
  void reject_trial() {}
};
// end snippet model

int main(int argc, char **argv) {
  try {
    const int n = polynomial::degree(polynomial::input(argc, argv));
    const auto x = polynomial::nodes(n);
    const auto full = polynomial::square(polynomial::differentiation(x));
    BoundaryProblem model{polynomial::Matrix(n - 1, polynomial::Vector(n - 1)), {}};
    for (int i = 1; i < n; ++i)
      for (int j = 1; j < n; ++j) model.d2[i - 1][j - 1] = full[i][j];
    // start snippet newton
    pfc::solvers::VectorOperations<> ops;
    pfc::solvers::NewtonControls controls;
    controls.relative_tolerance = 1e-11;
    controls.absolute_tolerance = 1e-11;
    controls.linear.restart = n - 1;
    polynomial::Vector interior(n - 1);
    const auto result = pfc::solvers::newton(ops, model, interior, controls);
    if (!result.converged()) throw std::runtime_error("Newton did not converge");
    // end snippet newton
    // start snippet exact
    const auto parameter = pfc::solvers::scalar_root(
        [](double a) {
          return pfc::solvers::ScalarValue{std::cos(a) - std::sqrt(2.0) * a,
                                           -std::sin(a) - std::sqrt(2.0)};
        },
        0.0, 1.0, 0.5);
    if (!parameter.converged()) throw std::runtime_error("Reference root failed");
    auto exact = [&](double point) {
      return 2 * std::log(std::cos(parameter.x) / std::cos(parameter.x * point));
    };
    // end snippet exact
    polynomial::Vector u(n + 1);
    std::copy(interior.begin(), interior.end(), u.begin() + 1);
    const polynomial::Interpolant interpolate(x, u);
    auto output = polynomial::output("nonlinear.csv");
    output << "kind,x,u,exact\n";
    for (int j = 0; j <= n; ++j)
      output << "node," << x[j] << ',' << u[j] << ',' << exact(x[j]) << '\n';
    for (int j = 0; j <= 400; ++j) {
      const double point = -1 + .005 * j;
      output << "curve," << point << ',' << interpolate(point) << ',' << exact(point)
             << '\n';
    }
    auto history = polynomial::output("newton.csv");
    history << "iteration,residual,step_length,linear_iterations\n";
    for (const auto &step : result.history)
      history << step.iteration << ',' << step.residual_after << ','
              << step.step_length << ',' << step.linear.iterations << '\n';
    std::cout << "Newton residual=" << result.residual << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
