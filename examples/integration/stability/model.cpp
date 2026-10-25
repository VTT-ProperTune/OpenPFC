// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <vector>

using Complex = std::complex<double>;
using Tableau = pfc::sim::steppers::ButcherTableau<double>;

// start snippet stability
Complex amplification(const Tableau &method, Complex z) {
  std::vector<Complex> stage(method.stage_count(), 1);
  Complex result = 1;
  for (unsigned i = 0; i < method.stage_count(); ++i) {
    for (unsigned j = 0; j < i; ++j) stage[i] += z * method.a(i, j) * stage[j];
    result += z * method.b(i) * stage[i];
  }
  return result;
}
// end snippet stability

// Integrate the Lagrange basis through the supplied time nodes on [0,1].
// These multistep reference weights are example-local, not OpenPFC steppers.
std::vector<double> integration_weights(const std::vector<int> &nodes) {
  std::vector<double> weights;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    std::vector<long double> polynomial{1};
    long double denominator = 1;
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      if (i == j) continue;
      std::vector<long double> product(polynomial.size() + 1);
      for (std::size_t k = 0; k < polynomial.size(); ++k) {
        product[k] -= nodes[j] * polynomial[k];
        product[k + 1] += polynomial[k];
      }
      polynomial = std::move(product);
      denominator *= nodes[i] - nodes[j];
    }
    long double integral = 0;
    for (std::size_t k = 0; k < polynomial.size(); ++k)
      integral += polynomial[k] / (k + 1);
    weights.push_back(static_cast<double>(integral / denominator));
  }
  return weights;
}

int main() {
  using namespace pfc::sim::steppers;
  const std::array<Tableau, 4> methods{
      Tableau(1, {0}, {1}, {0}, {}, "Euler", 1), make_rk2_midpoint<double>(),
      Tableau(3, {0, 0, 0, 0.5, 0, 0, -1, 2, 0}, {1.0 / 6, 2.0 / 3, 1.0 / 6},
              {0, 0.5, 1}, {}, "Kutta RK3", 3),
      make_rk4_classical<double>()};
  std::filesystem::create_directories("results");
  std::ofstream grid("results/rk.csv"), boundary("results/multistep.csv");
  grid << std::setprecision(17) << "real,imag,rk1,rk2,rk3,rk4\n";
  boundary << std::setprecision(17) << "family,order,theta,real,imag\n";
  // start snippet regions
  for (int iy = 0; iy <= 140; ++iy) {
    for (int ix = 0; ix <= 140; ++ix) {
      const Complex z{-5 + ix * 0.05, -3.5 + iy * 0.05};
      grid << z.real() << ',' << z.imag();
      for (const auto &method : methods)
        grid << ',' << std::abs(amplification(method, z));
      grid << '\n';
    }
  }
  // end snippet regions
  for (const auto family : {"AB", "AM", "BDF"}) {
    const bool ab = family == std::string("AB"), am = family == std::string("AM");
    for (int order = ab ? 1 : am ? 3 : 1; order <= (ab ? 3 : 6); ++order) {
      std::vector<int> nodes;
      if (ab || am) {
        for (int j = 0; j < order; ++j) nodes.push_back((am ? 1 : 0) - j);
      }
      const auto weights = integration_weights(nodes);
      // Quadrature moments independently constrain these reference weights.
      for (int degree = 0; degree < order && (ab || am); ++degree) {
        double moment = 0;
        for (std::size_t j = 0; j < nodes.size(); ++j)
          moment += weights[j] * std::pow(nodes[j], degree);
        if (std::abs(moment - 1.0 / (degree + 1)) > 1e-10) return 1;
      }
      // start snippet multistep
      for (int point = 0; point <= 400; ++point) {
        const double theta = 2 * std::numbers::pi * point / 400;
        const Complex w = std::polar(1.0, theta);
        Complex z = 0;
        if (ab || am) {
          Complex sigma = 0;
          for (std::size_t j = 0; j < nodes.size(); ++j)
            sigma += weights[j] * std::pow(w, nodes[j]);
          if (std::abs(sigma) < 1e-12)
            continue; // A pole is not a finite boundary point.
          z = (w - Complex{1}) / sigma;
        } else {
          const Complex difference = Complex{1} - Complex{1} / w;
          for (int j = 1; j <= order; ++j) z += std::pow(difference, j) / double(j);
        }
        boundary << family << ',' << order << ',' << theta << ',' << z.real() << ','
                 << z.imag() << '\n';
      }
      // end snippet multistep
    }
  }
  if (!grid || !boundary) throw std::runtime_error("Cannot write stability CSVs");
  std::cout << "Stability regions and root-locus reference curves written under "
               "results/\n";
}
