// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "collocation.hpp"
#include <iostream>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

namespace allen_cahn {
template <class RightBoundary>
int run(int argc, char **argv, bool lifted, RightBoundary right_boundary) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 20);
    const double epsilon = input.value("epsilon", .01);
    const double end = input.value("end", 100.0),
                 saveat = input.value("saveat", 2.0);
    if (!std::isfinite(epsilon) || epsilon <= 0 || epsilon > 1 ||
        !std::isfinite(end) || end <= 0 || !std::isfinite(saveat) || saveat <= 0 ||
        end / saveat > 10000)
      throw std::invalid_argument("Use positive finite times and epsilon in (0,1]");
    const auto x = polynomial::nodes(n);
    auto d2 = polynomial::square(polynomial::differentiation(x));
    // start snippet stepcap
    double diffusion_bound = 0;
    for (int i = 1; i < n; ++i) {
      double row_sum = 0;
      for (int j = 1; j < n; ++j) row_sum += std::abs(d2[i][j]);
      diffusion_bound = std::max(diffusion_bound, row_sum);
    }
    // |1-3u^2| <= 11 for |u| <= 2; retain a margin below Euler's limit.
    const double maximum_dt = std::min({.01, 50.0 / (std::pow(n, 4) * epsilon),
                                        1.8 / (epsilon * diffusion_bound + 11)});
    // end snippet stepcap
    const double requested_dt = input.value("dt", maximum_dt);
    if (!std::isfinite(requested_dt) || requested_dt <= 0 ||
        requested_dt > maximum_dt || saveat / requested_dt > 1000000)
      throw std::invalid_argument(
          "Reduce dt: it must satisfy the book and operator caps");
    // Preserve the book's rounding unless it would exceed the operator cap.
    const int gap = std::max({1, static_cast<int>(std::round(saveat / requested_dt)),
                              static_cast<int>(std::ceil(saveat / maximum_dt))});
    const int frames = std::max(1, static_cast<int>(std::round(end / saveat)));
    const double dt = saveat / gap;
    if (static_cast<double>(gap) * frames > 1000000)
      throw std::invalid_argument("Reduce dt or total timesteps");
    if (lifted) {
      std::fill(d2.at(0).begin(), d2.at(0).end(), 0);
      std::fill(d2.at(n).begin(), d2.at(n).end(), 0);
    }
    polynomial::Vector u(n + 1);
    for (int j = 0; j <= n; ++j)
      u[j] = .53 * x[j] + .47 * std::sin(-1.5 * std::numbers::pi * x[j]);
    u.at(0) = right_boundary(0);
    u.at(n) = -1;
    // start snippet reaction
    const auto rhs = [&](double, const auto &v, auto &dv) {
      auto homogeneous = v;
      if (lifted)
        for (int j = 0; j <= n; ++j) homogeneous[j] -= x[j];
      const auto curvature = polynomial::apply(d2, homogeneous);
      for (int j = 0; j <= n; ++j)
        dv[j] = epsilon * curvature[j] + v[j] - v[j] * v[j] * v[j];
    };
    pfc::sim::steppers::EulerStepper stepper(dt, u.size(), rhs);
    // end snippet reaction
    auto output = polynomial::output("allen_cahn.csv");
    output << "step,time,kind,x,u\n";
    const auto observe = [&](int step) {
      if (!std::ranges::all_of(u, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("Non-finite Allen-Cahn field");
      const double t = step * dt;
      for (int j = 0; j <= n; ++j)
        output << step << ',' << t << ",node," << x[j] << ',' << u[j] << '\n';
      const polynomial::Interpolant interpolate(x, u);
      for (int j = 0; j <= 80; ++j) {
        const double point = -1 + .025 * j;
        output << step << ',' << t << ",curve," << point << ',' << interpolate(point)
               << '\n';
      }
    };
    observe(0);
    // start snippet boundary
    for (int step = 1; step <= gap * frames; ++step) {
      stepper.step((step - 1) * dt, u);
      if (!lifted) {
        u.front() = right_boundary(step * dt);
        u.at(n) = -1;
      }
      if (step % gap == 0) observe(step);
    }
    // end snippet boundary
    std::cout << "dt=" << dt << " final=" << gap * frames * dt << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
} // namespace allen_cahn
