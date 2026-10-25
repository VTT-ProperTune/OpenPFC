// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = wave::input(argc, argv);
    const int n = input.value("n", 80);
    if (n < 8 || n > 128) throw std::invalid_argument("Use n in [8,128]");
    const double nominal_dt = input.value("dt", 8.0 / (n * n));
    const wave::Schedule clock(nominal_dt, input.value("saveat", .075),
                               input.value("end", 4.0));
    if (clock.dt * n * n > 8.01)
      throw std::invalid_argument("Use adjusted dt*n^2 <= 8");
    const auto profile = input.value("initial", std::string("gaussian"));
    const auto startup = input.value("startup", std::string("book"));
    if ((profile != "gaussian" && profile != "mode") ||
        (startup != "book" && startup != "taylor"))
      throw std::invalid_argument("Unknown initial condition or startup");
    wave::ChebyshevFFT derivative(n);
    wave::Vector u(n + 1), previous(n + 1), next(n + 1);
    for (int j = 0; j <= n; ++j) {
      const double x = derivative.x[j];
      u[j] = profile == "mode" ? std::cos(std::numbers::pi * x / 2)
                               : std::exp(-200 * x * x);
      previous[j] =
          profile == "mode" ? u[j] : std::exp(-200 * std::pow(x - nominal_dt, 2));
    }
    if (startup == "taylor") {
      const auto acceleration = derivative.derivative(u, 2);
      for (int j = 0; j <= n; ++j) {
        const double velocity =
            profile == "mode" ? 0 : -400 * derivative.x[j] * u[j];
        previous[j] =
            u[j] - clock.dt * velocity +
            (j > 0 && j < n ? .5 * clock.dt * clock.dt * acceleration[j] : 0);
      }
    }
    auto output = wave::output("wave.csv");
    output << "step,time,x,u\n";
    const auto observe = [&](int step) {
      if (!std::ranges::all_of(u, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("Non-finite field: refine space or time");
      for (int j = 0; j <= n; ++j)
        output << step << ',' << step * clock.dt << ',' << derivative.x[j] << ','
               << u[j] << '\n';
    };
    observe(0);
    // start snippet leapfrog
    for (int step = 1; step <= clock.frames * clock.gap; ++step) {
      auto acceleration = derivative.derivative(u, 2);
      acceleration.at(0) = acceleration.at(n) = 0;
      for (int j = 0; j <= n; ++j)
        next[j] = 2 * u[j] - previous[j] + clock.dt * clock.dt * acceleration[j];
      previous.swap(u);
      u.swap(next);
      if (step % clock.gap == 0) observe(step);
    }
    // end snippet leapfrog
    std::cout << "dt=" << clock.dt
              << " final=" << clock.frames * clock.gap * clock.dt << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
