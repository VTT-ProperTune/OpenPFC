// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = wave::input(argc, argv);
    const int n = input.value("n", 24);
    if (n < 8 || n > 64) throw std::invalid_argument("Use n in [8,64]");
    const wave::Schedule clock(input.value("dt", 6.0 / (n * n)),
                               input.value("saveat", 1.0 / 3),
                               input.value("end", 1.0));
    if (clock.dt * n * n > 6.01)
      throw std::invalid_argument("Use adjusted dt*n^2 <= 6");
    const auto profile = input.value("initial", std::string("gaussian"));
    const auto startup = input.value("startup", std::string("book"));
    if ((profile != "gaussian" && profile != "mode") ||
        (startup != "book" && startup != "taylor"))
      throw std::invalid_argument("Unknown initial condition or startup");
    wave::ChebyshevFFT derivative(n);
    const auto &x = derivative.x;
    const auto index = [n](int i, int j) { return i + (n + 1) * j; };
    wave::Vector u((n + 1) * (n + 1)), next(u.size()), line(n + 1);
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i)
        u[index(i, j)] =
            profile == "mode"
                ? std::cos(std::numbers::pi * x[i] / 2) *
                      std::cos(std::numbers::pi * x[j] / 2)
                : std::exp(-40 * (std::pow(x[i] - .4, 2) + x[j] * x[j]));
    // start snippet tensor
    const auto laplacian = [&](const auto &values) {
      wave::Vector result(values.size());
      for (int j = 1; j < n; ++j) {
        for (int i = 0; i <= n; ++i) line[i] = values.at(index(i, j));
        const auto dxx = derivative.derivative(line, 2);
        for (int i = 1; i < n; ++i) result[index(i, j)] = dxx.at(i);
      }
      for (int i = 1; i < n; ++i) {
        for (int j = 0; j <= n; ++j) line[j] = values.at(index(i, j));
        const auto dyy = derivative.derivative(line, 2);
        for (int j = 1; j < n; ++j) result[index(i, j)] += dyy.at(j);
      }
      return result;
    };
    // end snippet tensor
    auto previous = u;
    if (startup == "taylor") {
      const auto acceleration = laplacian(u);
      for (std::size_t j = 0; j < u.size(); ++j)
        previous[j] += .5 * clock.dt * clock.dt * acceleration[j];
    }
    auto output = wave::output("wave.csv");
    output << "step,time,x,y,u\n";
    const auto observe = [&](int step) {
      if (!std::ranges::all_of(u, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("Non-finite field: refine space or time");
      for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
          output << step << ',' << step * clock.dt << ',' << x[i] << ',' << x[j]
                 << ',' << u[index(i, j)] << '\n';
    };
    observe(0);
    // start snippet leapfrog
    for (int step = 1; step <= clock.frames * clock.gap; ++step) {
      const auto acceleration = laplacian(u);
      for (std::size_t j = 0; j < u.size(); ++j)
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
