// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

int main(int argc, char **argv) {
  try {
    const auto input = wave::input(argc, argv);
    const int n = input.value("n", 256);
    const double dt = input.value("dt", .4 / (n * n));
    const double end = input.value("end", .006);
    const double nonlinearity = input.value("nonlinearity", 1.0);
    if (n < 16 || n > 512 || n % 2)
      throw std::invalid_argument("Use even n in [16,512]");
    wave::positive(dt, "dt");
    wave::positive(end, "end");
    if (dt * n * n > .4001 || end / dt > 100000 || end / dt < 1 ||
        !std::isfinite(nonlinearity) || nonlinearity < 0 || nonlinearity > 1)
      throw std::invalid_argument(
          "Require dt*n^2 <= .4, 1..100000 steps and nonlinearity in [0,1]");
    const int steps = static_cast<int>(std::round(end / dt));
    const int gap = input.value(
        "save_every", std::max(1, static_cast<int>(std::floor((end / 25) / dt))));
    if (gap < 1 || gap > steps)
      throw std::invalid_argument("Use save_every in [1,steps]");
    const auto profile = input.value("initial", std::string("solitons"));
    if (profile != "solitons" && profile != "cosine" && profile != "constant")
      throw std::invalid_argument("Unknown initial condition");
    pfc::fft::LocalFFT fft({n, 1, 1});
    wave::Vector x(n), u(n), square(n);
    pfc::fft::ComplexVector transformed(n / 2 + 1), physical_modes(n / 2 + 1),
        nonlinear(n / 2 + 1);
    std::vector<std::complex<double>> linear(n / 2 + 1);
    for (int j = 0; j < n; ++j) {
      x[j] = 2 * std::numbers::pi * (j - n / 2) / n;
      u[j] = profile == "cosine" ? std::cos(2 * x[j])
             : profile == "constant"
                 ? 2
                 : 3 * 25 * 25 / std::pow(std::cosh(.5 * 25 * (x[j] + 2)), 2) +
                       3 * 16 * 16 / std::pow(std::cosh(.5 * 16 * (x[j] + 1)), 2);
    }
    fft.forward(u, transformed);
    // start snippet factor
    for (int k = 0; k <= n / 2; ++k)
      linear[k] = std::complex<double>(0, k == n / 2 ? 0 : k * k * k);
    const auto rhs = [&](double t, const auto &v, auto &dv) {
      for (int k = 0; k <= n / 2; ++k)
        physical_modes[k] = std::exp(t * linear[k]) * v[k];
      fft.backward(physical_modes, u);
      for (int j = 0; j < n; ++j) square[j] = u[j] * u[j];
      fft.forward(square, nonlinear);
      for (int k = 0; k <= n / 2; ++k) {
        const double odd_k = k == n / 2 ? 0 : k;
        dv[k] = std::exp(-t * linear[k]) *
                std::complex<double>(0, -.5 * nonlinearity * odd_k) * nonlinear[k];
      }
    };
    pfc::sim::steppers::ExplicitRKStepper<decltype(rhs), std::complex<double>>
        stepper(dt, transformed.size(),
                pfc::sim::steppers::make_rk4_classical<double>(), rhs);
    // end snippet factor
    auto output = wave::output("kdv.csv");
    auto diagnostics = wave::output("diagnostics.csv");
    output << "step,time,x,u\n";
    diagnostics << "step,time,mean,mean_square\n";
    const auto observe = [&](int step) {
      const double t = step * dt;
      for (int k = 0; k <= n / 2; ++k)
        physical_modes[k] = std::exp(t * linear[k]) * transformed[k];
      fft.backward(physical_modes, u);
      if (!std::ranges::all_of(u, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("Non-finite field: refine space or time");
      double mean = 0, mean_square = 0;
      for (int j = 0; j < n; ++j) {
        output << step << ',' << t << ',' << x[j] << ',' << u[j] << '\n';
        mean += u[j] / n;
        mean_square += u[j] * u[j] / n;
      }
      diagnostics << step << ',' << t << ',' << mean << ',' << mean_square << '\n';
    };
    observe(0);
    // start snippet evolve
    for (int step = 1; step <= steps; ++step) {
      stepper.step((step - 1) * dt, transformed);
      if (step % gap == 0 || step == steps) observe(step);
    }
    // end snippet evolve
    std::cout << "dt=" << dt << " steps=" << steps << " final=" << steps * dt
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
