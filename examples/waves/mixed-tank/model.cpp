// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../lesson.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = wave::input(argc, argv);
    const int nx = input.value("nx", 50), ny = input.value("ny", 15);
    const double half_length = input.value("half_length", 3.0);
    if (nx < 8 || nx > 128 || nx % 2 || ny < 4 || ny > 48)
      throw std::invalid_argument("Use even nx in [8,128] and ny in [4,48]");
    wave::positive(half_length, "half_length");
    const double nominal_dt = input.value("dt", 5.0 / (nx + ny * ny));
    const wave::Schedule clock(nominal_dt, input.value("saveat", 2.0),
                               input.value("end", 4.0));
    const auto profile = input.value("initial", std::string("gaussian"));
    const auto startup = input.value("startup", std::string("book"));
    if ((profile != "gaussian" && profile != "mode") ||
        (startup != "book" && startup != "taylor"))
      throw std::invalid_argument("Unknown initial condition or startup");
    wave::ChebyshevFFT chebyshev(ny);
    pfc::fft::LocalFFT fourier({nx, 1, 1});
    wave::Vector x(nx), u(nx * (ny + 1)), previous(u.size()), next(u.size());
    const auto index = [nx](int i, int j) { return i + nx * j; };
    for (int i = 0; i < nx; ++i)
      x[i] = -half_length + 2 * half_length * (i + 1) / nx;
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double y = chebyshev.x[j];
        u[index(i, j)] = profile == "mode"
                             ? std::cos(std::numbers::pi * x[i] / half_length) *
                                   std::cos(std::numbers::pi * y)
                             : std::exp(-8 * (std::pow(x[i] + 1.5, 2) + y * y));
        previous[index(i, j)] =
            profile == "mode"
                ? u[index(i, j)]
                : std::exp(-8 * (std::pow(x[i] + nominal_dt + 1.5, 2) + y * y));
      }
    wave::Vector top(ny + 1), bottom(ny + 1), basis(ny + 1);
    for (int j = 0; j <= ny; ++j) {
      basis[j] = 1;
      const auto column = chebyshev.derivative(basis, 1);
      top[j] = column.at(0);
      bottom[j] = column.at(ny);
      basis[j] = 0;
    }
    // start snippet boundary
    const double determinant = top[0] * bottom[ny] - top[ny] * bottom[0];
    const auto enforce_neumann = [&](auto &values) {
      for (int i = 0; i < nx; ++i) {
        double rtop = 0, rbottom = 0;
        for (int j = 1; j < ny; ++j) {
          rtop -= top[j] * values[index(i, j)];
          rbottom -= bottom[j] * values[index(i, j)];
        }
        values[index(i, 0)] = (rtop * bottom[ny] - top[ny] * rbottom) / determinant;
        values[index(i, ny)] = (top[0] * rbottom - bottom[0] * rtop) / determinant;
      }
    };
    // end snippet boundary
    // Bound the interior Neumann operator by its maximum absolute row sum.
    wave::Vector row_sums(ny + 1);
    for (int j = 1; j < ny; ++j) {
      std::fill(basis.begin(), basis.end(), 0);
      basis[j] = 1;
      basis[0] = (-top[j] * bottom[ny] + top[ny] * bottom[j]) / determinant;
      basis[ny] = (-top[0] * bottom[j] + bottom[0] * top[j]) / determinant;
      const auto curvature = chebyshev.derivative(basis, 2);
      for (int i = 1; i < ny; ++i) row_sums[i] += std::abs(curvature[i]);
    }
    const double bound = std::pow(std::numbers::pi * nx / (2 * half_length), 2) +
                         *std::ranges::max_element(row_sums);
    if (clock.dt * clock.dt * bound >= 4)
      throw std::invalid_argument("Reduce dt: dt^2*operator_bound must be below 4");
    wave::Vector row(nx), curvature(nx), column(ny + 1);
    pfc::fft::ComplexVector modes(nx / 2 + 1);
    // start snippet mixed
    const auto laplacian = [&](const auto &values) {
      wave::Vector result(values.size());
      for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) row[i] = values[index(i, j)];
        fourier.forward(row, modes);
        for (int k = 0; k <= nx / 2; ++k)
          modes[k] *= -std::pow(std::numbers::pi * k / half_length, 2);
        fourier.backward(modes, curvature);
        for (int i = 0; i < nx; ++i) result[index(i, j)] = curvature[i];
      }
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j <= ny; ++j) column[j] = values[index(i, j)];
        const auto dyy = chebyshev.derivative(column, 2);
        for (int j = 0; j <= ny; ++j) result[index(i, j)] += dyy[j];
      }
      return result;
    };
    // end snippet mixed
    if (startup == "taylor") {
      enforce_neumann(u);
      const auto acceleration = laplacian(u);
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const auto k = index(i, j);
          const double velocity = profile == "mode" ? 0 : 16 * (x[i] + 1.5) * u[k];
          previous[k] = u[k] - clock.dt * velocity +
                        .5 * clock.dt * clock.dt * acceleration[k];
        }
      enforce_neumann(previous);
    }
    auto output = wave::output("wave.csv");
    output << "step,time,x,y,u\n";
    const auto observe = [&](int step) {
      if (!std::ranges::all_of(u, [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("Non-finite field: refine space or time");
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i < nx; ++i)
          output << step << ',' << step * clock.dt << ',' << x[i] << ','
                 << chebyshev.x[j] << ',' << u[index(i, j)] << '\n';
    };
    observe(0);
    // start snippet leapfrog
    for (int step = 1; step <= clock.frames * clock.gap; ++step) {
      const auto acceleration = laplacian(u);
      for (std::size_t k = 0; k < u.size(); ++k)
        next[k] = 2 * u[k] - previous[k] + clock.dt * clock.dt * acceleration[k];
      previous.swap(u);
      u.swap(next);
      enforce_neumann(u);
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
