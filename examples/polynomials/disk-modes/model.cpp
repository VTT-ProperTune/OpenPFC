// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../disk.hpp"
#include "../eigensystem.hpp"
#include <iostream>

struct Mode {
  eigen_reference::Eigenpair pair;
  int k, radial_mode;
  bool sine;
};

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const disk::Grid grid(input.value("n", 25), input.value("m", 20));
    std::vector<Mode> modes;
    // start snippet modes
    for (int k = 0; k <= grid.m / 2; ++k) {
      auto a = grid.laplacian(k);
      for (auto &row : a)
        for (auto &value : row) value = -value;
      const auto radial_modes =
          eigen_reference::solve(eigen_reference::complex_matrix(a));
      for (int j = 0; j < grid.radial; ++j) {
        modes.push_back({radial_modes[j], k, j + 1, false});
        if (k != 0 && k != grid.m / 2)
          modes.push_back({radial_modes[j], k, j + 1, true});
      }
    }
    // end snippet modes
    std::stable_sort(modes.begin(), modes.end(), [](const Mode &a, const Mode &b) {
      return a.pair.value.real() < b.pair.value.real();
    });
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum << "mode,k,radial_mode,branch,real,imag,residual\n";
    auto radial = polynomial::output("radial.csv");
    radial << "mode,j,r,real,imag\n";
    for (int index = 0; index < static_cast<int>(modes.size()); ++index) {
      const auto &mode = modes[index];
      spectrum << index + 1 << ',' << mode.k << ',' << mode.radial_mode << ','
               << (mode.sine ? "sin" : "cos") << ',' << mode.pair.value.real() << ','
               << mode.pair.value.imag() << ',' << mode.pair.residual << '\n';
      for (int j = 0; j < grid.radial; ++j)
        radial << index + 1 << ',' << j << ',' << grid.r[j + 1] << ','
               << mode.pair.vector[j].real() << ',' << mode.pair.vector[j].imag()
               << '\n';
    }
    auto selected = polynomial::output("selected.csv");
    selected << "mode,k,radial_mode,branch,eigenvalue,frequency_ratio,residual\n";
    auto fields = polynomial::output("fields.csv");
    fields << "mode,kind,ir,it,r,theta,x,y,u\n";
    for (int index : {1, 3, 6, 10}) {
      const auto &mode = modes[index - 1];
      if (mode.pair.value.real() <= 0 || std::abs(mode.pair.value.imag()) > 1e-9)
        throw std::runtime_error("Selected disk mode is not real and positive");
      disk::Vector positive(grid.radial), u(grid.radial * grid.m);
      double peak = 0;
      const auto angular = [&](double theta) {
        return mode.sine ? std::sin(mode.k * theta) : std::cos(mode.k * theta);
      };
      for (int j = 0; j < grid.radial; ++j) {
        positive[j] = mode.pair.vector[j].real();
        for (int t = 0; t < grid.m; ++t) {
          u[t + grid.m * j] = positive[j] * angular(grid.angle(t));
          peak = std::max(peak, std::abs(u[t + grid.m * j]));
        }
      }
      if (!(peak > 0)) throw std::runtime_error("Zero disk mode");
      for (double &value : positive) value /= peak;
      for (double &value : u) value /= peak;
      const auto image = grid.apply(u);
      double residual2 = 0, image2 = 0;
      for (std::size_t j = 0; j < u.size(); ++j) {
        residual2 += std::pow(image[j] + mode.pair.value.real() * u[j], 2);
        image2 += image[j] * image[j];
      }
      const double residual = std::sqrt(residual2 / image2);
      if (residual > 1e-8)
        throw std::runtime_error("Full disk eigenmode residual is too large");
      selected << index << ',' << mode.k << ',' << mode.radial_mode << ','
               << (mode.sine ? "sin" : "cos") << ',' << mode.pair.value.real() << ','
               << std::sqrt(mode.pair.value.real() / modes.front().pair.value.real())
               << ',' << residual << '\n';
      // start snippet display
      const polynomial::Interpolant interpolate(
          grid.r, disk::extend(grid, positive, mode.k));
      const auto observe = [&](const char *kind, int ir, int it, double r,
                               double theta) {
        const double value =
            r == 0 && mode.k != 0 ? 0 : interpolate(r) * angular(theta);
        fields << index << ',' << kind << ',' << ir << ',' << it << ',' << r << ','
               << theta << ',' << r * std::cos(theta) << ',' << r * std::sin(theta)
               << ',' << value << '\n';
      };
      // end snippet display
      for (int ir = 0; ir <= grid.radial; ++ir)
        for (int it = 0; it < grid.m; ++it)
          observe("node", ir, it, grid.r[ir], grid.angle(it));
      for (int ir = 0; ir <= 64; ++ir)
        for (int it = 0; it <= 128; ++it)
          observe("curve", ir, it, ir / 64.0, 2 * std::numbers::pi * it / 128);
    }
    std::cout << "Disk spectrum: " << modes.size() << " modes, selected 1,3,6,10\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
