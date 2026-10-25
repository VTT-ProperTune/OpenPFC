// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const auto shape = input.value("shape", std::array{8, 9, 6});
    const double pi = std::numbers::pi;
    const auto length = input.value("length", std::array{2 * pi, 3 * pi, pi});
    const double dt = input.value("dt", 0.1);
    if (!std::isfinite(dt) || dt <= 0 || dt > 100)
      throw std::invalid_argument("Use dt in (0,100]");
    for (int axis = 0; axis < 3; ++axis) {
      if (shape[axis] < 3 || shape[axis] > 128 || !std::isfinite(length[axis]) ||
          length[axis] < 0.1 || length[axis] > 100)
        throw std::invalid_argument(
            "Use shape entries in [3,128] and lengths in [0.1,100]");
    }

    // start snippet domain
    const auto domain = pfc::domain::create(
        pfc::GridSize{shape}, pfc::PhysicalOrigin{{0, 0, 0}},
        pfc::GridSpacing{
            {length[0] / shape[0], length[1] / shape[1], length[2] / shape[2]}});
    const auto scale = pfc::fft::kspace::k_frequency_scaling(domain);
    // end snippet domain
    std::filesystem::create_directories("results");
    std::ofstream axes("results/axes.csv");
    if (!axes) throw std::runtime_error("Cannot write wave-number table");
    axes << std::setprecision(17) << "axis,index,k,k_odd,laplacian\n";
    for (int axis = 0; axis < 3; ++axis) {
      for (int index = 0; index < shape[axis]; ++index) {
        // start snippet components
        const double k =
            pfc::fft::kspace::k_component(index, shape[axis], scale[axis]);
        const double k_odd =
            pfc::fft::kspace::k_component_odd(index, shape[axis], scale[axis]);
        const double laplacian = pfc::fft::kspace::k_laplacian_value(k, 0, 0);
        // end snippet components
        axes << axis << ',' << index << ',' << k << ',' << k_odd << ',' << laplacian
             << '\n';
      }
    }

    std::ofstream symbols("results/symbols.csv");
    if (!symbols) throw std::runtime_error("Cannot write Fourier-symbol table");
    symbols << std::setprecision(17)
            << "offset,i,j,k,kx,ky,kz,laplacian,implicit,exponential\n";
    // This is the complete x-compressed r2c box, not a rank's allocated FFT.
    const pfc::fft::Box3i box{{0, 0, 0}, {shape[0] / 2, shape[1] - 1, shape[2] - 1}};
    // start snippet symbols
    pfc::fft::kspace::for_each_kpoint(
        box, domain,
        [&](std::size_t offset, double kx, double ky, double kz, int i, int j,
            int k) {
          const double lap = pfc::fft::kspace::k_laplacian_value(kx, ky, kz);
          const double implicit = 1 / (1 - dt * lap);
          const double exponential = std::exp(dt * lap);
          symbols << offset << ',' << i << ',' << j << ',' << k << ',' << kx << ','
                  << ky << ',' << kz << ',' << lap << ',' << implicit << ','
                  << exponential << '\n';
        });
    // end snippet symbols
    if (!axes || !symbols) throw std::runtime_error("Cannot finish Fourier tables");
    std::cout << "Wave numbers and heat multipliers written under results/\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
