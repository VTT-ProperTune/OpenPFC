// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openpfc/kernel/field/stencil_apply.hpp>

namespace stencil = pfc::field::stencil;

double sample(const std::string &profile, double x, double y, double z) {
  if (profile == "affine") return x + 2 * y - 3 * z;
  if (profile == "polynomial") return x * x * x + x * y * y + 0.5 * x * z * z;
  return x >= 0 ? 1 + 0.2 * y * y + 0.1 * z * z : 0;
}

int main(int argc, char **argv) {
  try {
    nlohmann::json input = nlohmann::json::object();
    if (argc > 1) {
      std::ifstream file(argv[1]);
      if (!file) throw std::runtime_error("Cannot open input JSON");
      file >> input;
    }
    const int n = input.value("n", 17);
    const double h = input.value("spacing", 0.25);
    const auto profile = input.value("profile", std::string("edge"));
    if (n < 5 || n > 65 || !std::isfinite(h) || h <= 0 || h > 10 ||
        (profile != "affine" && profile != "polynomial" && profile != "edge"))
      throw std::invalid_argument("Use n in [5,65], spacing in (0,10] and "
                                  "profile affine, polynomial or edge");

    // The generic kernel API consumes a contiguous brick and explicit strides.
    // Keep that layout here; no distributed Field or halo policy is implied.
    const auto offset = [n](int i, int j, int k) { return i + n * (j + n * k); };
    const auto coordinate = [=](int i) { return h * (i - 0.5 * (n - 1)); };
    std::vector<double> u(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          u[offset(i, j, k)] =
              sample(profile, coordinate(i), coordinate(j), coordinate(k));

    // start snippet weights
    const std::array derivative{-0.5 / h, 0.0, 0.5 / h};
    const std::array smoothing{0.25, 0.5, 0.25};
    double weights[3][3][3];
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
          weights[k][j][i] = derivative[i] * smoothing[j] * smoothing[k];
    // end snippet weights

    std::filesystem::create_directories("results");
    std::ofstream file("results/responses.csv");
    if (!file) throw std::runtime_error("Cannot write stencil responses");
    file << std::setprecision(17) << "x,y,z,u,central,separable,dense\n";
    const std::ptrdiff_t sx = 1, sy = n, sz = n * n;
    for (int k = 1; k < n - 1; ++k) {
      for (int j = 1; j < n - 1; ++j) {
        for (int i = 1; i < n - 1; ++i) {
          const auto c = offset(i, j, k);
          // start snippet kernels
          const double central = stencil::apply_1d_along<0>(derivative.data(), 1,
                                                            u.data(), c, sx, sy, sz);
          const double separable =
              stencil::apply_separable(derivative.data(), 1, smoothing.data(), 1,
                                       smoothing.data(), 1, u.data(), c, sx, sy, sz);
          const double dense =
              stencil::apply_dense(weights, u.data(), c, sx, sy, sz);
          // end snippet kernels
          file << coordinate(i) << ',' << coordinate(j) << ',' << coordinate(k)
               << ',' << u[c] << ',' << central << ',' << separable << ',' << dense
               << '\n';
        }
      }
    }
    if (!file) throw std::runtime_error("Cannot finish stencil responses");
    std::cout << "Interior responses written for " << profile << ", h=" << h << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
