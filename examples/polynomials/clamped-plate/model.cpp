// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../plate.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const auto input = polynomial::input(argc, argv);
    const int n = polynomial::degree(input, 17), m = n - 1;
    if (n < 6 || n > 20) throw std::invalid_argument("Use n in [6,20]");
    const int count = input.value("modes", 25);
    if (count < 1 || count > std::min(25, m * m))
      throw std::invalid_argument("Use modes in [1,min(25,(n-1)^2)]");
    // start snippet plate
    const auto op = clamped::operators(n);
    const auto modes = plate::modes(op);
    const double fundamental = modes.front().eigenvalue;
    // end snippet plate
    auto spectrum = polynomial::output("spectrum.csv");
    spectrum
        << "mode,eigenvalue,frequency_ratio,residual,parity_x,parity_y,exchange\n";
    auto output = polynomial::output("shapes.csv");
    output << "mode,kind,i,j,x,y,u,du_dx,du_dy\n";
    polynomial::Vector display(65);
    for (int j = 0; j < 65; ++j) display[j] = -1 + j / 32.0;
    for (int index = 0; index < count; ++index) {
      const auto &mode = modes[index];
      const auto image = plate::apply(op, mode.u);
      double residual2 = 0, image2 = 0;
      for (int j = 0; j < m * m; ++j) {
        residual2 += std::pow(image[j] - mode.eigenvalue * mode.u[j], 2);
        image2 += image[j] * image[j];
      }
      const double residual = std::sqrt(residual2 / image2);
      if (!std::isfinite(residual) || residual > 1e-8)
        throw std::runtime_error("Full plate eigenvector residual is too large");
      spectrum << index + 1 << ',' << mode.eigenvalue << ','
               << std::sqrt(mode.eigenvalue / fundamental) << ',' << residual << ','
               << mode.parity_x << ',' << mode.parity_y << ',' << mode.exchange
               << '\n';
      for (bool dense : {false, true}) {
        const auto &points = dense ? display : op.x;
        const auto basis = plate::evaluation(op.x, points);
        for (int j = 0; j < static_cast<int>(points.size()); ++j)
          for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            double u = 0, ux = 0, uy = 0;
            // start snippet reconstruction
            for (int b = 0; b < m; ++b)
              for (int a = 0; a < m; ++a) {
                const double coefficient = mode.u[a + m * b];
                u += basis.value[i][a] * basis.value[j][b] * coefficient;
                ux += basis.slope[i][a] * basis.value[j][b] * coefficient;
                uy += basis.value[i][a] * basis.slope[j][b] * coefficient;
              }
            // end snippet reconstruction
            output << index + 1 << ',' << (dense ? "curve" : "node") << ',' << i
                   << ',' << j << ',' << points[i] << ',' << points[j] << ',' << u
                   << ',' << ux << ',' << uy << '\n';
          }
      }
    }
    std::cout << count << " clamped square-plate modes, degree " << n << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
