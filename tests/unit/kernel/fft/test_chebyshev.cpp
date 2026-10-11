// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/fft/chebyshev.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Chebyshev Helmholtz remains accurate as degree increases",
          "[fft][chebyshev][helmholtz_stability]") {
  const double pi = std::acos(-1.0), k = pi / 2;
  for (int degree : {15, 31, 63, 127}) {
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    for (double reaction : {-1.0, 0.0, 0.1, 1.0, 6.854308612, 100.0, 900.0}) {
      for (bool neumann : {false, true}) {
        CAPTURE(degree, reaction, neumann);
        std::vector<double> exact(z.size()), f(z.size());
        for (std::size_t j = 0; j < z.size(); ++j) {
          exact[j] = neumann ? std::sin(k * z[j]) : std::cos(k * z[j]);
          f[j] = -(k * k + reaction) * exact[j];
        }
        auto u = neumann
                     ? pfc::fft::chebyshev_neumann_helmholtz(f, reaction, 0, 0)
                     : pfc::fft::chebyshev_dirichlet_helmholtz(f, reaction, 0, 0);
        auto d = pfc::fft::chebyshev_derivative(u);
        for (std::size_t j = 0; j < u.size(); ++j)
          REQUIRE_THAT(u[j], WithinAbs(exact[j], 1e-10));
        REQUIRE_THAT(neumann ? d.front() : u.front(), WithinAbs(0, 1e-9));
        REQUIRE_THAT(neumann ? d.back() : u.back(), WithinAbs(0, 1e-9));
      }
      // Outward derivatives have opposite signs at the two ends.
      std::vector<double> exact(z.size()), f(z.size());
      for (std::size_t j = 0; j < z.size(); ++j) {
        exact[j] = std::cos(k * z[j]);
        f[j] = -(k * k + reaction) * exact[j];
      }
      auto u = pfc::fft::chebyshev_robin_helmholtz(f, reaction, 1, 1, -k, 1, -1, -k);
      auto d = pfc::fft::chebyshev_derivative(u);
      CAPTURE(degree, reaction);
      REQUIRE_THAT(u.front() + d.front(), WithinAbs(-k, 1e-9));
      REQUIRE_THAT(u.back() - d.back(), WithinAbs(-k, 1e-9));
      for (std::size_t j = 0; j < u.size(); ++j)
        REQUIRE_THAT(u[j], WithinAbs(exact[j], 1e-10));
    }
  }
}

namespace {

[[nodiscard]] std::vector<double> mode_values(int degree, int mode) {
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = std::cos(mode * std::acos(std::clamp(nodes[j], -1.0, 1.0)));
  }
  return values;
}

[[nodiscard]] std::vector<double>
matrix_derivative(const std::vector<double> &values) {
  const int n = static_cast<int>(values.size()) - 1;
  const auto x = pfc::fft::chebyshev_lobatto(n);
  std::vector<double> c(values.size(), 1.0);
  c.front() = 2.0;
  c.back() = 2.0;
  for (int i = 0; i <= n; ++i) {
    if (i % 2 != 0) c[static_cast<std::size_t>(i)] *= -1.0;
  }
  std::vector<std::vector<double>> dense(values.size(),
                                         std::vector<double>(values.size()));
  for (int i = 0; i <= n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j <= n; ++j) {
      if (i == j) continue;
      const double entry =
          (c[static_cast<std::size_t>(i)] / c[static_cast<std::size_t>(j)]) /
          (x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(j)]);
      dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = entry;
      row_sum += entry;
    }
    dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = -row_sum;
  }
  std::vector<double> derivative(values.size());
  for (int i = 0; i <= n; ++i) {
    double sum = 0.0;
    for (int j = 0; j <= n; ++j) {
      sum += dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
             values[static_cast<std::size_t>(j)];
    }
    derivative[static_cast<std::size_t>(i)] = sum;
  }
  return derivative;
}

template <typename Sample>
[[nodiscard]] std::vector<double> tensor_field(int nx, int degree, double period,
                                               Sample sample) {
  const auto z = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(static_cast<std::size_t>(nx) * z.size());
  for (int iz = 0; iz <= degree; ++iz) {
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      values[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
             static_cast<std::size_t>(ix)] =
          sample(x, z[static_cast<std::size_t>(iz)]);
    }
  }
  return values;
}

template <typename Sample>
[[nodiscard]] std::vector<double> volume_field(int nx, int ny, int degree,
                                               double period_x, double period_y,
                                               Sample sample) {
  const auto z = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) * z.size());
  for (int iz = 0; iz <= degree; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        values[(static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
                static_cast<std::size_t>(iy)) *
                   static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix)] =
            sample(x, y, z[static_cast<std::size_t>(iz)]);
      }
    }
  }
  return values;
}

[[nodiscard]] double interval_integral(const std::vector<double> &values) {
  const auto coefficients = pfc::fft::chebyshev_coefficients(values);
  double sum = 0.0;
  for (int k = 0; k < static_cast<int>(coefficients.size()); ++k) {
    if (k % 2 != 0) continue;
    const double mode = static_cast<double>(k);
    const double weight = (k == 0) ? 2.0 : 2.0 / (1.0 - mode * mode);
    sum += coefficients[static_cast<std::size_t>(k)] * weight;
  }
  return sum;
}

[[nodiscard]] double max_abs_diff(const std::vector<double> &left,
                                  const std::vector<double> &right) {
  double peak = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    peak = std::max(peak, std::abs(left[i] - right[i]));
  }
  return peak;
}

} // namespace

TEST_CASE("stable Helmholtz satisfies an independent differentiation matrix",
          "[fft][chebyshev][helmholtz_stability]") {
  const double k = std::acos(-1.0) / 2, reaction = 6.854308612;
  const auto z = pfc::fft::chebyshev_lobatto(31);
  for (int boundary = 0; boundary < 3; ++boundary) {
    std::vector<double> f(z.size());
    for (std::size_t j = 0; j < z.size(); ++j)
      f[j] = -(k * k + reaction) *
             (boundary == 1 ? std::sin(k * z[j]) : std::cos(k * z[j]));
    auto u =
        boundary == 0 ? pfc::fft::chebyshev_dirichlet_helmholtz(f, reaction, 0, 0)
        : boundary == 1
            ? pfc::fft::chebyshev_neumann_helmholtz(f, reaction, 0, 0)
            : pfc::fft::chebyshev_robin_helmholtz(f, reaction, 1, 1, -k, 1, -1, -k);
    const auto second = matrix_derivative(matrix_derivative(u));
    for (std::size_t j = 1; j + 1 < u.size(); ++j)
      REQUIRE_THAT(second[j] - reaction * u[j], WithinAbs(f[j], 1e-9));
  }
}

TEST_CASE("integrated Helmholtz rejects nonfinite data and resonance",
          "[fft][chebyshev][helmholtz_stability]") {
  const double k = std::acos(-1.0) / 2;
  std::vector<double> f(64, 0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz(f, -k * k, 0, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_helmholtz(f, -k * k, 0, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz(
                        f, std::numeric_limits<double>::infinity(), 0, 0),
                    std::invalid_argument);
  f[2] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_robin_helmholtz(f, 1, 1, 1, 0, 1, -1, 0),
                    std::invalid_argument);
}

TEST_CASE("mixed Helmholtz keeps small Fourier forcing modes at high degree",
          "[fft][chebyshev][helmholtz_stability]") {
  const double pi = std::acos(-1.0), k = pi / 2;
  constexpr int nx = 7, ny = 5;
  for (int degree : {31, 63})
    for (double reaction : {0.0, 2.0})
      for (int boundary = 0; boundary < 3; ++boundary) {
        CAPTURE(degree, reaction, boundary);
        const auto h = [&](double x, double y) {
          return std::cos(2 * pi * x / 3) * std::sin(2 * pi * y / 4);
        };
        const auto exact =
            volume_field(nx, ny, degree, 3, 4, [&](double x, double y, double z) {
              return h(x, y) * (boundary == 1 ? std::sin(k * z) : std::cos(k * z));
            });
        auto f = exact;
        const double eigen = k * k + (2 * pi / 3) * (2 * pi / 3) +
                             (2 * pi / 4) * (2 * pi / 4) + reaction;
        for (double &v : f) v *= -eigen;
        std::vector<double> data(nx * ny, 0);
        if (boundary == 2)
          for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
              data[y * nx + x] = -k * h(3.0 * x / nx, 4.0 * y / ny);
        const auto u = pfc::fft::fourier_chebyshev_robin_helmholtz(
            f, nx, ny, 3, 4, reaction, boundary == 1 ? 0 : 1, boundary == 0 ? 0 : 1,
            data, boundary == 1 ? 0 : 1,
            boundary == 0 ? 0 : (boundary == 2 ? -1 : 1), data);
        REQUIRE(max_abs_diff(u, exact) < 1e-10);
      }
}

TEST_CASE("Chebyshev coefficients reproduce the Lobatto samples",
          "[fft][chebyshev]") {
  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  REQUIRE_THAT(nodes.front(), WithinAbs(1.0, 0.0));
  REQUIRE_THAT(nodes.back(), WithinAbs(-1.0, 1e-15));

  for (int mode = 0; mode <= 8; ++mode) {
    const auto values = mode_values(8, mode);
    const auto coefficients = pfc::fft::chebyshev_coefficients(values);
    for (int k = 0; k <= 8; ++k) {
      const double expect = k == mode ? 1.0 : 0.0;
      REQUIRE_THAT(coefficients[static_cast<std::size_t>(k)],
                   WithinAbs(expect, 1e-12));
    }
    const auto back = pfc::fft::chebyshev_values(coefficients);
    REQUIRE(max_abs_diff(back, values) < 1e-12);
  }
}

TEST_CASE("Chebyshev derivative matches the differentiation matrix",
          "[fft][chebyshev]") {
  const int degree = 12;
  const auto nodes = pfc::fft::chebyshev_lobatto(degree);
  std::vector<double> values(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    values[j] = std::exp(nodes[j]);
  }
  const auto spectral = pfc::fft::chebyshev_derivative(values);
  const auto matrix = matrix_derivative(values);
  REQUIRE(max_abs_diff(spectral, matrix) < 1e-10);

  const auto linear = mode_values(1, 1);
  const auto slope = pfc::fft::chebyshev_derivative(linear);
  REQUIRE_THAT(slope[0], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(slope[1], WithinAbs(1.0, 1e-12));

  const auto grid = pfc::fft::chebyshev_lobatto(6);
  const auto quadratic = mode_values(6, 2);
  const auto quad = pfc::fft::chebyshev_derivative(quadratic);
  REQUIRE_THAT(quad.front(), WithinAbs(4.0, 1e-11));
  REQUIRE_THAT(quad.back(), WithinAbs(-4.0, 1e-11));
  for (std::size_t j = 1; j + 1 < quad.size(); ++j) {
    REQUIRE_THAT(quad[j], WithinAbs(4.0 * grid[j], 1e-11));
  }
}

TEST_CASE("Chebyshev derivative of exp converges on the Lobatto grid",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> values(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) values[j] = std::exp(nodes[j]);
    const auto derivative = pfc::fft::chebyshev_derivative(values);
    double peak = 0.0;
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      peak = std::max(peak, std::abs(derivative[j] - values[j]));
    }
    return peak;
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Chebyshev Laplacian matches separable polynomials",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto laplace = [&](int nx, int degree, auto sample) {
    const auto values = tensor_field(nx, degree, period, sample);
    return pfc::fft::fourier_chebyshev_laplacian(values, nx, period);
  };

  const auto constant = laplace(8, 4, [](double, double) { return 1.0; });
  REQUIRE(max_abs_diff(constant, std::vector<double>(constant.size(), 0.0)) < 1e-12);

  const auto along_z =
      laplace(1, 8, [](double, double z) { return 2.0 * z * z - 1.0; });
  REQUIRE(max_abs_diff(along_z, std::vector<double>(along_z.size(), 4.0)) < 1e-10);

  const auto product = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (2.0 * z * z - 1.0);
  });
  const auto product_lap =
      pfc::fft::fourier_chebyshev_laplacian(product, 16, period);
  std::vector<double> product_expect(product.size());
  const auto z = pfc::fft::chebyshev_lobatto(8);
  for (int iz = 0; iz <= 8; ++iz) {
    const double t2 =
        2.0 * z[static_cast<std::size_t>(iz)] * z[static_cast<std::size_t>(iz)] -
        1.0;
    for (int ix = 0; ix < 16; ++ix) {
      const double x = period * static_cast<double>(ix) / 16.0;
      product_expect[static_cast<std::size_t>(iz) * 16 +
                     static_cast<std::size_t>(ix)] =
          std::cos(2.0 * x) * (4.0 - 4.0 * t2);
    }
  }
  REQUIRE(max_abs_diff(product_lap, product_expect) < 1e-9);

  const auto nyquist =
      tensor_field(8, 2, period, [](double x, double) { return std::cos(4.0 * x); });
  const auto nyquist_lap = pfc::fft::fourier_chebyshev_laplacian(nyquist, 8, period);
  std::vector<double> nyquist_expect(nyquist.size());
  for (std::size_t i = 0; i < nyquist.size(); ++i) {
    nyquist_expect[i] = -16.0 * nyquist[i];
  }
  REQUIRE(max_abs_diff(nyquist_lap, nyquist_expect) < 1e-9);
}

TEST_CASE("Fourier-Chebyshev Laplacian of a smooth mode converges in z",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_at = [&](int degree) {
    const auto values = tensor_field(8, degree, period, [](double x, double z) {
      return std::sin(2.0 * x) * std::exp(z);
    });
    const auto got = pfc::fft::fourier_chebyshev_laplacian(values, 8, period);
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    double peak = 0.0;
    for (int iz = 0; iz <= degree; ++iz) {
      for (int ix = 0; ix < 8; ++ix) {
        const double x = period * static_cast<double>(ix) / 8.0;
        const double expect =
            -3.0 * std::sin(2.0 * x) * std::exp(z[static_cast<std::size_t>(iz)]);
        const auto index =
            static_cast<std::size_t>(iz) * 8 + static_cast<std::size_t>(ix);
        peak = std::max(peak, std::abs(got[index] - expect));
      }
    }
    return peak;
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Chebyshev Laplacian rejects a bad grid", "[fft][chebyshev]") {
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian({}, 4, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 3, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 4, 0.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 4, -1.0),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Fourier-Chebyshev Laplacian matches separable polynomials",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto t2 = [](double z) { return 2.0 * z * z - 1.0; };

  const auto delegated_x =
      volume_field(16, 1, 8, period, period, [](double x, double, double z) {
        return std::cos(2.0 * x) * (2.0 * z * z - 1.0);
      });
  const auto one_direction =
      pfc::fft::fourier_chebyshev_laplacian(delegated_x, 16, period);
  REQUIRE(max_abs_diff(pfc::fft::fourier_chebyshev_laplacian(delegated_x, 16, 1,
                                                             period, period),
                       one_direction) == 0.0);

  const auto delegated_y =
      volume_field(1, 16, 8, period, period, [](double, double y, double z) {
        return std::cos(2.0 * y) * (2.0 * z * z - 1.0);
      });
  const auto along_y =
      pfc::fft::fourier_chebyshev_laplacian(delegated_y, 16, period);
  REQUIRE(max_abs_diff(pfc::fft::fourier_chebyshev_laplacian(delegated_y, 1, 16,
                                                             period, period),
                       along_y) == 0.0);

  const auto product =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (2.0 * z * z - 1.0);
      });
  const auto product_lap =
      pfc::fft::fourier_chebyshev_laplacian(product, 16, 12, period, period);
  std::vector<double> product_expect(product.size());
  const auto z = pfc::fft::chebyshev_lobatto(8);
  for (int iz = 0; iz <= 8; ++iz) {
    const double profile = t2(z[static_cast<std::size_t>(iz)]);
    for (int iy = 0; iy < 12; ++iy) {
      const double y = period * static_cast<double>(iy) / 12.0;
      for (int ix = 0; ix < 16; ++ix) {
        const double x = period * static_cast<double>(ix) / 16.0;
        product_expect[(static_cast<std::size_t>(iz) * 12 +
                        static_cast<std::size_t>(iy)) *
                           16 +
                       static_cast<std::size_t>(ix)] =
            std::cos(2.0 * x) * std::cos(y) * (4.0 - 5.0 * profile);
      }
    }
  }
  REQUIRE(max_abs_diff(product_lap, product_expect) < 1e-9);

  const auto along_x =
      volume_field(8, 8, 8, period, period, [](double x, double, double z) {
        return std::cos(2.0 * x) * (2.0 * z * z - 1.0);
      });
  const auto along_x_lap =
      pfc::fft::fourier_chebyshev_laplacian(along_x, 8, 8, period, period);
  std::vector<double> along_x_expect(along_x.size());
  for (int iz = 0; iz <= 8; ++iz) {
    const double profile = t2(z[static_cast<std::size_t>(iz)]);
    for (int iy = 0; iy < 8; ++iy) {
      for (int ix = 0; ix < 8; ++ix) {
        const double x = period * static_cast<double>(ix) / 8.0;
        along_x_expect[(static_cast<std::size_t>(iz) * 8 +
                        static_cast<std::size_t>(iy)) *
                           8 +
                       static_cast<std::size_t>(ix)] =
            std::cos(2.0 * x) * (4.0 - 4.0 * profile);
      }
    }
  }
  REQUIRE(max_abs_diff(along_x_lap, along_x_expect) < 1e-9);

  const auto along_only_y =
      volume_field(8, 8, 8, period, period, [](double, double y, double z) {
        return std::cos(y) * (2.0 * z * z - 1.0);
      });
  const auto along_only_y_lap =
      pfc::fft::fourier_chebyshev_laplacian(along_only_y, 8, 8, period, period);
  std::vector<double> along_only_y_expect(along_only_y.size());
  for (int iz = 0; iz <= 8; ++iz) {
    const double profile = t2(z[static_cast<std::size_t>(iz)]);
    for (int iy = 0; iy < 8; ++iy) {
      const double y = period * static_cast<double>(iy) / 8.0;
      for (int ix = 0; ix < 8; ++ix) {
        along_only_y_expect
            [(static_cast<std::size_t>(iz) * 8 + static_cast<std::size_t>(iy)) * 8 +
             static_cast<std::size_t>(ix)] = std::cos(y) * (4.0 - profile);
      }
    }
  }
  REQUIRE(max_abs_diff(along_only_y_lap, along_only_y_expect) < 1e-9);

  const double period_y = 4.0 * pi;
  const auto stretched =
      volume_field(8, 8, 2, period, period_y, [](double x, double y, double) {
        return std::cos(2.0 * x) * std::cos(0.5 * y);
      });
  const auto stretched_lap =
      pfc::fft::fourier_chebyshev_laplacian(stretched, 8, 8, period, period_y);
  std::vector<double> stretched_expect(stretched.size(), 0.0);
  for (std::size_t i = 0; i < stretched.size(); ++i) {
    stretched_expect[i] = -4.25 * stretched[i];
  }
  REQUIRE(max_abs_diff(stretched_lap, stretched_expect) < 1e-9);

  const auto odd =
      volume_field(5, 6, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(2.0 * y) * (2.0 * z * z - 1.0);
      });
  const auto odd_lap =
      pfc::fft::fourier_chebyshev_laplacian(odd, 5, 6, period, period);
  const auto z_odd = pfc::fft::chebyshev_lobatto(8);
  std::vector<double> odd_expect(odd.size());
  for (int iz = 0; iz <= 8; ++iz) {
    const double profile = t2(z_odd[static_cast<std::size_t>(iz)]);
    for (int iy = 0; iy < 6; ++iy) {
      const double y = period * static_cast<double>(iy) / 6.0;
      for (int ix = 0; ix < 5; ++ix) {
        const double x = period * static_cast<double>(ix) / 5.0;
        odd_expect[(static_cast<std::size_t>(iz) * 6 +
                    static_cast<std::size_t>(iy)) *
                       5 +
                   static_cast<std::size_t>(ix)] =
            std::cos(2.0 * x) * std::cos(2.0 * y) * (4.0 - 8.0 * profile);
      }
    }
  }
  REQUIRE(max_abs_diff(odd_lap, odd_expect) < 1e-9);

  const auto nyquist =
      volume_field(8, 4, 2, period, period, [](double x, double y, double) {
        return std::cos(4.0 * x) * std::cos(2.0 * y);
      });
  const auto nyquist_lap =
      pfc::fft::fourier_chebyshev_laplacian(nyquist, 8, 4, period, period);
  std::vector<double> nyquist_expect(nyquist.size());
  for (std::size_t i = 0; i < nyquist.size(); ++i) {
    nyquist_expect[i] = -20.0 * nyquist[i];
  }
  REQUIRE(max_abs_diff(nyquist_lap, nyquist_expect) < 1e-9);

  const auto sheet =
      volume_field(4, 4, 0, period, period,
                   [](double x, double, double) { return std::cos(2.0 * x); });
  const auto sheet_lap =
      pfc::fft::fourier_chebyshev_laplacian(sheet, 4, 4, period, period);
  std::vector<double> sheet_expect(sheet.size());
  for (std::size_t i = 0; i < sheet.size(); ++i) {
    sheet_expect[i] = -4.0 * sheet[i];
  }
  REQUIRE(max_abs_diff(sheet_lap, sheet_expect) < 1e-12);
}

TEST_CASE("Fourier-Fourier-Chebyshev Laplacian converges in z", "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_at = [&](int degree) {
    const auto values =
        volume_field(8, 8, degree, period, period, [](double x, double y, double z) {
          return std::sin(2.0 * x) * std::sin(y) * std::exp(z);
        });
    const auto got =
        pfc::fft::fourier_chebyshev_laplacian(values, 8, 8, period, period);
    const auto z = pfc::fft::chebyshev_lobatto(degree);
    double peak = 0.0;
    for (int iz = 0; iz <= degree; ++iz) {
      const double ez = std::exp(z[static_cast<std::size_t>(iz)]);
      for (int iy = 0; iy < 8; ++iy) {
        const double y = period * static_cast<double>(iy) / 8.0;
        for (int ix = 0; ix < 8; ++ix) {
          const double x = period * static_cast<double>(ix) / 8.0;
          const double expect = -4.0 * std::sin(2.0 * x) * std::sin(y) * ez;
          const auto index =
              (static_cast<std::size_t>(iz) * 8 + static_cast<std::size_t>(iy)) * 8 +
              static_cast<std::size_t>(ix);
          peak = std::max(peak, std::abs(got[index] - expect));
        }
      }
    }
    return peak;
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Fourier-Chebyshev Laplacian rejects a bad grid",
          "[fft][chebyshev]") {
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian({}, 4, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 0, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 2, 0, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 3, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 2, 2, 0.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_laplacian(values, 2, 2, 1.0, -1.0),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Fourier-Chebyshev Dirichlet Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;

  const auto traces_of = [](int nx, int ny, double period_x, double period_y,
                            auto exact) {
    std::vector<double> at_plus(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));
    std::vector<double> at_minus(at_plus.size());
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        at_plus[index] = exact(x, y, 1.0);
        at_minus[index] = exact(x, y, -1.0);
      }
    }
    return std::pair{at_plus, at_minus};
  };

  const auto along_x = [](double x, double, double z) {
    return std::cos(2.0 * x) * z;
  };
  const auto along_x_force =
      volume_field(16, 1, 8, period, period, [](double x, double, double z) {
        return -4.0 * std::cos(2.0 * x) * z;
      });
  const auto [along_x_plus, along_x_minus] =
      traces_of(16, 1, period, period, along_x);
  const auto along_x_got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      along_x_force, 16, 1, period, period, along_x_plus, along_x_minus);
  const auto along_x_one = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      along_x_force, 16, period, along_x_plus, along_x_minus);
  REQUIRE(max_abs_diff(along_x_got, along_x_one) == 0.0);

  const auto along_y = [](double, double y, double z) {
    return std::cos(2.0 * y) * z;
  };
  const auto along_y_force =
      volume_field(1, 16, 8, period, period, [](double, double y, double z) {
        return -4.0 * std::cos(2.0 * y) * z;
      });
  const auto [along_y_plus, along_y_minus] =
      traces_of(1, 16, period, period, along_y);
  const auto along_y_got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      along_y_force, 1, 16, period, period, along_y_plus, along_y_minus);
  const auto along_y_one = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      along_y_force, 16, period, along_y_plus, along_y_minus);
  REQUIRE(max_abs_diff(along_y_got, along_y_one) == 0.0);

  const auto error_of = [&](int nx, int ny, int degree, double period_x,
                            double period_y, auto exact, auto force) {
    const auto forcing = volume_field(nx, ny, degree, period_x, period_y, force);
    const auto truth = volume_field(nx, ny, degree, period_x, period_y, exact);
    const auto [at_plus, at_minus] = traces_of(nx, ny, period_x, period_y, exact);
    const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
        forcing, nx, ny, period_x, period_y, at_plus, at_minus);
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const auto plus =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        const auto minus =
            (static_cast<std::size_t>(degree) * static_cast<std::size_t>(ny) +
             static_cast<std::size_t>(iy)) *
                static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        REQUIRE_THAT(got[plus], WithinAbs(at_plus[plus], 1e-10));
        REQUIRE_THAT(got[minus], WithinAbs(at_minus[plus], 1e-10));
      }
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              16, 12, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 7.0);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 8, period, period,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -5.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period,
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (1.0 - z * z);
              },
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period,
              [](double, double y, double z) { return std::cos(y) * (1.0 - z * z); },
              [](double, double y, double z) {
                return std::cos(y) * (z * z - 3.0);
              }) < 1e-10);
  const double period_y = 4.0 * pi;
  REQUIRE(error_of(
              8, 8, 8, period, period_y,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (4.25 * z * z - 6.25);
              }) < 1e-10);
  REQUIRE(error_of(
              5, 6, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (8.0 * z * z - 10.0);
              }) < 1e-10);
  REQUIRE(error_of(
              5, 6, 8, period, period,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(2.0 * y);
              },
              [](double x, double y, double z) {
                return -8.0 * z * std::cos(2.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 4, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (20.0 * z * z - 22.0);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 4, 8, period, period,
              [](double x, double y, double z) {
                return z * std::cos(4.0 * x) * std::cos(2.0 * y);
              },
              [](double x, double y, double z) {
                return -20.0 * z * std::cos(4.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              4, 4, 8, period, period,
              [](double, double, double z) { return 1.0 - z * z; },
              [](double, double, double) { return -2.0; }) < 1e-10);

  const auto product_force =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 7.0);
      });
  const std::vector<double> zeros(static_cast<std::size_t>(16 * 12), 0.0);
  const auto solved = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      product_force, 16, 12, period, period, zeros, zeros);
  const auto residual =
      pfc::fft::fourier_chebyshev_laplacian(solved, 16, 12, period, period);
  REQUIRE(max_abs_diff(residual, product_force) < 1e-8);
}

TEST_CASE("Fourier-Fourier-Chebyshev Dirichlet Poisson converges in z",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto exact = [](double x, double y, double z) {
    return std::sin(2.0 * x) * std::sin(y) * std::exp(z);
  };
  const auto error_at = [&](int degree) {
    const auto forcing =
        volume_field(8, 8, degree, period, period, [](double x, double y, double z) {
          return -4.0 * std::sin(2.0 * x) * std::sin(y) * std::exp(z);
        });
    const auto truth = volume_field(8, 8, degree, period, period, exact);
    std::vector<double> at_plus(64);
    std::vector<double> at_minus(64);
    for (int iy = 0; iy < 8; ++iy) {
      const double y = period * static_cast<double>(iy) / 8.0;
      for (int ix = 0; ix < 8; ++ix) {
        const double x = period * static_cast<double>(ix) / 8.0;
        const auto index =
            static_cast<std::size_t>(iy) * 8 + static_cast<std::size_t>(ix);
        at_plus[index] = exact(x, y, 1.0);
        at_minus[index] = exact(x, y, -1.0);
      }
    }
    const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
        forcing, 8, 8, period, period, at_plus, at_minus);
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-8);
  REQUIRE(fine < coarse);
}

TEST_CASE("Fourier-Fourier-Chebyshev Dirichlet Poisson rejects a bad grid",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<double> one_plane{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  const std::vector<double> wide{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson({}, 2, 2, 1.0, 1.0,
                                                                  trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 0, 2, 1.0,
                                                                  1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 2, 0, 1.0,
                                                                  1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        one_plane, 3, 2, 1.0, 1.0, wide, wide),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 2, 2, 0.0,
                                                                  1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        forcing, 2, 2, 1.0, -1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        forcing, 2, 2, 1.0, 1.0, short_trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        one_plane, 2, 2, 1.0, 1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        forcing, 4, 1, 1.0, -1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 1, 4, 0.0,
                                                                  1.0, trace, trace),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Dirichlet Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, auto exact, auto force) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> at_plus(static_cast<std::size_t>(nx));
    std::vector<double> at_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      at_plus[static_cast<std::size_t>(ix)] = exact(x, 1.0);
      at_minus[static_cast<std::size_t>(ix)] = exact(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
        forcing, nx, period, at_plus, at_minus);
    for (int ix = 0; ix < nx; ++ix) {
      const auto plus = static_cast<std::size_t>(ix);
      const auto minus =
          static_cast<std::size_t>(degree) * static_cast<std::size_t>(nx) + plus;
      REQUIRE_THAT(got[plus], WithinAbs(at_plus[plus], 1e-10));
      REQUIRE_THAT(got[minus], WithinAbs(at_minus[plus], 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              16, 8,
              [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, [](double, double z) { return 1.0 - z * z + 0.25 * z; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (16.0 * z * z - 18.0);
              }) < 1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
  });
  const std::vector<double> zeros(16, 0.0);
  const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 16, period,
                                                                 zeros, zeros);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Dirichlet Poisson rejects a bad grid",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson({}, 4, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 0, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 3, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(forcing, 4, 0.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_dirichlet_poisson(
                        forcing, 4, 1.0, short_trace, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_dirichlet_poisson(one_line, 4, 1.0, trace, trace),
      std::invalid_argument);
}

TEST_CASE("Fourier-Fourier-Chebyshev Neumann Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;

  const auto slopes_of = [](int nx, int ny, double period_x, double period_y,
                            auto slope) {
    std::vector<double> at_plus(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));
    std::vector<double> at_minus(at_plus.size());
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        at_plus[index] = slope(x, y, 1.0);
        at_minus[index] = slope(x, y, -1.0);
      }
    }
    return std::pair{at_plus, at_minus};
  };

  const auto along_x_force =
      volume_field(16, 1, 8, period, period, [](double x, double, double z) {
        return -2.0 - 4.0 * z * std::cos(2.0 * x);
      });
  const auto [along_x_plus, along_x_minus] =
      slopes_of(16, 1, period, period, [](double x, double, double z) {
        return -2.0 * z + std::cos(2.0 * x);
      });
  const auto along_x_got = pfc::fft::fourier_chebyshev_neumann_poisson(
      along_x_force, 16, 1, period, period, along_x_plus, along_x_minus);
  const auto along_x_one = pfc::fft::fourier_chebyshev_neumann_poisson(
      along_x_force, 16, period, along_x_plus, along_x_minus);
  REQUIRE(max_abs_diff(along_x_got, along_x_one) == 0.0);

  const auto along_y_force =
      volume_field(1, 16, 8, period, period, [](double, double y, double z) {
        return -2.0 - 4.0 * z * std::cos(2.0 * y);
      });
  const auto [along_y_plus, along_y_minus] =
      slopes_of(1, 16, period, period, [](double, double y, double z) {
        return -2.0 * z + std::cos(2.0 * y);
      });
  const auto along_y_got = pfc::fft::fourier_chebyshev_neumann_poisson(
      along_y_force, 1, 16, period, period, along_y_plus, along_y_minus);
  const auto along_y_one = pfc::fft::fourier_chebyshev_neumann_poisson(
      along_y_force, 16, period, along_y_plus, along_y_minus);
  REQUIRE(max_abs_diff(along_y_got, along_y_one) == 0.0);

  const auto error_of = [&](int nx, int ny, int degree, double period_x,
                            double period_y, auto exact, auto force, auto slope) {
    const auto forcing = volume_field(nx, ny, degree, period_x, period_y, force);
    const auto truth = volume_field(nx, ny, degree, period_x, period_y, exact);
    const auto [at_plus, at_minus] = slopes_of(nx, ny, period_x, period_y, slope);
    const auto got = pfc::fft::fourier_chebyshev_neumann_poisson(
        forcing, nx, ny, period_x, period_y, at_plus, at_minus);
    double mean_integral = 0.0;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        std::vector<double> column(static_cast<std::size_t>(degree) + 1);
        for (int iz = 0; iz <= degree; ++iz) {
          column[static_cast<std::size_t>(iz)] =
              got[(static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
                   static_cast<std::size_t>(iy)) *
                      static_cast<std::size_t>(nx) +
                  static_cast<std::size_t>(ix)];
        }
        const auto derivative = pfc::fft::chebyshev_derivative(column);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        REQUIRE_THAT(derivative.front(), WithinAbs(at_plus[index], 1e-9));
        REQUIRE_THAT(derivative.back(), WithinAbs(at_minus[index], 1e-9));
        mean_integral += interval_integral(column);
      }
    }
    REQUIRE_THAT(mean_integral / static_cast<double>(nx * ny),
                 WithinAbs(0.0, 1e-10));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              16, 12, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 11.0 / 3.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 8, period, period,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double) {
                return std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period,
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (1.0 / 3.0 - z * z);
              },
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
              },
              [](double x, double, double z) {
                return -2.0 * z * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period,
              [](double, double y, double z) {
                return std::cos(y) * (1.0 / 3.0 - z * z);
              },
              [](double, double y, double z) {
                return std::cos(y) * (z * z - 7.0 / 3.0);
              },
              [](double, double y, double z) { return -2.0 * z * std::cos(y); }) <
          1e-10);
  const double period_y = 4.0 * pi;
  REQUIRE(error_of(
              8, 8, 8, period, period_y,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) *
                       (4.25 * z * z - 4.25 / 3.0 - 2.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(0.5 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              5, 6, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) *
                       (8.0 * z * z - 14.0 / 3.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 4, 8, period, period,
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) *
                       (20.0 * z * z - 26.0 / 3.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(4.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              4, 4, 8, period, period,
              [](double, double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double, double) { return -2.0; },
              [](double, double, double z) { return -2.0 * z; }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period,
              [](double x, double y, double z) {
                return 1.0 / 3.0 - z * z + z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -2.0 - 5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -2.0 * z + std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);

  const auto product_force =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 11.0 / 3.0);
      });
  const auto [product_plus, product_minus] =
      slopes_of(16, 12, period, period, [](double x, double y, double z) {
        return -2.0 * z * std::cos(2.0 * x) * std::cos(y);
      });
  const auto solved = pfc::fft::fourier_chebyshev_neumann_poisson(
      product_force, 16, 12, period, period, product_plus, product_minus);
  const auto residual =
      pfc::fft::fourier_chebyshev_laplacian(solved, 16, 12, period, period);
  REQUIRE(max_abs_diff(residual, product_force) < 1e-8);
}

TEST_CASE("Fourier-Fourier-Chebyshev Neumann Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<double> one_plane{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  const std::vector<double> wide{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson({}, 2, 2, 1.0, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 0, 2, 1.0,
                                                                1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 2, 0, 1.0,
                                                                1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(one_plane, 3, 2, 1.0,
                                                                1.0, wide, wide),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 2, 2, 0.0,
                                                                1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 2, 2, 1.0,
                                                                -1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(
                        forcing, 2, 2, 1.0, 1.0, short_trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(one_plane, 2, 2, 1.0,
                                                                1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 4, 1, 1.0,
                                                                -1.0, trace, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 1, 4, 0.0,
                                                                1.0, trace, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = volume_field(4, 4, 8, period, period,
                                  [](double, double, double) { return -2.0; });
  const std::vector<double> flat(16, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(field, 4, 4, period,
                                                                period, flat, flat),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Neumann Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, auto exact, auto force, auto slope) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> slope_plus(static_cast<std::size_t>(nx));
    std::vector<double> slope_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      slope_plus[static_cast<std::size_t>(ix)] = slope(x, 1.0);
      slope_minus[static_cast<std::size_t>(ix)] = slope(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_neumann_poisson(
        forcing, nx, period, slope_plus, slope_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(derivative.front(),
                   WithinAbs(slope_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(derivative.back(),
                   WithinAbs(slope_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
          },
          [](double x, double z) { return std::cos(2.0 * x) * (-2.0 * z); }) <
      1e-10);
  REQUIRE(error_of(
              16, 8, [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              8, 8, [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          8, 8,
          [](double x, double z) { return std::cos(4.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(4.0 * x) * (16.0 * z * z - 22.0 / 3.0);
          },
          [](double x, double z) { return std::cos(4.0 * x) * (-2.0 * z); }) <
      1e-10);
  REQUIRE(error_of(
              16, 8,
              [](double x, double z) {
                return 1.0 / 3.0 - z * z + z * std::cos(2.0 * x);
              },
              [](double x, double z) { return -2.0 - 4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) { return -2.0 * z + std::cos(2.0 * x); }) <
          1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
  });
  std::vector<double> slope_plus(16);
  std::vector<double> slope_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    slope_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    slope_minus[static_cast<std::size_t>(ix)] = 2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_neumann_poisson(
      forcing, 16, period, slope_plus, slope_minus);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Neumann Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson({}, 4, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 0, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 3, 1.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 4, 0.0, trace, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_neumann_poisson(forcing, 4, 1.0,
                                                                short_trace, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(one_line, 4, 1.0, trace, trace),
      std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_neumann_poisson(field, 8, period, flat, flat),
      std::invalid_argument);
}

TEST_CASE("Fourier-Fourier-Chebyshev Robin Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;

  const auto traces_of = [](int nx, int ny, double period_x, double period_y,
                            auto sample) {
    std::vector<double> at_plus(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));
    std::vector<double> at_minus(at_plus.size());
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        at_plus[index] = sample(x, y, 1.0);
        at_minus[index] = sample(x, y, -1.0);
      }
    }
    return std::pair{at_plus, at_minus};
  };

  const auto along_x_force =
      volume_field(16, 1, 8, period, period, [](double x, double, double z) {
        return -2.0 - 4.0 * z * std::cos(2.0 * x);
      });
  const auto [along_x_plus, along_x_minus] =
      traces_of(16, 1, period, period, [](double x, double, double z) {
        const double wave = std::cos(2.0 * x);
        return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
      });
  const auto along_x_got = pfc::fft::fourier_chebyshev_robin_poisson(
      along_x_force, 16, 1, period, period, 1.0, 1.0, along_x_plus, 1.0, -1.0,
      along_x_minus);
  const auto along_x_one = pfc::fft::fourier_chebyshev_robin_poisson(
      along_x_force, 16, period, 1.0, 1.0, along_x_plus, 1.0, -1.0, along_x_minus);
  REQUIRE(max_abs_diff(along_x_got, along_x_one) == 0.0);

  const auto along_y_force =
      volume_field(1, 16, 8, period, period, [](double, double y, double z) {
        return -2.0 - 4.0 * z * std::cos(2.0 * y);
      });
  const auto [along_y_plus, along_y_minus] =
      traces_of(1, 16, period, period, [](double, double y, double z) {
        const double wave = std::cos(2.0 * y);
        return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
      });
  const auto along_y_got = pfc::fft::fourier_chebyshev_robin_poisson(
      along_y_force, 1, 16, period, period, 1.0, 1.0, along_y_plus, 1.0, -1.0,
      along_y_minus);
  const auto along_y_one = pfc::fft::fourier_chebyshev_robin_poisson(
      along_y_force, 16, period, 1.0, 1.0, along_y_plus, 1.0, -1.0, along_y_minus);
  REQUIRE(max_abs_diff(along_y_got, along_y_one) == 0.0);

  const auto error_of = [&](int nx, int ny, int degree, double period_x,
                            double period_y, double value_plus, double slope_plus,
                            double value_minus, double slope_minus, auto exact,
                            auto force, auto data) {
    const auto forcing = volume_field(nx, ny, degree, period_x, period_y, force);
    const auto truth = volume_field(nx, ny, degree, period_x, period_y, exact);
    const auto [at_plus, at_minus] = traces_of(nx, ny, period_x, period_y, data);
    const auto got = pfc::fft::fourier_chebyshev_robin_poisson(
        forcing, nx, ny, period_x, period_y, value_plus, slope_plus, at_plus,
        value_minus, slope_minus, at_minus);
    double mean_integral = 0.0;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        std::vector<double> column(static_cast<std::size_t>(degree) + 1);
        for (int iz = 0; iz <= degree; ++iz) {
          column[static_cast<std::size_t>(iz)] =
              got[(static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
                   static_cast<std::size_t>(iy)) *
                      static_cast<std::size_t>(nx) +
                  static_cast<std::size_t>(ix)];
        }
        const auto derivative = pfc::fft::chebyshev_derivative(column);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                     WithinAbs(at_plus[index], 1e-9));
        REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                     WithinAbs(at_minus[index], 1e-9));
        mean_integral += interval_integral(column);
      }
    }
    if (value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx * ny),
                   WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              16, 12, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 7.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                const double wave = std::cos(2.0 * x) * std::cos(y);
                return (z > 0.0 ? 2.0 : -2.0) * wave;
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (1.0 - z * z);
              },
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
              },
              [](double x, double, double) { return -2.0 * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double, double y, double z) { return std::cos(y) * (1.0 - z * z); },
              [](double, double y, double z) { return std::cos(y) * (z * z - 3.0); },
              [](double, double y, double) { return -2.0 * std::cos(y); }) < 1e-10);
  const double period_y = 4.0 * pi;
  REQUIRE(error_of(
              8, 8, 8, period, period_y, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (4.25 * z * z - 6.25);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(0.5 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              5, 6, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (8.0 * z * z - 10.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 4, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (20.0 * z * z - 22.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(4.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              4, 4, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double, double, double z) { return 1.0 - z * z; },
              [](double, double, double) { return -2.0; },
              [](double, double, double) { return -2.0; }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return 1.0 - z * z + z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -2.0 - 5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                const double wave = std::cos(2.0 * x) * std::cos(y);
                return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 2.0, 0.0, 2.0, 0.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return 2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 0.0, 1.0, 0.0, 1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 11.0 / 3.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 1.0, 0.0, 0.0, 1.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -5.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double) {
                return std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);

  const auto product_force =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 7.0);
      });
  const auto [product_plus, product_minus] =
      traces_of(16, 12, period, period, [](double x, double y, double) {
        return -2.0 * std::cos(2.0 * x) * std::cos(y);
      });
  const auto solved = pfc::fft::fourier_chebyshev_robin_poisson(
      product_force, 16, 12, period, period, 1.0, 1.0, product_plus, 1.0, -1.0,
      product_minus);
  const auto residual =
      pfc::fft::fourier_chebyshev_laplacian(solved, 16, 12, period, period);
  REQUIRE(max_abs_diff(residual, product_force) < 1e-8);
}

TEST_CASE("Fourier-Fourier-Chebyshev Robin Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<double> one_plane{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  const std::vector<double> wide{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        {}, 2, 2, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 0, 2, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 2, 0, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        one_plane, 3, 2, 1.0, 1.0, 1.0, 1.0, wide, 1.0, -1.0, wide),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 2, 2, 0.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 2, 2, 1.0, -1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_poisson(forcing, 2, 2, 1.0, 1.0, 1.0, 1.0,
                                                short_trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(one_plane, 2, 2, 1.0,
                                                              1.0, 1.0, 1.0, trace,
                                                              1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 4, 1, 1.0, -1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 1, 4, 0.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = volume_field(4, 4, 8, period, period,
                                  [](double, double, double) { return -2.0; });
  const std::vector<double> flat(16, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 4, 4, period, period, 0.0, 0.0, flat, 1.0, 0.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 4, 4, period, period, 1.0, 0.0, flat, 0.5, 1.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 4, 4, period, period, 0.0, 1.0, flat, 0.0, 1.0, flat),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Robin Poisson recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, double value_plus, double slope_plus,
                            double value_minus, double slope_minus, auto exact,
                            auto force, auto data) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> data_plus(static_cast<std::size_t>(nx));
    std::vector<double> data_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      data_plus[static_cast<std::size_t>(ix)] = data(x, 1.0);
      data_minus[static_cast<std::size_t>(ix)] = data(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_robin_poisson(
        forcing, nx, period, value_plus, slope_plus, data_plus, value_minus,
        slope_minus, data_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                   WithinAbs(data_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                   WithinAbs(data_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    if (value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z + 0.5 * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return z > 0.0 ? -1.0 : -3.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 1.0, -1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (4.0 * z * z - 6.0); },
          [](double x, double) { return -2.0 * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (16.0 * z * z - 18.0);
              },
              [](double x, double) { return -2.0 * std::cos(4.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return 1.0 - z * z + z * std::cos(2.0 * x); },
              [](double x, double z) { return -2.0 - 4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return -2.0 + (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(
      error_of(
          16, 8, 1.0, 0.0, 1.0, 0.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (4.0 * z * z - 6.0); },
          [](double, double) { return 0.0; }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 2.0, 0.0, 2.0, 0.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double z) { return 2.0 * z * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 0.0, 1.0, 0.0, 1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 / 3.0 - z * z); },
          [](double x, double z) {
            return std::cos(2.0 * x) * (4.0 * z * z - 10.0 / 3.0);
          },
          [](double x, double z) { return -2.0 * z * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 0.0, 0.0, 1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -4.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (4.0 * z * z - 6.0);
  });
  std::vector<double> data_plus(16);
  std::vector<double> data_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    data_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    data_minus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_robin_poisson(
      forcing, 16, period, 1.0, 1.0, data_plus, 1.0, -1.0, data_minus);
  const auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Robin Poisson rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        {}, 4, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 3, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 4, 0.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        forcing, 4, 1.0, 1.0, 1.0, short_trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        one_line, 4, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 0.0, 0.0, flat, 1.0, 0.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 1.0, 0.0, flat, 0.5, 1.0, flat),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_poisson(
                        field, 8, period, 0.0, 1.0, flat, 0.0, 1.0, flat),
                    std::invalid_argument);
}

TEST_CASE("Fourier-Fourier-Chebyshev Robin Helmholtz recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const double lambda = 1.0;

  const auto traces_of = [](int nx, int ny, double period_x, double period_y,
                            auto sample) {
    std::vector<double> at_plus(static_cast<std::size_t>(nx) *
                                static_cast<std::size_t>(ny));
    std::vector<double> at_minus(at_plus.size());
    for (int iy = 0; iy < ny; ++iy) {
      const double y = period_y * static_cast<double>(iy) / static_cast<double>(ny);
      for (int ix = 0; ix < nx; ++ix) {
        const double x =
            period_x * static_cast<double>(ix) / static_cast<double>(nx);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        at_plus[index] = sample(x, y, 1.0);
        at_minus[index] = sample(x, y, -1.0);
      }
    }
    return std::pair{at_plus, at_minus};
  };

  const auto along_x_force =
      volume_field(16, 1, 8, period, period, [](double x, double, double z) {
        return z * z - 3.0 - 5.0 * z * std::cos(2.0 * x);
      });
  const auto [along_x_plus, along_x_minus] =
      traces_of(16, 1, period, period, [](double x, double, double z) {
        const double wave = std::cos(2.0 * x);
        return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
      });
  const auto along_x_got = pfc::fft::fourier_chebyshev_robin_helmholtz(
      along_x_force, 16, 1, period, period, lambda, 1.0, 1.0, along_x_plus, 1.0,
      -1.0, along_x_minus);
  const auto along_x_one = pfc::fft::fourier_chebyshev_robin_helmholtz(
      along_x_force, 16, period, lambda, 1.0, 1.0, along_x_plus, 1.0, -1.0,
      along_x_minus);
  REQUIRE(max_abs_diff(along_x_got, along_x_one) == 0.0);

  const auto along_y_force =
      volume_field(1, 16, 8, period, period, [](double, double y, double z) {
        return z * z - 3.0 - 5.0 * z * std::cos(2.0 * y);
      });
  const auto [along_y_plus, along_y_minus] =
      traces_of(1, 16, period, period, [](double, double y, double z) {
        const double wave = std::cos(2.0 * y);
        return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
      });
  const auto along_y_got = pfc::fft::fourier_chebyshev_robin_helmholtz(
      along_y_force, 1, 16, period, period, lambda, 1.0, 1.0, along_y_plus, 1.0,
      -1.0, along_y_minus);
  const auto along_y_one = pfc::fft::fourier_chebyshev_robin_helmholtz(
      along_y_force, 16, period, lambda, 1.0, 1.0, along_y_plus, 1.0, -1.0,
      along_y_minus);
  REQUIRE(max_abs_diff(along_y_got, along_y_one) == 0.0);

  const auto error_of = [&](int nx, int ny, int degree, double period_x,
                            double period_y, double helmholtz, double value_plus,
                            double slope_plus, double value_minus,
                            double slope_minus, auto exact, auto force, auto data) {
    const auto forcing = volume_field(nx, ny, degree, period_x, period_y, force);
    const auto truth = volume_field(nx, ny, degree, period_x, period_y, exact);
    const auto [at_plus, at_minus] = traces_of(nx, ny, period_x, period_y, data);
    const auto got = pfc::fft::fourier_chebyshev_robin_helmholtz(
        forcing, nx, ny, period_x, period_y, helmholtz, value_plus, slope_plus,
        at_plus, value_minus, slope_minus, at_minus);
    double mean_integral = 0.0;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        std::vector<double> column(static_cast<std::size_t>(degree) + 1);
        for (int iz = 0; iz <= degree; ++iz) {
          column[static_cast<std::size_t>(iz)] =
              got[(static_cast<std::size_t>(iz) * static_cast<std::size_t>(ny) +
                   static_cast<std::size_t>(iy)) *
                      static_cast<std::size_t>(nx) +
                  static_cast<std::size_t>(ix)];
        }
        const auto derivative = pfc::fft::chebyshev_derivative(column);
        const auto index =
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(ix);
        REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                     WithinAbs(at_plus[index], 1e-9));
        REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                     WithinAbs(at_minus[index], 1e-9));
        mean_integral += interval_integral(column);
      }
    }
    if (helmholtz == 0.0 && value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx * ny),
                   WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              16, 12, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (6.0 * z * z - 8.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -6.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                const double wave = std::cos(2.0 * x) * std::cos(y);
                return (z > 0.0 ? 2.0 : -2.0) * wave;
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (1.0 - z * z);
              },
              [](double x, double, double z) {
                return std::cos(2.0 * x) * (5.0 * z * z - 7.0);
              },
              [](double x, double, double) { return -2.0 * std::cos(2.0 * x); }) <
          1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double, double y, double z) { return std::cos(y) * (1.0 - z * z); },
              [](double, double y, double z) {
                return std::cos(y) * (2.0 * z * z - 4.0);
              },
              [](double, double y, double) { return -2.0 * std::cos(y); }) < 1e-10);
  const double period_y = 4.0 * pi;
  REQUIRE(error_of(
              8, 8, 8, period, period_y, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(0.5 * y) * (5.25 * z * z - 7.25);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(0.5 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              5, 6, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(2.0 * y) * (9.0 * z * z - 11.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(2.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 4, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (1.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(4.0 * x) * std::cos(2.0 * y) * (21.0 * z * z - 23.0);
              },
              [](double x, double y, double) {
                return -2.0 * std::cos(4.0 * x) * std::cos(2.0 * y);
              }) < 1e-10);
  REQUIRE(error_of(
              4, 4, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double, double, double z) { return 1.0 - z * z; },
              [](double, double, double z) { return z * z - 3.0; },
              [](double, double, double) { return -2.0; }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 1.0, 1.0, 1.0, -1.0,
              [](double x, double y, double z) {
                return 1.0 - z * z + z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return z * z - 3.0 - 6.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                const double wave = std::cos(2.0 * x) * std::cos(y);
                return z > 0.0 ? -2.0 + 2.0 * wave : -2.0 - 2.0 * wave;
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 2.0, 0.0, 2.0, 0.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -6.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return 2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 0.0, 1.0, 0.0, 1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (6.0 * z * z - 4.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, lambda, 1.0, 0.0, 0.0, 1.0,
              [](double x, double y, double z) {
                return z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double z) {
                return -6.0 * z * std::cos(2.0 * x) * std::cos(y);
              },
              [](double x, double y, double) {
                return std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 8, period, period, 0.0, 0.0, 1.0, 0.0, 1.0,
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (1.0 / 3.0 - z * z);
              },
              [](double x, double y, double z) {
                return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 11.0 / 3.0);
              },
              [](double x, double y, double z) {
                return -2.0 * z * std::cos(2.0 * x) * std::cos(y);
              }) < 1e-10);

  const auto product_force =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (6.0 * z * z - 8.0);
      });
  const auto [product_plus, product_minus] =
      traces_of(16, 12, period, period, [](double x, double y, double) {
        return -2.0 * std::cos(2.0 * x) * std::cos(y);
      });
  const auto solved = pfc::fft::fourier_chebyshev_robin_helmholtz(
      product_force, 16, 12, period, period, lambda, 1.0, 1.0, product_plus, 1.0,
      -1.0, product_minus);
  auto residual =
      pfc::fft::fourier_chebyshev_laplacian(solved, 16, 12, period, period);
  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] -= lambda * solved[i];
  }
  REQUIRE(max_abs_diff(residual, product_force) < 1e-8);

  const auto poisson_force =
      volume_field(16, 12, 8, period, period, [](double x, double y, double z) {
        return std::cos(2.0 * x) * std::cos(y) * (5.0 * z * z - 7.0);
      });
  const auto from_helmholtz = pfc::fft::fourier_chebyshev_robin_helmholtz(
      poisson_force, 16, 12, period, period, 0.0, 1.0, 1.0, product_plus, 1.0, -1.0,
      product_minus);
  const auto from_poisson = pfc::fft::fourier_chebyshev_robin_poisson(
      poisson_force, 16, 12, period, period, 1.0, 1.0, product_plus, 1.0, -1.0,
      product_minus);
  REQUIRE(max_abs_diff(from_helmholtz, from_poisson) == 0.0);
}

TEST_CASE("Fourier-Fourier-Chebyshev Robin Helmholtz rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<double> one_plane{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  const std::vector<double> wide{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        {}, 2, 2, 1.0, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 0, 2, 1.0, 1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 2, 0, 1.0, 1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(one_plane, 3, 2, 1.0, 1.0, 1.0,
                                                  1.0, 1.0, wide, 1.0, -1.0, wide),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 2, 2, 0.0, 1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 2, 2, 1.0, -1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(
          forcing, 2, 2, 1.0, 1.0, 1.0, 1.0, 1.0, short_trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(one_plane, 2, 2, 1.0, 1.0, 1.0,
                                                  1.0, 1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 4, 1, 1.0, -1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 1, 4, 0.0, 1.0, 1.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 4, 1, 1.0, -1.0, 0.0, 1.0,
                                                  1.0, trace, 1.0, -1.0, trace),
      std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const std::vector<double> slab(32, 0.0);
  const std::vector<double> plane(16, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(slab, 4, 4, period, period, 1.0,
                                                  0.0, 0.0, plane, 1.0, 0.0, plane),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(slab, 4, 4, period, period, 1.0,
                                                  1.0, 0.0, plane, 0.5, 1.0, plane),
      std::invalid_argument);
  const auto field = volume_field(4, 4, 8, period, period,
                                  [](double, double, double) { return -2.0; });
  const std::vector<double> flat(16, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(field, 4, 4, period, period, 0.0,
                                                  0.0, 1.0, flat, 0.0, 1.0, flat),
      std::invalid_argument);
}

TEST_CASE("Fourier-Chebyshev Robin Helmholtz recovers separable solutions",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto error_of = [&](int nx, int degree, double lambda, double value_plus,
                            double slope_plus, double value_minus,
                            double slope_minus, auto exact, auto force, auto data) {
    const auto forcing = tensor_field(nx, degree, period, force);
    const auto truth = tensor_field(nx, degree, period, exact);
    std::vector<double> data_plus(static_cast<std::size_t>(nx));
    std::vector<double> data_minus(static_cast<std::size_t>(nx));
    for (int ix = 0; ix < nx; ++ix) {
      const double x = period * static_cast<double>(ix) / static_cast<double>(nx);
      data_plus[static_cast<std::size_t>(ix)] = data(x, 1.0);
      data_minus[static_cast<std::size_t>(ix)] = data(x, -1.0);
    }
    const auto got = pfc::fft::fourier_chebyshev_robin_helmholtz(
        forcing, nx, period, lambda, value_plus, slope_plus, data_plus, value_minus,
        slope_minus, data_minus);
    double mean_integral = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
      std::vector<double> column(static_cast<std::size_t>(degree) + 1);
      for (int iz = 0; iz <= degree; ++iz) {
        column[static_cast<std::size_t>(iz)] =
            got[static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(ix)];
      }
      const auto derivative = pfc::fft::chebyshev_derivative(column);
      REQUIRE_THAT(value_plus * column.front() + slope_plus * derivative.front(),
                   WithinAbs(data_plus[static_cast<std::size_t>(ix)], 1e-9));
      REQUIRE_THAT(value_minus * column.back() + slope_minus * derivative.back(),
                   WithinAbs(data_minus[static_cast<std::size_t>(ix)], 1e-9));
      mean_integral += interval_integral(column);
    }
    if (lambda == 0.0 && value_plus == 0.0 && value_minus == 0.0) {
      REQUIRE_THAT(mean_integral / static_cast<double>(nx), WithinAbs(0.0, 1e-10));
    }
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              1, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double z) { return z * z - 3.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double z) { return z * z - 3.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (5.0 * z * z - 7.0); },
          [](double x, double) { return -2.0 * std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -5.0 * z * std::cos(2.0 * x); },
              [](double x, double z) {
                return (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return std::cos(4.0 * x) * (1.0 - z * z); },
              [](double x, double z) {
                return std::cos(4.0 * x) * (17.0 * z * z - 19.0);
              },
              [](double x, double) { return -2.0 * std::cos(4.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 1.0, 1.0, -1.0,
              [](double x, double z) { return 1.0 - z * z + z * std::cos(2.0 * x); },
              [](double x, double z) {
                return z * z - 3.0 - 5.0 * z * std::cos(2.0 * x);
              },
              [](double x, double z) {
                return -2.0 + (z > 0.0 ? 2.0 : -2.0) * std::cos(2.0 * x);
              }) < 1e-10);
  REQUIRE(
      error_of(
          16, 8, 1.0, 1.0, 0.0, 1.0, 0.0,
          [](double x, double z) { return std::cos(2.0 * x) * (1.0 - z * z); },
          [](double x, double z) { return std::cos(2.0 * x) * (5.0 * z * z - 7.0); },
          [](double, double) { return 0.0; }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 1.0, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double z) { return z * z - 7.0 / 3.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);
  REQUIRE(error_of(
              16, 8, 1.0, 1.0, 0.0, 0.0, 1.0,
              [](double x, double z) { return z * std::cos(2.0 * x); },
              [](double x, double z) { return -5.0 * z * std::cos(2.0 * x); },
              [](double x, double) { return std::cos(2.0 * x); }) < 1e-10);
  REQUIRE(error_of(
              8, 8, 0.0, 1.0, 1.0, 1.0, -1.0,
              [](double, double z) { return 1.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double) { return -2.0; }) < 1e-11);
  REQUIRE(error_of(
              8, 8, 0.0, 0.0, 1.0, 0.0, 1.0,
              [](double, double z) { return 1.0 / 3.0 - z * z; },
              [](double, double) { return -2.0; },
              [](double, double z) { return -2.0 * z; }) < 1e-11);

  const auto forcing = tensor_field(16, 8, period, [](double x, double z) {
    return std::cos(2.0 * x) * (5.0 * z * z - 7.0);
  });
  std::vector<double> data_plus(16);
  std::vector<double> data_minus(16);
  for (int ix = 0; ix < 16; ++ix) {
    const double x = period * static_cast<double>(ix) / 16.0;
    data_plus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
    data_minus[static_cast<std::size_t>(ix)] = -2.0 * std::cos(2.0 * x);
  }
  const auto got = pfc::fft::fourier_chebyshev_robin_helmholtz(
      forcing, 16, period, 1.0, 1.0, 1.0, data_plus, 1.0, -1.0, data_minus);
  auto residual = pfc::fft::fourier_chebyshev_laplacian(got, 16, period);
  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] -= got[i];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-8);
}

TEST_CASE("Fourier-Chebyshev Robin Helmholtz rejects a bad condition",
          "[fft][chebyshev]") {
  const std::vector<double> forcing{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> trace{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> short_trace{0.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        {}, 4, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 0, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 3, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        forcing, 4, 0.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(forcing, 4, 1.0, 1.0, 1.0, 1.0,
                                                  short_trace, 1.0, -1.0, trace),
      std::invalid_argument);
  const std::vector<double> one_line{1.0, 2.0, 3.0, 4.0};
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        one_line, 4, 1.0, 1.0, 1.0, 1.0, trace, 1.0, -1.0, trace),
                    std::invalid_argument);

  const double period = 2.0 * std::acos(-1.0);
  const auto field = tensor_field(8, 8, period, [](double, double) { return -2.0; });
  const std::vector<double> flat(8, 0.0);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        field, 8, period, 1.0, 0.0, 0.0, flat, 1.0, 0.0, flat),
                    std::invalid_argument);
  const std::vector<double> pair(8, 0.0);
  const std::vector<double> pair_trace(4, 0.0);
  REQUIRE_THROWS_AS(
      pfc::fft::fourier_chebyshev_robin_helmholtz(pair, 4, period, 1.0, 1.0, 0.0,
                                                  pair_trace, 0.5, 1.0, pair_trace),
      std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_robin_helmholtz(
                        field, 8, period, 0.0, 0.0, 1.0, flat, 0.0, 1.0, flat),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Dirichlet Poisson recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double at_plus,
                           double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_dirichlet_poisson(forcing, at_plus, at_minus);
    REQUIRE_THAT(got.front(), WithinAbs(at_plus, 1e-12));
    REQUIRE_THAT(got.back(), WithinAbs(at_minus, 1e-12));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; }, [](double x) { return 1.0 - x * x; },
              0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0, -1.0) <
          1e-12);
  REQUIRE(error_of(
              10, [](double x) { return -6.0 * x; },
              [](double x) { return x - x * x * x; }, 0.0, 0.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto solution = pfc::fft::chebyshev_dirichlet_poisson(forcing, 0.0, 0.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Dirichlet Poisson converges for a smooth forcing",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = std::exp(nodes[j]);
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_dirichlet_poisson(forcing, std::exp(1.0),
                                                           std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Dirichlet Poisson rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_poisson({}, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_poisson(one, 0.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Neumann Poisson recovers mean-zero polynomials",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double at_plus,
                           double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_neumann_poisson(forcing, at_plus, at_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(slope.front(), WithinAbs(at_plus, 1e-9));
    REQUIRE_THAT(slope.back(), WithinAbs(at_minus, 1e-9));
    REQUIRE_THAT(interval_integral(got), WithinAbs(0.0, 1e-10));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0, 1.0) <
          1e-12);
  REQUIRE(error_of(
              10, [](double x) { return -6.0 * x; },
              [](double x) { return x - x * x * x; }, -2.0, -2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x + x; }, -1.0, 3.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double) { return 0.0; }, 0.0, 0.0) <
          1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto solution = pfc::fft::chebyshev_neumann_poisson(forcing, -2.0, 2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Neumann Poisson rejects incompatible data",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson({}, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(one, 0.0, 0.0),
                    std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> constant(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(constant, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> zero(nodes.size(), 0.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_poisson(zero, 1.0, -1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double at_plus, double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_dirichlet_helmholtz(forcing, lambda, at_plus, at_minus);
    REQUIRE_THAT(got.front(), WithinAbs(at_plus, 1e-12));
    REQUIRE_THAT(got.back(), WithinAbs(at_minus, 1e-12));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x; }, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 3.0; },
              [](double x) { return 1.0 - x * x; }, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, -1.0, [](double x) { return x; }, [](double x) { return x; }, 1.0,
              -1.0) < 1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> forcing(nodes.size());
  std::vector<double> field(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    const double x = nodes[j];
    forcing[j] = x * x - 3.0;
    field[j] = 1.0 - x * x;
  }
  const auto solution =
      pfc::fft::chebyshev_dirichlet_helmholtz(forcing, 1.0, 0.0, 0.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-9);
  REQUIRE(max_abs_diff(solution, field) < 1e-11);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_dirichlet_helmholtz(
        forcing, 1.0, std::exp(1.0), std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Dirichlet Helmholtz rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz({}, 1.0, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_dirichlet_helmholtz(one, 1.0, 0.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Neumann Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double at_plus, double at_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_neumann_helmholtz(forcing, lambda, at_plus, at_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(slope.front(), WithinAbs(at_plus, 1e-9));
    REQUIRE_THAT(slope.back(), WithinAbs(at_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  const auto gauged = pfc::fft::chebyshev_lobatto(8);
  std::vector<double> gauged_force(gauged.size(), -2.0);
  const auto gauged_solution =
      pfc::fft::chebyshev_neumann_helmholtz(gauged_force, 0.0, -2.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged_solution), WithinAbs(0.0, 1e-10));

  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 7.0 / 3.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, -2.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 1.0,
              1.0) < 1e-12);

  const auto nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> forcing(nodes.size());
  for (std::size_t j = 0; j < nodes.size(); ++j) {
    forcing[j] = nodes[j] * nodes[j] - 7.0 / 3.0;
  }
  const auto solution =
      pfc::fft::chebyshev_neumann_helmholtz(forcing, 1.0, -2.0, 2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Neumann Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_neumann_helmholtz(
        forcing, 1.0, std::exp(1.0), std::exp(-1.0));
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Neumann Helmholtz rejects a one-point grid",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_helmholtz({}, 1.0, 0.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_neumann_helmholtz(one, 1.0, 1.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("Chebyshev Robin Poisson recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, auto force, auto exact, double value_plus,
                           double slope_plus, double data_plus, double value_minus,
                           double slope_minus, double data_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_robin_poisson(forcing, value_plus, slope_plus, data_plus,
                                          value_minus, slope_minus, data_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(value_plus * got.front() + slope_plus * slope.front(),
                 WithinAbs(data_plus, 1e-9));
    REQUIRE_THAT(value_minus * got.back() + slope_minus * slope.back(),
                 WithinAbs(data_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, [](double) { return -2.0; }, [](double x) { return 1.0 - x * x; },
              1.0, 0.0, 0.0, 1.0, 0.0, 0.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 2.0, 0.0,
              2.0, 2.0, 0.0, -2.0) < 1e-12);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, 0.0, 1.0, -2.0, 0.0, 1.0,
              2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return 0.0; }, [](double x) { return x; }, 1.0, 0.0,
              1.0, 0.0, 1.0, 1.0) < 1e-12);
  REQUIRE(error_of(
              8, [](double) { return -2.0; }, [](double x) { return 1.0 - x * x; },
              1.0, 1.0, -2.0, 1.0, 1.0, 2.0) < 1e-11);
  REQUIRE(error_of(
              8, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x + 0.5 * x; }, 1.0, 1.0, -1.0, 1.0,
              1.0, 2.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto gauged =
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 1.0, -2.0, 0.0, 1.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged), WithinAbs(0.0, 1e-10));
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(gauged));
  REQUIRE(max_abs_diff(second, forcing) < 1e-9);
}

TEST_CASE("Chebyshev Robin Poisson rejects an unusable condition",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson({}, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(one, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_poisson(forcing, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0),
      std::invalid_argument);
}

TEST_CASE("Chebyshev Robin Helmholtz recovers polynomial solutions",
          "[fft][chebyshev]") {
  const auto error_of = [](int degree, double lambda, auto force, auto exact,
                           double value_plus, double slope_plus, double data_plus,
                           double value_minus, double slope_minus,
                           double data_minus) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = force(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_robin_helmholtz(
        forcing, lambda, value_plus, slope_plus, data_plus, value_minus, slope_minus,
        data_minus);
    const auto slope = pfc::fft::chebyshev_derivative(got);
    REQUIRE_THAT(value_plus * got.front() + slope_plus * slope.front(),
                 WithinAbs(data_plus, 1e-9));
    REQUIRE_THAT(value_minus * got.back() + slope_minus * slope.back(),
                 WithinAbs(data_minus, 1e-9));
    return max_abs_diff(got, truth);
  };

  REQUIRE(error_of(
              8, 0.0, [](double) { return -2.0; },
              [](double x) { return 1.0 - x * x; }, 1.0, 1.0, -2.0, 1.0, 1.0,
              2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 3.0; },
              [](double x) { return 1.0 - x * x; }, 1.0, 1.0, -2.0, 1.0, -1.0,
              -2.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 0.5 * x - 3.0; },
              [](double x) { return 1.0 - x * x + 0.5 * x; }, 1.0, 1.0, -1.0, 1.0,
              -1.0, -3.0) < 1e-11);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 1.0,
              0.0, 1.0, 0.0, 1.0, 1.0) < 1e-12);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return -x; }, [](double x) { return x; }, 2.0,
              0.0, 2.0, 2.0, 0.0, -2.0) < 1e-12);
  REQUIRE(error_of(
              8, 1.0, [](double x) { return x * x - 7.0 / 3.0; },
              [](double x) { return 1.0 / 3.0 - x * x; }, 0.0, 1.0, -2.0, 0.0, 1.0,
              2.0) < 1e-11);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  const auto gauged = pfc::fft::chebyshev_robin_helmholtz(forcing, 0.0, 0.0, 1.0,
                                                          -2.0, 0.0, 1.0, 2.0);
  REQUIRE_THAT(interval_integral(gauged), WithinAbs(0.0, 1e-10));

  const auto residual_nodes = pfc::fft::chebyshev_lobatto(12);
  std::vector<double> residual_forcing(residual_nodes.size());
  for (std::size_t j = 0; j < residual_nodes.size(); ++j) {
    residual_forcing[j] = residual_nodes[j] * residual_nodes[j] - 3.0;
  }
  const auto solution = pfc::fft::chebyshev_robin_helmholtz(
      residual_forcing, 1.0, 1.0, 1.0, -2.0, 1.0, -1.0, -2.0);
  const auto second =
      pfc::fft::chebyshev_derivative(pfc::fft::chebyshev_derivative(solution));
  std::vector<double> residual(solution.size());
  for (std::size_t j = 0; j < solution.size(); ++j) {
    residual[j] = second[j] - solution[j];
  }
  REQUIRE(max_abs_diff(residual, residual_forcing) < 1e-9);
}

TEST_CASE("Chebyshev Robin Helmholtz converges for a smooth solution",
          "[fft][chebyshev]") {
  const auto error_at = [](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    const std::vector<double> forcing(nodes.size(), 0.0);
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      truth[j] = std::exp(nodes[j]);
    }
    const auto got = pfc::fft::chebyshev_robin_helmholtz(
        forcing, 1.0, 1.0, 1.0, 2.0 * std::exp(1.0), 1.0, -1.0, 0.0);
    return max_abs_diff(got, truth);
  };
  const double coarse = error_at(8);
  const double fine = error_at(16);
  REQUIRE(fine < 1e-10);
  REQUIRE(fine < coarse);
}

TEST_CASE("Chebyshev Robin Helmholtz rejects an unusable condition",
          "[fft][chebyshev]") {
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz({}, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);
  const std::vector<double> one{0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz(one, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      std::invalid_argument);

  const auto nodes = pfc::fft::chebyshev_lobatto(8);
  const std::vector<double> forcing(nodes.size(), -2.0);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_robin_helmholtz(forcing, 1.0, 0.0, 0.0, 0.0,
                                                        1.0, 0.0, 0.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::chebyshev_robin_helmholtz(forcing, 0.0, 0.0, 1.0, 0.0,
                                                        0.0, 1.0, 0.0),
                    std::invalid_argument);
  const std::vector<double> pair{0.0, 0.0};
  REQUIRE_THROWS_AS(
      pfc::fft::chebyshev_robin_helmholtz(pair, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0),
      std::invalid_argument);
}

// Uniform nodes z_i = 1 - i*(2/n), i = 0..n. Orders 4 and 8 solve
// u'' - λu = f with Dirichlet ends. An interior row uses the shipped
// even central stencil when that window fits. A row whose window would
// leave the grid uses a Fornberg closure of (order + 2) points. Known
// boundary samples move to the right-hand side. This records accuracy
// at equal point counts. It does not record wall time, memory, MPI, or
// a GPU.
namespace {

struct DifferenceRow {
  int begin = 0;
  std::vector<double> weight;
};

[[nodiscard]] std::vector<double> fornberg_second(double origin,
                                                  const std::vector<double> &nodes) {
  const int n = static_cast<int>(nodes.size()) - 1;
  constexpr int derivative = 2;
  std::vector<std::vector<double>> weight(static_cast<std::size_t>(derivative + 1),
                                          std::vector<double>(nodes.size(), 0.0));
  weight[0][0] = 1.0;
  double product = 1.0;
  double previous_offset = nodes[0] - origin;
  for (int i = 1; i <= n; ++i) {
    const int highest = std::min(i, derivative);
    double node_product = 1.0;
    const double saved_offset = previous_offset;
    previous_offset = nodes[static_cast<std::size_t>(i)] - origin;
    for (int j = 0; j < i; ++j) {
      const double separation =
          nodes[static_cast<std::size_t>(i)] - nodes[static_cast<std::size_t>(j)];
      node_product *= separation;
      if (j == i - 1) {
        for (int k = highest; k >= 1; --k) {
          const auto row = static_cast<std::size_t>(k);
          const auto col = static_cast<std::size_t>(i);
          const auto prev = static_cast<std::size_t>(i - 1);
          weight[row][col] = product *
                             (static_cast<double>(k) * weight[row - 1][prev] -
                              saved_offset * weight[row][prev]) /
                             node_product;
        }
        weight[0][static_cast<std::size_t>(i)] =
            -product * saved_offset * weight[0][static_cast<std::size_t>(i - 1)] /
            node_product;
      }
      for (int k = highest; k >= 1; --k) {
        const auto row = static_cast<std::size_t>(k);
        const auto col = static_cast<std::size_t>(j);
        weight[row][col] = (previous_offset * weight[row][col] -
                            static_cast<double>(k) * weight[row - 1][col]) /
                           separation;
      }
      weight[0][static_cast<std::size_t>(j)] =
          previous_offset * weight[0][static_cast<std::size_t>(j)] / separation;
    }
    product = node_product;
  }
  return weight[static_cast<std::size_t>(derivative)];
}

template <int Order>
[[nodiscard]] DifferenceRow central_row(int index, double spacing) {
  using Stencil = pfc::field::fd::EvenCentralD2<Order>;
  constexpr int half = Stencil::half_width;
  DifferenceRow row;
  row.begin = index - half;
  row.weight.assign(static_cast<std::size_t>(2 * half + 1), 0.0);
  const double scale =
      1.0 / (static_cast<double>(Stencil::denom) * spacing * spacing);
  row.weight[static_cast<std::size_t>(half)] =
      static_cast<double>(Stencil::coeffs[0]) * scale;
  for (int k = 1; k <= half; ++k) {
    const double value =
        static_cast<double>(Stencil::coeffs[static_cast<std::size_t>(k)]) * scale;
    row.weight[static_cast<std::size_t>(half - k)] = value;
    row.weight[static_cast<std::size_t>(half + k)] = value;
  }
  return row;
}

[[nodiscard]] DifferenceRow second_difference_row(int intervals, int order,
                                                  int index) {
  if (order != 4 && order != 8) {
    throw std::invalid_argument("uniform difference: order must be 4 or 8");
  }
  if (intervals < order + 1) {
    throw std::invalid_argument(
        "uniform difference: grid is shorter than the stencil");
  }
  const int half = order / 2;
  const double spacing = 2.0 / static_cast<double>(intervals);
  if (index >= half && index <= intervals - half) {
    return order == 4 ? central_row<4>(index, spacing)
                      : central_row<8>(index, spacing);
  }
  const int width = order + 2;
  int start = index - (width / 2 - 1);
  if (start < 0) start = 0;
  const int last = intervals - width + 1;
  if (start > last) start = last;
  std::vector<double> nodes(static_cast<std::size_t>(width));
  for (int j = 0; j < width; ++j) {
    nodes[static_cast<std::size_t>(j)] =
        1.0 - static_cast<double>(start + j) * spacing;
  }
  DifferenceRow row;
  row.begin = start;
  row.weight = fornberg_second(1.0 - static_cast<double>(index) * spacing, nodes);
  return row;
}

[[nodiscard]] std::vector<double> dense_solve(std::vector<double> matrix,
                                              std::vector<double> rhs) {
  const int size = static_cast<int>(rhs.size());
  for (int column = 0; column < size; ++column) {
    int pivot = column;
    double largest =
        std::abs(matrix[static_cast<std::size_t>(column * size + column)]);
    for (int row = column + 1; row < size; ++row) {
      const double value =
          std::abs(matrix[static_cast<std::size_t>(row * size + column)]);
      if (value > largest) {
        largest = value;
        pivot = row;
      }
    }
    if (!(largest > 0.0)) {
      throw std::runtime_error("uniform difference: singular stencil row");
    }
    if (pivot != column) {
      for (int j = column; j < size; ++j) {
        std::swap(matrix[static_cast<std::size_t>(column * size + j)],
                  matrix[static_cast<std::size_t>(pivot * size + j)]);
      }
      std::swap(rhs[static_cast<std::size_t>(column)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    const double diagonal = matrix[static_cast<std::size_t>(column * size + column)];
    for (int row = column + 1; row < size; ++row) {
      const double factor =
          matrix[static_cast<std::size_t>(row * size + column)] / diagonal;
      matrix[static_cast<std::size_t>(row * size + column)] = 0.0;
      for (int j = column + 1; j < size; ++j) {
        matrix[static_cast<std::size_t>(row * size + j)] -=
            factor * matrix[static_cast<std::size_t>(column * size + j)];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(column)];
    }
  }
  std::vector<double> solution(static_cast<std::size_t>(size));
  for (int row = size - 1; row >= 0; --row) {
    double sum = rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < size; ++column) {
      sum -= matrix[static_cast<std::size_t>(row * size + column)] *
             solution[static_cast<std::size_t>(column)];
    }
    solution[static_cast<std::size_t>(row)] =
        sum / matrix[static_cast<std::size_t>(row * size + row)];
  }
  return solution;
}

template <typename Force, typename Exact>
[[nodiscard]] double uniform_dirichlet_error(int intervals, int order, double lambda,
                                             Force force, Exact exact) {
  const double spacing = 2.0 / static_cast<double>(intervals);
  const int unknown = intervals - 1;
  std::vector<double> matrix(static_cast<std::size_t>(unknown * unknown), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(unknown), 0.0);
  const double at_plus = exact(1.0);
  const double at_minus = exact(-1.0);
  for (int index = 1; index <= unknown; ++index) {
    const DifferenceRow row = second_difference_row(intervals, order, index);
    const int equation = index - 1;
    const double node = 1.0 - static_cast<double>(index) * spacing;
    rhs[static_cast<std::size_t>(equation)] = force(node);
    for (int k = 0; k < static_cast<int>(row.weight.size()); ++k) {
      const int node_index = row.begin + k;
      const double value = row.weight[static_cast<std::size_t>(k)];
      if (node_index == 0) {
        rhs[static_cast<std::size_t>(equation)] -= value * at_plus;
      } else if (node_index == intervals) {
        rhs[static_cast<std::size_t>(equation)] -= value * at_minus;
      } else {
        matrix[static_cast<std::size_t>(equation * unknown + (node_index - 1))] +=
            value;
      }
    }
    matrix[static_cast<std::size_t>(equation * unknown + equation)] -= lambda;
  }
  const auto solution = dense_solve(std::move(matrix), std::move(rhs));
  double peak = 0.0;
  for (int index = 1; index <= unknown; ++index) {
    const double node = 1.0 - static_cast<double>(index) * spacing;
    peak = std::max(
        peak, std::abs(solution[static_cast<std::size_t>(index - 1)] - exact(node)));
  }
  return peak;
}

[[nodiscard]] double chebyshev_mode_error(int degree) {
  const int count = 8;
  const double period = 2.0 * std::acos(-1.0);
  const auto exact = [](double x, double z) {
    return std::cos(2.0 * x) * std::exp(z);
  };
  const auto forcing = tensor_field(count, degree, period, [](double x, double z) {
    return -3.0 * std::exp(z) * std::cos(2.0 * x);
  });
  const auto truth = tensor_field(count, degree, period, exact);
  std::vector<double> at_plus(static_cast<std::size_t>(count));
  std::vector<double> at_minus(static_cast<std::size_t>(count));
  for (int ix = 0; ix < count; ++ix) {
    const double x = period * static_cast<double>(ix) / static_cast<double>(count);
    at_plus[static_cast<std::size_t>(ix)] = exact(x, 1.0);
    at_minus[static_cast<std::size_t>(ix)] = exact(x, -1.0);
  }
  const auto got = pfc::fft::fourier_chebyshev_dirichlet_poisson(
      forcing, count, period, at_plus, at_minus);
  return max_abs_diff(got, truth);
}

void require_central_weights(int order) {
  const int intervals = 32;
  const int index = intervals / 2;
  const double spacing = 2.0 / static_cast<double>(intervals);
  const int half = order == 4 ? pfc::field::fd::EvenCentralD2<4>::half_width
                              : pfc::field::fd::EvenCentralD2<8>::half_width;
  const auto denom = order == 4 ? pfc::field::fd::EvenCentralD2<4>::denom
                                : pfc::field::fd::EvenCentralD2<8>::denom;
  const std::int64_t *coeffs = order == 4
                                   ? pfc::field::fd::EvenCentralD2<4>::coeffs.data()
                                   : pfc::field::fd::EvenCentralD2<8>::coeffs.data();
  std::vector<double> nodes(static_cast<std::size_t>(2 * half + 1));
  for (int j = 0; j <= 2 * half; ++j) {
    nodes[static_cast<std::size_t>(j)] =
        1.0 - static_cast<double>(index - half + j) * spacing;
  }
  const auto got =
      fornberg_second(1.0 - static_cast<double>(index) * spacing, nodes);
  const double scale = 1.0 / (static_cast<double>(denom) * spacing * spacing);
  REQUIRE(std::abs(got[static_cast<std::size_t>(half)] -
                   static_cast<double>(coeffs[0]) * scale) < 1e-9);
  for (int k = 1; k <= half; ++k) {
    const double expect = static_cast<double>(coeffs[k]) * scale;
    REQUIRE(std::abs(got[static_cast<std::size_t>(half - k)] - expect) < 1e-9);
    REQUIRE(std::abs(got[static_cast<std::size_t>(half + k)] - expect) < 1e-9);
  }
}

} // namespace

TEST_CASE("A uniform difference reproduces polynomials inside its order",
          "[fft][chebyshev]") {
  require_central_weights(4);
  require_central_weights(8);

  const auto quartic = [](double z) { return z * z * z * z; };
  const auto quartic_force = [](double z) { return 12.0 * z * z; };
  const auto octic = [](double z) {
    const double square = z * z;
    return square * square * square * square;
  };
  const auto octic_force = [](double z) {
    const double square = z * z;
    return 56.0 * square * square * square;
  };
  for (const int order : {4, 8}) {
    REQUIRE(uniform_dirichlet_error(16, order, 0.0, quartic_force, quartic) < 1e-10);
  }
  REQUIRE(uniform_dirichlet_error(16, 8, 0.0, octic_force, octic) < 1e-10);
  REQUIRE(uniform_dirichlet_error(16, 4, 0.0, octic_force, octic) > 1e-3);
}

TEST_CASE(
    "Chebyshev and a uniform difference solve the same smooth Dirichlet problem",
    "[fft][chebyshev]") {
  const auto exact = [](double z) { return std::exp(z); };
  const auto error_at = [&](int degree) {
    const auto nodes = pfc::fft::chebyshev_lobatto(degree);
    std::vector<double> forcing(nodes.size());
    std::vector<double> truth(nodes.size());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      forcing[j] = exact(nodes[j]);
      truth[j] = exact(nodes[j]);
    }
    const auto got =
        pfc::fft::chebyshev_dirichlet_poisson(forcing, exact(1.0), exact(-1.0));
    return max_abs_diff(got, truth);
  };
  const double chebyshev = error_at(16);
  const double order4_coarse = uniform_dirichlet_error(16, 4, 0.0, exact, exact);
  const double order4_fine = uniform_dirichlet_error(32, 4, 0.0, exact, exact);
  const double order8 = uniform_dirichlet_error(16, 8, 0.0, exact, exact);
  REQUIRE(chebyshev < 1e-10);
  REQUIRE(chebyshev < order8);
  REQUIRE(order4_fine * 8.0 < order4_coarse);
}

TEST_CASE("One Fourier mode of the smooth Dirichlet problem keeps that gap",
          "[fft][chebyshev]") {
  const auto modal = [](double z) { return std::exp(z); };
  const auto modal_force = [](double z) { return -3.0 * std::exp(z); };
  const double chebyshev = chebyshev_mode_error(16);
  const double order4_coarse =
      uniform_dirichlet_error(16, 4, 4.0, modal_force, modal);
  const double order4_fine = uniform_dirichlet_error(32, 4, 4.0, modal_force, modal);
  const double order8 = uniform_dirichlet_error(16, 8, 4.0, modal_force, modal);
  REQUIRE(chebyshev < 1e-10);
  REQUIRE(chebyshev < order8);
  REQUIRE(order4_fine * 8.0 < order4_coarse);
}

TEST_CASE("Fourier-Chebyshev gradient matches a separable polynomial",
          "[fft][chebyshev]") {
  const double pi = std::acos(-1.0);
  const double period = 2.0 * pi;
  const auto field =
      volume_field(8, 6, 4, period, period,
                   [](double x, double, double z) { return z * std::cos(2.0 * x); });
  const auto gradient =
      pfc::fft::fourier_chebyshev_gradient(field, 8, 6, period, period);
  const auto z = pfc::fft::chebyshev_lobatto(4);
  std::vector<double> expect_x(field.size());
  std::vector<double> expect_z(field.size());
  for (int iz = 0; iz <= 4; ++iz) {
    const double node = z[static_cast<std::size_t>(iz)];
    for (int iy = 0; iy < 6; ++iy) {
      for (int ix = 0; ix < 8; ++ix) {
        const double x = period * static_cast<double>(ix) / 8.0;
        const auto index =
            (static_cast<std::size_t>(iz) * 6 + static_cast<std::size_t>(iy)) * 8 +
            static_cast<std::size_t>(ix);
        expect_x[index] = -2.0 * node * std::sin(2.0 * x);
        expect_z[index] = std::cos(2.0 * x);
      }
    }
  }
  REQUIRE(max_abs_diff(gradient.x, expect_x) < 1e-12);
  REQUIRE(max_abs_diff(gradient.y, std::vector<double>(field.size(), 0.0)) < 1e-12);
  REQUIRE(max_abs_diff(gradient.z, expect_z) < 1e-12);

  const auto along_y =
      volume_field(4, 8, 2, period, period,
                   [](double, double y, double node) { return node * std::sin(y); });
  const auto slope_y =
      pfc::fft::fourier_chebyshev_gradient(along_y, 4, 8, period, period);
  const auto nodes = pfc::fft::chebyshev_lobatto(2);
  std::vector<double> expect_yy(along_y.size());
  std::vector<double> expect_yz(along_y.size());
  for (int iz = 0; iz <= 2; ++iz) {
    const double node = nodes[static_cast<std::size_t>(iz)];
    for (int iy = 0; iy < 8; ++iy) {
      const double y = period * static_cast<double>(iy) / 8.0;
      for (int ix = 0; ix < 4; ++ix) {
        const auto index =
            (static_cast<std::size_t>(iz) * 8 + static_cast<std::size_t>(iy)) * 4 +
            static_cast<std::size_t>(ix);
        expect_yy[index] = node * std::cos(y);
        expect_yz[index] = std::sin(y);
      }
    }
  }
  REQUIRE(max_abs_diff(slope_y.x, std::vector<double>(along_y.size(), 0.0)) < 1e-12);
  REQUIRE(max_abs_diff(slope_y.y, expect_yy) < 1e-12);
  REQUIRE(max_abs_diff(slope_y.z, expect_yz) < 1e-12);

  const auto wall_normal = volume_field(
      1, 4, 4, period, period, [](double, double, double node) { return node; });
  const auto wall_slope =
      pfc::fft::fourier_chebyshev_gradient(wall_normal, 1, 4, period, period);
  REQUIRE(max_abs_diff(wall_slope.x, std::vector<double>(wall_normal.size(), 0.0)) <
          1e-12);
  REQUIRE(max_abs_diff(wall_slope.z, std::vector<double>(wall_normal.size(), 1.0)) <
          1e-12);

  const auto nyquist =
      volume_field(8, 4, 2, period, period,
                   [](double x, double, double) { return std::cos(4.0 * x); });
  const auto nyquist_slope =
      pfc::fft::fourier_chebyshev_gradient(nyquist, 8, 4, period, period);
  REQUIRE(max_abs_diff(nyquist_slope.x, std::vector<double>(nyquist.size(), 0.0)) <
          1e-12);

  const auto odd = volume_field(
      5, 1, 2, period, period, [](double x, double, double) { return std::sin(x); });
  const auto odd_slope =
      pfc::fft::fourier_chebyshev_gradient(odd, 5, 1, period, period);
  std::vector<double> odd_expect(odd.size());
  for (int iz = 0; iz <= 2; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      const double x = period * static_cast<double>(ix) / 5.0;
      odd_expect[static_cast<std::size_t>(iz) * 5 + static_cast<std::size_t>(ix)] =
          std::cos(x);
    }
  }
  REQUIRE(max_abs_diff(odd_slope.x, odd_expect) < 1e-12);
  REQUIRE(max_abs_diff(odd_slope.y, std::vector<double>(odd.size(), 0.0)) < 1e-12);

  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_gradient({}, 4, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_gradient(std::vector<double>(4, 0.0),
                                                         0, 2, 1.0, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::fft::fourier_chebyshev_gradient(std::vector<double>(8, 1.0),
                                                         4, 2, 1.0, 1.0),
                    std::invalid_argument);
}
