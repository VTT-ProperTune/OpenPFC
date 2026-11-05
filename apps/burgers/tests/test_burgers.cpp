// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <burgers/model.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/field/central_derivatives.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double pi = std::numbers::pi;

void measure(const char *label, double value) {
  std::cout << "BURGERS_MEASURE " << label << " " << std::setprecision(17) << value
            << "\n";
}

[[nodiscard]] double bessel_i(int order, double argument) {
  double term = std::pow(0.5 * argument, order);
  for (int factor = 1; factor <= order; ++factor)
    term /= static_cast<double>(factor);
  double sum = term;
  for (int index = 1; index < 60; ++index) {
    term *= (0.5 * argument) * (0.5 * argument) /
            (static_cast<double>(index) * static_cast<double>(index + order));
    sum += term;
    if (std::abs(term) <= 1.0e-16 * std::abs(sum)) break;
  }
  return sum;
}

[[nodiscard]] double entropy_offset(double time) {
  if (!(time > 1.0)) {
    throw std::invalid_argument("entropy offset is defined after the shock");
  }
  double low = 1.0e-8;
  double high = pi - 1.0e-14;
  for (int iteration = 0; iteration < 80; ++iteration) {
    const double mid = 0.5 * (low + high);
    const double value = mid - time * std::sin(mid);
    if (value < 0.0)
      low = mid;
    else
      high = mid;
  }
  return 0.5 * (low + high);
}

[[nodiscard]] double characteristic_u(double x, double time) {
  constexpr double period = 2.0 * pi;
  x = std::fmod(x, period);
  if (x < 0.0) x += period;
  if (time > 1.0 && std::abs(x - pi) <= 1.0e-14) return 0.0;
  if (time < 1.0) {
    double xi = x;
    for (int iteration = 0; iteration < 30; ++iteration) {
      const double residual = xi + time * std::sin(xi) - x;
      const double slope = 1.0 + time * std::cos(xi);
      if (!(std::abs(slope) > 1.0e-14)) break;
      xi -= residual / slope;
    }
    const double residual = xi + time * std::sin(xi) - x;
    if (std::abs(residual) > 1.0e-10) {
      throw std::runtime_error("characteristic Newton did not converge");
    }
    return std::sin(xi);
  }
  const double delta = entropy_offset(time);
  double sign = 1.0;
  double place = x;
  if (place > pi) {
    place = period - place;
    sign = -1.0;
  }
  const double upper = pi - delta;
  double xi = std::min(place, upper);
  for (int iteration = 0; iteration < 40; ++iteration) {
    const double residual = xi + time * std::sin(xi) - place;
    const double slope = 1.0 + time * std::cos(xi);
    if (!(std::abs(slope) > 1.0e-14)) break;
    double next = xi - residual / slope;
    next = std::max(0.0, std::min(upper, next));
    xi = next;
  }
  const double residual = xi + time * std::sin(xi) - place;
  if (std::abs(residual) > 1.0e-8) {
    throw std::runtime_error("entropy characteristic did not converge");
  }
  return sign * std::sin(xi);
}

[[nodiscard]] double cole_hopf_u(double x, double time, double viscosity) {
  const double argument = 1.0 / (2.0 * viscosity);
  double profile = bessel_i(0, argument);
  double slope = 0.0;
  for (int mode = 1; mode <= 40; ++mode) {
    const double weight =
        bessel_i(mode, argument) * std::exp(-viscosity * mode * mode * time);
    profile += 2.0 * weight * std::cos(mode * x);
    slope += -2.0 * mode * weight * std::sin(mode * x);
  }
  return -2.0 * viscosity * slope / profile;
}

constexpr double gauss_node[] = {-0.96028985649753617615, -0.79666647741362672797,
                                 -0.52553240991632899082, -0.18343464249564978008,
                                 0.18343464249564978008,  0.52553240991632899082,
                                 0.79666647741362672797,  0.96028985649753617615};
constexpr double gauss_weight[] = {0.10122853629037706358, 0.22238103445337442654,
                                   0.31370664587788688049, 0.36268378337836165715,
                                   0.36268378337836165715, 0.31370664587788688049,
                                   0.22238103445337442654, 0.10122853629037706358};

template <class Sample>
[[nodiscard]] double cell_average_of(double left, double right, Sample &&sample) {
  const double mid = 0.5 * (left + right);
  const double half = 0.5 * (right - left);
  double sum = 0.0;
  for (int point = 0; point < 8; ++point) {
    sum += gauss_weight[point] * sample(mid + half * gauss_node[point]);
  }
  return 0.5 * sum;
}

[[nodiscard]] double cell_average(double left, double right, double time) {
  return cell_average_of(left, right,
                         [&](double x) { return characteristic_u(x, time); });
}

struct Norms {
  double l1;
  double l2;
  double linf;
};

[[nodiscard]] Norms nodal_error(const burgers::Simulation &simulation, double time) {
  Norms norms{0.0, 0.0, 0.0};
  const int count = simulation.count();
  const double spacing = simulation.spacing();
  for (int i = 0; i < count; ++i) {
    const double x = static_cast<double>(i) * spacing;
    const double error =
        simulation.state()[static_cast<std::size_t>(i)] - characteristic_u(x, time);
    norms.l1 += std::abs(error);
    norms.l2 += error * error;
    norms.linf = std::max(norms.linf, std::abs(error));
  }
  norms.l1 *= spacing;
  norms.l2 = std::sqrt(norms.l2 * spacing);
  return norms;
}

[[nodiscard]] Norms cell_error(const burgers::Simulation &simulation, double time) {
  Norms norms{0.0, 0.0, 0.0};
  const int count = simulation.count();
  const double spacing = simulation.spacing();
  for (int i = 0; i < count; ++i) {
    const double left = static_cast<double>(i) * spacing;
    const double error = simulation.state()[static_cast<std::size_t>(i)] -
                         cell_average(left, left + spacing, time);
    norms.l1 += std::abs(error);
    norms.l2 += error * error;
    norms.linf = std::max(norms.linf, std::abs(error));
  }
  norms.l1 *= spacing;
  norms.l2 = std::sqrt(norms.l2 * spacing);
  return norms;
}

[[nodiscard]] burgers::Simulation evolve(burgers::Method method, int count,
                                         double viscosity, double dt, double time) {
  burgers::Simulation simulation(method, count, viscosity);
  simulation.set_initial_sine();
  const double steps_real = time / dt;
  const int steps = static_cast<int>(std::llround(steps_real));
  REQUIRE_THAT(steps_real, WithinAbs(steps, 1.0e-12));
  simulation.advance(dt, steps);
  return simulation;
}

[[nodiscard]] double rate_sum(burgers::Simulation &simulation) {
  std::vector<double> rate(static_cast<std::size_t>(simulation.count()));
  simulation.rate(simulation.state(), rate);
  double sum = 0.0;
  for (double value : rate) sum += value;
  return sum;
}

[[nodiscard]] double max_rate_gap(burgers::Simulation &left,
                                  burgers::Simulation &right) {
  std::vector<double> left_rate(static_cast<std::size_t>(left.count()));
  std::vector<double> right_rate(static_cast<std::size_t>(right.count()));
  left.rate(left.state(), left_rate);
  right.rate(right.state(), right_rate);
  double gap = 0.0;
  for (std::size_t i = 0; i < left_rate.size(); ++i) {
    gap = std::max(gap, std::abs(left_rate[i] - right_rate[i]));
  }
  return gap;
}

[[nodiscard]] bool all_finite(std::span<const double> state) {
  for (double value : state) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

[[nodiscard]] double dissipation(std::span<const double> state, double spacing,
                                 double viscosity) {
  double sum = 0.0;
  const int count = static_cast<int>(state.size());
  for (int i = 0; i < count; ++i) {
    auto sample = [&](int dx, int, int) {
      int index = (i + dx) % count;
      if (index < 0) index += count;
      return state[static_cast<std::size_t>(index)];
    };
    const auto slope =
        pfc::field::fd::evaluate<2>(sample, spacing, spacing, spacing);
    sum += slope.x * slope.x;
  }
  return viscosity * sum * spacing;
}

} // namespace

TEST_CASE("Burgers references recover the sine wave and the Bessel values",
          "[burgers]") {
  REQUIRE_THAT(bessel_i(0, 1.0), WithinAbs(1.2660658777520084, 1.0e-12));
  REQUIRE_THAT(bessel_i(1, 1.0), WithinAbs(0.5651591039924850, 1.0e-12));
  REQUIRE_THAT(cole_hopf_u(0.3, 0.0, 0.5), WithinAbs(std::sin(0.3), 1.0e-12));
  REQUIRE_THAT(characteristic_u(1.1, 0.0), WithinAbs(std::sin(1.1), 1.0e-12));
  const double delta = entropy_offset(1.5);
  REQUIRE_THAT(delta, WithinAbs(1.5 * std::sin(delta), 1.0e-12));
  REQUIRE_THAT(characteristic_u(pi + 0.4, 1.5),
               WithinAbs(-characteristic_u(pi - 0.4, 1.5), 1.0e-12));
  double weight_sum = 0.0;
  for (double weight : gauss_weight) weight_sum += weight;
  REQUIRE_THAT(weight_sum, WithinAbs(2.0, 1.0e-14));
  const double spacing = 2.0 * pi / 32.0;
  const double left = 3.0 * spacing;
  const double exact = (std::cos(left) - std::cos(left + spacing)) / spacing;
  REQUIRE_THAT(cell_average(left, left + spacing, 0.0), WithinAbs(exact, 1.0e-12));
}

TEST_CASE("Dealiased Fourier tendency of the sine wave is the product rule",
          "[burgers]") {
  for (double viscosity : {0.0, 0.5}) {
    burgers::Simulation simulation(burgers::Method::spectral, 64, viscosity);
    simulation.set_initial_sine();
    std::vector<double> rate(64);
    simulation.rate(simulation.state(), rate);
    double peak = 0.0;
    for (int i = 0; i < 64; ++i) {
      const double x = simulation.spacing() * i;
      const double expected = -std::sin(x) * std::cos(x) - viscosity * std::sin(x);
      peak = std::max(peak, std::abs(rate[static_cast<std::size_t>(i)] - expected));
    }
    measure(viscosity == 0.0 ? "spectral_sine_rate_nu0" : "spectral_sine_rate_nu",
            peak);
    REQUIRE_THAT(peak, WithinAbs(0.0, 1.0e-10));
    REQUIRE_THAT(rate_sum(simulation), WithinAbs(0.0, 1.0e-10));
  }
}

TEST_CASE("A mode past the two-thirds rule is removed before the product",
          "[burgers]") {
  constexpr int count = 32;
  constexpr int mode = 11;
  burgers::Simulation dealiased(burgers::Method::spectral, count, 0.0);
  burgers::Simulation aliased(burgers::Method::spectral_aliased, count, 0.0);
  for (int i = 0; i < count; ++i) {
    const double value = std::sin(mode * dealiased.spacing() * i);
    dealiased.state()[static_cast<std::size_t>(i)] = value;
    aliased.state()[static_cast<std::size_t>(i)] = value;
  }
  std::vector<double> kept(static_cast<std::size_t>(count));
  std::vector<double> raw(static_cast<std::size_t>(count));
  dealiased.rate(dealiased.state(), kept);
  aliased.rate(aliased.state(), raw);
  REQUIRE_THAT(burgers::maximum_abs(kept), WithinAbs(0.0, 1.0e-10));
  const double alias_peak = burgers::maximum_abs(raw);
  measure("alias_peak", alias_peak);
  REQUIRE(alias_peak > 0.1);
}

TEST_CASE("FD2 and finite-volume tendencies converge on the initial sine wave",
          "[burgers]") {
  const auto fd_error = [](int count, double viscosity) {
    burgers::Simulation simulation(burgers::Method::fdm2, count, viscosity);
    simulation.set_initial_sine();
    std::vector<double> rate(static_cast<std::size_t>(count));
    simulation.rate(simulation.state(), rate);
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
      const double x = simulation.spacing() * i;
      const double expected = -std::sin(x) * std::cos(x) - viscosity * std::sin(x);
      const double error = rate[static_cast<std::size_t>(i)] - expected;
      sum += error * error;
    }
    return std::sqrt(sum * simulation.spacing());
  };
  const double fd_coarse = fd_error(32, 0.0);
  const double fd_fine = fd_error(64, 0.0);
  const double fd_viscous = fd_error(64, 0.5);
  const double fd_viscous_fine = fd_error(128, 0.5);
  measure("fd2_initial_l2_32", fd_coarse);
  measure("fd2_initial_l2_64", fd_fine);
  measure("fd2_initial_ratio", fd_coarse / fd_fine);
  measure("fd2_viscous_ratio", fd_viscous / fd_viscous_fine);
  REQUIRE(fd_coarse / fd_fine > 3.2);
  REQUIRE(fd_coarse / fd_fine < 4.8);
  REQUIRE(fd_viscous / fd_viscous_fine > 3.2);

  const auto fv_error = [](int count) {
    burgers::Simulation simulation(burgers::Method::fvm, count, 0.0);
    simulation.set_initial_sine();
    std::vector<double> rate(static_cast<std::size_t>(count));
    simulation.rate(simulation.state(), rate);
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
      const double left = simulation.spacing() * i;
      const double right = left + simulation.spacing();
      const double average_sin2 = (std::cos(2.0 * left) - std::cos(2.0 * right)) /
                                  (4.0 * simulation.spacing());
      const double expected = -average_sin2;
      const double error = rate[static_cast<std::size_t>(i)] - expected;
      sum += error * error;
    }
    return std::sqrt(sum * simulation.spacing());
  };
  const double fv_coarse = fv_error(32);
  const double fv_fine = fv_error(64);
  measure("fvm_initial_l2_32", fv_coarse);
  measure("fvm_initial_l2_64", fv_fine);
  measure("fvm_initial_ratio", fv_coarse / fv_fine);
  REQUIRE(fv_coarse / fv_fine > 1.6);
  REQUIRE(fv_coarse / fv_fine < 2.6);
}

TEST_CASE("Advective FD2 matches the stencil identity and leaves the flux form",
          "[burgers]") {
  burgers::Simulation conservative(burgers::Method::fdm2, 48, 0.0);
  burgers::Simulation advective(burgers::Method::fdm2_advective, 48, 0.0);
  conservative.set_initial_sine();
  for (int i = 0; i < 48; ++i) {
    const double x = conservative.spacing() * i;
    const double value =
        std::sin(x) + 0.2 * std::sin(2.0 * x) + 0.05 * std::cos(3.0 * x);
    conservative.state()[static_cast<std::size_t>(i)] = value;
    advective.state()[static_cast<std::size_t>(i)] = value;
  }
  const double defect = max_rate_gap(conservative, advective);
  measure("advective_point_defect", defect);
  REQUIRE(defect > 1.0e-4);
  REQUIRE_THAT(rate_sum(conservative), WithinAbs(0.0, 1.0e-10));
  REQUIRE_THAT(rate_sum(advective), WithinAbs(0.0, 1.0e-10));

  auto conservative_run = evolve(burgers::Method::fdm2, 32, 0.0, 0.001, 0.5);
  auto advective_run = evolve(burgers::Method::fdm2_advective, 32, 0.0, 0.001, 0.5);
  double separation = 0.0;
  for (int i = 0; i < 32; ++i) {
    separation = std::max(
        separation, std::abs(conservative_run.state()[static_cast<std::size_t>(i)] -
                             advective_run.state()[static_cast<std::size_t>(i)]));
  }
  measure("advective_evolved_separation", separation);
  measure("advective_mass",
          burgers::mass(advective_run.spacing(), advective_run.state()));
  measure("conservative_fd_mass",
          burgers::mass(conservative_run.spacing(), conservative_run.state()));
  REQUIRE(separation > 1.0e-4);
  REQUIRE_THAT(burgers::mass(advective_run.spacing(), advective_run.state()),
               WithinAbs(0.0, 1.0e-10));
}

TEST_CASE("Smooth Burgers flow matches the characteristic solution", "[burgers]") {
  constexpr double time = 0.5;
  constexpr double dt = 0.001;
  const auto spectral_32 = evolve(burgers::Method::spectral, 32, 0.0, dt, time);
  const auto spectral_64 = evolve(burgers::Method::spectral, 64, 0.0, dt, time);
  const auto fd_32 = evolve(burgers::Method::fdm2, 32, 0.0, dt, time);
  const auto fd_64 = evolve(burgers::Method::fdm2, 64, 0.0, dt, time);
  const auto fv_32 = evolve(burgers::Method::fvm, 32, 0.0, dt, time);
  const auto fv_64 = evolve(burgers::Method::fvm, 64, 0.0, dt, time);
  const Norms spectral_coarse = nodal_error(spectral_32, time);
  const Norms spectral_fine = nodal_error(spectral_64, time);
  const Norms fd_coarse = nodal_error(fd_32, time);
  const Norms fd_fine = nodal_error(fd_64, time);
  const Norms fv_coarse = cell_error(fv_32, time);
  const Norms fv_fine = cell_error(fv_64, time);
  measure("smooth_spectral_l2_32", spectral_coarse.l2);
  measure("smooth_spectral_l2_64", spectral_fine.l2);
  measure("smooth_fd2_l2_32", fd_coarse.l2);
  measure("smooth_fd2_l2_64", fd_fine.l2);
  measure("smooth_fd2_ratio", fd_coarse.l2 / fd_fine.l2);
  measure("smooth_fvm_l2_32", fv_coarse.l2);
  measure("smooth_fvm_l2_64", fv_fine.l2);
  measure("smooth_fvm_ratio", fv_coarse.l2 / fv_fine.l2);
  REQUIRE(spectral_coarse.l2 > 1.0e-14);
  REQUIRE(spectral_fine.l2 < spectral_coarse.l2 / 8.0);
  REQUIRE(fd_coarse.l2 / fd_fine.l2 > 3.2);
  REQUIRE(fd_coarse.l2 / fd_fine.l2 < 4.8);
  REQUIRE(fv_coarse.l2 / fv_fine.l2 > 1.6);
  REQUIRE(fv_coarse.l2 / fv_fine.l2 < 2.6);
  for (const burgers::Simulation *simulation : {&spectral_64, &fd_64, &fv_64}) {
    REQUIRE_THAT(burgers::mass(simulation->spacing(), simulation->state()),
                 WithinAbs(0.0, 1.0e-9));
  }
}

TEST_CASE("A shorter Burgers step reduces the smooth spectral error", "[burgers]") {
  // N=64 at t=0.5 is spatially small. dt=0.05 still resolves the retained
  // modes and puts the SSP-RK3 error above that floor; dt=0.02 did not.
  constexpr double time = 0.5;
  const auto coarse = evolve(burgers::Method::spectral, 64, 0.0, 0.05, time);
  const auto fine = evolve(burgers::Method::spectral, 64, 0.0, 0.025, time);
  const double coarse_error = nodal_error(coarse, time).l2;
  const double fine_error = nodal_error(fine, time).l2;
  measure("time_l2_dt", coarse_error);
  measure("time_l2_dt_half", fine_error);
  REQUIRE(std::isfinite(coarse_error));
  REQUIRE(coarse_error > 1.0e-8);
  REQUIRE(fine_error < 0.5 * coarse_error);
}

TEST_CASE("Viscous Burgers matches Cole-Hopf and the energy balance", "[burgers]") {
  constexpr double viscosity = 0.5;
  constexpr double time = 0.2;
  constexpr double dt = 0.0005;
  const auto spectral = evolve(burgers::Method::spectral, 64, viscosity, dt, time);
  const auto fd_coarse = evolve(burgers::Method::fdm2, 32, viscosity, dt, time);
  const auto fd_fine = evolve(burgers::Method::fdm2, 64, viscosity, dt, time);
  const auto fv_coarse = evolve(burgers::Method::fvm, 32, viscosity, dt, time);
  const auto fv_fine = evolve(burgers::Method::fvm, 64, viscosity, dt, time);
  const auto field_error = [&](const burgers::Simulation &simulation, bool cells) {
    double sum = 0.0;
    for (int i = 0; i < simulation.count(); ++i) {
      const double left = static_cast<double>(i) * simulation.spacing();
      const double reference =
          cells ? cell_average_of(
                      left, left + simulation.spacing(),
                      [&](double x) { return cole_hopf_u(x, time, viscosity); })
                : cole_hopf_u(left, time, viscosity);
      const double error =
          simulation.state()[static_cast<std::size_t>(i)] - reference;
      sum += error * error;
    }
    return std::sqrt(sum * simulation.spacing());
  };
  const double spectral_error = field_error(spectral, false);
  const double fd_coarse_error = field_error(fd_coarse, false);
  const double fd_fine_error = field_error(fd_fine, false);
  const double fv_coarse_error = field_error(fv_coarse, true);
  const double fv_fine_error = field_error(fv_fine, true);
  measure("viscous_spectral_l2", spectral_error);
  measure("viscous_fd2_ratio", fd_coarse_error / fd_fine_error);
  measure("viscous_fvm_l2_64", fv_fine_error);
  measure("viscous_fvm_ratio", fv_coarse_error / fv_fine_error);
  REQUIRE(spectral_error < 1.0e-8);
  REQUIRE(fd_coarse_error / fd_fine_error > 3.2);
  REQUIRE(fv_fine_error < fv_coarse_error);
  REQUIRE(fv_fine_error < 5.0e-2);

  burgers::Simulation balance(burgers::Method::spectral, 128, viscosity);
  balance.set_initial_sine();
  double previous = dissipation(balance.state(), balance.spacing(), viscosity);
  double accumulated = 0.0;
  const double energy_start = burgers::energy(balance.spacing(), balance.state());
  constexpr int samples = 200;
  constexpr double balance_dt = time / samples;
  for (int sample = 0; sample < samples; ++sample) {
    balance.advance(balance_dt, 1);
    const double current =
        dissipation(balance.state(), balance.spacing(), viscosity);
    accumulated += 0.5 * balance_dt * (previous + current);
    previous = current;
  }
  const double energy_end = burgers::energy(balance.spacing(), balance.state());
  const double residual = (energy_end - energy_start) + accumulated;
  measure("viscous_energy_drop", energy_start - energy_end);
  measure("viscous_energy_residual", residual);
  REQUIRE(energy_end < energy_start);
  REQUIRE(std::abs(residual) < 2.0e-3);
  REQUIRE_THAT(burgers::mass(balance.spacing(), balance.state()),
               WithinAbs(0.0, 1.0e-9));
}

TEST_CASE("Rusanov flux and a periodic Riemann problem stay entropy stable",
          "[burgers]") {
  REQUIRE_THAT(burgers::rusanov_flux(1.0, -1.0), WithinAbs(1.5, 1.0e-15));
  REQUIRE_THAT(burgers::rusanov_flux(1.0, 1.0), WithinAbs(0.5, 1.0e-15));
  REQUIRE_THAT(burgers::rusanov_flux(-1.0, 1.0), WithinAbs(-0.5, 1.0e-15));

  constexpr int count = 64;
  burgers::Simulation simulation(burgers::Method::fvm, count, 0.0);
  for (int i = 0; i < count; ++i) {
    simulation.state()[static_cast<std::size_t>(i)] = i < count / 2 ? 1.0 : -1.0;
  }
  simulation.advance(0.002, 150);
  REQUIRE(all_finite(simulation.state()));
  REQUIRE(simulation.state()[static_cast<std::size_t>(count / 4)] > 0.98);
  REQUIRE(simulation.state()[static_cast<std::size_t>(3 * count / 4)] < -0.98);
  int shock = 0;
  double jump = -1.0;
  for (int i = 0; i < count; ++i) {
    const double gap =
        std::abs(simulation.state()[static_cast<std::size_t>((i + 1) % count)] -
                 simulation.state()[static_cast<std::size_t>(i)]);
    if (gap > jump) {
      jump = gap;
      shock = i;
    }
  }
  const double shock_x = (static_cast<double>(shock) + 0.5) * simulation.spacing();
  measure("riemann_shock_x", shock_x);
  REQUIRE_THAT(shock_x, WithinAbs(pi, simulation.spacing()));
  REQUIRE_THAT(burgers::mass(simulation.spacing(), simulation.state()),
               WithinAbs(0.0, 1.0e-10));
}

TEST_CASE("The inviscid shock keeps finite-volume monotone and spectral oscillatory",
          "[burgers]") {
  constexpr double time = 1.5;
  constexpr double dt = 0.001;
  constexpr int count = 128;
  const auto spectral = evolve(burgers::Method::spectral, count, 0.0, dt, time);
  const auto finite_volume = evolve(burgers::Method::fvm, count, 0.0, dt, time);
  const auto fd = evolve(burgers::Method::fdm2, count, 0.0, dt, time);
  REQUIRE(all_finite(spectral.state()));
  REQUIRE(all_finite(finite_volume.state()));
  burgers::Simulation finite_volume_start(burgers::Method::fvm, count, 0.0);
  finite_volume_start.set_initial_sine();
  const double spectral_peak = burgers::maximum_abs(spectral.state());
  const double finite_volume_peak = burgers::maximum_abs(finite_volume.state());
  const double finite_volume_initial =
      burgers::maximum_abs(finite_volume_start.state());
  measure("shock_spectral_max", spectral_peak);
  measure("shock_fvm_max", finite_volume_peak);
  measure("shock_fvm_initial_max", finite_volume_initial);
  REQUIRE(finite_volume_peak <= finite_volume_initial + 1.0e-8);
  REQUIRE(spectral_peak > finite_volume_peak + 0.02);
  REQUIRE(spectral_peak > 1.02);

  const Norms spectral_error = nodal_error(spectral, time);
  const Norms finite_volume_error = cell_error(finite_volume, time);
  measure("shock_spectral_l1", spectral_error.l1);
  measure("shock_fvm_l1", finite_volume_error.l1);
  measure("shock_spectral_energy",
          burgers::energy(spectral.spacing(), spectral.state()));
  measure("shock_fvm_energy",
          burgers::energy(finite_volume.spacing(), finite_volume.state()));
  double reference_energy = 0.0;
  constexpr int panels = 256;
  const double panel = 2.0 * pi / panels;
  for (int i = 0; i < panels; ++i) {
    const double left = panel * i;
    const double right = left + panel;
    const double mid = 0.5 * (left + right);
    const double half = 0.5 * panel;
    double sum = 0.0;
    for (int point = 0; point < 8; ++point) {
      const double value = characteristic_u(mid + half * gauss_node[point], time);
      sum += gauss_weight[point] * 0.5 * value * value;
    }
    reference_energy += 0.5 * sum * panel;
  }
  measure("shock_entropy_energy", reference_energy);
  REQUIRE(reference_energy < 0.5 * pi - 0.05);
  REQUIRE(burgers::energy(finite_volume.spacing(), finite_volume.state()) <
          0.5 * pi - 0.02);
  REQUIRE(finite_volume_error.l1 < 0.25);

  int shock = 0;
  double jump = -1.0;
  for (int i = 0; i < finite_volume.count(); ++i) {
    const double gap =
        std::abs(finite_volume.state()[static_cast<std::size_t>((i + 1) % count)] -
                 finite_volume.state()[static_cast<std::size_t>(i)]);
    if (gap > jump) {
      jump = gap;
      shock = i;
    }
  }
  const double shock_x =
      (static_cast<double>(shock) + 0.5) * finite_volume.spacing();
  measure("shock_fvm_x", shock_x);
  REQUIRE_THAT(shock_x, WithinAbs(pi, finite_volume.spacing()));
  REQUIRE_THAT(burgers::mass(spectral.spacing(), spectral.state()),
               WithinAbs(0.0, 1.0e-8));
  REQUIRE_THAT(burgers::mass(finite_volume.spacing(), finite_volume.state()),
               WithinAbs(0.0, 1.0e-8));

  REQUIRE(all_finite(fd.state()));
  const double fd_peak = burgers::maximum_abs(fd.state());
  measure("shock_fd2_max", fd_peak);
  measure("shock_fd2_energy", burgers::energy(fd.spacing(), fd.state()));
  REQUIRE(fd_peak > finite_volume_peak + 0.01);
  REQUIRE_THAT(burgers::mass(fd.spacing(), fd.state()), WithinAbs(0.0, 1.0e-8));
}
