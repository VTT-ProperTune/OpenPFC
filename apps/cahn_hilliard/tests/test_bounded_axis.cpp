// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cahn_hilliard/bounded_axis.hpp>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>

using Catch::Approx;

namespace {

double pi() { return std::acos(-1.0); }

double mode_rate(const cahn_hilliard::BoundedAxisRequest &model) {
  const double wave = pi();
  return model.mobility * wave * wave *
         (model.curvature + model.kappa * wave * wave);
}

double stable_dt(int intervals, const cahn_hilliard::BoundedAxisRequest &model) {
  const double spacing = 1.0 / static_cast<double>(intervals);
  const double wavenumber = 4.0 / spacing;
  const double second = wavenumber * wavenumber;
  const double lambda =
      model.mobility * second * (std::abs(model.curvature) + model.kappa * second);
  return 2.0 / lambda;
}

void fill_mode(const pfc::field::fd::DiffusionAxis &axis, std::vector<double> &field,
               const cahn_hilliard::BoundedAxisRequest &model, double amplitude,
               double time) {
  const int n = axis.size();
  const double spacing = 1.0 / static_cast<double>(n - 1);
  const double decay = std::exp(-mode_rate(model) * time);
  for (int i = 0; i < n; ++i) {
    field[static_cast<std::size_t>(i)] =
        model.equilibrium +
        amplitude * std::cos(pi() * static_cast<double>(i) * spacing) * decay;
  }
}

double inventory(const pfc::field::fd::DiffusionAxis &axis,
                 const std::vector<double> &field, int begin, int end) {
  double sum = 0.0;
  for (int i = begin; i < end; ++i)
    sum += axis.weight(i) * field[static_cast<std::size_t>(i)];
  return sum;
}

double mode_error(const pfc::field::fd::DiffusionAxis &axis,
                  const std::vector<double> &field,
                  const cahn_hilliard::BoundedAxisRequest &model, double amplitude,
                  double time) {
  const int n = axis.size();
  const double spacing = 1.0 / static_cast<double>(n - 1);
  const double decay = std::exp(-mode_rate(model) * time);
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double exact =
        model.equilibrium +
        amplitude * std::cos(pi() * static_cast<double>(i) * spacing) * decay;
    const double diff = field[static_cast<std::size_t>(i)] - exact;
    error += axis.weight(i) * diff * diff;
  }
  return std::sqrt(error);
}

double laplacian_error(const pfc::field::fd::DiffusionAxis &axis,
                       const std::vector<double> &field, double amplitude) {
  const int n = axis.size();
  const double spacing = 1.0 / static_cast<double>(n - 1);
  const double wave2 = pi() * pi();
  double error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double exact =
        -wave2 * amplitude * std::cos(pi() * static_cast<double>(i) * spacing);
    const double got = cahn_hilliard::bounded_axis_laplacian(
        axis, i, [&](int j) { return field[static_cast<std::size_t>(j)]; }, 0.0);
    const double diff = got - exact;
    error += axis.weight(i) * diff * diff;
  }
  return std::sqrt(error);
}

cahn_hilliard::BoundedAxisStep
step_all(const pfc::field::fd::DiffusionAxis &axis, std::vector<double> &field,
         const cahn_hilliard::BoundedAxisRequest &model, double time, double dt,
         pfc::field::fd::FluxLedger &ledger) {
  return cahn_hilliard::advance_bounded_axis(
      axis, 0, axis.size(),
      [&](int i) { return field[static_cast<std::size_t>(i)]; },
      [&](int i, double value) { field[static_cast<std::size_t>(i)] = value; },
      [] {}, model, time, dt, ledger);
}

} // namespace

TEST_CASE("Linear Cahn-Hilliard no-flux axis conserves a decaying mode",
          "[bounded]") {
  cahn_hilliard::BoundedAxisRequest model;
  model.mobility = 1.0;
  model.kappa = 1.0e-4;
  model.curvature = 1.0;
  model.equilibrium = 0.5;
  const double amplitude = 0.05;
  const double horizon = 0.02;
  double previous_lap = 0.0;
  double previous_field = 0.0;
  for (int intervals : {32, 64}) {
    const int n = intervals + 1;
    pfc::field::fd::DiffusionAxis axis(n, 1.0 / static_cast<double>(intervals));
    std::vector<double> field(static_cast<std::size_t>(n));
    fill_mode(axis, field, model, amplitude, 0.0);
    const double lap = laplacian_error(axis, field, amplitude);
    const double idle = mode_error(axis, field, model, amplitude, horizon);
    const double dt_limit = stable_dt(intervals, model);
    const int steps = static_cast<int>(std::ceil(horizon / dt_limit));
    const double dt = horizon / static_cast<double>(steps);
    const double initial = inventory(axis, field, 0, n);
    pfc::field::fd::FluxLedger ledger;
    double reported = 0.0;
    for (int step = 0; step < steps; ++step) {
      const auto taken = step_all(axis, field, model, step * dt, dt, ledger);
      reported += taken.owned_inventory_change;
    }
    const double mass = inventory(axis, field, 0, n) - initial;
    const double field_error = mode_error(axis, field, model, amplitude, horizon);
    std::ostringstream record;
    record << std::setprecision(17) << "CH_BOUNDED noflux n=" << n
           << " steps=" << steps << " lap=" << lap << " field=" << field_error
           << " idle=" << idle << " mass=" << mass
           << " ledger=" << ledger.accepted();
    std::cout << record.str() << '\n';
    REQUIRE(std::isfinite(lap));
    REQUIRE(std::isfinite(field_error));
    REQUIRE(lap < 5.0e-4);
    REQUIRE(field_error < 1.0e-6);
    REQUIRE(field_error < 0.01 * idle);
    REQUIRE(std::abs(mass) < 1.0e-12);
    REQUIRE(std::abs(ledger.accepted()) < 1.0e-14);
    REQUIRE(reported == Approx(mass).margin(1.0e-12));
    if (previous_lap > 0.0) {
      REQUIRE(lap < previous_lap / 4.0);
      REQUIRE(field_error < previous_field / 8.0);
    }
    previous_lap = lap;
    previous_field = field_error;
  }

  pfc::field::fd::DiffusionAxis axis(33, 1.0 / 32.0);
  std::vector<double> flat(33, 0.5);
  cahn_hilliard::BoundedAxisRequest rest = model;
  pfc::field::fd::FluxLedger ledger;
  const double dt = stable_dt(32, rest);
  step_all(axis, flat, rest, 0.0, dt, ledger);
  for (double value : flat) REQUIRE(value == Approx(0.5).margin(1.0e-12));
  REQUIRE(ledger.accepted() == Approx(0.0).margin(1.0e-15));
}

TEST_CASE("Boundary sink inventory matches the imposed Cahn-Hilliard flux",
          "[bounded]") {
  cahn_hilliard::BoundedAxisRequest model;
  model.mobility = 1.0;
  model.kappa = 1.0e-4;
  model.curvature = 1.0;
  model.equilibrium = 0.5;
  model.outward_sink = 0.15;
  const double amplitude = 0.05;
  const double horizon = 0.02;
  constexpr int intervals = 32;
  const int n = intervals + 1;
  pfc::field::fd::DiffusionAxis axis(n, 1.0 / static_cast<double>(intervals));
  std::vector<double> field(static_cast<std::size_t>(n));
  fill_mode(axis, field, model, amplitude, 0.0);
  const double initial = inventory(axis, field, 0, n);
  const double dt_limit = stable_dt(intervals, model);
  const int steps = static_cast<int>(std::ceil(horizon / dt_limit));
  const double dt = horizon / static_cast<double>(steps);
  pfc::field::fd::FluxLedger ledger;
  for (int step = 0; step < steps; ++step)
    step_all(axis, field, model, step * dt, dt, ledger);
  const double mass = inventory(axis, field, 0, n) - initial;
  const double imposed = 2.0 * model.outward_sink * horizon;
  std::ostringstream record;
  record << std::setprecision(17) << "CH_BOUNDED sink mass=" << mass
         << " ledger=" << ledger.accepted() << " imposed=" << imposed;
  std::cout << record.str() << '\n';
  REQUIRE(ledger.accepted() == Approx(imposed).margin(1.0e-12));
  REQUIRE(mass == Approx(-imposed).margin(1.0e-12));
  REQUIRE(inventory(axis, field, 0, n) < initial);

  model.outward_sink = 0.0;
  model.exchange = 0.5;
  std::vector<double> rich(static_cast<std::size_t>(n), 0.7);
  const double rich_initial = inventory(axis, rich, 0, n);
  pfc::field::fd::FluxLedger exchange_ledger;
  for (int step = 0; step < 20; ++step)
    step_all(axis, rich, model, step * dt, dt, exchange_ledger);
  const double rich_mass = inventory(axis, rich, 0, n) - rich_initial;
  REQUIRE(exchange_ledger.accepted() > 1.0e-6);
  REQUIRE(rich_mass == Approx(-exchange_ledger.accepted()).margin(1.0e-10));
  REQUIRE(inventory(axis, rich, 0, n) < rich_initial);

  pfc::field::fd::DiffusionAxis small(17, 1.0 / 16.0);
  std::vector<double> sample(17, 0.4);
  pfc::field::fd::FluxLedger unused;
  cahn_hilliard::BoundedAxisRequest bad = model;
  bad.mobility = 0.0;
  REQUIRE_THROWS_AS(step_all(small, sample, bad, 0.0, dt, unused),
                    std::invalid_argument);
  bad.mobility = 1.0;
  bad.kappa = -1.0;
  REQUIRE_THROWS_AS(step_all(small, sample, bad, 0.0, dt, unused),
                    std::invalid_argument);
  bad.kappa = 1.0e-4;
  REQUIRE_THROWS_AS(step_all(small, sample, bad, 0.0, 0.0, unused),
                    std::invalid_argument);
  REQUIRE(unused.accepted() == 0.0);
  REQUIRE_FALSE(unused.active());
  for (double value : sample) REQUIRE(value == 0.4);
}

TEST_CASE("Rejected bounded RK stages restore the owned field and ledger",
          "[bounded]") {
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) SKIP("single-rank rollback injection");
  constexpr int intervals = 32;
  constexpr int n = intervals + 1;
  pfc::field::fd::DiffusionAxis axis(n, 1.0 / intervals);
  cahn_hilliard::BoundedAxisRequest model;
  model.mobility = 1.0;
  model.kappa = 1.0e-4;
  model.curvature = 1.0;
  model.equilibrium = 0.5;
  std::vector<double> field(static_cast<std::size_t>(n));
  fill_mode(axis, field, model, 0.05, 0.0);
  const auto before = field;
  pfc::field::fd::FluxLedger ledger;
  int exchanges = 0;
  const auto refresh = [&] {
    ++exchanges;
    if (exchanges == 2) throw std::runtime_error("injected halo failure");
  };
  const auto read = [&](int i) { return field[static_cast<std::size_t>(i)]; };
  const auto write = [&](int i, double value) {
    field[static_cast<std::size_t>(i)] = value;
  };
  const double dt = stable_dt(intervals, model);
  using cahn_hilliard::advance_bounded_axis;
  const auto failure = [&] {
    advance_bounded_axis(axis, 0, n, read, write, refresh, model, 0.0, dt, ledger);
  };
  REQUIRE_THROWS_AS(failure(), std::runtime_error);
  REQUIRE(exchanges == 2);
  REQUIRE(field == before);
  REQUIRE_FALSE(ledger.active());
  REQUIRE(ledger.accepted() == 0.0);
}

TEST_CASE("Bounded Cahn-Hilliard is independent of the owned split",
          "[bounded][mpi]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  constexpr int intervals = 32;
  constexpr int n = intervals + 1;
  pfc::field::fd::DiffusionAxis axis(n, 1.0 / static_cast<double>(intervals));
  cahn_hilliard::BoundedAxisRequest model;
  model.mobility = 1.0;
  model.kappa = 1.0e-4;
  model.curvature = 1.0;
  model.equilibrium = 0.5;
  model.outward_sink = 0.05;
  const double amplitude = 0.05;
  const double dt = stable_dt(intervals, model);
  constexpr int steps = 4;

  std::vector<double> serial(static_cast<std::size_t>(n));
  fill_mode(axis, serial, model, amplitude, 0.0);
  pfc::field::fd::FluxLedger serial_ledger;
  for (int step = 0; step < steps; ++step)
    step_all(axis, serial, model, step * dt, dt, serial_ledger);

  std::vector<int> offsets;
  offsets.reserve(static_cast<std::size_t>(size) + 1U);
  offsets.push_back(0);
  if (size > 1) {
    for (int part = 1; part < size; ++part)
      offsets.push_back((n - 1) * part / (size - 1));
  }
  offsets.push_back(n);
  const int begin = offsets[static_cast<std::size_t>(rank)];
  const int end = offsets[static_cast<std::size_t>(rank + 1)];

  std::vector<double> local(static_cast<std::size_t>(n),
                            std::numeric_limits<double>::quiet_NaN());
  for (int i = begin; i < end; ++i)
    local[static_cast<std::size_t>(i)] =
        model.equilibrium + amplitude * std::cos(pi() * static_cast<double>(i) /
                                                 static_cast<double>(intervals));

  std::vector<pfc::halo::RemoteHalo<double>> halos;
  for (int peer = 0; peer < size; ++peer) {
    if (peer == rank) continue;
    std::vector<std::size_t> send;
    std::vector<std::size_t> recv;
    for (int i = begin; i < end; ++i) send.push_back(static_cast<std::size_t>(i));
    const int peer_begin = offsets[static_cast<std::size_t>(peer)];
    const int peer_end = offsets[static_cast<std::size_t>(peer + 1)];
    for (int i = peer_begin; i < peer_end; ++i)
      recv.push_back(static_cast<std::size_t>(i));
    pfc::halo::RemoteHalo<double> halo;
    halo.peer_rank = peer;
    halo.send_tag = 253;
    halo.recv_tag = 253;
    halo.send_values = pfc::core::SparseVector<pfc::backend::CPUTag, double>(send);
    halo.recv_values = pfc::core::SparseVector<pfc::backend::CPUTag, double>(recv);
    halo.scatter_after_recv = true;
    halos.push_back(std::move(halo));
  }
  pfc::comm::SparseExchange<pfc::HostSpace, double> exchange(std::move(halos), rank,
                                                             MPI_COMM_WORLD);
  auto refresh = [&] { exchange.exchange(std::span<double>(local)); };

  pfc::field::fd::FluxLedger ledger;
  for (int step = 0; step < steps; ++step) {
    cahn_hilliard::advance_bounded_axis(
        axis, begin, end, [&](int i) { return local[static_cast<std::size_t>(i)]; },
        [&](int i, double value) { local[static_cast<std::size_t>(i)] = value; },
        refresh, model, step * dt, dt, ledger);
  }

  double local_gap = 0.0;
  for (int i = begin; i < end; ++i) {
    local_gap = std::max(local_gap, std::abs(local[static_cast<std::size_t>(i)] -
                                             serial[static_cast<std::size_t>(i)]));
  }
  const double owned_mass = inventory(axis, local, begin, end);
  const double accepted = ledger.accepted();
  double global_gap = 0.0;
  double total_mass = 0.0;
  double total_accepted = 0.0;
  MPI_Allreduce(&local_gap, &global_gap, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&owned_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&accepted, &total_accepted, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double serial_mass = inventory(axis, serial, 0, n);
  const double imposed = 2.0 * model.outward_sink * steps * dt;
  REQUIRE(global_gap < 1.0e-11);
  REQUIRE(total_mass == Approx(serial_mass).margin(1.0e-11));
  REQUIRE(total_accepted == Approx(serial_ledger.accepted()).margin(1.0e-11));
  REQUIRE(total_accepted == Approx(imposed).margin(1.0e-12));
  REQUIRE(std::abs(total_mass - inventory(axis, serial, 0, n) +
                   serial_ledger.accepted() - total_accepted) < 1.0e-11);
}
