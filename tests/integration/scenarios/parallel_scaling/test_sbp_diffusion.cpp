// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <mpi.h>
#include <openpfc/kernel/decomposition/comm_sparse_exchange.hpp>
#include <openpfc/kernel/field/reaction_flux.hpp>
#include <openpfc/kernel/field/sbp_diffusion.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>
#include <set>
using pfc::field::fd::DiffusionAxis;
TEST_CASE("Physical SBP faces survive thin distributed partitions", "[mpi][sbp]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  constexpr int n = 33;
  DiffusionAxis axis(n, 1. / 32);
  std::vector<int> offsets(size + 1);
  offsets[0] = 0;
  offsets[size] = n;
  // Last rank owns one node, so it must obtain a full physical closure remotely.
  for (int p = 1; p < size; ++p) offsets[p] = (n - 1) * p / (size - 1);
  auto needs = [&](int p) {
    std::set<int> result;
    for (int i = offsets[p]; i < offsets[p + 1]; ++i)
      axis.coefficients(i, [&](int j, int k, double) {
        result.insert(j);
        result.insert(k);
      });
    return result;
  };
  std::vector<pfc::halo::RemoteHalo<double>> halos;
  auto local = needs(rank);
  for (int peer = 0; peer < size; ++peer)
    if (peer != rank) {
      std::vector<std::size_t> send, recv;
      for (int i : needs(peer))
        if (i >= offsets[rank] && i < offsets[rank + 1]) send.push_back(i);
      for (int i : local)
        if (i >= offsets[peer] && i < offsets[peer + 1]) recv.push_back(i);
      pfc::halo::RemoteHalo<double> h;
      h.peer_rank = peer;
      h.send_tag = 173;
      h.recv_tag = 173;
      h.send_values = pfc::core::SparseVector<pfc::backend::CPUTag, double>(send);
      h.recv_values = pfc::core::SparseVector<pfc::backend::CPUTag, double>(recv);
      h.scatter_after_recv = true;
      halos.push_back(std::move(h));
    }
  pfc::comm::SparseExchange<pfc::HostSpace, double> exchange(std::move(halos), rank,
                                                             MPI_COMM_WORLD);
  std::vector<double> u(n, std::numeric_limits<double>::quiet_NaN()), d = u;
  auto exact_u = [](int i) { return std::sin(i * .17) + i * .02; };
  auto exact_d = [](int i) { return 1. + i * .01; };
  for (int i = offsets[rank]; i < offsets[rank + 1]; ++i) {
    u[i] = exact_u(i);
    d[i] = exact_d(i);
  }
  pfc::communication::StagePreparationService<double> prepare;
  prepare.bind("u", [&] { exchange.exchange(std::span<double>(u)); });
  prepare.bind("D", [&] { exchange.exchange(std::span<double>(d)); });
  pfc::integrator::StageContext stage;
  stage.time = .3;
  stage.needs_halo_exchange = true;
  double prepared_time = -1;
  prepare.set_boundary_hook([&](std::string_view) { prepared_time = stage.time; });
  auto req = pfc::integrator::requirements_from(stage, true);
  req.ordering = pfc::communication::BoundaryHaloOrder::HaloThenBoundary;
  const std::array<std::string_view, 2> names{"u", "D"};
  prepare.prepare(req, names);
  REQUIRE(prepared_time == stage.time);
  pfc::field::fd::FaceCondition left{
      pfc::field::fd::BoundaryQuantity::ConstitutiveFlux,
      [](double t) { return t; }};
  pfc::field::fd::FaceCondition right{
      pfc::field::fd::BoundaryQuantity::ConstitutiveFlux,
      [](double t) { return -2 * t; }};
  double error = 0, mass = 0, scale = 1;
  for (int i = offsets[rank]; i < offsets[rank + 1]; ++i) {
    double value = axis.apply(
        i, [&](int j) { return u[j]; }, [&](int j) { return d[j]; }, .3, &left,
        &right);
    error = std::max(
        error, std::abs(value - axis.apply(i, exact_u, exact_d, .3, &left, &right)));
    mass += axis.weight(i) * value;
    scale += std::abs(axis.weight(i) * value);
  }
  double total, largest, total_scale;
  MPI_Allreduce(&mass, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&error, &largest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&scale, &total_scale, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(largest <= 1e-11 * total_scale);
  REQUIRE(std::abs(total - .3) <= 1e-11 * total_scale);
  // State-dependent reactions are evaluated only by physical endpoint owners.
  using namespace pfc::field::fd;
  FluxLedger ledger;
  ledger.begin();
  auto law = [](const ReactionPoint &p, double k) {
    return ReactionRate{k * p.state * p.state + p.stage_time * p.outward_normal[0]};
  };
  auto valid = [](const ReactionPoint &, double) { return true; };
  FaceCondition reaction_left, reaction_right;
  if (rank == 0) {
    reaction_left = reaction_face({u[0], {}, {0, 0, 0}, {-1, 0, 0}, stage.time}, .2,
                                  law, valid, ledger);
    ledger.stage(reaction_left.flux(stage.time, d[0]), 1, .1);
  }
  if (rank == size - 1) {
    reaction_right = reaction_face({u[n - 1], {}, {1, 0, 0}, {1, 0, 0}, stage.time},
                                   .2, law, valid, ledger);
    ledger.stage(reaction_right.flux(stage.time, d[n - 1]), 1, .1);
  }
  mass = 0;
  for (int i = offsets[rank]; i < offsets[rank + 1]; ++i)
    mass += axis.weight(i) * axis.apply(
                                 i, [&](int j) { return u[j]; },
                                 [&](int j) { return d[j]; }, stage.time,
                                 &reaction_left, &reaction_right);
  ledger.accept();
  double accepted = ledger.accepted(), integrated = 0;
  MPI_Allreduce(&mass, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&accepted, &integrated, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double exact_flux =
      .2 * (exact_u(0) * exact_u(0) + exact_u(n - 1) * exact_u(n - 1));
  REQUIRE(std::abs(total + exact_flux) <= 1e-11 * total_scale);
  REQUIRE(std::abs(integrated - .1 * exact_flux) <= 1e-11 * total_scale);
  REQUIRE(std::abs(.1 * total + integrated) <= 1e-11 * total_scale);
  ledger.begin();
  ledger.stage(1000, 1, 1);
  ledger.reject();
  REQUIRE(ledger.accepted() == accepted);
}

TEST_CASE("Supplied circular surface source has unique distributed owners",
          "[mpi][sbp][reaction]") {
  using namespace pfc::field::fd;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int n = 128;
  const double h = 2. / n, R = .3, eps = std::sqrt(R * h), dt = .01;
  FluxLedger ledger;
  ledger.begin();
  double change = 0, area = 0, serial = 0;
  auto law = [](const ReactionPoint &p, double) {
    return ReactionRate{p.state * p.state - 1 + p.stage_time};
  };
  auto valid = [](const ReactionPoint &p, double) { return p.state >= 0; };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double x = -1 + (i + .25) * h, y = -1 + (j + .25) * h, r = std::hypot(x, y),
             d = r - R;
      if (std::abs(d) >= eps) continue;
      double rho = (1 + std::cos(std::acos(-1.) * d / eps)) / (2 * eps);
      double u = 2 + .2 * x / r;
      serial += (u * u - 1 + .3) * rho * h * h * dt;
      if (i * size / n != rank) continue;
      ReactionPoint p{u, {}, {x, y, 0}, {x / r, y / r, 0}, .3};
      double rhs = reaction_source(p, rho, h * h, dt, 0., law, valid, ledger);
      change += rhs * h * h * dt;
      area += rho * h * h;
    }
  ledger.accept();
  double local[3] = {change, ledger.accepted(), area}, global[3];
  MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(std::abs(global[0] + global[1]) < 1e-12);
  REQUIRE(std::abs(global[1] - serial) < 1e-12);
  REQUIRE(std::abs(global[2] / (2 * std::acos(-1.) * R) - 1) < 1e-3);
  ledger.begin();
  reaction_source({2, {}, {0, 0, 0}, {.6, .8, 0}, 0}, 1, 1, .1, 0., law, valid,
                  ledger);
  ledger.reject();
  REQUIRE(ledger.accepted() == local[1]);
}
